//! Whose kernels run a session's linear layers, per bin of a run's packed
//! tokens, and the choices line that names them (docs/levelzero.md,
//! Kernel choices).

use super::Linear;
use crate::tuning::tcap_bin;

/// The token bins, as the choices and timings lines name them.
pub(super) const BINS: [&str; 5] = ["le256", "le1k", "le4k", "le16k", "gt16k"];

/// The linear layers' kernels of each token bin.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Linears(pub(super) [Linear; 5]);

impl Linears {
    pub(super) fn all(l: Linear) -> Linears {
        Linears([l; 5])
    }

    /// The kernels a run of `tokens` packed tokens takes.
    #[cfg_attr(not(feature = "levelzero-onednn"), allow(dead_code))]
    pub(super) fn at(self, tokens: u32) -> Linear {
        self.0[tcap_bin(tokens) as usize]
    }

    /// Whether a run at any bin takes `l`.
    pub(super) fn uses(self, l: Linear) -> bool {
        self.0.contains(&l)
    }

    /// The choices line of a session whose bins go up to `top`:
    /// `linear=` the top bin's kernels, then `<bin>:linear=` each bin below
    /// it that differs, then `forced=`, naming `linear` when the
    /// environment fixed it.
    pub(super) fn line(self, top: u32, forced: bool) -> String {
        let base = self.0[top as usize];
        let mut line = format!("linear={}", base.name());
        for (bin, l) in BINS.iter().zip(self.0).take(top as usize).filter(|&(_, l)| l != base) {
            line.push_str(&format!(";{bin}:linear={}", l.name()));
        }
        line + if forced { ";forced=linear" } else { ";forced=" }
    }
}

/// What a choices line names: the kernels of every bin (`linear=`) and of
/// single bins (`<bin>:linear=`), which win over the first whatever their
/// order.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(super) struct Picks {
    all: Option<Linear>,
    bins: [Option<Linear>; 5],
}

impl Picks {
    /// Each bin's kernels: the line's, else `default`'s.
    pub(super) fn or(self, default: Linears) -> Linears {
        let mut out = default;
        for (b, o) in out.0.iter_mut().enumerate() {
            *o = self.bins[b].or(self.all).unwrap_or(*o);
        }
        out
    }
}

/// The linear items of a choices line, among `;`-separated items:
/// `linear=own` or `linear=onednn`, alone or after a bin and a colon
/// (`le256:linear=own`). A `forced=` item, as a session reports it, is
/// passed over. None when it names no linear item.
pub(super) fn parse_linears(line: &str) -> Result<Option<Picks>, String> {
    let mut picks = Picks::default();
    let mut any = false;
    for item in line.split(';').filter(|i| !i.is_empty() && !i.starts_with("forced=")) {
        let (bin, choice) = match item.split_once(':') {
            Some((bin, choice)) => (Some(bin), choice),
            None => (None, item),
        };
        let l = match choice.split_once('=') {
            Some(("linear", "own")) => Linear::Own,
            Some(("linear", "onednn")) => Linear::Onednn,
            _ => return Err(format!("{item:?} is not linear=own or linear=onednn, alone or after a bin")),
        };
        match bin {
            None => picks.all = Some(l),
            Some(bin) => match BINS.iter().position(|&b| b == bin) {
                Some(b) => picks.bins[b] = Some(l),
                None => return Err(format!("{item:?} names no token bin; the bins are {}", BINS.join(", "))),
            },
        }
        any = true;
    }
    Ok(any.then_some(picks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use Linear::*;

    fn parsed(line: &str) -> Linears {
        parse_linears(line).unwrap().unwrap().or(Linears::all(Own))
    }

    #[test]
    fn a_line_names_the_top_bin_then_each_bin_below_that_differs() {
        let l = Linears([Own, Onednn, Onednn, Onednn, Onednn]);
        assert_eq!(l.line(3, false), "linear=onednn;le256:linear=own;forced=");
        assert_eq!(l.line(0, false), "linear=own;forced=");
        assert_eq!(Linears::all(Onednn).line(4, true), "linear=onednn;forced=linear");
        // A bin past the top is not the session's and is not named.
        assert_eq!(Linears([Onednn, Onednn, Own, Own, Own]).line(1, false), "linear=onednn;forced=");
    }

    #[test]
    fn a_reported_line_parses_back_to_the_same_choices() {
        for l in [Linears([Own, Onednn, Own, Onednn, Onednn]), Linears::all(Own), Linears::all(Onednn)] {
            assert_eq!(parsed(&l.line(4, false)), l);
            assert_eq!(parsed(&l.line(4, true)), l);
        }
    }

    #[test]
    fn a_line_of_one_choice_holds_for_every_bin_and_a_bin_wins_over_it() {
        assert_eq!(parsed("linear=onednn;forced="), Linears::all(Onednn));
        assert_eq!(parsed("le1k:linear=onednn;linear=own"), Linears([Own, Onednn, Own, Own, Own]));
        // A bin alone leaves the others as they were.
        let only = parse_linears("gt16k:linear=onednn").unwrap().unwrap();
        assert_eq!(only.or(Linears::all(Own)), Linears([Own, Own, Own, Own, Onednn]));
        assert_eq!(parse_linears("forced=linear").unwrap(), None);
        assert_eq!(parse_linears("").unwrap(), None);
    }

    #[test]
    fn anything_else_is_refused() {
        for line in ["linear=fast", "le2k:linear=own", "attn=auto", "le256:linear", "linear"] {
            assert!(parse_linears(line).is_err(), "{line:?}");
        }
    }

    #[test]
    fn a_run_takes_its_tokens_bin() {
        let l = Linears([Own, Onednn, Own, Onednn, Own]);
        let at: Vec<Linear> = [1, 256, 257, 1024, 1025, 4096, 4097, 16384, 16385].map(|t| l.at(t)).to_vec();
        assert_eq!(at, [Own, Own, Onednn, Onednn, Own, Own, Onednn, Onednn, Own]);
        assert!(l.uses(Onednn) && Linears::all(Own).uses(Own) && !Linears::all(Own).uses(Onednn));
    }
}
