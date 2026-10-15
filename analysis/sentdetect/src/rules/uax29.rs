//! The UAX #29 sentence boundary rules SB1 to SB998 over a sequence of
//! classified characters.
//!
//! Rule SB5 attaches `Extend` and `Format` characters to the character
//! before them, unless that is the start of the text or a paragraph
//! separator. The other rules then look only at the characters that are
//! not attached, here called bases. The pattern `SATerm Close* Sp*` that
//! rules SB8 to SB11 share is tracked as a state over the bases seen so
//! far, so a text is one pass; the look-ahead of SB8 remembers where its
//! last scan stopped, so the scans never cover the same character twice.

use turbo_unicode::SentenceBreak::{self, *};

use super::units::{Unit, is_para_sep};

/// No terminator: the break is a paragraph break (SB4).
pub(crate) const NO_TERM: u32 = u32::MAX;

/// A sentence boundary before unit `at`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Break {
    /// The unit the next sentence starts with.
    pub at: u32,
    /// The last terminator of the `SATerm Close* Sp*` run the break
    /// follows (SB11), or [`NO_TERM`] after a paragraph separator (SB4).
    pub term: u32,
}

/// Where the bases are in the `SATerm Close* Sp*` pattern.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Phase {
    /// Not in it.
    Out,
    /// Right after the terminator.
    Term,
    /// In the closing punctuation after it.
    Close,
    /// In the spaces after that.
    Sp,
}

/// Whether SB8's look-ahead stops at this class.
#[inline]
fn stops_sb8(class: SentenceBreak) -> bool {
    matches!(class, OLetter | Upper | Lower | ATerm | STerm) || is_para_sep(class)
}

/// Appends to `out` every boundary inside `units` (the ones at the start
/// and end of the text, SB1 and SB2, are left out), in text order.
pub(crate) fn boundaries(units: &[Unit], out: &mut Vec<Break>) {
    out.clear();
    let n = units.len();
    if n == 0 {
        return;
    }
    // The last two bases before the current one, and the pattern state
    // after the last one.
    let mut prev: SentenceBreak = units[0].class;
    let mut prev2: Option<SentenceBreak> = None;
    let (mut phase, mut term_class, mut term) = (Phase::Out, Other, NO_TERM);
    advance(units[0].class, 0, &mut phase, &mut term_class, &mut term);
    // The unit SB8's last scan stopped at (or `n`), and the first unit it
    // covered.
    let mut sb8_from = usize::MAX;
    let mut sb8_stop = n;
    let mut i = 1;
    while i < n {
        let class = units[i].class;
        // SB3 and SB4.
        if prev == Cr && class == Lf {
            prev = Lf;
            prev2 = Some(Cr);
            (phase, term) = (Phase::Out, NO_TERM);
            i += 1;
            continue;
        }
        if is_para_sep(prev) {
            out.push(Break { at: i as u32, term: NO_TERM });
            prev2 = Some(prev);
            prev = class;
            (phase, term) = (Phase::Out, NO_TERM);
            advance(class, i, &mut phase, &mut term_class, &mut term);
            i += 1;
            continue;
        }
        // SB5: attached to the base before.
        if matches!(class, Extend | Format) {
            i += 1;
            continue;
        }
        // Each rule that keeps the characters together, in order; past
        // them, SB10 keeps a space or paragraph separator and SB11 breaks.
        let sb8 = |sb8_from: &mut usize, sb8_stop: &mut usize| {
            if !(*sb8_from <= i && i <= *sb8_stop) {
                *sb8_from = i;
                *sb8_stop = (i..n).find(|&j| stops_sb8(units[j].class)).unwrap_or(n);
            }
            *sb8_stop < n && units[*sb8_stop].class == Lower
        };
        let keep = phase == Phase::Out
            || (prev == ATerm && class == Numeric) // SB6
            || (prev == ATerm && class == Upper && matches!(prev2, Some(Upper | Lower))) // SB7
            || (term_class == ATerm && sb8(&mut sb8_from, &mut sb8_stop)) // SB8
            || matches!(class, SContinue | ATerm | STerm) // SB8a
            || (matches!(phase, Phase::Term | Phase::Close) && class == Close) // SB9
            || class == Sp
            || is_para_sep(class); // SB9, SB10
        if !keep {
            out.push(Break { at: i as u32, term });
            (phase, term) = (Phase::Out, NO_TERM);
        }
        prev2 = Some(prev);
        prev = class;
        advance(class, i, &mut phase, &mut term_class, &mut term);
        i += 1;
    }
}

/// Moves the `SATerm Close* Sp*` state over one more base.
#[inline]
fn advance(class: SentenceBreak, i: usize, phase: &mut Phase, term_class: &mut SentenceBreak, term: &mut u32) {
    *phase = match (class, *phase) {
        (ATerm | STerm, _) => {
            *term_class = class;
            *term = i as u32;
            Phase::Term
        }
        (Close, Phase::Term | Phase::Close) => Phase::Close,
        (Sp, Phase::Term | Phase::Close | Phase::Sp) => Phase::Sp,
        _ => {
            *term = NO_TERM;
            Phase::Out
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breaks(classes: &[SentenceBreak]) -> Vec<(u32, u32)> {
        let units: Vec<Unit> = classes.iter().enumerate().map(|(i, &class)| Unit { start: i as u32, class }).collect();
        let mut out = Vec::new();
        boundaries(&units, &mut out);
        out.iter().map(|b| (b.at, b.term)).collect()
    }

    #[test]
    fn terminators_spaces_and_paragraphs() {
        // "a. B": SB11 before B, after the terminator at 1.
        assert_eq!(breaks(&[Lower, ATerm, Sp, Upper]), [(3, 1)]);
        // "a. b": SB8.
        assert_eq!(breaks(&[Lower, ATerm, Sp, Lower]), []);
        // "a! b": STerm has no SB8.
        assert_eq!(breaks(&[Lower, STerm, Sp, Lower]), [(3, 1)]);
        // "a\nb": SB4.
        assert_eq!(breaks(&[Lower, Lf, Lower]), [(2, NO_TERM)]);
        // "a.\r\nb": SB9 keeps the CR, SB3 the LF, SB4 breaks after it.
        assert_eq!(breaks(&[Lower, ATerm, Cr, Lf, Lower]), [(4, NO_TERM)]);
        // "3.4" SB6, "U.S" SB7, "a.." SB8a.
        assert_eq!(breaks(&[Numeric, ATerm, Numeric]), []);
        assert_eq!(breaks(&[Upper, ATerm, Upper]), []);
        assert_eq!(breaks(&[Lower, ATerm, ATerm, Sp, Upper]), [(4, 2)]);
        // An Extend after a terminator is attached to it.
        assert_eq!(breaks(&[Lower, STerm, Extend, Sp, Upper]), [(4, 1)]);
    }
}
