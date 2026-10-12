//! The chunker's context generator and sequence validator
//! (`opennlp.tools.chunker`). `ChunkerFactory` has no other choice, so
//! every chunker model uses these.

use crate::beam::{ContextGenerator, SequenceValidator};
use crate::features::Features;

/// The model entry of the classifier.
pub const MODEL_ENTRY_NAME: &str = "chunker.model";
/// `ChunkerME.DEFAULT_BEAM_SIZE`.
pub const DEFAULT_BEAM_SIZE: usize = 10;

/// A token and its part-of-speech tag: `TokenTag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenTag<'a> {
    pub token: &'a str,
    pub tag: &'a str,
}

/// `DefaultChunkerContextGenerator`: the words and tags in a window of two
/// on each side, the two previous chunk tags, and their combinations, 41
/// features in a fixed order.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultChunkerContext;

/// One of the window's features before it is combined: a name such as
/// `w_1` and either a word, a tag, an outcome, or a boundary marker
/// (`bos`, `eos`).
#[derive(Clone, Copy)]
struct Part<'a> {
    name: &'static str,
    value: &'a str,
}

impl DefaultChunkerContext {
    /// Writes `name=value` for each part, concatenated, as one feature.
    fn add(out: &mut Features, parts: &[Part<'_>]) {
        for p in parts {
            out.push_str(p.name).push_str("=").push_str(p.value);
        }
        out.finish();
    }

    /// The features of item `i`.
    pub fn features(&self, out: &mut Features, i: usize, sequence: &[TokenTag<'_>], prior: &[&str]) {
        let word = |name, at: Option<usize>, eos| Part {
            name,
            value: match at {
                Some(j) if j < sequence.len() => sequence[j].token,
                _ => eos,
            },
        };
        let tag = |name, at: Option<usize>, eos| Part {
            name,
            value: match at {
                Some(j) if j < sequence.len() => sequence[j].tag,
                _ => eos,
            },
        };
        let w_2 = word("w_2", i.checked_sub(2), "bos");
        let w_1 = word("w_1", i.checked_sub(1), "bos");
        let w0 = word("w0", Some(i), "");
        let w1 = word("w1", Some(i + 1), "eos");
        let w2 = word("w2", Some(i + 2), "eos");
        let t_2 = tag("t_2", i.checked_sub(2), "bos");
        let t_1 = tag("t_1", i.checked_sub(1), "bos");
        let t0 = tag("t0", Some(i), "");
        let t1 = tag("t1", Some(i + 1), "eos");
        let t2 = tag("t2", Some(i + 2), "eos");
        let p_2 = Part { name: "p_2", value: i.checked_sub(2).map_or("bos", |j| prior[j]) };
        let p_1 = Part { name: "p_1", value: i.checked_sub(1).map_or("bos", |j| prior[j]) };

        for parts in [
            &[w_2][..],
            &[w_1],
            &[w0],
            &[w1],
            &[w2],
            &[w_1, w0],
            &[w0, w1],
            &[t_2],
            &[t_1],
            &[t0],
            &[t1],
            &[t2],
            &[t_2, t_1],
            &[t_1, t0],
            &[t0, t1],
            &[t1, t2],
            &[t_2, t_1, t0],
            &[t_1, t0, t1],
            &[t0, t1, t2],
            &[p_2],
            &[p_1],
            &[p_2, p_1],
            &[p_1, t_2],
            &[p_1, t_1],
            &[p_1, t0],
            &[p_1, t1],
            &[p_1, t2],
            &[p_1, t_2, t_1],
            &[p_1, t_1, t0],
            &[p_1, t0, t1],
            &[p_1, t1, t2],
            &[p_1, t_2, t_1, t0],
            &[p_1, t_1, t0, t1],
            &[p_1, t0, t1, t2],
            &[p_1, w_2],
            &[p_1, w_1],
            &[p_1, w0],
            &[p_1, w1],
            &[p_1, w2],
            &[p_1, w_1, w0],
            &[p_1, w0, w1],
        ] {
            Self::add(out, parts);
        }
    }
}

impl<'a> ContextGenerator<TokenTag<'a>> for DefaultChunkerContext {
    fn context(&mut self, index: usize, sequence: &[TokenTag<'a>], prior: &[&str], out: &mut Features) {
        self.features(out, index, sequence, prior);
    }
}

/// `DefaultChunkerSequenceValidator`: an `I-` tag must follow a tag other
/// than `O` whose text after its first two characters is the same (an
/// `I-NP` follows a `B-NP` or an `I-NP`).
///
/// Java takes the previous tag's `substring(2)` and throws when that tag
/// is shorter than two UTF-16 units; here such a tag simply does not
/// match.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkerValidator;

impl<T> SequenceValidator<T> for ChunkerValidator {
    fn valid_sequence(&self, _index: usize, _sequence: &[T], prior: &[&str], outcome: &str) -> bool {
        if !outcome.starts_with("I-") {
            return true;
        }
        match prior.last() {
            None => false,
            Some(&"O") => false,
            Some(previous) => match skip_two_units(previous) {
                Some(rest) => rest == &outcome[2..],
                None => false,
            },
        }
    }
}

/// `s.substring(2)`, or `None` when `s` has fewer than two UTF-16 units
/// (Java throws) or the cut falls inside a surrogate pair (the rest starts
/// with a lone low surrogate, which equals no Rust string, so `None` gives
/// the same answer).
fn skip_two_units(s: &str) -> Option<&str> {
    let mut units = 0;
    for (at, c) in s.char_indices() {
        if units == 2 {
            return Some(&s[at..]);
        }
        units += c.len_utf16();
    }
    (units == 2).then_some("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_has_41_features() {
        let seq = [
            TokenTag { token: "He", tag: "PRP" },
            TokenTag { token: "reckons", tag: "VBZ" },
            TokenTag { token: "the", tag: "DT" },
        ];
        let mut out = Features::new();
        DefaultChunkerContext.features(&mut out, 1, &seq, &["B-NP"]);
        assert_eq!(out.len(), 41);
        let f = |i| out.to_string_lossy(i);
        assert_eq!(f(0), "w_2=bos");
        assert_eq!(f(1), "w_1=He");
        assert_eq!(f(4), "w2=eos");
        assert_eq!(f(5), "w_1=Hew0=reckons");
        assert_eq!(f(21), "p_2=bosp_1=B-NP");
        assert_eq!(f(40), "p_1=B-NPw0=reckonsw1=the");
    }

    #[test]
    fn inside_tags_continue_their_chunk() {
        let v = ChunkerValidator;
        let ok = |prior: &[&str], o: &str| SequenceValidator::<()>::valid_sequence(&v, 0, &[], prior, o);
        assert!(ok(&[], "B-NP"));
        assert!(!ok(&[], "I-NP"));
        assert!(!ok(&["O"], "I-NP"));
        assert!(ok(&["B-NP"], "I-NP"));
        assert!(!ok(&["B-VP"], "I-NP"));
        assert!(!ok(&["B"], "I-NP"));
    }
}
