//! OpenNLP's `DictionaryDetokenizer`: rebuilds running text from tokens.
//!
//! A detokenizer dictionary gives some tokens an operation that says how
//! the token attaches to its neighbours: a full stop attaches to the
//! token on its left, an opening bracket to the token on its right, a
//! hyphen to both, and a straight quote alternates, attaching to the
//! right the first time it is seen and to the left the next time. Every
//! other token is separated from its neighbours by one space.
//!
//! - [`dictionary`]: [`DetokenizerDictionary`], read from OpenNLP's
//!   dictionary XML or built entry by entry.
//! - [`detokenizer`]: [`Detokenizer`], which resolves the operation of
//!   each token and writes the rebuilt text, optionally with a split
//!   marker wherever two tokens were joined without a space. Its offset
//!   form also records where each token starts in the rebuilt text, so
//!   the text maps back to the tokens it came from.
//!
//! Detokenization produces text, not a layer, so there is no annotator
//! here. A `Tokens` layer keeps its spans into the original view, so a
//! pipeline never needs to rebuild the text it started from; the
//! detokenizer is for tokens that arrive without that text, such as the
//! output of a model or a training corpus.

pub mod detokenizer;
pub mod dictionary;

pub use detokenizer::Detokenizer;
pub use dictionary::{DetokenizerDictionary, DictionaryError};

/// How a token attaches to its neighbours.
///
/// A dictionary maps tokens to [`MergeToLeft`](Operation::MergeToLeft),
/// [`MergeToRight`](Operation::MergeToRight),
/// [`MergeBoth`](Operation::MergeBoth) or
/// [`RightLeftMatching`](Operation::RightLeftMatching). The detokenizer
/// resolves `RightLeftMatching` to `MergeToRight` or `MergeToLeft` for
/// each occurrence, and gives `NoOperation` to every token the dictionary
/// does not hold, so the operations it reports never contain
/// `RightLeftMatching`.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Attach to the token on the left. `MOVE_LEFT` in the XML,
    /// `MERGE_TO_LEFT` in OpenNLP's results.
    MergeToLeft,
    /// Attach to the token on the right. `MOVE_RIGHT` in the XML,
    /// `MERGE_TO_RIGHT` in OpenNLP's results.
    MergeToRight,
    /// Attach to the tokens on both sides. `MOVE_BOTH` in the XML,
    /// `MERGE_BOTH` in OpenNLP's results.
    MergeBoth,
    /// Attach to the right on the first occurrence of the token and to
    /// the left on the next, alternating. `RIGHT_LEFT_MATCHING` in the
    /// XML; only ever found in a dictionary.
    RightLeftMatching,
    /// No attachment of its own; a neighbour may still attach to this
    /// token. `NO_OPERATION` in OpenNLP's results; never in the XML.
    NoOperation,
}

impl Operation {
    /// The operation an `operation` attribute of the dictionary XML
    /// names, compared exactly, as `DetokenizationDictionary.Operation.parse`
    /// does. `None` for any other value.
    pub fn from_attribute(value: &str) -> Option<Operation> {
        match value {
            "MOVE_LEFT" => Some(Operation::MergeToLeft),
            "MOVE_RIGHT" => Some(Operation::MergeToRight),
            "MOVE_BOTH" => Some(Operation::MergeBoth),
            "RIGHT_LEFT_MATCHING" => Some(Operation::RightLeftMatching),
            _ => None,
        }
    }

    /// The value of the `operation` attribute that names this operation,
    /// or `None` for [`NoOperation`](Operation::NoOperation), which the
    /// XML cannot express.
    pub fn attribute(self) -> Option<&'static str> {
        match self {
            Operation::MergeToLeft => Some("MOVE_LEFT"),
            Operation::MergeToRight => Some("MOVE_RIGHT"),
            Operation::MergeBoth => Some("MOVE_BOTH"),
            Operation::RightLeftMatching => Some("RIGHT_LEFT_MATCHING"),
            Operation::NoOperation => None,
        }
    }

    /// Whether the token attaches to the token on its left.
    pub(crate) fn merges_left(self) -> bool {
        matches!(self, Operation::MergeToLeft | Operation::MergeBoth)
    }

    /// Whether the token attaches to the token on its right.
    pub(crate) fn merges_right(self) -> bool {
        matches!(self, Operation::MergeToRight | Operation::MergeBoth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Operation; 5] = [
        Operation::MergeToLeft,
        Operation::MergeToRight,
        Operation::MergeBoth,
        Operation::RightLeftMatching,
        Operation::NoOperation,
    ];

    #[test]
    fn attribute_names_round_trip() {
        for op in ALL {
            match op.attribute() {
                Some(name) => assert_eq!(Operation::from_attribute(name), Some(op)),
                None => assert_eq!(op, Operation::NoOperation),
            }
        }
    }

    #[test]
    fn attribute_names_are_exact() {
        assert_eq!(Operation::from_attribute("MOVE_LEFT"), Some(Operation::MergeToLeft));
        assert_eq!(Operation::from_attribute("move_left"), None);
        assert_eq!(Operation::from_attribute(" MOVE_LEFT"), None);
        assert_eq!(Operation::from_attribute("NO_OPERATION"), None);
        assert_eq!(Operation::from_attribute(""), None);
    }

    #[test]
    fn merge_directions() {
        let left: Vec<bool> = ALL.iter().map(|o| o.merges_left()).collect();
        let right: Vec<bool> = ALL.iter().map(|o| o.merges_right()).collect();
        assert_eq!(left, [true, false, true, false, false]);
        assert_eq!(right, [false, true, true, false, false]);
    }
}
