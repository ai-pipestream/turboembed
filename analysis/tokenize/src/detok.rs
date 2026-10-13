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

/// What a dictionary says about a token: OpenNLP's
/// `DetokenizationDictionary.Operation`. The detokenizer resolves it to an
/// [`Operation`] for each occurrence of the token.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DictionaryOperation {
    /// Attach to the token on the left. `MOVE_LEFT` in the XML.
    MoveLeft,
    /// Attach to the token on the right. `MOVE_RIGHT` in the XML.
    MoveRight,
    /// Attach to the tokens on both sides. `MOVE_BOTH` in the XML.
    MoveBoth,
    /// Attach to the right on the first occurrence of the token and to
    /// the left on the next, alternating. `RIGHT_LEFT_MATCHING` in the
    /// XML.
    RightLeftMatching,
}

impl DictionaryOperation {
    /// The operation an `operation` attribute of the dictionary XML
    /// names, compared exactly, as `DetokenizationDictionary.Operation.parse`
    /// does. `None` for any other value.
    pub fn from_attribute(value: &str) -> Option<DictionaryOperation> {
        match value {
            "MOVE_LEFT" => Some(DictionaryOperation::MoveLeft),
            "MOVE_RIGHT" => Some(DictionaryOperation::MoveRight),
            "MOVE_BOTH" => Some(DictionaryOperation::MoveBoth),
            "RIGHT_LEFT_MATCHING" => Some(DictionaryOperation::RightLeftMatching),
            _ => None,
        }
    }

    /// The value of the `operation` attribute that names this operation.
    pub fn attribute(self) -> &'static str {
        match self {
            DictionaryOperation::MoveLeft => "MOVE_LEFT",
            DictionaryOperation::MoveRight => "MOVE_RIGHT",
            DictionaryOperation::MoveBoth => "MOVE_BOTH",
            DictionaryOperation::RightLeftMatching => "RIGHT_LEFT_MATCHING",
        }
    }
}

/// How one occurrence of a token attaches to its neighbours: OpenNLP's
/// `DetokenizationOperation`, what the detokenizer reports per token.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Attach to the token on the left. `MERGE_TO_LEFT` in OpenNLP.
    MergeToLeft,
    /// Attach to the token on the right. `MERGE_TO_RIGHT` in OpenNLP.
    MergeToRight,
    /// Attach to the tokens on both sides. `MERGE_BOTH` in OpenNLP.
    MergeBoth,
    /// No attachment of its own, for a token the dictionary does not hold;
    /// a neighbour may still attach to it. `NO_OPERATION` in OpenNLP.
    NoOperation,
}

impl Operation {
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

    const ALL: [DictionaryOperation; 4] = [
        DictionaryOperation::MoveLeft,
        DictionaryOperation::MoveRight,
        DictionaryOperation::MoveBoth,
        DictionaryOperation::RightLeftMatching,
    ];

    #[test]
    fn attribute_names_round_trip() {
        for op in ALL {
            assert_eq!(DictionaryOperation::from_attribute(op.attribute()), Some(op));
        }
    }

    #[test]
    fn attribute_names_are_exact() {
        assert_eq!(DictionaryOperation::from_attribute("MOVE_LEFT"), Some(DictionaryOperation::MoveLeft));
        assert_eq!(DictionaryOperation::from_attribute("move_left"), None);
        assert_eq!(DictionaryOperation::from_attribute(" MOVE_LEFT"), None);
        assert_eq!(DictionaryOperation::from_attribute("NO_OPERATION"), None);
        assert_eq!(DictionaryOperation::from_attribute("MERGE_TO_LEFT"), None);
        assert_eq!(DictionaryOperation::from_attribute(""), None);
    }

    #[test]
    fn merge_directions() {
        let all = [Operation::MergeToLeft, Operation::MergeToRight, Operation::MergeBoth, Operation::NoOperation];
        let left: Vec<bool> = all.iter().map(|o| o.merges_left()).collect();
        let right: Vec<bool> = all.iter().map(|o| o.merges_right()).collect();
        assert_eq!(left, [true, false, true, false]);
        assert_eq!(right, [false, true, true, false]);
    }
}
