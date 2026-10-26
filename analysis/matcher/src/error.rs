//! Why a pattern was refused, and why matching a document failed.

use std::fmt;

use turbo_analysis::ViewId;

use crate::attr::Field;

/// A pattern or phrase the matcher cannot take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatternError {
    /// A pattern with no token, or a phrase with no token.
    Empty,
    /// A repeat whose most is below its least, or zero.
    Repeat { min: u32, max: u32 },
    /// A pattern whose repeats expand to more states than the matcher
    /// takes ([`crate::Matcher::MAX_STATES`]).
    TooLarge { states: usize },
    /// A phrase over a field a phrase's own text cannot give: a
    /// stopword or number flag, which come from a layer, or whether a
    /// token starts a sentence or is followed by a space.
    NotFromText(Field),
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternError::Empty => f.write_str("a pattern needs at least one token"),
            PatternError::Repeat { min, max } => write!(f, "a repeat of at least {min} and at most {max}"),
            PatternError::TooLarge { states } => write!(f, "a pattern of {states} states is too large"),
            PatternError::NotFromText(field) => write!(f, "a phrase's text gives no {field:?}"),
        }
    }
}

impl std::error::Error for PatternError {}

/// Why a document could not be matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchError {
    /// A pattern reads a layer the input does not have: `opennlp:pos` for a
    /// part-of-speech test, `turbo:stopwords` for the stopword flag.
    Missing(&'static str),
    /// A layer given to the input that is not one item per token.
    Misaligned { layer: &'static str, len: usize, tokens: usize },
    /// A span-bearing layer (`opennlp:entities`, `turbo:numbers`, or
    /// `opennlp:tokens`) belongs to a different view than the tokens.
    ViewMismatch { layer: &'static str, expected: ViewId, actual: ViewId },
    /// A token span outside the text, or off a character boundary.
    Token { index: usize, start: u32, end: u32 },
    /// A token range past the input's tokens.
    Range { start: usize, end: usize, tokens: usize },
    /// A label the matcher does not have.
    NoLabel(String),
    /// Invalid dependency columns, forest, or sentence alignment.
    Dependencies(String),
}

impl fmt::Display for MatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchError::Missing(layer) => write!(f, "a pattern reads {layer}, which the input does not have"),
            MatchError::Misaligned { layer, len, tokens } => {
                write!(f, "{layer} has {len} items for {tokens} tokens")
            }
            MatchError::ViewMismatch { layer, expected, actual } => {
                write!(f, "{layer} is in view {}, expected view {}", actual.0, expected.0)
            }
            MatchError::Token { index, start, end } => {
                write!(f, "token {index} at {start}..{end} is outside the text or cuts a character")
            }
            MatchError::Range { start, end, tokens } => {
                write!(f, "tokens {start}..{end} are not inside the {tokens} tokens")
            }
            MatchError::Dependencies(reason) => write!(f, "invalid dependency forest: {reason}"),
            MatchError::NoLabel(label) => write!(f, "no label {label:?}"),
        }
    }
}

impl std::error::Error for MatchError {}
