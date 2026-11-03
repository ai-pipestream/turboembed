//! Why a dictionary, a pattern or a finder was refused.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A token pattern the matcher cannot take. `Tok` lists a pattern is
    /// built from refuse through this variant.
    Pattern(turbo_matcher::PatternError),
    /// Matching a document failed: a `Tok` tests an attribute the token
    /// layer does not carry, such as `Attr::Pos` over bare tokens.
    Match(turbo_matcher::MatchError),
    /// A dictionary line the line format refuses. `at` is the 0-based
    /// line number.
    Dictionary { at: usize, message: &'static str },
    /// `DictionaryNameFinder::new` over a dictionary with no entries:
    /// there is no type a name could carry.
    EmptyDictionary,
    /// `RegexNameFinder(Pattern[], String)` with no patterns.
    NoPatterns,
    /// The same entity type twice in what stands for a map.
    DuplicateType(Box<str>),
    /// More entity types than a `TagSet` holds.
    TooManyTypes,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Pattern(e) => write!(f, "token pattern: {e}"),
            Error::Match(e) => write!(f, "token pattern match: {e}"),
            Error::Dictionary { at, message } => write!(f, "dictionary line {at}: {message}"),
            Error::EmptyDictionary => f.write_str("a dictionary finder needs at least one entry"),
            Error::NoPatterns => f.write_str("patterns must not be empty"),
            Error::DuplicateType(t) => write!(f, "the entity type {t:?} is given twice"),
            Error::TooManyTypes => f.write_str("more than 65536 entity types"),
        }
    }
}

impl std::error::Error for Error {}

impl From<turbo_matcher::PatternError> for Error {
    fn from(e: turbo_matcher::PatternError) -> Error {
        Error::Pattern(e)
    }
}
