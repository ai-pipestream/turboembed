//! Legal corpus readers with exact spans in their source formats.

mod bouvier;
mod caselaw;
mod json;
mod text;

pub use bouvier::BouvierReader;
pub use caselaw::CaselawReader;

/// The deepest nesting of JSON objects and arrays a legal reader accepts.
/// Deeper input is refused with [`LegalError::TooDeep`] instead of exhausting the stack.
pub const MAX_JSON_DEPTH: usize = 128;

/// A malformed legal corpus document or one missing a required field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LegalError {
    Json { offset: usize, expected: &'static str },
    Markup { offset: usize, expected: &'static str },
    MissingField(&'static str),
    TooDeep { offset: usize, limit: usize },
}

impl std::fmt::Display for LegalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LegalError::Json { offset, expected } => write!(f, "JSON byte {offset}: expected {expected}"),
            LegalError::Markup { offset, expected } => write!(f, "HTML byte {offset}: expected {expected}"),
            LegalError::MissingField(field) => write!(f, "missing required field {field}"),
            LegalError::TooDeep { offset, limit } => write!(f, "JSON byte {offset}: nesting deeper than {limit}"),
        }
    }
}

impl std::error::Error for LegalError {}
