//! Legal corpus readers with exact spans in their source formats.

mod bouvier;
mod caselaw;
mod json;
mod text;

pub use bouvier::BouvierReader;
pub use caselaw::CaselawReader;

/// A malformed legal corpus document or one missing a required field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LegalError {
    Json { offset: usize, expected: &'static str },
    Markup { offset: usize, expected: &'static str },
    MissingField(&'static str),
}

impl std::fmt::Display for LegalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LegalError::Json { offset, expected } => write!(f, "JSON byte {offset}: expected {expected}"),
            LegalError::Markup { offset, expected } => write!(f, "HTML byte {offset}: expected {expected}"),
            LegalError::MissingField(field) => write!(f, "missing required field {field}"),
        }
    }
}

impl std::error::Error for LegalError {}
