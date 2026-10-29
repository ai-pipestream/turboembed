use crate::{Error, Result};

/// Explicit contract for how dependency relation strings encode pseudo-projective paths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabelEncoding {
    /// Relation labels are literal strings; only in-memory `projectivize`
    /// sentinel frames are decoded.
    Plain,
    /// Percent-escaped pseudo-projective encoding (`%25`, `%5C`, `%5E`, `%3C`,
    /// `%3E`, `%3A`).
    #[default]
    EscapedV1,
}

impl LabelEncoding {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::EscapedV1 => "escaped-v1",
        }
    }
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim() {
            "plain" => Ok(Self::Plain),
            "escaped-v1" => Ok(Self::EscapedV1),
            other => Err(Error(format!("unknown depparse label.encoding {other}"))),
        }
    }
}
