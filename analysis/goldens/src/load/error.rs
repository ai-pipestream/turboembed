//! Why a golden layer could not be loaded onto a document.

use std::fmt;

use turbo_analysis::KeyId;

use crate::compare::Mismatch;
use crate::hex::to_hex;

/// Why a golden layer was not loaded. Each names the layer, as the
/// document names it (`opennlp:pos@golden`), or the golden block.
#[derive(Clone, Debug, PartialEq)]
pub enum LoadError {
    /// The text is not the golden's input.
    Input { expected_len: u32, found_len: usize, expected_sha256: [u8; 32], found_sha256: [u8; 32] },
    /// The golden file has no block for the layer.
    MissingGolden { layer: KeyId, available: Vec<String> },
    /// The golden block's anchor or columns are not the layer type's.
    Shape(Mismatch),
    /// The layer is aligned to, or indexes, a layer that is not on the
    /// document under the same key name.
    MissingAnchor { layer: KeyId, anchor: KeyId },
    /// Row `row` has a token range that ends before it starts, or past the
    /// `tokens` tokens the anchor has.
    TokenRange { layer: KeyId, row: usize, start: u32, end: u32, tokens: usize },
    /// Token `token`, with the source span `start..end`, is not inside a
    /// sentence of the sentences loaded under the same name, taken in text
    /// order.
    OutsideSentences { layer: KeyId, token: usize, start: u32, end: u32 },
    /// A golden block with more distinct labels than a tag set holds.
    Labels { layer: KeyId, column: &'static str, count: usize },
    /// A document layer whose golden block has no row.
    Empty { layer: KeyId },
    /// A golden block of a layer type nothing loads.
    NoLoader { layer: String },
    /// The document refused the layer.
    Document(turbo_analysis::Error),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Input { expected_len, found_len, expected_sha256, found_sha256 } => write!(
                f,
                "the text is not the golden's input: {found_len} bytes with SHA-256 {}, expected {expected_len} \
                 bytes with SHA-256 {}",
                to_hex(found_sha256),
                to_hex(expected_sha256)
            ),
            LoadError::MissingGolden { layer, available } => {
                write!(f, "{layer}: the golden has no such layer; it has [{}]", available.join(", "))
            }
            LoadError::Shape(m) => write!(f, "{m}"),
            LoadError::MissingAnchor { layer, anchor } => {
                write!(f, "{layer}: the document has no {anchor} layer to load it onto; load that first")
            }
            LoadError::TokenRange { layer, row, start, end, tokens } => write!(
                f,
                "{layer} row {row}: the token range {start}..{end} is not inside the {tokens} tokens it indexes"
            ),
            LoadError::OutsideSentences { layer, token, start, end } => write!(
                f,
                "{layer} token {token}: bytes {start}..{end} are not inside a sentence, in text order, of the \
                 sentences loaded under the same name"
            ),
            LoadError::Labels { layer, column, count } => {
                write!(f, "{layer}: {count} distinct values in {column}, more than a tag set holds")
            }
            LoadError::Empty { layer } => write!(f, "{layer}: the golden layer has no row"),
            LoadError::NoLoader { layer } => write!(f, "{layer}: no layer type loads this golden layer"),
            LoadError::Document(e) => write!(f, "the document refused the layer: {e}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Document(e) => Some(e),
            _ => None,
        }
    }
}
