//! Glossary readers: delimited term lists and TBX termbases, and the TBX
//! writer the readers are checked against.

mod delimited;
mod tbx;
mod write;
mod xml;

use turbo_analysis::Span;
use turbo_glossary_layers::Glossary;

pub use delimited::{DelimitedReader, Delimiter, FirstRecord};
pub use tbx::TbxReader;
pub use write::{TbxDialect, WriteError, write_tbx};

use crate::error::ReadError;

/// The entries of a glossary file, in file order, and where each entry's
/// term is in the file: a byte span per entry. For a term written with
/// escapes (a doubled quote, an XML reference) the span covers what is
/// written, not the term it stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlossaryFile {
    pub glossary: Glossary,
    pub term_at: Vec<Span>,
}

/// The UTF-8 BOM.
const BOM: &[u8] = b"\xEF\xBB\xBF";

/// The text of `bytes` without a byte order mark, and the number of bytes
/// the mark took. Anything that is not UTF-8 is refused.
fn decode(bytes: &[u8]) -> Result<(&str, u32), ReadError> {
    let base = if bytes.starts_with(BOM) { BOM.len() } else { 0 };
    match std::str::from_utf8(&bytes[base..]) {
        Ok(text) => Ok((text, base as u32)),
        Err(e) => Err(ReadError::Utf8 { offset: base + e.valid_up_to() }),
    }
}
