//! Opening a document from bytes: settle the encoding, decode, keep the
//! bytes below the text, and record the encoding as `turbo:encoding`.

use std::sync::Arc;

use turbo_analysis::{Document, Error, Key, Provenance};
use turbo_encoding_layers::{Encoding, EncodingEvidence, SourceEncoding};

use crate::decode::{bom_sniff, decode};
use crate::detect::detect;

/// The annotator name in the provenance of the `turbo:encoding` layer.
pub const OPEN_ID: &str = "turbo:encoding:open";

/// The encoding `bytes` are read with, in this order: a byte order mark;
/// the `declared` label; otherwise the detector, which prefers valid
/// UTF-8 to every legacy encoding and reads pure ASCII as UTF-8.
pub fn settle(bytes: &[u8], declared: Option<Encoding>) -> SourceEncoding {
    if let Some((encoding, _)) = bom_sniff(bytes) {
        return SourceEncoding { encoding, confidence: 1.0, evidence: EncodingEvidence::Bom };
    }
    if let Some(encoding) = declared {
        return SourceEncoding { encoding, confidence: 1.0, evidence: EncodingEvidence::Declared };
    }
    let d = detect(bytes);
    let evidence = match d.encoding {
        Encoding::Utf8 if d.non_ascii => EncodingEvidence::ValidUtf8,
        Encoding::Utf8 => EncodingEvidence::Ascii,
        _ => EncodingEvidence::Detected,
    };
    SourceEncoding { encoding: d.encoding, confidence: d.confidence, evidence }
}

/// Decodes `bytes` into a document: view 0 is the text, the bytes sit
/// below it (`Document::offset_to_source_bytes` maps any offset back to
/// them), and the `turbo:encoding` layer records the encoding used and
/// why. Fails only for input over `u32::MAX` bytes, or text that would
/// be (`Error::Source`).
pub fn open(bytes: impl Into<Arc<[u8]>>, declared: Option<Encoding>) -> Result<Document, Error> {
    let bytes: Arc<[u8]> = bytes.into();
    let settled = settle(&bytes, declared);
    let decoded = decode(&bytes, settled.encoding).map_err(Error::Source)?;
    let mut doc = decoded.into_document(bytes)?;
    doc.add(&Key::<SourceEncoding>::DEFAULT, settled, Provenance::by(OPEN_ID))?;
    Ok(doc)
}
