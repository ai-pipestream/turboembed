//! The document graph codec for this crate's layer, so a document's
//! encoding crosses the export boundary with the rest of the graph.

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, Reader, Writer, get_variant, put_variant,
};
use turbo_analysis::{Document, Layer};

use crate::{Encoding, EncodingEvidence, SourceEncoding};

/// The evidence kinds in wire order: a variant crosses as its position
/// here, so a position outside the table is refused rather than read
/// as some other kind.
const EVIDENCES: [EncodingEvidence; 5] = [
    EncodingEvidence::Bom,
    EncodingEvidence::Declared,
    EncodingEvidence::ValidUtf8,
    EncodingEvidence::Ascii,
    EncodingEvidence::Detected,
];

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

impl LayerCodec for SourceEncoding {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_variant(&mut w, &Encoding::ALL, self.encoding, "encoding")?;
        w.u32(self.confidence.to_bits());
        put_variant(&mut w, &EVIDENCES, self.evidence, "encoding evidence")?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let layer = SourceEncoding {
            encoding: get_variant(&mut r, &Encoding::ALL, "encoding")?,
            confidence: f32::from_bits(r.u32()?),
            evidence: get_variant(&mut r, &EVIDENCES, "encoding evidence")?,
        };
        if !r.is_done() {
            return Err(malformed("trailing bytes in a source encoding payload"));
        }
        // The layer's own invariants settle the rest: the confidence
        // finite and in [0, 1], and the evidence agreeing with the
        // encoding and the confidence.
        layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
        Ok(layer)
    }
}

/// Registers this crate's layer codec on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<SourceEncoding>();
}

/// A registry holding this crate's codec and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
