//! The document graph codec for this crate's layer, so the stopword
//! flags cross the export boundary with the rest of the graph.

use turbo_analysis::export::{Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, Reader, Writer};
use turbo_analysis::{Document, Layer};
use turbo_analysis_layers::Tokens;

use crate::{StopwordLanguage, Stopwords};

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

/// The item count of the already restored `Target` instance an
/// aligned layer on this record is aligned to: the instance the
/// record's provenance resolves `Target` to, else the default.
fn aligned_len<Target: Layer>(cx: &ImportCx) -> Result<usize, ExportError> {
    let name = Target::NAME.to_string();
    let instance = cx.resolved_instance(&name).to_owned();
    cx.layer_len(&LayerRef { layer: name, instance })
}

impl LayerCodec for Stopwords {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.u8(self.language as u8);
        w.usize(self.stop.len())?;
        for &stop in &self.stop {
            w.u8(stop as u8);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let at = r.u8()? as usize;
        let language =
            StopwordLanguage::ALL.get(at).copied().ok_or_else(|| malformed(format!("a stopword language of {at}")))?;
        let mut stop = Vec::new();
        for _ in 0..r.usize()? {
            stop.push(match r.u8()? {
                0 => false,
                1 => true,
                tag => return Err(malformed(format!("a boolean tag of {tag}"))),
            });
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a stopwords payload"));
        }
        let want = aligned_len::<Tokens>(cx)?;
        if stop.len() != want {
            return Err(malformed(format!("{} stopword flags for {want} tokens", stop.len())));
        }
        let layer = Stopwords { language, stop };
        layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
        Ok(layer)
    }
}

/// Registers this crate's layer codec on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<Stopwords>();
}

/// A registry holding this crate's codec and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
