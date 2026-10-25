use crate::Decisions;
use turbo_analysis::export::{Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef};
use turbo_analysis::{Document, Layer};
use turbo_analysis_layers::Entities;
impl LayerCodec for Decisions {
    fn export(&self, _: &ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(ExportError::Malformed)?;
        let mut out = self.threshold.to_le_bytes().to_vec();
        for &v in &self.logits {
            out.extend(v.to_le_bytes());
        }
        Ok(out)
    }
    fn import(data: &[u8], cx: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        if data.len() < 4 || !data.len().is_multiple_of(4) || data.len() > 4 * (1 + 384 * 32 * 32) {
            return Err(ExportError::Malformed("gliner decision payload size limit".into()));
        }
        let mut values = data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b));
        let out = Self { threshold: values.next().unwrap(), logits: values.collect() };
        out.check().map_err(ExportError::Malformed)?;
        // One decision per entity of the instance this layer is aligned to.
        let name = Entities::NAME.to_string();
        let entities = LayerRef { instance: cx.resolved_instance(&name).to_owned(), layer: name };
        let len = cx.layer_len(&entities)?;
        if len != out.len() {
            return Err(ExportError::InvalidLayer {
                layer: cx.layer().to_owned(),
                detail: format!("{} decisions for the {len} entities of {entities}", out.len()),
            });
        }
        Ok(out)
    }
}

/// Registers the codec for this crate's own layer, [`Decisions`]; the
/// entities it is aligned to cross with `turbo_analysis_layers`' codecs.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<Decisions>();
}

/// The shared layers' codecs plus this crate's.
pub fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}
