use crate::Decisions;
use turbo_analysis::export::{ExportCx, ExportError, ImportCx, LayerCodec};
use turbo_analysis::{Document, Layer};
impl LayerCodec for Decisions {
    fn export(&self, _: &ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(ExportError::Malformed)?;
        let mut out = self.threshold.to_le_bytes().to_vec();
        for &v in &self.logits {
            out.extend(v.to_le_bytes());
        }
        Ok(out)
    }
    fn import(data: &[u8], _: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        if data.len() < 4 || !data.len().is_multiple_of(4) || data.len() > 4 * (1 + 384 * 32 * 32) {
            return Err(ExportError::Malformed("gliner decision payload size limit".into()));
        }
        let mut values = data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b));
        let out = Self { threshold: values.next().unwrap(), logits: values.collect() };
        out.check().map_err(ExportError::Malformed)?;
        Ok(out)
    }
}
