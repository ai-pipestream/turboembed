//! The wire form of `Edus`: the unit spans, then each unit's sentence.
//! The `Discourse` and `ClaimLinks` edges use the core edge codec.

use super::*;
use crate::rhetoric::Edus;

impl LayerCodec for Edus {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        for s in &self.sentence {
            w.u32(s.get() as u32);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans: turbo_analysis::Spans<Edus> = get_spans(&mut r, cx)?;
        let mut sentence = Vec::with_capacity(spans.len());
        for _ in 0..spans.len() {
            sentence.push(Ix::new(r.u32()?));
        }
        done(&r, "discourse units")?;
        let out = Edus { spans, sentence };
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
