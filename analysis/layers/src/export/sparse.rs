use std::marker::PhantomData;

use super::*;
use crate::SparseTerms;
use turbo_analysis::Span;

impl<L: Layer> LayerCodec for SparseTerms<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(malformed)?;
        let mut w = Writer::new();
        put_u32s(&mut w, &self.row)?;
        put_u32s(&mut w, &self.term)?;
        w.usize(self.weight.len())?;
        for weight in &self.weight {
            w.u32(weight.to_bits());
        }
        w.usize(self.span.len())?;
        for span in &self.span {
            w.u8(u8::from(span.is_some()));
            if let Some(span) = span {
                w.u32(span.start);
                w.u32(span.end);
            }
        }
        put_u32s(&mut w, &self.occ_at)?;
        w.usize(self.occ.len())?;
        for span in &self.occ {
            w.u32(span.start);
            w.u32(span.end);
        }
        put_model(&mut w, &self.model)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let row = get_u32s(&mut r)?;
        let term = get_u32s(&mut r)?;
        let mut weight = Vec::new();
        for _ in 0..r.usize()? {
            weight.push(f32::from_bits(r.u32()?));
        }
        let mut span = Vec::new();
        for _ in 0..r.usize()? {
            span.push(match r.u8()? {
                0 => None,
                1 => Some(Span { start: r.u32()?, end: r.u32()? }),
                tag => return Err(malformed(format!("a sparse optional span tag of {tag}"))),
            });
        }
        let occ_at = get_u32s(&mut r)?;
        let mut occ = Vec::new();
        for _ in 0..r.usize()? {
            occ.push(Span { start: r.u32()?, end: r.u32()? });
        }
        let model = get_model(&mut r)?;
        done(&r, "sparse terms")?;
        let terms = SparseTerms { row, term, weight, span, occ_at, occ, model, _of: PhantomData };
        terms.check().map_err(malformed)?;
        Ok(terms)
    }
}
