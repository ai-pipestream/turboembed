use super::*;

impl<L: Layer> LayerCodec for Vectors<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.u32(self.dim);
        w.usize(self.len())?;
        w.usize(self.data.len())?;
        for v in &self.data {
            w.u32(v.to_bits());
        }
        w.usize(self.empty.len())?;
        for e in &self.empty {
            w.u32(e.get() as u32);
        }
        put_model(&mut w, &self.model)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let dim = r.u32()?;
        if dim == 0 {
            return Err(malformed("a vector dimension of zero"));
        }
        let rows = r.usize()?;
        let want = rows.checked_mul(dim as usize).ok_or_else(|| malformed("a vector block that overflows"))?;
        let count = r.usize()?;
        if count != want {
            return Err(malformed(format!("{count} vector values for {rows} rows of dimension {dim}")));
        }
        let mut data = Vec::new();
        for _ in 0..count {
            data.push(f32::from_bits(r.u32()?));
        }
        let mut empty = Vec::new();
        for _ in 0..r.usize()? {
            let e = r.u32()?;
            if e as usize >= rows {
                return Err(malformed(format!("an empty row {e} outside {rows} rows")));
            }
            if empty.last().is_some_and(|&last: &Ix<L>| last.get() as u32 >= e) {
                return Err(malformed("empty rows out of order or repeated"));
            }
            empty.push(Ix::new(e));
        }
        let model = get_model(&mut r)?;
        done(&r, "vectors")?;
        Ok(Vectors::new(dim, data, empty, model))
    }
}

impl<L: Layer> LayerCodec for Confidence<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_f64s(&mut w, &self.value)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let value = get_f64s(&mut r)?;
        done(&r, "confidence")?;
        Ok(Confidence::new(value))
    }
}

impl<L: Layer> LayerCodec for Utf16Spans<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_u32s(&mut w, &self.start16)?;
        put_u32s(&mut w, &self.end16)?;
        put_bools(&mut w, &self.cut)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let start16 = get_u32s(&mut r)?;
        let end16 = get_u32s(&mut r)?;
        let cut = get_bools(&mut r)?;
        want_len(end16.len(), start16.len(), "UTF-16 ends")?;
        want_len(cut.len(), start16.len(), "cut flags")?;
        done(&r, "utf16 spans")?;
        Ok(Utf16Spans::new(start16, end16, cut))
    }
}
