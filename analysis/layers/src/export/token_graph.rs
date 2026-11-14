use super::*;
use crate::{TokenArcKind, TokenGraph};

impl LayerCodec for TokenGraph {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_strs(&mut w, &self.terms)?;
        put_u32s(&mut w, &self.position)?;
        put_u32s(&mut w, &self.position_length)?;
        w.u32(self.num_positions);
        w.usize(self.kind.len())?;
        for &k in &self.kind {
            w.u8(k.to_byte());
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let terms = get_strs(&mut r)?;
        want_len(terms.len(), n, "token graph terms")?;
        let position = get_u32s(&mut r)?;
        want_len(position.len(), n, "token graph positions")?;
        let position_length = get_u32s(&mut r)?;
        want_len(position_length.len(), n, "token graph position_lengths")?;
        let num_positions = r.u32()?;
        let kind_len = r.usize()?;
        want_len(kind_len, n, "token graph kinds")?;
        let mut kind = Vec::with_capacity(kind_len);
        for _ in 0..kind_len {
            let b = r.u8()?;
            let k = TokenArcKind::from_byte(b).ok_or_else(|| malformed(format!("unknown TokenArcKind byte {b}")))?;
            kind.push(k);
        }
        done(&r, "token graph")?;
        Ok(TokenGraph { spans, terms, position, position_length, num_positions, kind })
    }
}
