use super::*;

impl<L: Layer> LayerCodec for Boilerplate<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.len())?;
        for k in &self.kind {
            w.u8(k.map_or(0, |k| k.code() + 1));
        }
        for s in &self.score {
            w.u32(s.to_bits());
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let n = r.usize()?;
        let mut kind = Vec::new();
        for _ in 0..n {
            kind.push(match r.u8()? {
                0 => None,
                code => Some(
                    BoilerplateKind::from_code(code - 1)
                        .ok_or_else(|| malformed(format!("a boilerplate kind code of {code}")))?,
                ),
            });
        }
        let mut score = Vec::new();
        for _ in 0..n {
            let s = f32::from_bits(r.u32()?);
            if !(0.0..=1.0).contains(&s) {
                return Err(malformed(format!("a boilerplate score of {s} outside [0, 1]")));
            }
            score.push(s);
        }
        done(&r, "boilerplate")?;
        Ok(Boilerplate::new(kind, score))
    }
}
