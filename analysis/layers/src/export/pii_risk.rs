//! The `turbo:pii-risk` codec: the score, its counts and the per-kind
//! rows, with the custom kind table they index. No value crosses.

use super::*;
use crate::{PiiRisk, PiiRiskKind, PiiRiskLevel, PiiSeverity, PiiSuggestedRewrite};

fn put_f32(w: &mut Writer, x: f32) {
    w.u32(x.to_bits());
}

fn get_f32(r: &mut Reader<'_>) -> Result<f32, ExportError> {
    r.u32().map(f32::from_bits)
}

impl LayerCodec for PiiRisk {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(malformed)?;
        let mut w = Writer::new();
        put_f32(&mut w, self.score);
        w.u8(self.level.id());
        w.u32(self.mentions);
        w.u32(self.chars);
        put_f32(&mut w, self.density_per_1000_chars);
        w.u64(self.severity_table_digest);
        w.usize(self.custom_kinds.len())?;
        for name in self.custom_kinds.names() {
            w.str(name)?;
        }
        w.usize(self.kinds.len())?;
        for k in &self.kinds {
            w.u32(k.kind.code());
            w.u32(k.mentions);
            w.u8(k.severity.id());
            put_f32(&mut w, k.weighted_count);
            put_f32(&mut w, k.contribution);
            w.u8(k.suggested.id());
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let score = get_f32(&mut r)?;
        let level = r.u8()?;
        let level =
            PiiRiskLevel::from_id(u32::from(level)).ok_or_else(|| malformed(format!("a risk level of {level}")))?;
        let mentions = r.u32()?;
        let chars = r.u32()?;
        let density_per_1000_chars = get_f32(&mut r)?;
        let severity_table_digest = r.u64()?;
        let n_custom = r.usize()?;
        if n_custom > CustomKinds::MAX {
            return Err(malformed(KindError::Full.to_string()));
        }
        let mut names = Vec::with_capacity(n_custom);
        for _ in 0..n_custom {
            names.push(r.str()?);
        }
        let custom_kinds = CustomKinds::from_names(names).map_err(|e| malformed(e.to_string()))?;
        let n = r.usize()?;
        if n > data.len() {
            return Err(malformed("more pii risk kinds than the payload holds"));
        }
        let mut kinds = Vec::with_capacity(n);
        for _ in 0..n {
            let kind = custom_kinds.from_code(r.u32()?).map_err(|e| malformed(e.to_string()))?;
            let mentions = r.u32()?;
            let s = r.u8()?;
            let severity = PiiSeverity::from_id(u32::from(s)).ok_or_else(|| malformed(format!("a severity of {s}")))?;
            let weighted_count = get_f32(&mut r)?;
            let contribution = get_f32(&mut r)?;
            let g = r.u8()?;
            let suggested = PiiSuggestedRewrite::from_id(u32::from(g))
                .ok_or_else(|| malformed(format!("a suggested rewrite of {g}")))?;
            kinds.push(PiiRiskKind { kind, mentions, severity, weighted_count, contribution, suggested });
        }
        done(&r, "pii risk")?;
        let risk = PiiRisk {
            score,
            level,
            mentions,
            chars,
            density_per_1000_chars,
            kinds,
            custom_kinds,
            severity_table_digest,
        };
        risk.check().map_err(malformed)?;
        Ok(risk)
    }
}
