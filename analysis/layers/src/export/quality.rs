use std::borrow::Cow;

use super::*;
use crate::quality::grammar::check_rule_id;
use crate::{
    GrammarCategory, GrammarDiagnostic, GrammarDiagnostics, MAX_DIAGNOSTIC_REPLACEMENTS, QualityFlags, QualityMetrics,
    SpanQuality, TextQuality, builtin_rule_id,
};

fn put_metrics(w: &mut Writer, m: QualityMetrics) {
    for x in [
        m.composite_score,
        m.mtld_lexical_diversity,
        m.flesch_kincaid_grade,
        m.automated_readability_index,
        m.coleman_liau_index,
        m.information_density,
        m.alpha_ratio,
        m.digit_ratio,
        m.symbol_ratio,
        m.whitespace_ratio,
        m.replacement_ratio,
    ] {
        w.u32(x.to_bits());
    }
    w.u16(m.flags.0);
}

fn get_metrics(r: &mut Reader<'_>) -> Result<QualityMetrics, ExportError> {
    let mut next = || r.u32().map(f32::from_bits);
    let m = QualityMetrics {
        composite_score: next()?,
        mtld_lexical_diversity: next()?,
        flesch_kincaid_grade: next()?,
        automated_readability_index: next()?,
        coleman_liau_index: next()?,
        information_density: next()?,
        alpha_ratio: next()?,
        digit_ratio: next()?,
        symbol_ratio: next()?,
        whitespace_ratio: next()?,
        replacement_ratio: next()?,
        flags: QualityFlags(r.u16()?),
    };
    m.check().map_err(malformed)?;
    Ok(m)
}

impl LayerCodec for TextQuality {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_metrics(&mut w, self.metrics);
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let metrics = get_metrics(&mut r)?;
        done(&r, "text quality")?;
        Ok(Self { spans, metrics })
    }
}

impl LayerCodec for SpanQuality {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.metrics.len())?;
        for &m in &self.metrics {
            put_metrics(&mut w, m);
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = r.usize()?;
        want_len(n, spans.len(), "quality rows")?;
        let mut metrics = Vec::new();
        for _ in 0..n {
            metrics.push(get_metrics(&mut r)?);
        }
        done(&r, "span quality")?;
        Ok(Self { spans, metrics })
    }
}

impl LayerCodec for GrammarDiagnostics {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.diagnostics.len())?;
        for d in &self.diagnostics {
            d.check().map_err(malformed)?;
            w.str(&d.rule_id)?;
            w.u8(d.category.as_u8());
            w.str(&d.message)?;
            w.usize(d.replacements.len())?;
            for rep in &d.replacements {
                w.str(rep)?;
            }
            w.u32(d.confidence.to_bits());
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = r.usize()?;
        want_len(n, spans.len(), "grammar diagnostics")?;
        let mut diagnostics = Vec::with_capacity(n);
        for (_, span) in spans.iter() {
            let raw_rule = r.str()?;
            check_rule_id(raw_rule).map_err(malformed)?;
            let rule_id = builtin_rule_id(raw_rule).map_or_else(|| Cow::Owned(raw_rule.to_owned()), Cow::Borrowed);
            let category = GrammarCategory::from_byte(r.u8()?)
                .ok_or_else(|| malformed("unknown grammar diagnostic category byte"))?;
            let message: Box<str> = r.str()?.into();
            let rep_count = r.usize()?;
            if rep_count > MAX_DIAGNOSTIC_REPLACEMENTS {
                return Err(malformed("too many grammar diagnostic replacements"));
            }
            let mut replacements = Vec::with_capacity(rep_count);
            for _ in 0..rep_count {
                replacements.push(Box::<str>::from(r.str()?));
            }
            let confidence = f32::from_bits(r.u32()?);
            let diag = GrammarDiagnostic {
                span,
                rule_id,
                category,
                message,
                replacements: replacements.into_boxed_slice(),
                confidence,
            };
            diag.check().map_err(malformed)?;
            diagnostics.push(diag);
        }
        done(&r, "grammar diagnostics")?;
        let layer = Self { spans, diagnostics };
        layer.check().map_err(malformed)?;
        Ok(layer)
    }
}
