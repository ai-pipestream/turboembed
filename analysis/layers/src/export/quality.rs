use super::*;
use crate::{QualityFlags, QualityMetrics, SpanQuality, TextQuality};

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
