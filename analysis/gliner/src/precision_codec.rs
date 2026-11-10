use turbo_analysis::export::{ExportError, ImportCx, LayerCodec, Reader, Writer};
use turbo_analysis::{Document, IxRange, Layer, Span, TagSet, ViewId};
use turbo_analysis_layers::{PiiRule, Tokens};

use crate::precision::{PrecisionCandidates, PrecisionDecision, PrecisionRejection};

const MAX_CANDIDATES: usize = 1 << 20;

fn put_span(w: &mut Writer, span: Span) {
    w.u32(span.start);
    w.u32(span.end);
}

fn get_span(r: &mut Reader) -> Result<Span, ExportError> {
    Ok(Span::new(r.u32()?, r.u32()?))
}

fn put_range(w: &mut Writer, range: IxRange<Tokens>) {
    w.u32(range.start.get() as u32);
    w.u32(range.end.get() as u32);
}

fn get_range(r: &mut Reader) -> Result<IxRange<Tokens>, ExportError> {
    Ok(IxRange::new(r.u32()?, r.u32()?))
}

fn put_kinds(w: &mut Writer, kinds: &TagSet) -> Result<(), ExportError> {
    w.str(&kinds.name)?;
    w.usize(kinds.len())?;
    for label in kinds.labels() {
        w.str(label)?;
    }
    Ok(())
}

fn get_kinds(r: &mut Reader) -> Result<std::sync::Arc<TagSet>, ExportError> {
    let name = r.string()?;
    let n = r.usize()?;
    if n > 128 {
        return Err(ExportError::Malformed("precision kind count exceeds limit".into()));
    }
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        labels.push(r.string()?);
    }
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    Ok(TagSet::new(&name, &refs))
}

fn put_decision(w: &mut Writer, decision: PrecisionDecision) {
    match decision {
        PrecisionDecision::Accepted => {
            w.u8(0);
            w.u32(0);
        }
        PrecisionDecision::Changed => {
            w.u8(1);
            w.u32(0);
        }
        PrecisionDecision::Rejected(PrecisionRejection::NonEntity) => {
            w.u8(2);
            w.u32(0);
        }
        PrecisionDecision::Rejected(PrecisionRejection::NoCompatibleLink) => {
            w.u8(3);
            w.u32(0);
        }
        PrecisionDecision::Rejected(PrecisionRejection::PiiConflict(kind)) => {
            w.u8(4);
            w.u32(kind.code());
        }
    };
}

fn get_decision(r: &mut Reader) -> Result<PrecisionDecision, ExportError> {
    let tag = r.u8()?;
    let code = r.u32()?;
    match tag {
        0 => Ok(PrecisionDecision::Accepted),
        1 => Ok(PrecisionDecision::Changed),
        2 => Ok(PrecisionDecision::Rejected(PrecisionRejection::NonEntity)),
        3 => Ok(PrecisionDecision::Rejected(PrecisionRejection::NoCompatibleLink)),
        4 => {
            // A custom kind comes back as the index it was written with,
            // into the custom-kind table of the PII layer the rule read.
            match PiiRule::from_code(code) {
                Some(PiiRule::Kind(kind)) => Ok(PrecisionDecision::Rejected(PrecisionRejection::PiiConflict(kind))),
                _ => Err(ExportError::Malformed(format!("unknown PII conflict kind {code}"))),
            }
        }
        other => Err(ExportError::Malformed(format!("unknown precision decision {other}"))),
    }
}

impl LayerCodec for PrecisionCandidates {
    fn export(&self, _cx: &turbo_analysis::export::ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(ExportError::Malformed)?;
        let mut w = Writer::new();
        w.u32(u32::from(self.spans.view().0));
        w.usize(self.spans.len())?;
        for i in 0..self.spans.len() {
            put_span(&mut w, self.original_spans[i]);
            put_span(&mut w, self.spans.get(turbo_analysis::Ix::new(i as u32)));
            put_range(&mut w, self.original_tokens[i]);
            put_range(&mut w, self.tokens[i]);
            w.u16(self.kind[i].0);
            w.u32(self.logits[i].to_bits());
            w.f64(self.score[i]);
            put_decision(&mut w, self.decision[i]);
        }
        put_kinds(&mut w, &self.kinds)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let view =
            ViewId(r.u32()?.try_into().map_err(|_| ExportError::Malformed("precision view id exceeds u16".into()))?);
        let n = r.usize()?;
        if n > MAX_CANDIDATES {
            return Err(ExportError::Malformed("precision candidate count exceeds limit".into()));
        }
        let mut out = PrecisionCandidates::new(view, std::sync::Arc::new(TagSet::default()));
        for _ in 0..n {
            out.original_spans.push(get_span(&mut r)?);
            out.spans.push(get_span(&mut r)?);
            out.original_tokens.push(get_range(&mut r)?);
            out.tokens.push(get_range(&mut r)?);
            let kind = turbo_analysis::Tag(r.u16()?);
            out.kind.push(kind);
            out.logits.push(f32::from_bits(r.u32()?));
            out.score.push(r.f64()?);
            out.decision.push(get_decision(&mut r)?);
        }
        out.kinds = get_kinds(&mut r)?;
        if !r.is_done() {
            return Err(ExportError::Malformed("trailing precision payload bytes".into()));
        }
        out.check().map_err(ExportError::Malformed)?;
        Ok(out)
    }
}
