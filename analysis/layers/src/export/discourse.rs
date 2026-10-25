//! The wire form of `Quotations`: the content spans, then per row the
//! optional speaker and cue spans, the optional speaker chain, the kind
//! and the confidence bits.

use turbo_analysis::Span;

use super::*;
use crate::{Quotations, QuoteKind};

fn put_opt_span(w: &mut Writer, s: Option<Span>) {
    match s {
        None => w.u8(0),
        Some(s) => {
            w.u8(1);
            w.span(s);
        }
    }
}

fn get_opt_span(r: &mut Reader<'_>) -> Result<Option<Span>, ExportError> {
    match r.u8()? {
        0 => Ok(None),
        1 => Ok(Some(r.span()?)),
        tag => Err(malformed(format!("an optional span tag of {tag}"))),
    }
}

impl LayerCodec for Quotations {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        for q in self.iter() {
            put_opt_span(&mut w, q.speaker);
            put_opt_span(&mut w, q.cue_verb);
            match q.speaker_chain {
                None => w.u8(0),
                Some(c) => {
                    w.u8(1);
                    w.u32(c);
                }
            }
            w.u8(q.kind.to_byte());
            w.u32(q.confidence.to_bits());
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut out = Quotations::new(spans.view());
        let n = spans.len();
        out.spans = spans;
        for _ in 0..n {
            out.speaker.push(get_opt_span(&mut r)?);
            out.cue_verb.push(get_opt_span(&mut r)?);
            out.speaker_chain.push(match r.u8()? {
                0 => None,
                1 => Some(r.u32()?),
                tag => return Err(malformed(format!("an optional chain tag of {tag}"))),
            });
            let kind = r.u8()?;
            out.kind.push(QuoteKind::from_byte(kind).ok_or_else(|| malformed(format!("a quote kind of {kind}")))?);
            out.confidence.push(f32::from_bits(r.u32()?));
        }
        done(&r, "quotations")?;
        let text = doc.text(out.spans.view());
        let all = out.iter().flat_map(|q| [Some(q.content), q.speaker, q.cue_verb]).flatten();
        for s in all.collect::<Vec<_>>() {
            if text.get(s.start as usize..s.end as usize).is_none() || s.start > s.end {
                return Err(malformed(format!("span {}..{} outside the view text", s.start, s.end)));
            }
        }
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
