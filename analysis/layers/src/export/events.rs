//! The wire form of `Events`: the trigger spans and lemmas, then per
//! event its sentence, confidence bits and arguments, each argument its
//! span, role and anchored layer, then the anchored index when the
//! layer is not `None`.

use super::*;
use crate::{AnchoredLayerKind, ArgumentRole, EventArgument, Events};

impl LayerCodec for Events {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_strs(&mut w, &self.lemma)?;
        for f in self.iter() {
            w.u32(f.sentence_ix);
            w.u32(f.confidence.to_bits());
            w.u32(f.arguments.len() as u32);
            for a in f.arguments {
                w.span(a.span);
                w.u8(a.role as u8);
                w.u8(a.anchored_layer as u8);
                if let Some(i) = a.anchored_index {
                    w.u32(i);
                }
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let mut out = Events::new(spans.view());
        out.spans = spans;
        out.lemma = get_strs(&mut r)?;
        want_len(out.lemma.len(), n, "event lemmas")?;
        let text = doc.text(out.spans.view());
        for _ in 0..n {
            out.sentence.push(r.u32()?);
            out.confidence.push(f32::from_bits(r.u32()?));
            let count = r.u32()?;
            for _ in 0..count {
                let span = r.span()?;
                if text.get(span.range()).is_none() {
                    return Err(malformed(format!(
                        "argument {}..{} outside view or off UTF-8 boundaries",
                        span.start, span.end
                    )));
                }
                let role = r.u8()?;
                let role = ArgumentRole::from_byte(role).ok_or_else(|| malformed(format!("argument role {role}")))?;
                let layer = r.u8()?;
                let anchored_layer =
                    AnchoredLayerKind::from_byte(layer).ok_or_else(|| malformed(format!("anchored layer {layer}")))?;
                let anchored_index = match anchored_layer {
                    AnchoredLayerKind::None => None,
                    _ => Some(r.u32()?),
                };
                out.arguments.push(EventArgument { span, role, anchored_layer, anchored_index });
            }
            out.arg_end.push(out.arguments.len() as u32);
        }
        done(&r, "events")?;
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
