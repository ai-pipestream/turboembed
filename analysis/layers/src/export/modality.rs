//! The wire form of `Modality`: the scope spans, then per scope its
//! polarity, epistemic and deontic bytes, an optional cue span and the
//! confidence bits.

use super::*;
use crate::modality::{DeonticKind, EpistemicStatus, Modality, ModalityTag, Polarity};

impl LayerCodec for Modality {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        for (_, t) in self.iter() {
            w.u8(t.polarity as u8);
            w.u8(t.epistemic as u8);
            w.u8(t.deontic as u8);
            match t.cue {
                None => w.u8(0),
                Some(c) => {
                    w.u8(1);
                    w.span(c);
                }
            }
            w.u32(t.confidence.to_bits());
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans: turbo_analysis::Spans<Modality> = get_spans(&mut r, cx)?;
        let text_len = doc.text(spans.view()).len() as u32;
        let mut out = Modality::new(spans.view());
        for (_, scope) in spans.iter() {
            let byte = |b: u8, what: &str| malformed(format!("a {what} of {b}"));
            let b = r.u8()?;
            let polarity = Polarity::from_byte(b).ok_or_else(|| byte(b, "polarity"))?;
            let b = r.u8()?;
            let epistemic = EpistemicStatus::from_byte(b).ok_or_else(|| byte(b, "epistemic status"))?;
            let b = r.u8()?;
            let deontic = DeonticKind::from_byte(b).ok_or_else(|| byte(b, "deontic kind"))?;
            let cue = match r.u8()? {
                0 => None,
                1 => Some(r.span()?),
                b => return Err(byte(b, "cue tag")),
            };
            if cue.is_some_and(|c| c.end > text_len) {
                return Err(malformed("a modality cue outside the view text"));
            }
            let confidence = f32::from_bits(r.u32()?);
            out.push(scope, ModalityTag { polarity, epistemic, deontic, cue, confidence });
        }
        done(&r, "modality")?;
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
