//! The wire form of `PiiEdits`: the spans in the derived view, then per
//! edit its source span, member count and members, rule code
//! ([`PiiRule::code`]) and strategy id ([`PiiStrategyKind::id`]), then the
//! policy revision. An unknown rule code or strategy id is refused on
//! import; the document then checks the edits against the view's
//! alignment and the pii layer they cover.

use super::*;
use crate::{PiiEdit, PiiEdits, PiiRule, PiiStrategyKind};

impl LayerCodec for PiiEdits {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.source.len())?;
        for e in self.edits() {
            w.span(e.source);
            w.usize(e.members.len())?;
            for m in e.members {
                w.u32(m.get() as u32);
            }
            w.u32(e.rule.code());
            w.u8(e.strategy.id());
        }
        w.u64(self.policy_revision);
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans: turbo_analysis::Spans<PiiEdits> = get_spans(&mut r, cx)?;
        let n = r.usize()?;
        want_len(n, spans.len(), "pii edit sources")?;
        let mut out = PiiEdits::new(spans.view(), 0);
        let mut members: Vec<Ix<Pii>> = Vec::new();
        for (_, span) in spans.iter() {
            let source = r.span()?;
            members.clear();
            for _ in 0..r.usize()? {
                members.push(Ix::new(r.u32()?));
            }
            let code = r.u32()?;
            let rule = PiiRule::from_code(code).ok_or_else(|| malformed(format!("pii edit rule code {code}")))?;
            let id = r.u8()?;
            let strategy = PiiStrategyKind::from_id(id).ok_or_else(|| malformed(format!("pii edit strategy {id}")))?;
            out.push(PiiEdit { span, source, members: &members, rule, strategy });
        }
        out.policy_revision = r.u64()?;
        done(&r, "pii edits")?;
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
