//! The wire form of `PiiEdits`: the spans in the derived view, then per
//! edit its source span, member count and members, rule code
//! ([`PiiRule::code`]) and strategy id ([`PiiStrategyKind::id`]), then the
//! policy revision. A layer that names a redaction policy ends with a
//! tail: the tag 1, every edit's rule id, and the policy digest (a
//! presence byte, then 32 bytes); a layer that names none has no tail, so
//! its bytes are the same as before the tail existed. An unknown rule
//! code or strategy id is refused on import; the document then checks the
//! edits against the view's alignment and the pii layer they cover.

use super::*;
use crate::{PiiEdit, PiiEdits, PiiRule, PiiStrategyKind};
use turbo_analysis::PolicyDigest;

/// The tag of the policy tail.
const TAIL_POLICY: u8 = 1;

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
        if self.policy_digest.is_some() || self.rule_id.iter().any(|id| !id.is_empty()) {
            w.u8(TAIL_POLICY);
            put_strs(&mut w, &self.rule_id)?;
            match &self.policy_digest {
                Some(d) => {
                    w.u8(1);
                    w.bytes(d.as_bytes())?;
                }
                None => w.u8(0),
            }
        }
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
            out.push(PiiEdit { span, source, members: &members, rule, strategy, rule_id: "" });
        }
        out.policy_revision = r.u64()?;
        if !r.is_done() {
            let tag = r.u8()?;
            if tag != TAIL_POLICY {
                return Err(malformed(format!("pii edits tail tag {tag}")));
            }
            let ids = get_strs(&mut r)?;
            want_len(ids.len(), spans.len(), "pii edit rule ids")?;
            out.rule_id = ids;
            out.policy_digest = match r.u8()? {
                0 => None,
                1 => {
                    let b: [u8; 32] =
                        r.bytes()?.try_into().map_err(|_| malformed("pii edits policy digest is not 32 bytes"))?;
                    Some(PolicyDigest::from_bytes(b))
                }
                v => return Err(malformed(format!("pii edits policy digest presence {v}"))),
            };
        }
        done(&r, "pii edits")?;
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
