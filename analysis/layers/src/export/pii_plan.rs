//! The wire form of `PiiPlan`: the spans, then per item its rule code,
//! strategy id and rule id, then the policy revision and the policy digest
//! (a presence byte, then 32 bytes).

use super::*;
use crate::{PiiPlan, PiiRule, PiiStrategyKind};
use turbo_analysis::PolicyDigest;

impl LayerCodec for PiiPlan {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        for i in 0..self.spans.len() {
            w.u32(self.rule[i].code());
            w.u8(self.strategy[i].id());
            w.str(self.rule_id.get(i))?;
        }
        w.u64(self.policy_revision);
        match &self.policy_digest {
            Some(d) => {
                w.u8(1);
                w.bytes(d.as_bytes())?;
            }
            None => w.u8(0),
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans: turbo_analysis::Spans<PiiPlan> = get_spans(&mut r, cx)?;
        let mut out = PiiPlan::new(spans.view(), 0, None);
        for (_, span) in spans.iter() {
            let code = r.u32()?;
            let rule = PiiRule::from_code(code).ok_or_else(|| malformed(format!("pii plan rule code {code}")))?;
            let id = r.u8()?;
            let strategy = PiiStrategyKind::from_id(id).ok_or_else(|| malformed(format!("pii plan strategy {id}")))?;
            let rule_id = r.str()?;
            out.push(span, rule, strategy, rule_id);
        }
        out.policy_revision = r.u64()?;
        out.policy_digest = match r.u8()? {
            0 => None,
            1 => {
                let b: [u8; 32] =
                    r.bytes()?.try_into().map_err(|_| malformed("pii plan policy digest is not 32 bytes"))?;
                Some(PolicyDigest::from_bytes(b))
            }
            v => return Err(malformed(format!("pii plan policy digest presence {v}"))),
        };
        done(&r, "pii plan")?;
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
