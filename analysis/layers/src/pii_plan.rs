//! What a PII rewrite would do (`turbo:pii-plan`), written by a dry run in
//! place of the derived view and its edits: one item per mention of the
//! `turbo:pii` layer, in the layer's order, with the rule, rule id and
//! strategy the edit covering that mention would record. A mention that
//! loses its cluster takes the winner's choice, as its edit would. The
//! layer holds no text and no value.

use std::any::TypeId;

use turbo_analysis::{Document, Fnv1a, Key, KeyId, Layer, Overlap, PolicyDigest, Spans, ViewId};

use crate::{Pii, PiiRule, PiiStrategyKind, Strs};

/// The planned rewrite of each mention of one pii layer.
pub struct PiiPlan {
    /// The mentions' spans, one for one with the pii layer.
    pub spans: Spans<PiiPlan>,
    pub rule: Vec<PiiRule>,
    pub strategy: Vec<PiiStrategyKind>,
    /// The policy's id for each rule; empty for the default.
    pub rule_id: Strs,
    pub policy_revision: u64,
    pub policy_digest: Option<PolicyDigest>,
}

impl PiiPlan {
    /// An empty plan over `view`, the pii layer's view.
    pub fn new(view: ViewId, policy_revision: u64, policy_digest: Option<PolicyDigest>) -> PiiPlan {
        PiiPlan {
            spans: Spans::new(view),
            rule: Vec::new(),
            strategy: Vec::new(),
            rule_id: Strs::default(),
            policy_revision,
            policy_digest,
        }
    }

    /// Appends the plan of the next mention.
    pub fn push(&mut self, span: turbo_analysis::Span, rule: PiiRule, strategy: PiiStrategyKind, rule_id: &str) {
        self.spans.push(span);
        self.rule.push(rule);
        self.strategy.push(strategy);
        self.rule_id.push(rule_id);
    }
}

impl Layer for PiiPlan {
    const ID: &'static str = "turbo:pii-plan";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(PiiPlan);

    /// The planned pii layer, so the document binds it.
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Pii>::DEFAULT.id(), self.spans.len())]
    }

    fn content_hash(&self) -> Option<u64> {
        let mut h = Fnv1a::new();
        h.usize(self.spans.len()).u64(self.policy_revision);
        match &self.policy_digest {
            Some(d) => h.u8(1).bytes(d.as_bytes()),
            None => h.u8(0),
        };
        for i in 0..self.spans.len() {
            h.u32(self.rule[i].code()).u8(self.strategy[i].id()).str(self.rule_id.get(i));
        }
        Some(h.finish())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.rule.len() != n || self.strategy.len() != n || self.rule_id.len() != n {
            return Err("every column as long as the spans".into());
        }
        Ok(())
    }

    /// The pii layer is over the same view, with exactly these spans in
    /// this order, and every rule kind is in its table.
    fn check_document(&self, doc: &Document, on: &[KeyId]) -> Result<(), String> {
        let key = on.iter().find(|k| k.ty == TypeId::of::<Pii>()).ok_or("a pii plan needs its pii layer")?;
        let pii = doc.get_key(&Key::<Pii>::named(key.name)).ok_or("a pii plan needs its pii layer")?;
        if pii.spans.view() != self.spans.view() || pii.spans.len() != self.spans.len() {
            return Err("a pii plan lists every mention of its pii layer, on the same view".into());
        }
        for ((i, s), (_, m)) in self.spans.iter().zip(pii.spans.iter()) {
            if s != m {
                return Err(format!("pii plan item {} is not over its mention", i.get()));
            }
            if let PiiRule::Kind(k) = self.rule[i.get()]
                && pii.custom_kinds.check(k).is_err()
            {
                return Err(format!("pii plan item {} names a rule kind outside the pii layer's table", i.get()));
            }
        }
        Ok(())
    }
}
