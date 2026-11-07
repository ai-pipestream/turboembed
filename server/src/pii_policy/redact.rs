//! The `turbo-pii:redact` step: a request's [`RedactionPolicy`] applied to
//! the `turbo:pii` layer, writing #1902's derived `redacted` view and its
//! `turbo:pii-edits` layer. Each edit names the rule id that chose its
//! strategy, the policy revision and the policy digest.
//!
//! The digest is in the step's cache key twice: as the SHA-256 of the
//! step's [`ModelRef`], which the provenance digest chains, and folded
//! into the local FNV [`Annotator::config_hash`].

use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Error, Fnv1a, ModelRef, Refs, View};
use turbo_pii::rewrite::{Choice, REDACTED, StrategyChooser, rewrite_chosen};
use turbo_pii::{Pii, PiiEdits, PiiMention, PiiPlan, PiiRule, PiiStrategyKind};

use super::{MentionFacts, RedactionPolicy};

/// The step id.
pub const REDACT_STEP: &str = "turbo-pii:redact";

/// Chooses each mention's strategy by the policy's rules over its facts.
struct PolicyChooser<'p> {
    policy: &'p RedactionPolicy,
}

impl StrategyChooser for PolicyChooser<'_> {
    fn choose<'r>(&'r self, pii: &Pii, m: &PiiMention<'_>) -> Choice<'r> {
        let (detector_id, detector_version) = pii.detectors.get(m.detector).unwrap_or(("", ""));
        let facts = MentionFacts { evidence: m.evidence, detector_id, detector_version, score: m.score.as_ref() };
        // A layer's custom kind is matched to the policy's by name; one
        // the policy does not list takes the default.
        let (strategy, rule_id) = match self.policy.custom_kinds.translate(&pii.custom_kinds, m.kind) {
            Some(k) => self.policy.strategy_for(k, &facts),
            None => (&self.policy.default_strategy, None),
        };
        match rule_id {
            Some(id) => Choice { strategy, rule: PiiRule::Kind(m.kind), rule_id: id },
            None => Choice { strategy, rule: PiiRule::Default, rule_id: "" },
        }
    }
}

/// Rewrites the pii layer over the view it reads with one policy.
#[derive(Clone, Debug)]
pub struct RedactStep {
    policy: Arc<RedactionPolicy>,
}

impl RedactStep {
    pub fn new(policy: Arc<RedactionPolicy>) -> RedactStep {
        RedactStep { policy }
    }

    pub fn policy(&self) -> &RedactionPolicy {
        &self.policy
    }

    /// The step for a pipeline, writing the view and its edits as
    /// `redacted`.
    pub fn bound(self) -> Bound<RedactStep> {
        Bound::new(self).write(0, REDACTED.name).write(1, REDACTED.name)
    }

    /// The rewrite of `pii`, a layer over `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<(View, PiiEdits), Error> {
        let chooser = PolicyChooser { policy: &self.policy };
        let rewrite = rewrite_chosen(src, pii, &chooser, self.policy.revision)
            .map_err(|e| Error::Annotator { annotator: REDACT_STEP, detail: e.to_string() })?;
        Ok(rewrite.with_policy_digest(self.policy.digest).into_parts())
    }

    /// What the rewrite of `pii` would record for each mention, without
    /// the view: the same rewrite is run, so a plan is refused exactly
    /// where the rewrite is.
    pub fn plan(&self, src: &View, pii: &Pii) -> Result<PiiPlan, Error> {
        let (_, edits) = self.rewrite(src, pii)?;
        let mut per: Vec<Option<(PiiRule, PiiStrategyKind, &str)>> = vec![None; pii.spans.len()];
        for e in edits.edits() {
            for m in e.members {
                per[m.get()] = Some((e.rule, e.strategy, e.rule_id));
            }
        }
        let mut plan = PiiPlan::new(pii.spans.view(), self.policy.revision, Some(self.policy.digest));
        for ((_, span), p) in pii.spans.iter().zip(per) {
            // Every mention is under exactly one edit; the edits layer's
            // own check holds the rewrite to that.
            let (rule, strategy, rule_id) = p.ok_or_else(|| Error::Annotator {
                annotator: REDACT_STEP,
                detail: "a mention is under no edit".into(),
            })?;
            plan.push(span, rule, strategy, rule_id);
        }
        Ok(plan)
    }

    /// The dry-run step for a pipeline, writing the default `turbo:pii-plan`.
    pub fn dry_run(self) -> RedactPlanStep {
        RedactPlanStep { inner: self }
    }

    fn fold(&self, h: &mut Fnv1a) {
        h.bytes(self.policy.digest.as_bytes()).u64(self.policy.revision);
    }

    fn model_ref(&self) -> ModelRef {
        ModelRef {
            id: self.policy.id.clone(),
            revision: self.policy.revision.to_string(),
            sha256: self.policy.digest.to_hex(),
        }
    }
}

/// The dry run of a [`RedactStep`]: the plan, and no view or edits.
#[derive(Clone, Debug)]
pub struct RedactPlanStep {
    inner: RedactStep,
}

impl Annotator for RedactPlanStep {
    const ID: &'static str = REDACT_STEP;
    const VERSION: &'static str = "1";
    type Reads = (View, Pii);
    type Writes = (PiiPlan,);

    fn annotate(&self, (src, pii): Refs<'_, Self>) -> Result<(PiiPlan,), Error> {
        Ok((self.inner.plan(src, pii)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.inner.model_ref())
    }

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.str("dry-run");
        self.inner.fold(&mut h);
        h.finish()
    }
}

impl Annotator for RedactStep {
    const ID: &'static str = REDACT_STEP;
    const VERSION: &'static str = "1";
    type Reads = (View, Pii);
    type Writes = (View, PiiEdits);

    fn annotate(&self, (src, pii): Refs<'_, Self>) -> Result<(View, PiiEdits), Error> {
        self.rewrite(src, pii)
    }

    /// The policy by id, revision and canonical SHA-256 digest: the digest
    /// enters every provenance digest the step writes.
    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref())
    }

    /// FNV-1a over the digest and revision, the local cache hash.
    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        self.fold(&mut h);
        h.finish()
    }
}
