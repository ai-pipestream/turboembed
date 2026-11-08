//! Per-mention policy selection survives a same-kind overlap.

#[path = "pii_policy/common.rs"]
mod common;

use common::*;
use std::sync::Arc;
use turbo_analysis::{Annotator, Document, Ix, Layer, Span, View, ViewId};
use turbo_kserve::pii_policy::{NoRegistry, RedactPlanStep, RedactStep, RedactionPolicy};
use turbo_kserve::proto::result as pb;
use turbo_pii::{Pii, PiiDetectorRef, PiiEvidence, PiiKind, PiiMention, PiiScore, PiiScoreKind};

#[test]
fn score_conditioned_phone_overlap_masks_the_stricter_members_digits() {
    use pb::pii_rewrite_strategy::Strategy;
    use pb::pii_strategy_condition::Test;
    let msg = policy(vec![rule(
        "phone",
        builtin(pb::PiiKindId::Phone),
        vec![
            pb::PiiStrategyEntry {
                when: vec![pb::PiiStrategyCondition {
                    test: Some(Test::ScoreAtLeast(pb::PiiScoreCondition {
                        kind: pb::PiiScoreKind::Probability as i32,
                        min: 0.5,
                        calibrated: false,
                    })),
                }],
                strategy: Some(strategy(Strategy::LastN(pb::PiiLastNStrategy {
                    count: 4,
                    mask_char: "*".into(),
                    keep_format: false,
                }))),
            },
            entry(Strategy::Mask(pb::PiiMaskStrategy::default())),
        ],
    )]);
    let policy = Arc::new(RedactionPolicy::from_proto(&msg, &NoRegistry).unwrap());
    let step = RedactStep::new(policy.clone());
    let doc = Document::new("555-1234");
    let source = doc.get::<View>().unwrap();
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("fixture", "1").unwrap();
    for (span, score, normalized) in [(Span::new(0, 8), 0.9, "5551234"), (Span::new(4, 8), 0.1, "1234")] {
        pii.push(PiiMention {
            span,
            kind: PiiKind::Phone,
            detector,
            evidence: &[PiiEvidence::Shape],
            normalized,
            score: Some(PiiScore {
                kind: PiiScoreKind::Probability,
                value: score,
                model: PiiDetectorRef::new(0),
                calibrated: false,
            }),
        });
    }
    let (view, edits) = step.rewrite(source, &pii).unwrap();
    assert_eq!(view.text(), "********");
    assert_eq!(edits.get(Ix::new(0)).members, &[Ix::new(0), Ix::new(1)]);
    assert_eq!(edits.get(Ix::new(0)).rule_id, "phone");
    assert_eq!(edits.check(), Ok(()));
    let plan = step.plan(source, &pii).unwrap();
    assert_eq!(plan.spans.len(), 2);
    assert_eq!(plan.policy_digest, Some(policy.digest));
    assert!(plan.rule_id.iter().all(|r| r == "phone"));
    assert_eq!(RedactStep::VERSION, "2");
    assert_eq!(RedactPlanStep::VERSION, "2");
}
