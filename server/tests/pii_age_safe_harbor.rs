//! The `PiiAgeSafeHarborStrategy` arm (issue #1884): it converts to the
//! native strategy and back, its mask is checked and in the digest, and
//! the redact step writes ages over 89 as `90+` with edits that record
//! `PII_STRATEGY_AGE_SAFE_HARBOR` on the wire.

#[path = "pii_policy/common.rs"]
mod common;
#[path = "pii_redact/fixture.rs"]
mod fixture;

use common::*;
use fixture::*;
use pb::pii_rewrite_strategy::Strategy;
use turbo_analysis::ViewId;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::pii_policy::{NoRegistry, PolicyError, RedactionPolicy, strategy_to_proto};
use turbo_kserve::proto::result as pb;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::results::typed::document_to_typed;
use turbo_pii::PiiStrategyKind;
use turbo_pii::rewrite::{AgeSafeHarbor, RewriteStrategy};

fn age(mask_char: &str) -> Strategy {
    Strategy::AgeSafeHarbor(pb::PiiAgeSafeHarborStrategy { mask_char: mask_char.to_owned() })
}

/// One rule for ages with `s` as its only strategy.
fn age_policy(s: Strategy) -> pb::RedactionPolicy {
    policy(vec![rule("age", builtin(pb::PiiKindId::Age), vec![entry(s)])])
}

fn native(s: Strategy) -> Result<RewriteStrategy, PolicyError> {
    let p = RedactionPolicy::from_proto(&age_policy(s), &NoRegistry)?;
    Ok(p.rules[0].strategies[0].strategy.clone())
}

#[test]
fn the_arm_converts_and_round_trips() {
    let plain = native(age("")).unwrap();
    assert_eq!(plain, RewriteStrategy::AgeSafeHarbor(AgeSafeHarbor::new()));
    assert_eq!(strategy_to_proto(&plain).unwrap().strategy, Some(age("")), "the default mask is written empty");
    let hash = native(age("#")).unwrap();
    assert_eq!(hash, RewriteStrategy::AgeSafeHarbor(AgeSafeHarbor::new().with_mask('#')));
    assert_eq!(strategy_to_proto(&hash).unwrap().strategy, Some(age("#")));
    assert_eq!(native(age("*")).unwrap(), plain, "the default mask spelled out is the same strategy");
    assert_eq!(hash.kind(), PiiStrategyKind::AgeSafeHarbor);
    assert_eq!(turbo_kserve::pii_policy::edit_strategy(&age("")), pb::PiiStrategy::AgeSafeHarbor);
}

#[test]
fn a_mask_of_more_than_one_character_is_refused() {
    let err = native(age("**")).expect_err("refused");
    assert!(matches!(err, PolicyError::Invalid { .. }), "{err:?}");
}

#[test]
fn the_digest_covers_the_mask() {
    let digest = |s| RedactionPolicy::from_proto(&age_policy(s), &NoRegistry).unwrap().digest;
    assert_eq!(digest(age("")), digest(age("")));
    assert_ne!(digest(age("")), digest(age("#")));
    let zip = Strategy::ZipSafeHarbor(pb::PiiZipSafeHarborStrategy::default());
    assert_ne!(digest(age("")), digest(zip), "the arm is part of the digest");
}

#[test]
fn the_redact_step_aggregates_ages_and_the_wire_edits_say_so() {
    let text = "Seen today: a 93-year-old man and a woman aged 34; age: 104 on file.";
    let p = convert(&age_policy(age("")), &NoRegistry);
    let doc = run_with(text, p.clone(), false);
    let (view, edits) = redacted(&doc);
    assert_eq!(view.text(), "Seen today: a 90+-year-old man and a woman aged 34; age: 90+ on file.");
    assert!(!view.text().contains("93") && !view.text().contains("104"));
    assert_eq!(edits.edits().count(), 3);
    assert!(edits.edits().all(|e| e.strategy == PiiStrategyKind::AgeSafeHarbor && e.rule_id == "age"));

    let typed = document_to_typed("redact", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let layer = typed.layers.iter().find(|l| l.id == "turbo:pii-edits").expect("the edits layer");
    assert_eq!(layer.annotations.len(), 3);
    for a in &layer.annotations {
        let Some(Value::PiiEdit(e)) = &a.value else { panic!("not an edit") };
        assert_eq!(e.strategy, pb::PiiStrategy::AgeSafeHarbor as i32);
        assert_eq!(e.policy_digest, p.digest.as_bytes().to_vec());
    }
}
