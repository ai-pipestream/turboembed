//! The `PiiZipSafeHarborStrategy` arm (issue #1884): it converts to the
//! native strategy and back, its table is checked, the digest covers it
//! as a set, and the redact step writes the generalized ZIP with edits
//! that record `PII_STRATEGY_ZIP_SAFE_HARBOR` on the wire.

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
use turbo_pii::rewrite::{RewriteStrategy, ZipSafeHarbor};

fn zip(mask_char: &str, prefixes: &[&str]) -> Strategy {
    Strategy::ZipSafeHarbor(pb::PiiZipSafeHarborStrategy {
        mask_char: mask_char.to_owned(),
        restricted_prefixes: prefixes.iter().map(|p| (*p).to_owned()).collect(),
    })
}

/// One rule for US ZIP codes with `s` as its only strategy.
fn zip_policy(s: Strategy) -> pb::RedactionPolicy {
    policy(vec![rule("zip", builtin(pb::PiiKindId::UsZip), vec![entry(s)])])
}

fn native(s: Strategy) -> Result<RewriteStrategy, PolicyError> {
    let p = RedactionPolicy::from_proto(&zip_policy(s), &NoRegistry)?;
    Ok(p.rules[0].strategies[0].strategy.clone())
}

#[test]
fn the_arm_converts_and_round_trips() {
    let bundled = native(zip("", &[])).unwrap();
    assert_eq!(bundled, RewriteStrategy::ZipSafeHarbor(ZipSafeHarbor::hhs_2000()));
    assert_eq!(strategy_to_proto(&bundled).unwrap().strategy, Some(zip("", &[])), "the bundled table is written empty");

    let own = native(zip("#", &["627", "036"])).unwrap();
    let want = ZipSafeHarbor::with_restricted(["036", "627"]).unwrap().with_mask('#');
    assert_eq!(own, RewriteStrategy::ZipSafeHarbor(want));
    assert_eq!(strategy_to_proto(&own).unwrap().strategy, Some(zip("#", &["036", "627"])), "sorted on the way back");
    assert_eq!(own.kind(), PiiStrategyKind::ZipSafeHarbor);
    assert_eq!(turbo_kserve::pii_policy::edit_strategy(&zip("", &[])), pb::PiiStrategy::ZipSafeHarbor);
}

#[test]
fn a_bad_table_or_mask_is_refused() {
    for bad in [zip("", &["62"]), zip("", &["627", "62a"]), zip("", &["6270"]), zip("**", &[])] {
        let err = native(bad.clone()).expect_err("refused");
        assert!(matches!(err, PolicyError::Invalid { .. }), "{bad:?}: {err:?}");
    }
    let every: Vec<String> = (0..1000).map(|n| format!("{n:03}")).collect();
    let all: Vec<&str> = every.iter().map(String::as_str).collect();
    assert!(native(zip("", &all)).is_ok(), "every three-digit area may be listed");
    let mut too_many = all.clone();
    too_many.push("000");
    let err = native(zip("", &too_many)).expect_err("refused");
    assert!(err.to_string().contains("1000"), "{err}");
}

#[test]
fn the_digest_covers_the_table_as_a_set() {
    let digest = |s| RedactionPolicy::from_proto(&zip_policy(s), &NoRegistry).unwrap().digest;
    let a = digest(zip("", &["627", "036"]));
    assert_eq!(a, digest(zip("", &["036", "627", "627"])), "order and repeats do not change it");
    assert_ne!(a, digest(zip("", &["036"])));
    assert_ne!(a, digest(zip("#", &["627", "036"])));
    assert_ne!(digest(zip("", &[])), digest(zip("*", &["036"])));
}

#[test]
fn the_redact_step_generalizes_zips_and_the_wire_edits_say_so() {
    let text = "Ship to Springfield, IL 62704-1234 or Hanover, NH 03601 today.";
    let p = convert(&zip_policy(zip("", &[])), &NoRegistry);
    let doc = run_with(text, p.clone(), false);
    let (view, edits) = redacted(&doc);
    assert_eq!(view.text(), "Ship to Springfield, IL 627**-**** or Hanover, NH 000** today.");
    assert!(!view.text().contains("62704") && !view.text().contains("1234") && !view.text().contains("03601"));
    assert_eq!(edits.edits().count(), 2);
    assert!(edits.edits().all(|e| e.strategy == PiiStrategyKind::ZipSafeHarbor && e.rule_id == "zip"));

    let typed = document_to_typed("redact", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let layer = typed.layers.iter().find(|l| l.id == "turbo:pii-edits").expect("the edits layer");
    assert_eq!(layer.annotations.len(), 2);
    for a in &layer.annotations {
        let Some(Value::PiiEdit(e)) = &a.value else { panic!("not an edit") };
        assert_eq!(e.strategy, pb::PiiStrategy::ZipSafeHarbor as i32);
        assert_eq!(e.policy_digest, p.digest.as_bytes().to_vec());
    }
}
