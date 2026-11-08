//! A redaction policy configures `turbo-pii:pii` (issue #1882 part B):
//! disabled kinds are not detected, allow lists keep values out, deny
//! lists and custom identifiers find the policy's custom kinds, and each
//! is then rewritten by its rule.

#[path = "pii_policy/common.rs"]
mod common;
#[path = "pii_redact/fixture.rs"]
mod fixture;

use common::*;
use fixture::*;
use pb::pii_rewrite_strategy::Strategy;
use turbo_kserve::pii_policy::{IDENTIFIERS_DETECTOR, NoRegistry};
use turbo_kserve::proto::result as pb;
use turbo_pii::{Pii, PiiCandidates, PiiDisposition, PiiKind};

fn kinds(doc: &turbo_analysis::Document) -> Vec<PiiKind> {
    let pii = doc.get::<Pii>().unwrap();
    pii.kind.clone()
}

#[test]
fn deny_lists_and_custom_identifiers_find_the_policy_s_custom_kinds() {
    let policy = convert(&rich(), &TestRegistry);
    let text = "Badge EMP-123456 works at Initech, mail jane@example.com.";
    let doc = run_with(text, policy.clone(), false);
    let pii = doc.get::<Pii>().unwrap();
    let employee = PiiKind::Custom(turbo_pii::CustomKindId::new(0));
    let project = PiiKind::Custom(turbo_pii::CustomKindId::new(1));
    let found: Vec<(&str, PiiKind)> = pii.spans.iter().map(|(i, s)| (&text[s.range()], pii.kind[i.get()])).collect();
    assert!(found.contains(&("EMP-123456", employee)), "{found:?}");
    assert!(found.contains(&("Initech", project)), "{found:?}");
    let ids = pii.spans.iter().find(|(i, _)| pii.kind[i.get()] == employee).map(|(i, _)| pii.get(i).detector).unwrap();
    assert_eq!(pii.detectors.get(ids).map(|d| d.0), Some(IDENTIFIERS_DETECTOR));
    let (view, edits) = redacted(&doc);
    for needle in ["EMP-123456", "Initech", EMAIL] {
        assert!(!view.text().contains(needle), "{needle} is rewritten: {}", view.text());
    }
    assert!(edits.edits().any(|e| e.rule_id == "employee"));
}

#[test]
fn a_disabled_kind_is_not_detected() {
    let mut msg = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    msg.kind_rules[0].state = pb::PiiRuleState::Disabled as i32;
    let doc = run(convert(&msg, &NoRegistry));
    assert!(!kinds(&doc).contains(&PiiKind::Email));
    assert!(kinds(&doc).contains(&PiiKind::Card), "other kinds are still found");
}

#[test]
fn a_kind_allow_list_keeps_its_term_out() {
    let mut msg = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    msg.kind_rules[0].allow = Some(pb::PiiAllowList {
        terms: vec![pb::PiiTermSource {
            source: Some(pb::pii_term_source::Source::Terms(pb::PiiTerms { terms: vec![EMAIL.to_owned()] })),
        }],
        case_sensitive: false,
        patterns: Vec::new(),
    });
    let doc = run(convert(&msg, &NoRegistry));
    assert!(!kinds(&doc).contains(&PiiKind::Email));
    let candidates = doc.get::<PiiCandidates>().unwrap();
    assert!(candidates.disposition.iter().any(|d| matches!(d, PiiDisposition::AllowedByRule { .. })));
}
