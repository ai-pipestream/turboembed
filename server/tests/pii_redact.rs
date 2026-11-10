//! The `turbo-pii:redact` step (issue #1882 part B): every strategy writes
//! a redacted view that maps back to the source through the wire result's
//! edit chain, every edit names its rule id, revision and digest, the
//! digest keys the step's cache, and a named policy matches the same
//! policy inline.

#[path = "pii_policy/common.rs"]
mod common;
#[path = "pii_redact/fixture.rs"]
mod fixture;

use common::*;
use fixture::*;
use pb::pii_rewrite_strategy::Strategy;
use turbo_analysis::{Document, Ix, Span, ViewId};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::pii_policy::{NamedPolicies, NoRegistry, SelectError, select};
use turbo_kserve::progressive::compile_analyzer_with_redaction;
use turbo_kserve::proto::result as pb;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::results::typed::chain::WireChain;
use turbo_kserve::results::typed::document_to_typed;
use turbo_pii::PiiRule;

fn every_strategy() -> Vec<Strategy> {
    vec![
        Strategy::Mask(pb::PiiMaskStrategy::default()),
        Strategy::Format(pb::PiiFormatStrategy { template: "[%t]".to_owned() }),
        Strategy::Map(Box::new(pb::PiiMapStrategy {
            table: Some(pb::pii_map_strategy::Table::Entries(map_entries(&[(EMAIL, "J")]))),
            case_sensitive: false,
            target: pb::PiiMapTarget::NormalizedOrSurface as i32,
            fallback: None,
        })),
        Strategy::Truncate(pb::PiiTruncateStrategy {
            side: pb::PiiKeepSide::Start as i32,
            count: 2,
            mask_char: String::new(),
            keep_format: false,
        }),
        Strategy::LastN(pb::PiiLastNStrategy { count: 4, mask_char: String::new(), keep_format: true }),
        Strategy::Abbreviate(pb::PiiAbbreviateStrategy::default()),
        keyed("tokens", 1),
        Strategy::Sha256(pb::PiiSha256Strategy::default()),
        Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[E]".to_owned() }),
        Strategy::Pseudonym(pb::PiiPseudonymStrategy::default()),
        Strategy::ZipSafeHarbor(pb::PiiZipSafeHarborStrategy::default()),
        Strategy::AgeSafeHarbor(pb::PiiAgeSafeHarborStrategy::default()),
        Strategy::Date(pb::PiiDateStrategy {
            mode: Some(pb::pii_date_strategy::Mode::Year(pb::PiiDateYear {})),
            mask_char: String::new(),
        }),
    ]
}

#[test]
fn every_strategy_maps_back_to_the_source_through_the_wire_chain() {
    for s in every_strategy() {
        let policy = convert(&one(s.clone()), &TestRegistry);
        let doc = run(policy.clone());
        let (view, edits) = redacted(&doc);
        assert!(!view.text().contains(EMAIL), "{s:?}: the email is rewritten");
        assert!(edits.edits().count() >= 2, "{s:?}: the email and the card are edited");
        assert_eq!(edits.policy_digest, Some(policy.digest));
        assert_eq!(edits.policy_revision, policy.revision);
        let typed = document_to_typed("redact", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
        let chain = WireChain::from_payload(typed.mapping.as_ref()).unwrap();
        for e in edits.edits() {
            assert_eq!(chain.map_span(view.id(), ViewId::SOURCE, e.span).unwrap(), e.source, "{s:?}");
            assert_eq!(chain.map_span(ViewId::SOURCE, view.id(), e.source).unwrap(), e.span, "{s:?}");
            let email = matches!(e.rule, PiiRule::Kind(k) if k == turbo_pii::PiiKind::Email);
            assert_eq!(e.rule_id, if email { "email" } else { "" }, "{s:?}");
        }
        // Text outside every edit maps to itself shifted, never into an edit.
        let start = Span::new(0, 4);
        assert_eq!(chain.map_span(ViewId::SOURCE, view.id(), start).unwrap(), start);
        // The wire edits carry the rule id and the digest.
        let layer = typed.layers.iter().find(|l| l.id == "turbo:pii-edits").unwrap();
        for a in &layer.annotations {
            let Some(Value::PiiEdit(e)) = &a.value else { panic!("not an edit") };
            assert_eq!(e.policy_digest, policy.digest.as_bytes().to_vec());
            assert!(e.rule_id.is_empty() || e.rule_id == "email");
        }
    }
}

#[test]
fn a_policy_digest_change_misses_the_cache_and_the_same_policy_hits_it() {
    let a = convert(&one(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[A]".into() })), &NoRegistry);
    let mut b_msg = one(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[A]".into() }));
    b_msg.revision = 2;
    let b = convert(&b_msg, &NoRegistry);
    assert_ne!(a.digest, b.digest);
    let build = |p| {
        let r = turbo_kserve::pii_policy::Redaction { policy: p, dry_run: false };
        compile_analyzer_with_redaction(&Document::new(""), STEPS, &r).unwrap()
    };
    let mut doc = Document::new(TEXT);
    let first = build(a.clone()).analyze_incremental(&mut doc).unwrap();
    assert!(!first.was_skipped("turbo-pii:redact"));
    let again = build(a.clone()).analyze_incremental(&mut doc).unwrap();
    assert!(again.was_skipped("turbo-pii:redact"), "the same digest hits the cache");
    let changed = build(b.clone()).analyze_incremental(&mut doc).unwrap();
    assert!(!changed.was_skipped("turbo-pii:redact"), "a new digest misses it");
    assert!(changed.was_skipped("turbo-pii:pii"), "detection is not rerun");
    assert_eq!(redacted(&doc).1.policy_digest, Some(b.digest));
}

#[test]
fn a_named_policy_and_the_same_policy_inline_give_one_digest_and_output() {
    let msg = rich();
    let named = NamedPolicies::from_messages(std::slice::from_ref(&msg), &TestRegistry).unwrap();
    let by_name = pb::PiiRedactionSelection {
        policy: Some(pb::pii_redaction_selection::Policy::Name(msg.id.clone())),
        dry_run: false,
    };
    let inline =
        pb::PiiRedactionSelection { policy: Some(pb::pii_redaction_selection::Policy::Inline(msg)), dry_run: false };
    let a = select(&by_name, &named, &TestRegistry).unwrap();
    let b = select(&inline, &NamedPolicies::default(), &TestRegistry).unwrap();
    assert_eq!(a.policy.digest, b.policy.digest);
    let (da, db) = (run(a.policy), run(b.policy));
    assert_eq!(redacted(&da).0.text(), redacted(&db).0.text());
    let rows = |d: &Document| {
        redacted(d).1.edits().map(|e| (e.span, e.rule, e.strategy, e.rule_id.to_owned())).collect::<Vec<_>>()
    };
    assert_eq!(rows(&da), rows(&db));
}

#[test]
fn an_unknown_name_or_an_unset_selection_is_refused_without_echoing_the_name() {
    let named = NamedPolicies::default();
    let sel = |name: &str| pb::PiiRedactionSelection {
        policy: Some(pb::pii_redaction_selection::Policy::Name(name.to_owned())),
        dry_run: false,
    };
    let err = select(&sel(EMAIL), &named, &NoRegistry).unwrap_err();
    assert_eq!(err, SelectError::UnknownName);
    assert!(!err.to_string().contains(EMAIL));
    assert_eq!(select(&pb::PiiRedactionSelection::default(), &named, &NoRegistry).unwrap_err(), SelectError::Unset);
}

#[test]
fn named_policies_refuse_a_bad_policy_and_a_repeated_id_at_load() {
    let bad = pb::RedactionPolicy { id: String::new(), ..one(Strategy::Mask(pb::PiiMaskStrategy::default())) };
    assert!(NamedPolicies::from_messages(&[bad], &NoRegistry).unwrap_err().contains("pii policy 0"));
    let ok = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    let err = NamedPolicies::from_messages(&[ok.clone(), ok], &NoRegistry).unwrap_err();
    assert!(err.contains("pii policy 1") && err.contains("repeats"), "{err}");
    assert!(
        NamedPolicies::from_encoded(&[vec![0xff, 0xff]], &NoRegistry).unwrap_err().contains("not a RedactionPolicy")
    );
    // A keyed token whose key is not registered fails the load, naming
    // the reference only.
    let err = NamedPolicies::from_messages(&[one(keyed("tokens", 9))], &TestRegistry).unwrap_err();
    assert!(err.contains("not registered"), "{err}");
}

#[test]
fn the_redact_step_needs_a_policy_and_a_policy_needs_the_step() {
    let Err(none) = turbo_kserve::progressive::compile_analyzer(&Document::new(""), STEPS) else { panic!("built") };
    assert!(none.message.contains("pii_redaction"), "{}", none.message);
    let r = turbo_kserve::pii_policy::Redaction {
        policy: convert(&one(Strategy::Mask(pb::PiiMaskStrategy::default())), &NoRegistry),
        dry_run: false,
    };
    let Err(stray) = compile_analyzer_with_redaction(&Document::new(""), &["turbo-pii:pii"], &r) else {
        panic!("built")
    };
    assert!(stray.message.contains("turbo-pii:redact"), "{}", stray.message);
}

#[test]
fn a_cluster_takes_the_winner_s_rule_for_every_member() {
    let doc =
        run(convert(&one(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[E]".into() })), &NoRegistry));
    let (_, edits) = redacted(&doc);
    let first = edits.get(Ix::new(0));
    assert_eq!((first.rule_id, first.strategy), ("email", turbo_pii::PiiStrategyKind::Static));
}
