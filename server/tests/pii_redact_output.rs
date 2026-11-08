//! What a `turbo-pii:redact` run lets out (issue #1882 part B): a dry run
//! writes no view and plans every kept candidate, the findings-only
//! default shows no redacted text, a keyed-token run without a grant
//! releases no value, unknown kinds and strategies are refused unless the
//! policy opts into redact-and-flag, and a planted value appears in no
//! edit, decision, receipt or error.

#[path = "pii_policy/common.rs"]
mod common;
#[path = "pii_redact/fixture.rs"]
mod fixture;

use common::*;
use fixture::*;
use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use turbo_analysis::{Key, ViewId};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::pii_policy::{NamedPolicies, NoRegistry, PolicyFlag, SelectError, select};
use turbo_kserve::proto::result as pb;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::typed::{document_to_typed, document_to_typed_with};
use turbo_pii::rewrite::REDACTED;
use turbo_pii::{Pii, PiiCandidates, PiiDisposition, PiiEdits, PiiPlan};

const REWRITTEN_AND_DETAIL: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: true };

fn contains(hay: &[u8], needle: &str) -> bool {
    hay.windows(needle.len()).any(|w| w == needle.as_bytes())
}

#[test]
fn a_dry_run_writes_no_view_and_plans_every_kept_candidate() {
    let policy = convert(&one(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[E]".into() })), &NoRegistry);
    let doc = run_with(TEXT, policy.clone(), true);
    assert!(doc.get_key(&REDACTED).is_none(), "no redacted view");
    assert!(doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).is_none(), "no edits");
    let pii = doc.get::<Pii>().unwrap();
    let plan = doc.get::<PiiPlan>().unwrap();
    assert_eq!(plan.spans.len(), pii.spans.len());
    assert_eq!(plan.policy_digest, Some(policy.digest));
    let typed = document_to_typed_with("dry", &doc, ViewId::SOURCE, OffsetMode::Utf8, &REWRITTEN_AND_DETAIL).unwrap();
    let layer = typed.layers.iter().find(|l| l.id == "turbo:pii-candidates").expect("the candidates");
    let candidates = doc.get::<PiiCandidates>().unwrap();
    let mut kept = 0;
    for (i, a) in layer.annotations.iter().enumerate() {
        let Some(Value::PiiCandidate(c)) = &a.value else { panic!("not a candidate") };
        let native = candidates.get(turbo_analysis::Ix::new(i as u32));
        match native.disposition {
            PiiDisposition::Kept { mention } => {
                kept += 1;
                let p = c.planned.as_ref().expect("a kept candidate is planned");
                let m = mention.get();
                assert_eq!(p.strategy, i32::from(plan.strategy[m].id()));
                assert_eq!(p.rule_id, plan.rule_id.get(m));
                assert_eq!(p.policy_digest, policy.digest.as_bytes().to_vec());
                assert_eq!(p.policy_revision, policy.revision);
            }
            _ => assert!(c.planned.is_none(), "only kept candidates are planned"),
        }
    }
    assert_eq!(kept, pii.spans.len(), "every mention's candidate is listed with its decision");
    assert!(plan.rule_id.iter().any(|id| id == "email"));
}

#[test]
fn the_findings_only_default_shows_no_redacted_text() {
    let doc =
        run(convert(&one(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[E]".into() })), &NoRegistry));
    let (view, _) = redacted(&doc);
    assert!(view.text().contains("[E]"));
    let findings = document_to_typed("findings", &doc, view.id(), OffsetMode::Utf8).unwrap();
    assert!(findings.raw_text.is_empty(), "findings only: the redacted text is withheld");
    assert!(findings.withheld_text_len.is_some());
    let bytes = findings.encode_to_vec();
    assert!(!contains(&bytes, "[E]") && !contains(&bytes, EMAIL));
    let asked = document_to_typed_with("asked", &doc, view.id(), OffsetMode::Utf8, &REWRITTEN_AND_DETAIL).unwrap();
    assert_eq!(asked.raw_text, view.text(), "asked for and allowed: the redacted text is shown");
}

#[test]
fn a_keyed_token_run_without_a_grant_releases_no_value() {
    let doc = run(convert(&one(keyed("tokens", 1)), &TestRegistry));
    let (view, _) = redacted(&doc);
    let typed = document_to_typed_with("keyed", &doc, view.id(), OffsetMode::Utf8, &REWRITTEN_AND_DETAIL).unwrap();
    let pii = typed.layers.iter().find(|l| l.id == "turbo:pii").expect("the mentions");
    for a in &pii.annotations {
        let Some(Value::Pii(m)) = &a.value else { panic!("not a mention") };
        let form = m.value.as_ref().and_then(|v| v.form.as_ref());
        assert!(
            matches!(form, None | Some(pb::pii_value::Form::Withheld(_))),
            "no value leaves without a grant: {form:?}"
        );
    }
    let bytes = typed.encode_to_vec();
    assert!(!contains(&bytes, EMAIL) && !contains(&bytes, CARD));
}

#[test]
fn a_planted_value_appears_in_no_edit_decision_receipt_or_error() {
    let mut msg = one(Strategy::Map(Box::new(pb::PiiMapStrategy {
        table: Some(pb::pii_map_strategy::Table::Entries(map_entries(&[(EMAIL, "J")]))),
        case_sensitive: false,
        target: pb::PiiMapTarget::NormalizedOrSurface as i32,
        fallback: None,
    })));
    msg.revision = 3;
    let policy = convert(&msg, &NoRegistry);
    for dry_run in [false, true] {
        let doc = run_with(TEXT, policy.clone(), dry_run);
        let target = doc.get_key(&REDACTED).map_or(ViewId::SOURCE, |v| v.id());
        let typed = document_to_typed_with("plant", &doc, target, OffsetMode::Utf8, &REWRITTEN_AND_DETAIL).unwrap();
        assert!(typed.receipt.is_some(), "results carry receipts");
        let bytes = typed.encode_to_vec();
        assert!(!contains(&bytes, EMAIL), "dry_run={dry_run}: the email leaves in no message");
        assert!(!contains(&bytes, CARD), "dry_run={dry_run}: the card leaves in no message");
    }
    // Errors and debug output name the policy, never its terms or keys.
    assert!(!format!("{policy:?}").contains(EMAIL));
    let named = NamedPolicies::from_messages(&[msg.clone()], &NoRegistry).unwrap();
    assert!(!format!("{named:?}").contains(EMAIL));
    let mut bad = msg;
    bad.default_strategy = Some(pb::PiiDefaultStrategy::default());
    let sel =
        pb::PiiRedactionSelection { policy: Some(pb::pii_redaction_selection::Policy::Inline(bad)), dry_run: false };
    let err = select(&sel, &NamedPolicies::default(), &NoRegistry).unwrap_err();
    assert!(!err.to_string().contains(EMAIL), "{err}");
}

fn with_unknown_kind_and_strategy(flag: bool) -> pb::RedactionPolicy {
    let mut p = policy(vec![
        rule("future-kind", Some(pb::PiiKindRef { kind: Some(pb::pii_kind_ref::Kind::Builtin(9999)) }), Vec::new()),
        rule(
            "email",
            builtin(pb::PiiKindId::Email),
            vec![pb::PiiStrategyEntry { when: Vec::new(), strategy: Some(pb::PiiRewriteStrategy { strategy: None }) }],
        ),
    ]);
    p.redact_and_flag_unknown = flag;
    p
}

#[test]
fn unknown_kinds_and_strategies_are_refused_unless_the_policy_opts_into_redact_and_flag() {
    let inline =
        |p| pb::PiiRedactionSelection { policy: Some(pb::pii_redaction_selection::Policy::Inline(p)), dry_run: false };
    let refused = select(&inline(with_unknown_kind_and_strategy(false)), &NamedPolicies::default(), &NoRegistry);
    assert!(matches!(refused, Err(SelectError::Policy(_))), "refused by default");
    assert!(NamedPolicies::from_messages(&[with_unknown_kind_and_strategy(false)], &NoRegistry).is_err());
    let flagged =
        select(&inline(with_unknown_kind_and_strategy(true)), &NamedPolicies::default(), &NoRegistry).unwrap();
    assert_eq!(
        flagged.policy.flags,
        vec![
            PolicyFlag::UnknownKind { rule_id: "future-kind".into(), id: 9999 },
            PolicyFlag::UnknownStrategy { rule_id: "email".into(), entry: 0 }
        ]
    );
    // The flagged strategy is a whole-value mask: the email is masked,
    // never passed through, and its edit names the rule.
    let doc = run(flagged.policy);
    let (view, edits) = redacted(&doc);
    assert!(!view.text().contains(EMAIL));
    assert!(view.text().contains(&"*".repeat(EMAIL.len())));
    assert!(edits.edits().any(|e| e.rule_id == "email"));
}

fn edit(t: &mut pb::TypedDocumentResult) -> &mut pb::PiiEditAnnotation {
    let l = t.layers.iter_mut().find(|l| l.id == "turbo:pii-edits").unwrap();
    match &mut l.annotations[0].value {
        Some(Value::PiiEdit(e)) => e,
        _ => panic!("not an edit"),
    }
}

#[test]
fn the_validator_refuses_a_bad_digest_a_stray_rule_id_and_a_stray_plan() {
    use turbo_kserve::results::typed::validate_typed_document;
    let policy = convert(&one(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[E]".into() })), &NoRegistry);
    let doc = run(policy.clone());
    let typed = document_to_typed("v", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    validate_typed_document(&typed).unwrap();
    let mut short = typed.clone();
    edit(&mut short).policy_digest.pop();
    assert!(validate_typed_document(&short).unwrap_err().message.contains("not 32 bytes"));
    let mut stray = typed.clone();
    let e = edit(&mut stray);
    e.policy_digest.clear();
    e.rule_id = "email".into();
    assert!(validate_typed_document(&stray).unwrap_err().message.contains("rule_id"));

    let dry = run_with(TEXT, policy, true);
    let typed = document_to_typed_with("v", &dry, ViewId::SOURCE, OffsetMode::Utf8, &REWRITTEN_AND_DETAIL).unwrap();
    validate_typed_document(&typed).unwrap();
    let mut no_digest = typed.clone();
    let l = no_digest.layers.iter_mut().find(|l| l.id == "turbo:pii-candidates").unwrap();
    for a in &mut l.annotations {
        if let Some(Value::PiiCandidate(c)) = &mut a.value
            && let Some(p) = &mut c.planned
        {
            p.policy_digest.clear();
        }
    }
    assert!(validate_typed_document(&no_digest).unwrap_err().message.contains("policy_digest is missing"));
}
