//! Release audit events name the redaction policy per released mention
//! (#1882 part C): a release from a redacted graph copies each mention's
//! rule id, revision and digest from its covering `turbo:pii-edits` edit,
//! a graph with no redaction leaves them empty, a policy change is a new
//! event id while a retry is still one event, and an edit naming a rule
//! with no policy digest refuses the release before any audit call.

#[path = "pii_release/certs.rs"]
mod certs;
mod common;
#[allow(dead_code)]
#[path = "pii_release/fixture.rs"]
mod fixture;
#[path = "pii_policy/common.rs"]
mod policy;

use std::sync::Arc;
use std::time::Duration;

use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Key, Layer, Provenance, View};
use turbo_kserve::pii_policy::{NoRegistry, REDACT_STEP, RedactStep, RedactionPolicy};
use turbo_kserve::proto::audit::{PiiAccessEvent, RecordPiiAccessRequest};
use turbo_kserve::proto::result::{
    self as pb, InspectDocumentGraphRequest, OffsetMode, PiiOutputOptions, PiiValueForm,
};
use turbo_kserve::release::POLICY_ID;
use turbo_kserve::results::document::codecs;
use turbo_pii::rewrite::REDACTED;
use turbo_pii::{Pii, PiiEdits};

use certs::client_fingerprint;
use fixture::*;
use policy::{builtin, entry, rule};

const SECOND: Duration = Duration::from_secs(1);

fn redaction(revision: u64) -> Arc<RedactionPolicy> {
    let mut p = policy::policy(vec![rule(
        "email",
        builtin(pb::PiiKindId::Email),
        vec![entry(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[E]".into() }))],
    )]);
    p.revision = revision;
    Arc::new(RedactionPolicy::from_proto(&p, &NoRegistry).unwrap())
}

/// `graph()` with the redacted view and edits of `policy`; with
/// `drop_digest`, its edits keep their rule ids but lose the digest.
fn redacted(policy: Arc<RedactionPolicy>, drop_digest: bool) -> Vec<u8> {
    let mut doc: Document = import(&DocumentGraph::from_bytes(&graph()).unwrap(), codecs()).unwrap();
    let (view, mut edits) = {
        let src = doc.get_key(&Key::<View>::DEFAULT).unwrap();
        RedactStep::new(policy).rewrite(src, doc.get::<Pii>().unwrap()).unwrap()
    };
    if drop_digest {
        edits.policy_digest = None;
    }
    doc.add(&REDACTED, view, Provenance::by(REDACT_STEP)).unwrap();
    // The document gave the view its id; the edits are bound to it as the
    // analyzer binds a step's writes.
    edits.bind_pending_view(doc.get_key(&REDACTED).unwrap().id());
    doc.add(&Key::<PiiEdits>::named(REDACTED.name), edits, Provenance::by(REDACT_STEP)).unwrap();
    export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

fn ask(graph: Vec<u8>, operation_key: &str) -> InspectDocumentGraphRequest {
    InspectDocumentGraphRequest {
        doc_id: "doc-release-policy".into(),
        graph,
        view: 0,
        offsets: OffsetMode::Utf8 as i32,
        operation_key: operation_key.into(),
        pii_output: Some(PiiOutputOptions { values: PiiValueForm::Token as i32, ..Default::default() }),
        ..Default::default()
    }
}

fn sent(audit: &Audit) -> Vec<PiiAccessEvent> {
    let calls = audit.calls.lock().unwrap().clone();
    calls.iter().map(|c| RecordPiiAccessRequest::decode(c.as_slice()).unwrap().event.unwrap()).collect()
}

#[tokio::test]
async fn each_released_mention_names_its_covering_redaction_policy() {
    let files = Files::create("policy-refs");
    let (audit, at) = audit(Answer::Ack).await;
    let grant = format!("{}:token:*", client_fingerprint());
    let s = server(&files, release(&files, &[grant], at, SECOND), open_policy()).await;
    let mut c = fixture::connect(&s, true).await.unwrap();
    let (r1, r2) = (redaction(1), redaction(2));

    // No redact step: the mentions name no redaction policy.
    c.inspect_document_graph(ask(graph(), "op-plain")).await.unwrap();
    // After the redact step: the email names its rule, the phone the
    // policy default; both name the revision and the digest.
    c.inspect_document_graph(ask(redacted(r1.clone(), false), "op-redact")).await.unwrap();
    // A retry of that release, and the same release under revision 2.
    c.inspect_document_graph(ask(redacted(r1.clone(), false), "op-redact")).await.unwrap();
    c.inspect_document_graph(ask(redacted(r2.clone(), false), "op-redact")).await.unwrap();
    let e = sent(&audit);
    assert_eq!(e.len(), 4);

    for m in &e[0].mentions {
        assert_eq!((m.rule_id.as_str(), m.policy_revision, m.redaction_policy_digest.len()), ("", 0, 0));
    }
    let by_kind = |ev: &PiiAccessEvent, kind: pb::PiiKindId| {
        ev.mentions.iter().find(|m| m.builtin_kind == kind as u32).cloned().expect("the kind is released")
    };
    let email = by_kind(&e[1], pb::PiiKindId::Email);
    assert_eq!(email.rule_id, "email");
    assert_eq!(email.policy_revision, 1);
    assert_eq!(email.redaction_policy_digest, r1.digest.as_bytes().to_vec());
    let phone = by_kind(&e[1], pb::PiiKindId::Phone);
    assert_eq!((phone.rule_id.as_str(), phone.policy_revision), ("", 1), "the default strategy names no rule");
    assert_eq!(phone.redaction_policy_digest, r1.digest.as_bytes().to_vec());
    // The event-level policy stays the server ceiling and grant.
    assert!(e.iter().all(|ev| ev.policy_id == POLICY_ID && ev.policy_digest == e[0].policy_digest));

    assert_eq!(e[1].event_id, e[2].event_id, "a retry is one event");
    assert_ne!(e[1].event_id, e[3].event_id, "a policy change is another event");
    assert_eq!(by_kind(&e[3], pb::PiiKindId::Email).redaction_policy_digest, r2.digest.as_bytes().to_vec());
    let events = audit.events.lock().unwrap().clone();
    assert_eq!(events.len(), 3, "plain, revision 1 once, revision 2");
    s.stop().await;
}

#[tokio::test]
async fn an_edit_naming_a_rule_without_a_policy_digest_refuses_before_the_audit() {
    let files = Files::create("policy-bad");
    let (audit, at) = audit(Answer::Ack).await;
    let grant = format!("{}:token:*", client_fingerprint());
    let s = server(&files, release(&files, &[grant], at, SECOND), open_policy()).await;
    let mut c = fixture::connect(&s, true).await.unwrap();
    let err = c.inspect_document_graph(ask(redacted(redaction(1), true), "op-bad")).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err:?}");
    assert!(err.message().contains("policy_digest is empty"), "{}", err.message());
    assert!(!holds(err.message().as_bytes(), EMAIL.as_bytes()), "{}", err.message());
    assert!(audit.calls.lock().unwrap().is_empty(), "no audit call was made");
    s.stop().await;
}
