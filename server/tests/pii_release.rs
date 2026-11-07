//! PII value release (#1909) over real mTLS: a calling service is its
//! verified client certificate, a grant covers the kinds it names, the
//! audit service acknowledges every release before a value leaves, and a
//! retry has one event. Metadata confers nothing, and no planted value or
//! key byte reaches an event, an error or a `Debug` string.

#[path = "pii_release/certs.rs"]
mod certs;
mod common;
#[path = "pii_release/fixture.rs"]
mod fixture;

use std::time::Duration;

use prost::Message;
use tonic::Code;
use turbo_kserve::proto::audit::{PiiAccessForm, RecordPiiAccessRequest};
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{
    DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, PiiOutputOptions, PiiValueForm, TypedDocumentResult,
    pii_value,
};
use turbo_kserve::release::POLICY_ID;
use turbo_kserve::release::caps::{Form, Forms};
use turbo_kserve::results::disclosure::PiiOutputPolicy;
use turbo_pii::PiiKind;

use certs::client_fingerprint;
use fixture::*;

const SECOND: Duration = Duration::from_secs(1);

fn ask(form: PiiValueForm, operation_key: &str) -> InspectDocumentGraphRequest {
    InspectDocumentGraphRequest {
        doc_id: "doc-release".into(),
        graph: graph(),
        view: 0,
        offsets: OffsetMode::Utf8 as i32,
        operation_key: operation_key.into(),
        pii_output: Some(PiiOutputOptions { values: form as i32, ..Default::default() }),
        ..Default::default()
    }
}

/// The value form of each PII mention, in layer order.
fn forms(typed: &TypedDocumentResult) -> Vec<pii_value::Form> {
    let pii = typed.layers.iter().find(|l| l.id == "turbo:pii").expect("the pii layer");
    pii.annotations
        .iter()
        .map(|a| match &a.value {
            Some(Value::Pii(m)) => m.value.clone().and_then(|v| v.form).expect("a value form"),
            _ => panic!("a pii annotation"),
        })
        .collect()
}

fn withheld(f: &pii_value::Form) -> bool {
    matches!(f, pii_value::Form::Withheld(_))
}

#[tokio::test]
async fn a_grant_for_one_kind_releases_that_kind_only_after_the_audit_ack() {
    let files = Files::create("kind");
    let (audit, at) = audit(Answer::Ack).await;
    let grant = format!("{}:token+clear:email", client_fingerprint());
    let s = server(&files, release(&files, &[grant], at, SECOND), open_policy()).await;
    let mut c = connect(&s, true).await.unwrap();

    let typed = c.inspect_document_graph(ask(PiiValueForm::Clear, "op-clear")).await.unwrap().into_inner();
    let instance = typed.layers.iter().find(|l| l.id == "turbo:pii").unwrap().instance.clone();
    let got = forms(&typed);
    assert_eq!(got[0], pii_value::Form::Clear(EMAIL.into()), "the granted kind is released");
    assert!(withheld(&got[1]), "the phone kind is not granted");
    assert!(!holds(&typed.encode_to_vec(), PHONE.as_bytes()));
    assert!(typed.raw_text.is_empty(), "values do not turn on the text");

    let typed = c.inspect_document_graph(ask(PiiValueForm::Token, "op-token")).await.unwrap().into_inner();
    let got = forms(&typed);
    let pii_value::Form::Token(t) = &got[0] else { panic!("a token: {got:?}") };
    assert_eq!(t.token.len(), 32);
    let key = t.key.clone().unwrap();
    assert_eq!((key.id.as_str(), key.version), ("tok", 4));
    assert!(withheld(&got[1]));
    assert!(!holds(&typed.encode_to_vec(), EMAIL.as_bytes()), "a token release carries no clear value");
    assert!(!holds(&typed.encode_to_vec(), KEY_BYTES));

    let events = audit.events.lock().unwrap().clone();
    assert_eq!(events.len(), 2);
    for e in events.values() {
        assert_eq!(e.service_fingerprint, certs::fingerprint_bytes().to_vec());
        assert_eq!(e.mentions.len(), 1, "only the released mention is listed");
        let m = &e.mentions[0];
        assert_eq!((m.layer_instance.as_str(), m.index), (instance.as_str(), 0));
        assert_eq!(m.builtin_kind, u32::from(PiiKind::Email.builtin_id().unwrap()));
        assert_eq!(e.effective.len(), 1);
        assert_eq!((e.policy_id.as_str(), e.policy_digest.len(), e.doc_id.as_str()), (POLICY_ID, 32, "doc-release"));
        assert!(e.time_unix_nanos > 0 && e.event_id.len() == 64);
    }
    let forms_seen: Vec<i32> = events.values().map(|e| e.mentions[0].form).collect();
    assert!(forms_seen.contains(&(PiiAccessForm::Clear as i32)) && forms_seen.contains(&(PiiAccessForm::Token as i32)));
    for call in audit.calls.lock().unwrap().iter() {
        for planted in [EMAIL.as_bytes(), PHONE.as_bytes(), KEY_BYTES] {
            assert!(!holds(call, planted), "an audit event carries a planted value or key byte");
        }
    }
    s.stop().await;
}

#[tokio::test]
async fn a_retried_release_records_one_event() {
    let files = Files::create("retry");
    let (audit, at) = audit(Answer::Ack).await;
    let grant = format!("{}:token:*", client_fingerprint());
    let s = server(&files, release(&files, &[grant], at, SECOND), open_policy()).await;
    let mut c = connect(&s, true).await.unwrap();
    // Two attempts at one operation: same key, and each answer's receipt
    // names its own attempt id.
    let first = c.inspect_document_graph(ask(PiiValueForm::Token, "op-retry")).await.unwrap().into_inner();
    let again = c.inspect_document_graph(ask(PiiValueForm::Token, "op-retry")).await.unwrap().into_inner();
    assert_eq!(forms(&first), forms(&again), "a retry answers the same tokens");
    let attempt = |t: &TypedDocumentResult| t.receipt.as_ref().expect("a receipt").attempt_id.clone();
    assert_ne!(attempt(&first), attempt(&again), "two attempts");
    assert!(forms(&first).iter().all(|f| matches!(f, pii_value::Form::Token(_))), "`*` grants every kind");
    let calls = audit.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 2, "each attempt is recorded before it answers");
    let sent: Vec<_> =
        calls.iter().map(|c| RecordPiiAccessRequest::decode(c.as_slice()).unwrap().event.unwrap()).collect();
    assert_eq!(sent[0].event_id, sent[1].event_id, "the attempt is not part of the event id");
    assert_eq!((sent[0].attempt_id.clone(), sent[1].attempt_id.clone()), (attempt(&first), attempt(&again)));
    let events = audit.events.lock().unwrap().clone();
    assert_eq!(events.len(), 1, "one recorded event");
    let e = events.values().next().unwrap();
    assert_eq!((e.mentions.len(), e.attempt_id.clone(), e.input_digest.len()), (2, attempt(&first), 32));
    assert_eq!(e.operation_key, "op-retry");
    // Another operation is another event.
    c.inspect_document_graph(ask(PiiValueForm::Token, "op-other")).await.unwrap();
    assert_eq!(audit.events.lock().unwrap().len(), 2);
    s.stop().await;
}

#[tokio::test]
async fn spoofed_identity_metadata_has_no_effect() {
    let files = Files::create("spoof");
    let (audit, at) = audit(Answer::Ack).await;
    // The grant names a service that is not the connecting one.
    let granted = "ab".repeat(32);
    let s = server(&files, release(&files, &[format!("{granted}:clear:*")], at, SECOND), open_policy()).await;
    let mut c = connect(&s, true).await.unwrap();
    let mut req = tonic::Request::new(ask(PiiValueForm::Clear, "op-spoof"));
    for header in ["x-client-cert-sha256", "x-forwarded-client-cert", "x-ssl-client-fingerprint", "x-turbo-service"] {
        req.metadata_mut().insert(header, granted.parse().unwrap());
    }
    let typed = c.inspect_document_graph(req).await.unwrap().into_inner();
    assert!(forms(&typed).iter().all(withheld), "metadata confers nothing");
    assert!(!holds(&typed.encode_to_vec(), EMAIL.as_bytes()));
    assert!(audit.calls.lock().unwrap().is_empty(), "nothing released, nothing recorded");
    // A connection without a client certificate does not reach the call.
    let refused = match connect(&s, false).await {
        Err(_) => true,
        Ok(mut c) => c.inspect_document_graph(ask(PiiValueForm::Clear, "op-spoof")).await.is_err(),
    };
    assert!(refused, "mTLS refuses a client without a certificate");
    s.stop().await;
}

#[tokio::test]
async fn an_audit_failure_blocks_the_release() {
    let grant = format!("{}:clear:email", client_fingerprint());
    let gone = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    for (tag, answer) in
        [("fail", Some(Answer::Fail)), ("wrong", Some(Answer::WrongId)), ("slow", Some(Answer::Slow)), ("gone", None)]
    {
        let files = Files::create(tag);
        let (calls, at) = match answer {
            Some(a) => {
                let (audit, at) = audit(a).await;
                (Some(audit), at)
            }
            None => (None, gone),
        };
        let s = server(
            &files,
            release(&files, std::slice::from_ref(&grant), at, Duration::from_millis(300)),
            open_policy(),
        )
        .await;
        let mut c = connect(&s, true).await.unwrap();
        let e = c.inspect_document_graph(ask(PiiValueForm::Clear, "op-fail")).await.unwrap_err();
        assert_eq!(e.code(), Code::Unavailable, "{tag}: {e:?}");
        assert!(e.message().contains("no value released"), "{tag}: {e:?}");
        for planted in [EMAIL, PHONE] {
            assert!(!e.message().contains(planted) && !format!("{e:?}").contains(planted), "{tag}");
        }
        if let Some(a) = calls {
            assert!(a.events.lock().unwrap().is_empty() || answer == Some(Answer::Slow), "{tag}");
        }
        s.stop().await;
    }
}

#[tokio::test]
async fn releases_are_refused_where_they_are_not_offered() {
    let files = Files::create("refuse");
    let (audit, at) = audit(Answer::Ack).await;
    let grant = format!("{}:token+clear:*", client_fingerprint());
    let config = release(&files, &[grant], at, SECOND);
    let debug = format!("{config:?}");
    assert!(!holds(debug.as_bytes(), KEY_BYTES), "{debug}");
    let token_only = PiiOutputPolicy { values: Forms::of(Form::Token), ..PiiOutputPolicy::ALL };
    let s = server(&files, config, token_only).await;
    let mut c = connect(&s, true).await.unwrap();
    let code = |r: Result<tonic::Response<TypedDocumentResult>, tonic::Status>| r.unwrap_err().code();
    assert_eq!(code(c.inspect_document_graph(ask(PiiValueForm::Ciphertext, "op")).await), Code::Unimplemented);
    assert_eq!(code(c.inspect_document_graph(ask(PiiValueForm::Clear, "op")).await), Code::FailedPrecondition);
    // Without an operation key the release is refused and the findings
    // are served; nothing is recorded.
    let typed = c.inspect_document_graph(ask(PiiValueForm::Token, "")).await.unwrap().into_inner();
    assert!(forms(&typed).iter().all(withheld));
    // The default path on a release server withholds every value.
    let mut plain = ask(PiiValueForm::Unspecified, "op");
    plain.pii_output = None;
    let typed = c.inspect_document_graph(plain).await.unwrap().into_inner();
    assert!(forms(&typed).iter().all(withheld));
    // Only InspectDocumentGraph releases values.
    let options = Some(PiiOutputOptions { values: PiiValueForm::Token as i32, ..Default::default() });
    let e = c
        .round_trip_document_graph(DocumentGraphRequest { graph: graph().into(), pii_output: options })
        .await
        .unwrap_err();
    assert_eq!(e.code(), Code::Unimplemented, "{e:?}");
    assert!(audit.calls.lock().unwrap().is_empty());
    s.stop().await;
}
