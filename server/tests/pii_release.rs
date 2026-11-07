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
use turbo_kserve::proto::audit::PiiAccessForm;
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

fn ask(form: PiiValueForm, operation_key: &str, attempt: u32) -> InspectDocumentGraphRequest {
    InspectDocumentGraphRequest {
        doc_id: "doc-release".into(),
        graph: graph(),
        view: 0,
        offsets: OffsetMode::Utf8 as i32,
        pii_output: Some(PiiOutputOptions {
            values: form as i32,
            operation_key: operation_key.into(),
            attempt,
            ..Default::default()
        }),
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

    let typed = c.inspect_document_graph(ask(PiiValueForm::Clear, "op-clear", 0)).await.unwrap().into_inner();
    let instance = typed.layers.iter().find(|l| l.id == "turbo:pii").unwrap().instance.clone();
    let got = forms(&typed);
    assert_eq!(got[0], pii_value::Form::Clear(EMAIL.into()), "the granted kind is released");
    assert!(withheld(&got[1]), "the phone kind is not granted");
    assert!(!holds(&typed.encode_to_vec(), PHONE.as_bytes()));
    assert!(typed.raw_text.is_empty(), "values do not turn on the text");

    let typed = c.inspect_document_graph(ask(PiiValueForm::Token, "op-token", 0)).await.unwrap().into_inner();
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
    let first = c.inspect_document_graph(ask(PiiValueForm::Token, "op-retry", 3)).await.unwrap().into_inner();
    let again = c.inspect_document_graph(ask(PiiValueForm::Token, "op-retry", 3)).await.unwrap().into_inner();
    assert_eq!(first, again, "a retry answers the same tokens");
    assert!(forms(&first).iter().all(|f| matches!(f, pii_value::Form::Token(_))), "`*` grants every kind");
    assert_eq!(audit.calls.lock().unwrap().len(), 2, "each attempt is recorded before it answers");
    let events = audit.events.lock().unwrap().clone();
    assert_eq!(events.len(), 1, "both calls carry one event id");
    assert_eq!(events.values().next().unwrap().mentions.len(), 2);
    // Another attempt is another event.
    c.inspect_document_graph(ask(PiiValueForm::Token, "op-retry", 4)).await.unwrap();
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
    let mut req = tonic::Request::new(ask(PiiValueForm::Clear, "op-spoof", 0));
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
        Ok(mut c) => c.inspect_document_graph(ask(PiiValueForm::Clear, "op-spoof", 1)).await.is_err(),
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
        let e = c.inspect_document_graph(ask(PiiValueForm::Clear, "op-fail", 0)).await.unwrap_err();
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
    assert_eq!(code(c.inspect_document_graph(ask(PiiValueForm::Ciphertext, "op", 0)).await), Code::Unimplemented);
    assert_eq!(code(c.inspect_document_graph(ask(PiiValueForm::Clear, "op", 0)).await), Code::FailedPrecondition);
    assert_eq!(code(c.inspect_document_graph(ask(PiiValueForm::Token, "", 0)).await), Code::InvalidArgument);
    // Only InspectDocumentGraph releases values.
    let options =
        Some(PiiOutputOptions { values: PiiValueForm::Token as i32, operation_key: "op".into(), ..Default::default() });
    let e = c
        .round_trip_document_graph(DocumentGraphRequest { graph: graph().into(), pii_output: options })
        .await
        .unwrap_err();
    assert_eq!(e.code(), Code::Unimplemented, "{e:?}");
    assert!(audit.calls.lock().unwrap().is_empty());
    s.stop().await;
}
