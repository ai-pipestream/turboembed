//! The per-document PII risk (#1889) through the server: the
//! `turbo-pii:risk` step writes `turbo:pii-risk`, its wire form validates
//! and carries no value, a tampered one is refused, the suggested policy
//! converts and redacts every mention, and the findings-only gRPC answer
//! carries the risk.

mod common;
#[path = "pii_redact/fixture.rs"]
mod fixture;

use fixture::{CARD, EMAIL, TEXT};
use prost::Message;
use tokio_stream::StreamExt;
use turbo_analysis::{Document, ViewId};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::pii_policy::{NoRegistry, RedactionPolicy, SUGGESTED_POLICY_ID, suggested_policy};
use turbo_kserve::progressive::compile_analyzer;
use turbo_kserve::proto::result as pb;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::results::typed::{document_to_typed, validate_typed_document};
use turbo_kserve::{Serve, Server};
use turbo_pii::rewrite::REDACTED;

const STEPS: &[&str] = &["turbo-pii:pii", "turbo-pii:risk"];

fn analyzed(text: &str) -> Document {
    let analyzer = compile_analyzer(&Document::new(""), STEPS).expect("the pipeline builds");
    let mut docs = vec![Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.expect("the pipeline runs");
    }
    docs.pop().unwrap()
}

fn typed(text: &str) -> pb::TypedDocumentResult {
    document_to_typed("risk", &analyzed(text), ViewId::SOURCE, OffsetMode::Utf8).unwrap()
}

fn risk_layer(t: &pb::TypedDocumentResult) -> &pb::TypedLayer {
    t.layers.iter().find(|l| l.id == "turbo:pii-risk").expect("the risk layer")
}

fn risk(t: &pb::TypedDocumentResult) -> &pb::PiiRiskAnnotation {
    match &risk_layer(t).annotations[0].value {
        Some(Value::PiiRisk(r)) => r,
        other => panic!("a pii risk annotation, not {other:?}"),
    }
}

fn holds(bytes: &[u8], needle: &str) -> bool {
    bytes.windows(needle.len()).any(|w| w == needle.as_bytes())
}

#[test]
fn the_step_writes_a_valid_risk_with_no_value() {
    let t = typed(TEXT);
    validate_typed_document(&t).unwrap();
    let layer = risk_layer(&t);
    assert_eq!(layer.annotations.len(), 1);
    assert!(layer.annotations[0].spans.is_empty(), "the risk covers the document");
    let r = risk(&t);
    assert_eq!(r.level, pb::PiiRiskLevel::Critical as i32, "a Luhn-checked card is critical");
    assert_eq!(r.mentions, 2);
    assert_eq!(r.kinds.len(), 2);
    let bytes = layer.encode_to_vec();
    assert!(!holds(&bytes, EMAIL) && !holds(&bytes, "4111"), "the risk layer holds no value");
    let plain = typed("Nothing personal here.");
    assert_eq!(risk(&plain).level, pb::PiiRiskLevel::None as i32);
    assert!(risk(&plain).kinds.is_empty());
}

#[test]
fn the_same_text_gives_the_same_risk_bytes() {
    assert_eq!(risk(&typed(TEXT)).encode_to_vec(), risk(&typed(TEXT)).encode_to_vec());
}

/// Whether the validator refuses the result of `TEXT` once `f` changed its
/// risk layer.
fn refused_after(f: impl FnOnce(&mut pb::TypedLayer)) -> bool {
    let mut t = typed(TEXT);
    f(t.layers.iter_mut().find(|l| l.id == "turbo:pii-risk").unwrap());
    validate_typed_document(&t).is_err()
}

/// [`refused_after`] a change to the risk annotation itself.
fn refused_with(f: impl FnOnce(&mut pb::PiiRiskAnnotation)) -> bool {
    refused_after(|l| match &mut l.annotations[0].value {
        Some(Value::PiiRisk(r)) => f(r),
        _ => unreachable!(),
    })
}

#[test]
fn a_tampered_risk_is_refused() {
    assert!(refused_with(|r| r.level = pb::PiiRiskLevel::Low as i32), "a level its score does not give");
    assert!(refused_with(|r| r.level = 0), "an unspecified level");
    assert!(refused_with(|r| r.mentions += 1), "a count that is not the sum");
    assert!(refused_with(|r| r.kinds[0].severity = 9), "an unknown severity");
    assert!(refused_with(|r| r.kinds[0].suggested = 0), "an unspecified suggestion");
    assert!(refused_with(|r| r.kinds.swap(0, 1)), "kinds out of order");
    assert!(refused_with(|r| r.score = f32::NAN), "a nan score");
    let none = |r: &mut pb::PiiRiskAnnotation| {
        r.score = 0.0;
        r.level = pb::PiiRiskLevel::None as i32;
    };
    assert!(refused_with(none), "a zero score and NONE over positive contributions");
    assert!(refused_with(|r| r.kinds[0].weighted_count = 0.0), "no weight for a mention");
    assert!(refused_with(|r| r.kinds[0].contribution *= 2.0), "a contribution its weight does not give");
    assert!(refused_with(|r| r.density_per_1000_chars = 0.0), "no density for mentions");
    let custom = Some(pb::PiiKindRef { kind: Some(pb::pii_kind_ref::Kind::Custom(0)) });
    assert!(refused_with(|r| r.kinds[0].kind = custom), "a custom index outside the table");
    assert!(refused_after(|l| l.annotations.push(l.annotations[0].clone())), "a second annotation");
    assert!(refused_after(|l| l.annotations[0].spans.push(pb::SpanFragment::exact(0, 4))), "a span");
    assert!(!refused_after(|_| ()), "control: the untouched result validates");
}

#[test]
fn the_suggested_policy_converts_and_redacts_every_mention() {
    let t = typed(TEXT);
    let msg = suggested_policy(risk_layer(&t)).expect("a suggested policy");
    assert_eq!(msg.id, SUGGESTED_POLICY_ID);
    assert_eq!(msg.kind_rules.len(), 2);
    assert!(
        msg.kind_rules.iter().all(|r| r.release_ceiling.is_none()),
        "a suggested policy caps no release form (#1943)"
    );
    let pii = t.layers.iter().find(|l| l.id == "turbo:pii").unwrap();
    assert!(suggested_policy(pii).is_none(), "only a risk layer suggests a policy");
    let policy = RedactionPolicy::from_proto(&msg, &NoRegistry).expect("the suggestion is a valid policy");
    assert!(policy.flags.is_empty());
    let doc = fixture::run(std::sync::Arc::new(policy));
    let text = doc.get_key(&REDACTED).unwrap().text();
    assert!(!text.contains(EMAIL) && !text.contains(CARD), "{text}");
    assert!(text.contains("[EMAIL]"), "{text}");
    assert!(text.contains("**** **** **** 1111"), "{text}");
}

fn model() -> ModelConfig {
    ModelConfig {
        bundle: common::tiny().to_str().unwrap().into(),
        name: None,
        device: Device::Index(common::cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

#[tokio::test]
async fn the_findings_only_answer_carries_the_risk() {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, Serve::default()).await.unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let req = pb::AnalyzePipelineRequest {
        doc_id: "doc-risk".into(),
        text: TEXT.into(),
        offsets: pb::OffsetMode::Utf8 as i32,
        steps: STEPS.iter().map(|&s| s.into()).collect(),
        ..Default::default()
    };
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut complete = None;
    while let Some(ev) = stream.next().await {
        let ev = ev.unwrap();
        assert!(!holds(&ev.encode_to_vec(), EMAIL), "an event carries the planted value");
        if let Some(Event::Complete(done)) = ev.event {
            complete = Some(done);
        }
    }
    let findings = complete.expect("a complete answer").findings.expect("findings stand in for the graph");
    let r = risk(&findings);
    assert_eq!(r.level, pb::PiiRiskLevel::Critical as i32);
    assert_eq!(r.mentions, 2);
    s.stop().await;
}
