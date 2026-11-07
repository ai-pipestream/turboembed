//! The PII disclosure decision (#1904) on every server surface: a planted
//! value is searched for in the encoded bytes of each default answer, and
//! each output option is checked to need the request and the server's
//! policy both. The default is findings only: kinds, confidences and
//! spans, with every value withheld and no text or graph bytes.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::tiny;
use prost::Message;
use tokio_stream::StreamExt;
use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Provenance, Span, View, ViewId};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{
    AnalyzePipelineRequest, DocumentGraphRequest, DocumentGraphResponse, FormatDocumentRequest,
    InspectDocumentGraphRequest, OffsetMode, PiiOutputOptions, PiiValueForm, TypedDocumentResult, pii_value,
};
use turbo_kserve::proto::sink::DocumentSinkItem;
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::document::codecs;
use turbo_kserve::sink::{SinkConfig, SinkReceiver};
use turbo_kserve::{Serve, Server};
use turbo_pii::{Pii, PiiConfidence, PiiKind, PiiMention};

/// The planted value: an address the PII detector finds, lowercase so its
/// normalized form is the same bytes.
const PLANTED: &str = "alice.planted@example.com";
const TEXT: &str = "Write to alice.planted@example.com before noon.";
/// The masked view the `turbo-pii:masked` step writes.
const MASKED: u32 = 1;

fn model() -> ModelConfig {
    ModelConfig {
        bundle: tiny().to_str().unwrap().into(),
        name: None,
        device: Device::Index(common::cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

async fn server(serve: Serve) -> Server {
    Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap()
}

async fn client(s: &Server) -> AnalysisServiceClient<tonic::transport::Channel> {
    AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap()
}

fn holds(bytes: &[u8], needle: &str) -> bool {
    bytes.windows(needle.len()).any(|w| w == needle.as_bytes())
}

fn options(full_text: bool, rewritten_text: bool) -> Option<PiiOutputOptions> {
    Some(PiiOutputOptions { full_text, rewritten_text, ..Default::default() })
}

/// Runs the PII scan and mask steps on TEXT under `pii_output` and answers
/// every event's encoded bytes and the complete answer.
async fn progressive(s: &Server, pii_output: Option<PiiOutputOptions>) -> (Vec<Vec<u8>>, DocumentGraphResponse) {
    let req = AnalyzePipelineRequest {
        doc_id: "doc-pii".into(),
        text: TEXT.into(),
        offsets: OffsetMode::Utf8 as i32,
        steps: vec!["turbo-pii:pii".into(), "turbo-pii:masked".into()],
        pii_output,
        ..Default::default()
    };
    let mut stream = client(s).await.analyze_document_progressive(req).await.unwrap().into_inner();
    let (mut events, mut complete) = (Vec::new(), None);
    while let Some(ev) = stream.next().await {
        let ev = ev.unwrap();
        events.push(ev.encode_to_vec());
        if let Some(Event::Complete(done)) = ev.event {
            complete = Some(done);
        }
    }
    (events, complete.expect("the call ends with its complete answer"))
}

/// The graph of TEXT with its PII layer and masked view, asked for with
/// `full_text` from a server that allows it.
async fn pii_graph(s: &Server) -> Vec<u8> {
    let (_, complete) = progressive(s, options(true, false)).await;
    assert!(holds(&complete.graph, PLANTED), "positive control: the full graph holds the value");
    complete.graph.to_vec()
}

/// Asserts `typed` holds the email finding with its value withheld.
fn assert_findings(typed: &TypedDocumentResult) {
    let pii = typed.layers.iter().find(|l| l.id == "turbo:pii").expect("the pii layer");
    assert!(!pii.annotations.is_empty());
    for a in &pii.annotations {
        let Some(Value::Pii(m)) = &a.value else { panic!("a pii annotation") };
        assert!(matches!(m.value.as_ref().and_then(|v| v.form.as_ref()), Some(pii_value::Form::Withheld(_))));
        assert!(!a.spans.is_empty(), "the finding keeps its span");
    }
}

async fn inspect(s: &Server, graph: &[u8], view: u32, pii_output: Option<PiiOutputOptions>) -> TypedDocumentResult {
    let req = InspectDocumentGraphRequest {
        doc_id: "doc-pii".into(),
        graph: graph.to_vec(),
        view,
        offsets: OffsetMode::Utf8 as i32,
        pii_output,
        binding: None,
        operation_key: String::new(),
    };
    client(s).await.inspect_document_graph(req).await.unwrap().into_inner()
}

#[tokio::test]
async fn progressive_events_and_the_complete_answer_hold_no_value_by_default() {
    let s = server(Serve::default()).await;
    let (events, complete) = progressive(&s, None).await;
    for (i, bytes) in events.iter().enumerate() {
        assert!(!holds(bytes, PLANTED), "event {i} carries the planted value");
    }
    assert!(complete.graph.is_empty() && complete.source_text.is_empty());
    let findings = complete.findings.expect("the findings stand in for the graph");
    assert!(findings.raw_text.is_empty());
    assert_findings(&findings);
    s.stop().await;
}

#[tokio::test]
async fn inspect_projects_findings_without_text_or_values_on_every_view() {
    let s = server(Serve::default()).await;
    let graph = pii_graph(&s).await;
    for view in [0, MASKED] {
        let typed = inspect(&s, &graph, view, None).await;
        assert!(!holds(&typed.encode_to_vec(), PLANTED), "view {view}");
        assert!(typed.raw_text.is_empty(), "view {view}");
        assert_findings(&typed);
    }
    // A withheld text keeps its length, so the result still validates.
    let typed = inspect(&s, &graph, 0, None).await;
    assert_eq!(typed.withheld_text_len, Some(TEXT.len() as u32));
    let mut c = client(&s).await;
    assert_eq!(c.round_trip_typed_document(typed.clone()).await.unwrap().into_inner(), typed);
    let short = TypedDocumentResult { withheld_text_len: Some(3), ..typed.clone() };
    assert_eq!(c.round_trip_typed_document(short).await.unwrap_err().code(), tonic::Code::InvalidArgument);
    // The retired clear field is refused if a producer sets it.
    let mut leaky = typed.clone();
    let pii = leaky.layers.iter_mut().find(|l| l.id == "turbo:pii").unwrap();
    let Some(Value::Pii(m)) = &mut pii.annotations[0].value else { panic!("a pii annotation") };
    #[allow(deprecated)]
    {
        m.normalized = PLANTED.into();
    }
    let e = c.round_trip_typed_document(leaky).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::InvalidArgument);
    assert!(e.message().contains("normalized") && !holds(e.message().as_bytes(), PLANTED), "{e:?}");
    let both = TypedDocumentResult { raw_text: TEXT.into(), ..typed };
    assert_eq!(c.round_trip_typed_document(both).await.unwrap_err().code(), tonic::Code::InvalidArgument);
    // rewritten_text gives the masked view's text, never the source's.
    let masked = inspect(&s, &graph, MASKED, options(false, true)).await;
    assert!(!masked.raw_text.is_empty() && !holds(&masked.encode_to_vec(), PLANTED), "{}", masked.raw_text);
    assert!(inspect(&s, &graph, 0, options(false, true)).await.raw_text.is_empty());
    // full_text is the caller's own text back.
    assert_eq!(inspect(&s, &graph, 0, options(true, false)).await.raw_text, TEXT);
    s.stop().await;
}

#[tokio::test]
async fn round_trip_withholds_graph_and_source_text_unless_asked() {
    let s = server(Serve::default()).await;
    let graph = pii_graph(&s).await;
    let mut c = client(&s).await;
    let req = DocumentGraphRequest { graph: graph.clone().into(), pii_output: None };
    let out = c.round_trip_document_graph(req).await.unwrap().into_inner();
    assert!(!holds(&out.encode_to_vec(), PLANTED));
    assert!(out.graph.is_empty() && out.source_text.is_empty());
    assert!(out.layers.iter().any(|l| l.layer == "turbo:pii"), "the listing stays");
    assert_findings(out.findings.as_ref().unwrap());
    let full = DocumentGraphRequest { graph: graph.into(), pii_output: options(true, false) };
    let out = c.round_trip_document_graph(full).await.unwrap().into_inner();
    assert_eq!(out.source_text, TEXT);
    assert!(out.findings.is_none());
    s.stop().await;
}

#[tokio::test]
async fn values_are_refused_and_the_policy_caps_every_option() {
    let s = server(Serve::default()).await;
    let graph = pii_graph(&s).await;
    for form in [PiiValueForm::Token, PiiValueForm::Ciphertext, PiiValueForm::Clear] {
        let asked = Some(PiiOutputOptions { values: form as i32, ..Default::default() });
        let req = InspectDocumentGraphRequest { graph: graph.clone(), pii_output: asked, ..Default::default() };
        let e = client(&s).await.inspect_document_graph(req).await.unwrap_err();
        assert_eq!(e.code(), tonic::Code::Unimplemented, "{form:?}: {e:?}");
        assert!(!holds(e.message().as_bytes(), PLANTED));
    }
    s.stop().await;
    let closed = server(Serve { pii_output: PiiOutputPolicy::NONE, ..Serve::default() }).await;
    let req =
        InspectDocumentGraphRequest { graph: graph.clone(), pii_output: options(true, false), ..Default::default() };
    let e = client(&closed).await.inspect_document_graph(req).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::FailedPrecondition, "{e:?}");
    assert!(e.message().contains("full_text"), "{e:?}");
    let req = AnalyzePipelineRequest { text: TEXT.into(), pii_output: options(false, true), ..Default::default() };
    let e = client(&closed).await.analyze_document_progressive(req).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::FailedPrecondition, "{e:?}");
    // Findings only is always served.
    assert_findings(&inspect(&closed, &graph, 0, None).await);
    closed.stop().await;
    // A sink asking for more than the policy allows refuses the start.
    let sink = SinkConfig {
        endpoint: "http://127.0.0.1:1".into(),
        max_queue_items: 1,
        max_queue_bytes: 1 << 20,
        documents: false,
        typed: None,
        pii_output: Disclosure::FULL,
    };
    let serve = Serve { sink: Some(sink), pii_output: PiiOutputPolicy::NONE, ..Serve::default() };
    assert!(Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.is_err());
}

#[tokio::test]
async fn sink_items_carry_findings_and_no_value_by_default() {
    let got: Arc<Mutex<Vec<DocumentSinkItem>>> = Arc::default();
    let keep = got.clone();
    let receiver = SinkReceiver::new(move |item| keep.lock().unwrap().push(item));
    let sink = server(Serve { sink_receiver: Some(receiver), ..Serve::default() }).await;
    let config = SinkConfig {
        endpoint: format!("http://{}", sink.local_addr()),
        max_queue_items: 16,
        max_queue_bytes: 64 << 20,
        documents: true,
        typed: None,
        pii_output: Disclosure::FINDINGS,
    };
    let s = server(Serve { sink: Some(config), ..Serve::default() }).await;
    let graph = pii_graph(&s).await;
    assert!(s.forward_document("doc-fwd", graph).await.unwrap());
    // The progressive task offers its graph after the stream ends, so wait
    // for both offers before closing the sink.
    let start = std::time::Instant::now();
    while s.sink_stats().unwrap().offered < 2 && start.elapsed() < Duration::from_secs(10) {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.accepted_count, summary.rejected_count), (2, 0), "{:?}", summary.errors);
    let items = got.lock().unwrap().clone();
    assert_eq!(items.len(), 2, "the progressive call's graph and the forwarded one");
    for item in &items {
        assert!(!holds(&item.encode_to_vec(), PLANTED), "{}", item.doc_id);
        assert!(item.graph.is_empty());
        let typed = item.typed.as_ref().expect("the findings projection");
        assert_eq!(typed.doc_id, item.doc_id);
        assert_findings(typed);
    }
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn format_document_needs_full_text_and_no_format_writes_a_value() {
    let s = server(Serve::default()).await;
    let graph = pii_graph(&s).await;
    let mut c = client(&s).await;
    let req = |format_id: &str, pii_output| FormatDocumentRequest {
        doc_id: "doc-pii".into(),
        graph: graph.clone(),
        format_id: format_id.into(),
        offsets: OffsetMode::Utf8 as i32,
        pii_output,
    };
    let e = c.format_document(req("jsonld", None)).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::FailedPrecondition, "{e:?}");
    assert!(e.message().contains("full_text") && !holds(e.message().as_bytes(), PLANTED), "{e:?}");
    // With the text asked for, the text holds the value, but no PII
    // finding repeats it.
    let jsonld = c.format_document(req("jsonld", options(true, false))).await.unwrap().into_inner().content;
    let jsonld = String::from_utf8(jsonld).unwrap();
    assert!(jsonld.contains("turbo:PiiMention") && !jsonld.contains("\"normalized\""), "{jsonld}");
    let md = c.format_document(req("markdown", options(true, false))).await.unwrap().into_inner().content;
    let md = String::from_utf8(md).unwrap();
    let summary = md.split("## PII Summary").nth(1).expect("the PII summary").split("\n## ").next().unwrap();
    assert!(summary.contains("**email**") && !summary.contains(PLANTED), "{summary}");
    s.stop().await;
}

/// `detail` is decided and capped here; nothing reads it until #1903's
/// candidates and evidence (#1916) land, so today it changes no answer.
/// When #1916 and this change meet, that branch wires it and this test
/// moves to pinning what `detail` adds.
#[tokio::test]
async fn detail_is_decided_and_capped_but_adds_nothing_yet() {
    let s = server(Serve::default()).await;
    let graph = pii_graph(&s).await;
    let detail = Some(PiiOutputOptions { detail: true, ..Default::default() });
    // Each answer carries its own attempt's receipt (#1908); the rest is
    // the same.
    let without_receipt = |mut t: TypedDocumentResult| {
        t.receipt = None;
        t
    };
    assert_eq!(
        without_receipt(inspect(&s, &graph, 0, detail).await),
        without_receipt(inspect(&s, &graph, 0, None).await)
    );
    s.stop().await;
    let closed = server(Serve { pii_output: PiiOutputPolicy::NONE, ..Serve::default() }).await;
    let req = InspectDocumentGraphRequest {
        graph: graph.clone(),
        pii_output: Some(PiiOutputOptions { detail: true, ..Default::default() }),
        ..Default::default()
    };
    let e = client(&closed).await.inspect_document_graph(req).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::FailedPrecondition, "{e:?}");
    assert!(e.message().contains("detail"), "{e:?}");
    closed.stop().await;
}

#[tokio::test]
async fn the_started_event_sanitizes_the_views_of_an_initial_graph_holding_pii() {
    // An initial graph that already holds PII, beside a view declined with
    // a free-text reason that carries the planted value.
    let mut doc = Document::new(TEXT);
    let mut pii = Pii::new(ViewId::SOURCE);
    let at = TEXT.find(PLANTED).unwrap() as u32;
    let span = Span::new(at, at + PLANTED.len() as u32);
    pii.push(PiiMention { span, kind: PiiKind::Email, confidence: PiiConfidence::Format, normalized: PLANTED });
    doc.add(&Key::DEFAULT, pii, Provenance::by("turbo-pii:pii")).unwrap();
    let declined = Key::<View>::named("declined").id();
    doc.skip(declined, format!("declined near {PLANTED}"), Provenance::by("test:view")).unwrap();
    let graph = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let s = server(Serve::default()).await;
    let req = AnalyzePipelineRequest {
        doc_id: "doc-pii".into(),
        initial_graph: graph,
        offsets: OffsetMode::Utf8 as i32,
        steps: vec!["turbo-sentdetect:rules".into()],
        ..Default::default()
    };
    let mut stream = client(&s).await.analyze_document_progressive(req).await.unwrap().into_inner();
    let Some(Event::Started(started)) = stream.next().await.unwrap().unwrap().event else { panic!("started first") };
    let view = started.views.iter().find(|v| v.instance == "declined").expect("the declined view is listed");
    assert!(view.error.is_some(), "the declined view says why, as a typed error");
    assert!(!holds(&started.encode_to_vec(), PLANTED));
    while let Some(ev) = stream.next().await {
        let ev = ev.unwrap();
        assert!(!holds(&ev.encode_to_vec(), PLANTED), "a later event carries the planted value: {ev:?}");
    }
    s.stop().await;
}

#[tokio::test]
async fn snapshots_before_the_pii_step_carry_no_graph_either() {
    // A step that runs before the PII scan: its snapshot holds the text,
    // and the document holds no findings yet, so the request decides.
    let s = server(Serve::default()).await;
    let run = |pii_output: Option<PiiOutputOptions>| {
        let req = AnalyzePipelineRequest {
            doc_id: "doc-pii".into(),
            text: TEXT.into(),
            offsets: OffsetMode::Utf8 as i32,
            steps: vec!["turbo-sentdetect:rules".into(), "turbo-pii:pii".into()],
            pii_output,
            ..Default::default()
        };
        let s = &s;
        async move {
            let mut stream = client(s).await.analyze_document_progressive(req).await.unwrap().into_inner();
            let mut batches = Vec::new();
            while let Some(ev) = stream.next().await {
                if let Some(Event::LayersReady(b)) = ev.unwrap().event {
                    batches.push(b);
                }
            }
            batches
        }
    };
    let batches = run(None).await;
    let first =
        batches.iter().find(|b| b.completed_step == "turbo-sentdetect:rules").expect("the sentence step's snapshot");
    assert!(first.layers.iter().all(|l| l.layer != "turbo:pii"), "positive control: it ran before the PII scan");
    for b in &batches {
        assert!(b.graph.is_empty() && !holds(&b.encode_to_vec(), PLANTED), "{} carries the text", b.completed_step);
    }
    // Asked for, the text does leave, from the first snapshot on.
    let full = run(options(true, false)).await;
    assert!(full.iter().any(|b| b.completed_step == "turbo-sentdetect:rules" && holds(&b.graph, PLANTED)));
    s.stop().await;
}
