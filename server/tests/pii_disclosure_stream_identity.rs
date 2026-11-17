//! Over the wire, no answer about a document holding PII carries an
//! unkeyed digest of the text it withholds (#2403): the progressive
//! stream's events and its terminal receipt, `InspectDocumentGraph`, and
//! the items a sink is offered. A stream that withheld its graph is still
//! complete by its receipt, and with `full_text` every digest is back.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::tiny;
use prost::Message;
use tokio_stream::StreamExt;
use turbo_analysis::ViewTextDigest;
use turbo_analysis::export::graph_content_digest;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{
    AnalyzePipelineRequest, AnalyzeProgressiveEvent, InspectDocumentGraphRequest, OffsetMode, PiiOutputOptions,
    ProcessingStatus,
};
use turbo_kserve::proto::sink::DocumentSinkItem;
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::receipt::stream::{Incomplete, terminal_receipt};
use turbo_kserve::sink::{SinkConfig, SinkReceiver};
use turbo_kserve::{Serve, Server};

/// A source short enough to guess: the whole text is the address.
const TEXT: &str = "alice.planted@example.com";
const KEY: &str = "batch-9/doc-3";
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

fn options(full_text: bool, rewritten_text: bool) -> Option<PiiOutputOptions> {
    Some(PiiOutputOptions { full_text, rewritten_text, ..Default::default() })
}

/// Runs the PII scan and mask steps on TEXT under `pii_output`.
async fn progressive(s: &Server, pii_output: Option<PiiOutputOptions>) -> Vec<AnalyzeProgressiveEvent> {
    let req = AnalyzePipelineRequest {
        doc_id: "doc-pii".into(),
        text: TEXT.into(),
        offsets: OffsetMode::Utf8 as i32,
        steps: vec!["turbo-pii:pii".into(), "turbo-pii:masked".into()],
        pii_output,
        operation_key: KEY.into(),
        ..Default::default()
    };
    let mut stream = client(s).await.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.unwrap());
    }
    events
}

fn graph_of(events: &[AnalyzeProgressiveEvent]) -> Vec<u8> {
    events
        .iter()
        .find_map(|e| match &e.event {
            Some(Event::Complete(c)) => Some(c.graph.to_vec()),
            _ => None,
        })
        .expect("the stream has its complete answer")
}

fn holds(bytes: &[u8], needle: &[u8]) -> bool {
    bytes.windows(needle.len()).any(|w| w == needle)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Asserts `bytes` carry neither `digest` nor its hexadecimal.
fn assert_no_digest(bytes: &[u8], digest: &[u8], what: &str) {
    assert!(!holds(bytes, digest), "{what}: the SHA-256 left");
    assert!(!holds(bytes, hex(digest).as_bytes()), "{what}: the hexadecimal SHA-256 left");
}

#[tokio::test]
async fn a_stream_that_withholds_its_graph_names_no_digest_and_is_still_complete() {
    let s = server(Serve::default()).await;
    let full = progressive(&s, options(true, false)).await;
    let graph = graph_of(&full);
    let text_digest = ViewTextDigest::of_text(TEXT);
    let graph_digest = graph_content_digest(&graph);

    // Positive control: with full_text the receipt names the input and the
    // delivered graph by digest.
    let receipt = terminal_receipt(&full, KEY, TEXT).expect("a terminal receipt");
    assert_eq!(receipt.input.as_ref().unwrap().source_text_digest, text_digest.as_bytes());
    assert_eq!(receipt.result.as_ref().unwrap().content_digest, graph_digest.as_bytes());
    assert!(receipt.layers.iter().all(|l| l.provenance_digest.is_some()));

    for pii_output in [None, options(false, true)] {
        let events = progressive(&s, pii_output).await;
        assert!(graph_of(&events).is_empty(), "the graph is withheld");
        for (i, e) in events.iter().enumerate() {
            let bytes = e.encode_to_vec();
            assert_no_digest(&bytes, text_digest.as_bytes(), &format!("event {i}, source text"));
            assert_no_digest(&bytes, graph_digest.as_bytes(), &format!("event {i}, graph bytes"));
        }
        let receipt = terminal_receipt(&events, KEY, TEXT).expect("a withheld stream ends with its receipt");
        assert_eq!(receipt.status, ProcessingStatus::Complete as i32);
        let input = receipt.input.as_ref().unwrap();
        assert!(input.source_text_digest.is_empty() && input.raw_bytes_digest.is_empty(), "{input:?}");
        assert_eq!(input.source_text_len as usize, TEXT.len());
        let result = receipt.result.as_ref().expect("the attempt produced a result");
        assert!(result.content_digest.is_empty() && result.result_id.is_empty(), "{result:?}");
        assert!(receipt.layers.iter().all(|l| l.provenance_digest.is_none()), "{:?}", receipt.layers);
        // The receipt still binds what it can: the key and the length.
        assert_eq!(terminal_receipt(&events, "other-key", TEXT).unwrap_err(), Incomplete::OtherOperation);
        assert_eq!(terminal_receipt(&events, KEY, "a longer text than that").unwrap_err(), Incomplete::OtherInput);
        assert_eq!(
            terminal_receipt(&events[..events.len() - 1], KEY, TEXT).unwrap_err(),
            Incomplete::NoTerminalReceipt
        );
    }

    // A receipt that withholds its digests binds no delivered graph: put
    // on the stream that carried the bytes, it is not that stream's.
    let withheld = progressive(&s, None).await;
    let mut forged = full.clone();
    *forged.last_mut().unwrap() = withheld.last().unwrap().clone();
    assert_eq!(terminal_receipt(&forged, KEY, TEXT).unwrap_err(), Incomplete::OtherResult);
    s.stop().await;
}

#[tokio::test]
async fn inspect_and_sink_items_carry_no_digest_of_the_withheld_source() {
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
    let graph = graph_of(&progressive(&s, options(true, false)).await);
    let text_digest = ViewTextDigest::of_text(TEXT);
    let graph_digest = graph_content_digest(&graph);

    for (view, pii_output) in [(0, None), (MASKED, None), (0, options(false, true)), (MASKED, options(false, true))] {
        let req = InspectDocumentGraphRequest {
            doc_id: "doc-pii".into(),
            graph: graph.clone(),
            view,
            offsets: OffsetMode::Utf8 as i32,
            pii_output,
            binding: None,
            operation_key: String::new(),
            pii_policy: None,
        };
        let typed = client(&s).await.inspect_document_graph(req).await.unwrap().into_inner();
        let bytes = typed.encode_to_vec();
        assert_no_digest(&bytes, text_digest.as_bytes(), &format!("inspect view {view}, source text"));
        assert_no_digest(&bytes, graph_digest.as_bytes(), &format!("inspect view {view}, graph bytes"));
    }

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
        let bytes = item.encode_to_vec();
        assert!(item.graph.is_empty() && item.typed.is_some(), "{}: findings stand in for the graph", item.doc_id);
        assert_no_digest(&bytes, text_digest.as_bytes(), &format!("sink item {}, source text", item.doc_id));
        assert_no_digest(&bytes, graph_digest.as_bytes(), &format!("sink item {}, graph bytes", item.doc_id));
    }
    s.stop().await;
    sink.stop().await;
}
