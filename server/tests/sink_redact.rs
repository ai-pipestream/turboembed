//! Sink items of a `turbo-pii:redact` document (#1882 part C): under the
//! findings-only default (`--sink-pii-output`) an item carries the edits
//! layer with its policy rule ids and digest, the fragment outcomes and
//! the receipt, and neither a planted value nor the redacted text; the
//! redacted text appears only when the sink output allows rewritten text.

mod common;
#[path = "pii_redact/fixture.rs"]
mod fixture;
#[path = "pii_policy/common.rs"]
mod policy;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use turbo_analysis::export::export;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::pii_policy::NoRegistry;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{self as pb, OffsetMode, SpanOutcome, TypedDocumentResult};
use turbo_kserve::proto::sink::DocumentSinkItem;
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::document::codecs;
use turbo_kserve::sink::{SinkConfig, SinkReceiver, SinkTyped};
use turbo_kserve::{Serve, Server};

use fixture::{CARD, EMAIL, convert, redacted, run};
use policy::one;

const REPLACEMENT: &str = "[E]";

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

fn holds(bytes: &[u8], needle: &str) -> bool {
    bytes.windows(needle.len()).any(|w| w == needle.as_bytes())
}

/// The redacted graph, its redacted view's id and the policy digest.
fn graph() -> (Vec<u8>, u32, Vec<u8>) {
    let policy =
        convert(&one(Strategy::StaticText(pb::PiiStaticStrategy { replacement: REPLACEMENT.into() })), &NoRegistry);
    let doc = run(policy.clone());
    let view = u32::from(redacted(&doc).0.id().0);
    (export(&doc, codecs()).unwrap().to_bytes().unwrap(), view, policy.digest.as_bytes().to_vec())
}

/// Forwards the redacted graph through a sink that projects `view` under
/// `pii_output`, and answers the item the sink accepted.
async fn forward(view: u32, pii_output: Disclosure) -> DocumentSinkItem {
    let got: Arc<Mutex<Vec<DocumentSinkItem>>> = Arc::default();
    let keep = got.clone();
    let receiver = SinkReceiver::new(move |item| keep.lock().unwrap().push(item));
    let start = |serve| Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve);
    let sink = start(Serve { sink_receiver: Some(receiver), ..Serve::default() }).await.unwrap();
    let config = SinkConfig {
        endpoint: format!("http://{}", sink.local_addr()),
        max_queue_items: 16,
        max_queue_bytes: 64 << 20,
        documents: false,
        typed: Some(SinkTyped { view, offsets: OffsetMode::Utf8 }),
        pii_output,
    };
    let s = start(Serve { sink: Some(config), ..Serve::default() }).await.unwrap();
    assert!(s.forward_document("doc-redact", graph().0).await.unwrap());
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.accepted_count, summary.rejected_count), (1, 0), "{:?}", summary.errors);
    s.stop().await;
    sink.stop().await;
    got.lock().unwrap().pop().unwrap()
}

/// The edits layer carries every edit's rule id and the policy digest.
fn assert_edits(typed: &TypedDocumentResult, digest: &[u8]) {
    let edits = typed.layers.iter().find(|l| l.id == "turbo:pii-edits").expect("the edits layer");
    assert!(!edits.annotations.is_empty());
    for a in &edits.annotations {
        let Some(Value::PiiEdit(e)) = &a.value else { panic!("not an edit") };
        assert_eq!(e.policy_digest, digest);
        assert!(e.rule_id.is_empty() || e.rule_id == "email", "{}", e.rule_id);
        assert_eq!(e.policy_revision, 1);
        for f in a.spans.iter().chain(e.source.iter()) {
            assert_ne!(f.outcome, SpanOutcome::Unspecified as i32, "every fragment has its outcome");
        }
    }
    assert!(edits.annotations.iter().any(|a| matches!(&a.value, Some(Value::PiiEdit(e)) if e.rule_id == "email")));
}

#[tokio::test]
async fn a_findings_only_sink_item_carries_the_edits_and_receipt_but_no_text() {
    let (_, view, digest) = graph();
    for target in [0, view] {
        let item = forward(target, Disclosure::FINDINGS).await;
        let typed = item.typed.as_ref().expect("the typed projection");
        assert_edits(typed, &digest);
        assert!(typed.receipt.is_some(), "the item carries the receipt");
        assert!(typed.raw_text.is_empty(), "view {target}: findings only withholds the text");
        assert!(item.graph.is_empty(), "view {target}: findings only withholds the graph");
        let bytes = item.encode_to_vec();
        for planted in [EMAIL, CARD, REPLACEMENT] {
            assert!(!holds(&bytes, planted), "view {target}: the item holds {planted}");
        }
    }
}

#[tokio::test]
async fn the_redacted_text_reaches_the_sink_only_when_its_output_allows_it() {
    let (_, view, digest) = graph();
    let rewritten = Disclosure { rewritten_text: true, ..Disclosure::FINDINGS };
    let item = forward(view, rewritten).await;
    let typed = item.typed.as_ref().expect("the typed projection");
    assert_edits(typed, &digest);
    assert!(typed.raw_text.contains(REPLACEMENT), "the redacted view's text: {}", typed.raw_text);
    let bytes = item.encode_to_vec();
    assert!(!holds(&bytes, EMAIL) && !holds(&bytes, CARD), "rewritten text never carries the value");
    // The source view's text is the full text, which rewritten_text does
    // not allow.
    let source = forward(0, rewritten).await;
    assert!(source.typed.as_ref().unwrap().raw_text.is_empty());
    assert!(!holds(&source.encode_to_vec(), EMAIL));
}
