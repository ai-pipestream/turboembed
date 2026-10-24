//! Progressive single-document pipeline analysis (`AnalyzeDocumentProgressive`,
//! issue `#1218`) over a real tonic gRPC channel.

mod common;

use tokio_stream::StreamExt as _;
use tonic::Code;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis_layers::{Chunks, Language, Lemmas, PosTags, Sentences, Stems, Terms, Tokens, Words};
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;

use common::*;

#[tokio::test]
async fn progressive_stream_emits_strict_1_based_sequence_from_started_to_complete() {
    let (server, mut client) = serve().await;

    let req = AnalyzePipelineRequest {
        doc_id: "doc-prog-1".into(),
        text: "TurboEmbed streams DAG branches progressively. Fast stages arrive first!".into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "default".into(),
        steps: Vec::new(),
        model: String::new(),
        initial_graph: Vec::new(),
    };

    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.unwrap());
    }

    // Must have `started` (seq 1) + 10 `layers_ready` events (seq 2..=11) + `complete` (seq 12).
    assert_eq!(events.len(), 12);
    for (idx, ev) in events.iter().enumerate() {
        assert_eq!(ev.sequence, (idx + 1) as u64, "strictly increasing 1-based sequence at index {idx}");
    }

    // First event is AnalysisStarted.
    let Some(Event::Started(started)) = &events[0].event else {
        panic!("expected first event to be AnalysisStarted, got {:?}", events[0]);
    };
    assert_eq!(started.doc_id, "doc-prog-1");
    assert_eq!(started.step_ids.len(), 10);
    assert_eq!(started.views.len(), 1);

    // Middle events are AnalysisLayerBatch in valid topological order.
    let mut completed_steps = Vec::new();
    for ev in &events[1..events.len() - 1] {
        let Some(Event::LayersReady(batch)) = &ev.event else {
            panic!("expected LayersReady event, got {ev:?}");
        };
        assert!(!batch.layers.is_empty());
        assert!(!batch.graph.is_empty());
        completed_steps.push(batch.completed_step.as_str());
    }

    let pos_of = |step: &str| completed_steps.iter().position(|&s| s == step).unwrap();
    assert!(pos_of("turbo:sentences") < pos_of("turbo:tokens"));
    assert!(pos_of("turbo:sentences") < pos_of("turbo:boundaries"));
    assert!(pos_of("turbo:boundaries") < pos_of("turbo:chunks"));
    assert!(pos_of("turbo:tokens") < pos_of("turbo:words"));
    assert!(pos_of("turbo:tokens") < pos_of("turbo:terms"));
    assert!(pos_of("turbo:terms") < pos_of("turbo:stems"));
    assert!(pos_of("turbo:tokens") < pos_of("turbo:pos"));
    assert!(pos_of("turbo:pos") < pos_of("turbo:lemmas"));

    // Final event is DocumentGraphResponse complete.
    let Some(Event::Complete(complete)) = &events.last().unwrap().event else {
        panic!("expected final event to be Complete");
    };
    assert_eq!(complete.source_text, "TurboEmbed streams DAG branches progressively. Fast stages arrive first!");
    assert_eq!(complete.layers.len(), 10);

    let graph = DocumentGraph::from_bytes(&complete.graph).unwrap();
    let doc = import(&graph, codecs()).unwrap();
    assert!(doc.get::<Language>().is_some());
    assert!(doc.get::<Sentences>().is_some());
    assert!(doc.get::<Tokens>().is_some());
    assert!(doc.get::<Words>().is_some());
    assert!(doc.get::<Terms>().is_some());
    assert!(doc.get::<Stems>().is_some());
    assert!(doc.get::<PosTags>().is_some());
    assert!(doc.get::<Lemmas>().is_some());
    assert!(doc.get::<Chunks>().is_some());

    server.stop().await;
}

#[tokio::test]
async fn progressive_stream_isolates_step_failure_and_completes_sibling_branches() {
    let (server, mut client) = serve().await;

    let req = AnalyzePipelineRequest {
        doc_id: "doc-fail-pos".into(),
        text: "This document triggers [FAIL_POS] while sibling branches complete.".into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "default".into(),
        steps: Vec::new(),
        model: String::new(),
        initial_graph: Vec::new(),
    };

    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.unwrap());
    }

    for (idx, ev) in events.iter().enumerate() {
        assert_eq!(ev.sequence, (idx + 1) as u64);
    }

    let mut step_failures = Vec::new();
    let mut ready_steps = Vec::new();
    for ev in &events {
        match &ev.event {
            Some(Event::StepFailed(fail)) => step_failures.push(fail.clone()),
            Some(Event::LayersReady(batch)) => ready_steps.push(batch.completed_step.clone()),
            _ => {}
        }
    }

    assert_eq!(step_failures.len(), 1);
    let fail = &step_failures[0];
    assert_eq!(fail.step_id, "turbo:pos");
    assert_eq!(fail.skipped_dependents, vec!["turbo:lemmas".to_string()]);
    assert!(fail.failure.as_ref().unwrap().message.contains("POS tagger branch failure"));

    // Independent sibling branches all completed and emitted LayersReady.
    for sibling in [
        "turbo:language",
        "turbo:sentences",
        "turbo:tokens",
        "turbo:words",
        "turbo:terms",
        "turbo:stems",
        "turbo:boundaries",
        "turbo:chunks",
    ] {
        assert!(ready_steps.iter().any(|s| s == sibling), "missing sibling step {sibling}");
    }

    let Some(Event::Complete(complete)) = &events.last().unwrap().event else {
        panic!("expected final Complete event");
    };
    let pos_info = complete.layers.iter().find(|l| l.layer == "opennlp:pos").unwrap();
    assert_eq!(pos_info.status, LayerWireStatus::Failed as i32);
    let lemma_info = complete.layers.iter().find(|l| l.layer == "opennlp:lemmas").unwrap();
    assert_eq!(lemma_info.status, LayerWireStatus::Skipped as i32);

    server.stop().await;
}

#[tokio::test]
async fn progressive_stream_cancellation_and_validation_refusals() {
    let (server, mut client) = serve().await;

    // 1. Dropping the stream after the first event cancels remaining work cleanly.
    let req = AnalyzePipelineRequest {
        doc_id: "doc-cancel".into(),
        text: "Cancel this progressive stream right after started.".into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "default".into(),
        steps: Vec::new(),
        model: "slow-lang".into(),
        initial_graph: Vec::new(),
    };
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.sequence, 1);
    assert!(matches!(first.event, Some(Event::Started(_))));
    drop(stream);

    // 2. Unknown step or profile is refused with INVALID_ARGUMENT and turbo-code.
    let bad_step_req = AnalyzePipelineRequest {
        doc_id: "doc-bad".into(),
        text: "Hello.".into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: String::new(),
        steps: vec!["turbo:nonexistent".into()],
        model: String::new(),
        initial_graph: Vec::new(),
    };
    let err = client.analyze_document_progressive(bad_step_req).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("unknown pipeline step"));

    server.stop().await;
}
