//! Progressive single-document pipeline analysis (`AnalyzeDocumentProgressive`,
//! issue `#1218`) over a real tonic gRPC channel using first-party workspace
//! annotators (`turbo-structure`, `turbo-sentdetect`, `turbo-tokenize`,
//! `turbo-abbrev`, `turbo-stem`, `turbo-stopword`, `turbo-keywords`,
//! `turbo-chunker`).

mod common;

use tokio_stream::StreamExt as _;
use tonic::Code;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Ix, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Abbreviations, Boundaries, Sentences, Stems, Structure, Terms, Tokens};
use turbo_keywords::Keywords;
use turbo_kserve::api::TURBO_E_INTERNAL;
use turbo_kserve::progressive::compile_analyzer;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_stopword_layers::Stopwords;

use common::*;

const DEFAULT_STEP_IDS: &[&str] = &[
    "turbo-structure:blocks",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo-abbrev:schwartz-hearst",
    "turbo-stem:terms",
    "turbo-stem:stemmer",
    "turbo-stopword:bundled",
    "turbo-keywords:yake",
    "turbo:structure-boundaries",
    "turbo:lexical-boundaries",
];

#[tokio::test]
async fn progressive_stream_emits_strict_1_based_sequence_and_matches_analyzer_run() {
    let (server, mut client) = serve().await;
    let source_text =
        "# Heading\n\nWorld Health Organization (WHO) streams DAG branches progressively. Fast stages arrive first!";

    let req = AnalyzePipelineRequest {
        doc_id: "doc-prog-1".into(),
        text: source_text.into(),
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

    let Some(Event::Started(started)) = &events[0].event else {
        panic!("expected first event to be AnalysisStarted, got {:?}", events[0]);
    };
    assert_eq!(started.doc_id, "doc-prog-1");
    assert_eq!(started.step_ids.len(), 10);
    assert_eq!(started.views.len(), 1);

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
    assert!(pos_of("turbo-sentdetect:rules") < pos_of("turbo-tokenize:uax29-untyped"));
    assert!(pos_of("turbo-sentdetect:rules") < pos_of("turbo:structure-boundaries"));
    assert!(pos_of("turbo-structure:blocks") < pos_of("turbo:structure-boundaries"));
    assert!(pos_of("turbo-tokenize:uax29-untyped") < pos_of("turbo-abbrev:schwartz-hearst"));
    assert!(pos_of("turbo-tokenize:uax29-untyped") < pos_of("turbo-stem:terms"));
    assert!(pos_of("turbo-tokenize:uax29-untyped") < pos_of("turbo-stem:stemmer"));
    assert!(pos_of("turbo-tokenize:uax29-untyped") < pos_of("turbo-stopword:bundled"));
    assert!(pos_of("turbo-stem:terms") < pos_of("turbo-keywords:yake"));
    assert!(pos_of("turbo-stem:terms") < pos_of("turbo:lexical-boundaries"));

    let Some(Event::Complete(complete)) = &events.last().unwrap().event else {
        panic!("expected final event to be Complete");
    };
    assert_eq!(complete.source_text, source_text);
    // 10 steps write 11 layers (`AbbreviationAnnotator` writes both `Abbreviations` and its link `Edges`).
    assert_eq!(complete.layers.len(), 11);

    // Compare terminal Complete graph byte-for-byte with an independent Analyzer::run export.
    let mut ref_doc = Document::new(source_text);
    let ref_analyzer = compile_analyzer(&ref_doc, DEFAULT_STEP_IDS).unwrap();
    let _ = ref_analyzer.run(std::slice::from_mut(&mut ref_doc));
    let expected_graph_bytes = export(&ref_doc, codecs()).unwrap().to_bytes().unwrap();
    assert_eq!(complete.graph, expected_graph_bytes, "progressive Complete graph must equal Analyzer::run export");

    let graph = DocumentGraph::from_bytes(&complete.graph).unwrap();
    let doc = import(&graph, codecs()).unwrap();
    assert!(doc.get::<Structure>().is_some());
    assert!(doc.get::<Sentences>().is_some());
    assert!(doc.get::<Tokens>().is_some());
    assert!(doc.get::<Abbreviations>().is_some());
    assert!(doc.get::<Terms>().is_some());
    assert!(doc.get::<Stems>().is_some());
    assert!(doc.get::<Stopwords>().is_some());
    assert!(doc.get::<Keywords>().is_some());
    assert!(doc.get::<Boundaries>().is_some());
    assert!(doc.get_key(&Key::<Boundaries>::named("lexical")).is_some());

    server.stop().await;
}

#[tokio::test]
async fn progressive_stream_isolates_real_annotator_failure_and_completes_sibling_branches() {
    let (server, mut client) = serve().await;

    // Build an initial document where `Sentences` sits on ViewId::SOURCE (0)
    // and `Tokens` sits on a derived view (ViewId(1)). When `turbo-stem:terms`
    // runs against default View(0) and Tokens(1), it genuinely fails with
    // `Error::Annotator { annotator: "turbo-stem:terms", detail: "the tokens are in another view" }`,
    // cascading `Skipped` to `turbo-keywords:yake` while sibling steps
    // `turbo-structure:blocks` and `turbo:structure-boundaries` succeed.
    let text = "# Section\n\nAlpha beta gamma.";
    let mut init_doc = Document::new(text);
    let mut vb = ViewBuilder::new(text);
    vb.keep(text.len());
    let (derived_text, align) = vb.finish();
    let src_view = init_doc.get::<View>().unwrap().clone();
    let child_key = Key::<View>::named("child");
    init_doc.add(&child_key, View::derived(&src_view, derived_text, align), Provenance::by("test:view")).unwrap();
    let child_view = init_doc.get_key(&child_key).unwrap().id();

    let mut sent_spans = Spans::<Sentences>::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, text.len() as u32));
    init_doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test:sent")).unwrap();

    let mut tok_spans = Spans::<Tokens>::new(child_view);
    tok_spans.push(Span::new(11, 16));
    tok_spans.push(Span::new(17, 21));
    init_doc
        .add(
            &Key::DEFAULT,
            Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 2]) },
            Provenance::by("test:tok"),
        )
        .unwrap();

    let initial_graph = export(&init_doc, codecs()).unwrap().to_bytes().unwrap();
    let steps = vec![
        "turbo-structure:blocks".to_string(),
        "turbo:structure-boundaries".to_string(),
        "turbo-stem:terms".to_string(),
        "turbo-keywords:yake".to_string(),
    ];

    let req = AnalyzePipelineRequest {
        doc_id: "doc-branch-fail".into(),
        text: String::new(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: String::new(),
        steps: steps.clone(),
        model: String::new(),
        initial_graph,
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
    assert_eq!(fail.step_id, "turbo-stem:terms");
    assert_eq!(fail.skipped_dependents, vec!["turbo-keywords:yake".to_string()]);
    let failure_info = fail.failure.as_ref().unwrap();
    assert_eq!(failure_info.turbo_code, TURBO_E_INTERNAL);
    assert!(failure_info.message.contains("the tokens are in another view"), "got: {}", failure_info.message);

    for sibling in ["turbo-structure:blocks", "turbo:structure-boundaries"] {
        assert!(ready_steps.iter().any(|s| s == sibling), "missing sibling step {sibling}");
    }

    let Some(Event::Complete(complete)) = &events.last().unwrap().event else {
        panic!("expected final Complete event");
    };
    let terms_info = complete.layers.iter().find(|l| l.layer == "turbo:terms").unwrap();
    assert_eq!(terms_info.status, LayerWireStatus::Failed as i32);
    let kw_info = complete.layers.iter().find(|l| l.layer == "turbo:keywords").unwrap();
    assert_eq!(kw_info.status, LayerWireStatus::Skipped as i32);

    // Compare terminal Complete graph byte-for-byte with Analyzer::run on the same failure case.
    let step_refs: Vec<&str> = steps.iter().map(String::as_str).collect();
    let ref_analyzer = compile_analyzer(&init_doc, &step_refs).unwrap();
    let _ = ref_analyzer.run(std::slice::from_mut(&mut init_doc));
    let expected_bytes = export(&init_doc, codecs()).unwrap().to_bytes().unwrap();
    assert_eq!(complete.graph, expected_bytes);

    server.stop().await;
}

#[tokio::test]
async fn progressive_stream_cancellation_and_validation_refusals() {
    let (server, mut client) = serve().await;

    let req = AnalyzePipelineRequest {
        doc_id: "doc-cancel".into(),
        text: "Cancel this progressive stream right after started. Second sentence here.".into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "default".into(),
        steps: Vec::new(),
        model: String::new(),
        initial_graph: Vec::new(),
    };
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.sequence, 1);
    assert!(matches!(first.event, Some(Event::Started(_))));
    drop(stream);

    // Unknown step is refused with INVALID_ARGUMENT.
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

    // Non-empty model is refused on built-in progressive profiles.
    let bad_model_req = AnalyzePipelineRequest {
        doc_id: "doc-bad-model".into(),
        text: "Hello.".into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "minimal".into(),
        steps: Vec::new(),
        model: "unknown-bundle".into(),
        initial_graph: Vec::new(),
    };
    let err = client.analyze_document_progressive(bad_model_req).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);

    server.stop().await;
}
