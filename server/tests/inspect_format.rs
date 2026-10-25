//! End-to-end gRPC integration tests for `InspectDocumentWindow`,
//! `ProjectChunks`, `ListOutputFormats`, and `FormatDocument` on
//! `AnalysisService` (issue #1220).

use std::sync::Arc;

use tonic::Code;
use turbo_analysis::export::{export, import};
use turbo_analysis::{
    Annotator, Document, Ix, IxRange, Key, ModelRef as CoreModelRef, Provenance, Span, Spans, Tag, TagSet, View,
    ViewBuilder, ViewId,
};
use turbo_analysis_layers::{
    Boundaries, Categories, Chunks, Cut, Entities, Language, Lemmas, Pii, PiiConfidence, PiiKind, PiiMention, PosTags,
    Sentences, Strs, Subwords, Tokens, Vectors,
};
use turbo_chunker::terms::{TermCounter, TermIndexer, Vocabulary};
use turbo_chunker::{Budget, Chunker};
use turbo_kserve::api::{TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT};
use turbo_kserve::config::{Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{
    BudgetSpec, DocumentWindowRequest, FormatDocumentRequest, ListOutputFormatsRequest, OffsetMode,
    ProjectChunksRequest, XrayEditKindProto,
};
use turbo_kserve::results::MAX_WINDOWS;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};

fn test_model_config() -> ModelConfig {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root");
    let bundle = root.join("testdata/tiny-bert-bundle");
    ModelConfig {
        bundle: bundle.to_string_lossy().into_owned(),
        name: Some("tiny".into()),
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

fn check_turbo_error(status: &tonic::Status, expected_grpc: Code, expected_turbo: i32) {
    assert_eq!(status.code(), expected_grpc, "unexpected gRPC code: {status:?}");
    let tc = status
        .metadata()
        .get("turbo-code")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i32>().ok())
        .expect("turbo-code metadata header must be present");
    assert_eq!(tc, expected_turbo, "unexpected turbo-code on {status:?}");
}

/// Builds a rich multi-view, multi-layer `Document` and serializes it to
/// `DocumentGraph` bytes.
fn build_rich_graph_bytes() -> (Document, Vec<u8>) {
    // Multibyte UTF-8 ("Café" = 5 bytes, 4 chars, 4 UTF-16 units) + astral
    // emoji ("😀" = 4 bytes, 1 char, 2 UTF-16 units).
    let text = "Café 😀 contact alice@example.com in Paris. Second topic discusses retrieval engines.";
    let mut doc = Document::new(text);

    // Add a derived view (View 1) that replaces "Café" with "Cafe" and inserts "[NORM] " at the start.
    let src_view = doc.get::<View>().expect("source view").clone();
    let mut vb = ViewBuilder::new(src_view.text());
    vb.replace(0, "[NORM] ");
    vb.keep(3); // "Caf"
    vb.replace(5, "e"); // "é" (bytes 3..5) -> "e"
    let (derived_text, derived_align) = vb.finish();
    let derived_view = View::derived(&src_view, derived_text, derived_align);
    doc.add(&Key::<View>::named("normalized"), derived_view, Provenance::by("test:normalizer")).unwrap();

    // Sentences on View 0:
    // S0: "Café 😀 contact alice@example.com in Paris." (0..46)
    // S1: "Second topic discusses retrieval engines." (47..88)
    let mut sent_spans = Spans::<Sentences>::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, 46));
    sent_spans.push(Span::new(47, 88));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test:sentences")).unwrap();

    // Tokens on View 0.
    let token_ranges: &[(u32, u32, u32, &str, &str)] = &[
        (0, 5, 0, "NNP", "cafe"),
        (6, 10, 0, "SYM", "emoji"),
        (11, 18, 0, "VB", "contact"),
        (19, 36, 0, "NN", "alice@example.com"),
        (37, 39, 0, "IN", "in"),
        (40, 45, 0, "NNP", "paris"),
        (47, 53, 1, "JJ", "second"),
        (54, 59, 1, "NN", "topic"),
        (60, 69, 1, "VBZ", "discuss"),
        (70, 79, 1, "NN", "retrieval"),
        (80, 87, 1, "NNS", "engine"),
    ];
    let mut tok_spans = Spans::<Tokens>::new(ViewId::SOURCE);
    let mut tok_sents = Vec::new();
    let mut sub_spans = Spans::<Subwords>::new(ViewId::SOURCE);
    let mut sub_ids = Vec::new();
    let mut sub_tokens = Vec::new();
    let mut pos_tags = Vec::new();
    let mut lemmas = Strs::default();
    let pos_tagset = TagSet::new("upos", &["NNP", "SYM", "VB", "NN", "IN", "JJ", "VBZ", "NNS"]);
    for (idx, &(s, e, sent_idx, pos, lemma)) in token_ranges.iter().enumerate() {
        tok_spans.push(Span::new(s, e));
        tok_sents.push(Ix::new(sent_idx));
        sub_spans.push(Span::new(s, e));
        sub_ids.push(idx as i32 + 1);
        sub_tokens.push(Ix::new(idx as u32));
        pos_tags.push(pos_tagset.tag(pos).unwrap());
        lemmas.push(lemma);
    }
    let model_ref = CoreModelRef { id: "test-embed".into(), revision: "r1".into(), sha256: "0123456789abcdef".into() };
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: Some(tok_sents) }, Provenance::by("test:tokens"))
        .unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords { spans: sub_spans, ids: sub_ids, token: Some(sub_tokens), model: model_ref.clone() },
        Provenance::by("test:subwords"),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        PosTags { tagset: Arc::clone(&pos_tagset), tags: pos_tags, prob: Some(vec![0.95; token_ranges.len()]) },
        Provenance::by("test:pos"),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Lemmas { lemma: lemmas }, Provenance::by("test:lemmas")).unwrap();

    // Entities on View 0, including a discontinuous entity across "Café" (0..5) and "Paris" (40..45).
    let ent_kinds = TagSet::new("ner", &["LOC", "MISC"]);
    let mut ent_spans = Spans::<Entities>::new(ViewId::SOURCE);
    ent_spans.push(Span::new(40, 45)); // "Paris"
    ent_spans.push_fragments(&[Span::new(0, 5), Span::new(40, 45)]); // discontinuous "Café ... Paris"
    doc.add(
        &Key::DEFAULT,
        Entities {
            spans: ent_spans,
            tokens: vec![IxRange::new(5, 6), IxRange::new(0, 6)],
            kind: vec![ent_kinds.tag("LOC").unwrap(), ent_kinds.tag("MISC").unwrap()],
            kinds: ent_kinds,
            score: vec![0.98, 0.87],
        },
        Provenance::by("test:entities"),
    )
    .unwrap();

    // PII on View 0: "alice@example.com" (19..36).
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(PiiMention {
        span: Span::new(19, 36),
        kind: PiiKind::Email,
        confidence: PiiConfidence::Checksum,
        normalized: "alice@example.com",
    });
    doc.add(&Key::DEFAULT, pii, Provenance::by("test:pii")).unwrap();

    // Categories on View 0 and Categories<Sentences>.
    let cat_tags = TagSet::new("sentiment", &["positive", "negative"]);
    let mut doc_cats = Categories::<View>::new(Arc::clone(&cat_tags));
    doc_cats.push(Tag(0), &[0.88, 0.12]);
    doc.add(&Key::DEFAULT, doc_cats, Provenance::by("test:doccat")).unwrap();

    let mut sent_cats = Categories::<Sentences>::new(Arc::clone(&cat_tags));
    sent_cats.push(Tag(0), &[0.85, 0.15]);
    sent_cats.push(Tag(1), &[0.20, 0.80]);
    doc.add(&Key::DEFAULT, sent_cats, Provenance::by("test:sentcat")).unwrap();

    // Document Language.
    let mut lang_codes = Strs::default();
    lang_codes.push("eng");
    doc.add(&Key::DEFAULT, Language { code: lang_codes, score: vec![0.99] }, Provenance::by("test:lang")).unwrap();

    // Boundaries + Chunker (budget cutting S0 and S1 into 2 chunks).
    doc.add(
        &Key::DEFAULT,
        Boundaries { cost: vec![Boundaries::UNIT, Boundaries::UNIT], reason: vec![Cut::Sentence, Cut::Sentence] },
        Provenance::by("test:bounds"),
    )
    .unwrap();

    let budget = Budget { max: 8, target: 5, size_weight: 1000, overhead: 1 };
    let chunks = Chunker { budget }
        .annotate((
            doc.get::<Sentences>().unwrap(),
            doc.get::<Subwords>().unwrap(),
            doc.get::<Boundaries>().unwrap(),
            None,
            None,
        ))
        .unwrap()
        .0;
    assert_eq!(chunks.spans.len(), 2, "budget max=8 cuts the 2 sentences into 2 chunks");
    doc.add(&Key::DEFAULT, chunks, Provenance::by("test:chunker")).unwrap();

    // TermIndexer + TermCounter<Chunks> + TermCounter<View>.
    let (vocab, indexed) = TermIndexer.annotate((doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap())).unwrap();
    doc.add(&Key::DEFAULT, vocab, Provenance::by("test:vocab")).unwrap();
    doc.add(&Key::DEFAULT, indexed, Provenance::by("test:indexed")).unwrap();

    let chunk_stats = TermCounter::<Chunks>::new()
        .annotate((
            doc.get::<Tokens>().unwrap(),
            doc.get::<turbo_chunker::terms::IndexedTerms>().unwrap(),
            doc.get::<Vocabulary>().unwrap(),
            doc.get::<Chunks>().unwrap(),
        ))
        .unwrap()
        .0;
    doc.add(&Key::DEFAULT, chunk_stats, Provenance::by("test:chunk-terms")).unwrap();

    let doc_stats = TermCounter::<View>::new()
        .annotate((
            doc.get::<Tokens>().unwrap(),
            doc.get::<turbo_chunker::terms::IndexedTerms>().unwrap(),
            doc.get::<Vocabulary>().unwrap(),
            doc.get::<View>().unwrap(),
        ))
        .unwrap()
        .0;
    doc.add(&Key::DEFAULT, doc_stats, Provenance::by("test:doc-terms")).unwrap();

    // Vectors<Sentences> and Vectors<Chunks> (dim = 4).
    let model_ref = CoreModelRef { id: "test-embed".into(), revision: "r1".into(), sha256: "0123456789abcdef".into() };
    let sent_vecs = Vectors::<Sentences>::new(
        4,
        vec![
            1.0, 0.0, 0.2, 0.1, // S0
            0.0, 1.0, -0.2, 0.3, // S1
        ],
        Vec::new(),
        model_ref.clone(),
    );
    doc.add(&Key::DEFAULT, sent_vecs, Provenance::by("test:sent-vecs")).unwrap();

    let chunk_vecs = Vectors::<Chunks>::new(
        4,
        vec![
            1.0, 0.0, 0.2, 0.1, // C0
            0.0, 1.0, -0.2, 0.3, // C1
        ],
        Vec::new(),
        model_ref,
    );
    doc.add(&Key::DEFAULT, chunk_vecs, Provenance::by("test:chunk-vecs")).unwrap();

    let graph = export(&doc, codecs()).expect("export graph");
    let bytes = graph.to_bytes().expect("encode graph bytes");
    let _ = import(&graph, codecs()).expect("round-trip import check");
    (doc, bytes)
}

#[tokio::test]
async fn inspect_project_and_format_rpcs_end_to_end() {
    let srv = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![test_model_config()],
        8 * 1024 * 1024,
        Serve { admin: false, ..Serve::default() },
    )
    .await
    .expect("start server");
    let addr = srv.local_addr();
    let endpoint = format!("http://{addr}");
    let mut client = AnalysisServiceClient::connect(endpoint).await.expect("connect AnalysisServiceClient");

    let (_doc, graph_bytes) = build_rich_graph_bytes();

    // 1. InspectDocumentWindow across UTF-8, UTF-16, and CodePoint OffsetModes,
    //    with NormalizationXray against View 1 and both exact and binned heatmaps.
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let resp = client
            .inspect_document_window(DocumentWindowRequest {
                graph: graph_bytes.clone(),
                view: 0,
                offsets: mode as i32,
                page_size: 16_000,
                page: 0,
                filter_layers: Vec::new(),
                heatmap_buckets: 0,
                xray_target_view: Some(1),
            })
            .await
            .expect("InspectDocumentWindow succeeds")
            .into_inner();

        let summary = resp.summary.expect("summary present");
        assert_eq!(summary.view, 0);
        assert_eq!(summary.total_bytes, 88);
        assert_eq!(summary.view_count, 2);
        assert!(!summary.layers.is_empty());
        // "Café 😀" saves 3 bytes in UTF-16 (5+4=9 bytes vs 4+2=6 UTF-16 units -> 88 - 3 = 85)
        // and 4 bytes in CodePoint (5+4=9 bytes vs 4+1=5 code points -> 88 - 4 = 84).
        match mode {
            OffsetMode::Utf8 => assert_eq!(summary.total_units, 88),
            OffsetMode::Utf16 => assert_eq!(summary.total_units, 85),
            OffsetMode::CodePoint => assert_eq!(summary.total_units, 84),
        }

        let win = resp.window.expect("window present");
        assert_eq!(win.page, 0);
        assert_eq!(win.page_count, 1);
        assert_eq!(win.start, 0);
        assert_eq!(win.end, summary.total_units);

        // Verify discontinuous entity ("Café" 0..5 and "Paris" 40..45) is NOT active in the gap ("contact" 11..18).
        assert!(!resp.segments.is_empty());
        let contact_seg = resp.segments.iter().find(|s| s.text == "contact").expect("elementary segment for 'contact'");
        assert!(
            !contact_seg.annotations.iter().any(|a| a.layer == "opennlp:entities" && a.label == "MISC"),
            "discontinuous entity gap must not carry the MISC entity annotation"
        );
        let paris_seg = resp.segments.iter().find(|s| s.text == "Paris").expect("elementary segment for 'Paris'");
        assert!(paris_seg.annotations.iter().any(|a| a.layer == "opennlp:entities" && a.label == "LOC"));
        assert!(paris_seg.annotations.iter().any(|a| a.layer == "opennlp:entities" && a.label == "MISC"));

        // Verify document chips and heatmap lanes.
        assert!(resp.document_chips.iter().any(|c| c.layer == "turbo:categories" && c.label == "positive"));
        assert!(resp.document_chips.iter().any(|c| c.layer == "opennlp:language" && c.label == "eng"));
        assert!(resp.heatmap_lanes.iter().any(|l| l.id == "tokens"));
        assert!(resp.heatmap_lanes.iter().any(|l| l.id == "entities"));
        assert!(resp.heatmap_lanes.iter().any(|l| l.id == "pii"));
        assert!(resp.heatmap_lanes.iter().any(|l| l.id == "sentiment"));
        assert!(resp.heatmap_lanes.iter().any(|l| l.id == "chunks"));

        // Verify NormalizationXray diff runs from View 0 to View 1.
        let xray = resp.xray.expect("xray present");
        assert_eq!(xray.from_view, 0);
        assert_eq!(xray.to_view, 1);
        assert_eq!(xray.steps.len(), 1);
        assert_eq!(xray.steps[0].instance, "normalized");
        assert!(xray.runs.iter().any(|r| r.kind == XrayEditKindProto::Inserted as i32 && r.child_text == "[NORM] "));
        assert!(
            xray.runs
                .iter()
                .any(|r| r.kind == XrayEditKindProto::Replaced as i32 && r.parent_text == "é" && r.child_text == "e")
        );
        for r in &xray.runs {
            if r.kind == XrayEditKindProto::Equal as i32 {
                assert_eq!(r.parent_text, r.child_text, "Equal xray run texts must match byte for byte");
            }
        }
    }

    // Check binned heatmap mode (heatmap_buckets = 4) and filter_layers.
    let binned = client
        .inspect_document_window(DocumentWindowRequest {
            graph: graph_bytes.clone(),
            view: 0,
            offsets: OffsetMode::Utf8 as i32,
            page_size: 32,
            page: 1,
            filter_layers: vec!["turbo:pii".into()],
            heatmap_buckets: 4,
            xray_target_view: None,
        })
        .await
        .expect("binned InspectDocumentWindow succeeds")
        .into_inner();
    assert_eq!(binned.window.as_ref().unwrap().page, 1);
    assert_eq!(binned.window.as_ref().unwrap().page_count, 3);
    for seg in &binned.segments {
        for ann in &seg.annotations {
            assert_eq!(ann.layer, "turbo:pii");
        }
    }
    for lane in &binned.heatmap_lanes {
        assert_eq!(lane.buckets.len(), 4);
    }

    // Verify structured refusals on InspectDocumentWindow.
    let bad_view = client
        .inspect_document_window(DocumentWindowRequest {
            graph: graph_bytes.clone(),
            view: 99,
            offsets: OffsetMode::Utf8 as i32,
            page_size: 16_000,
            page: 0,
            filter_layers: Vec::new(),
            heatmap_buckets: 0,
            xray_target_view: None,
        })
        .await
        .unwrap_err();
    check_turbo_error(&bad_view, Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);

    let over_buckets = client
        .inspect_document_window(DocumentWindowRequest {
            graph: graph_bytes.clone(),
            view: 0,
            offsets: OffsetMode::Utf8 as i32,
            page_size: 16_000,
            page: 0,
            filter_layers: Vec::new(),
            heatmap_buckets: (MAX_WINDOWS as u32) + 1,
            xray_target_view: None,
        })
        .await
        .unwrap_err();
    check_turbo_error(&over_buckets, Code::OutOfRange, TURBO_E_CAPACITY);

    // 2. ProjectChunks in 2D and 3D with drift curve, term stacks, and strategy matrix.
    for pca_dims in [2u32, 3u32] {
        let proj_resp = client
            .project_chunks(ProjectChunksRequest {
                graph: graph_bytes.clone(),
                offsets: OffsetMode::Utf16 as i32,
                pca_dims,
                max_stack_segments: 3,
                include_drift_curve: true,
                include_strategy_matrix: true,
                strategy_budgets: vec![
                    BudgetSpec {
                        name: "single".into(),
                        max_tokens: 32,
                        target_tokens: 16,
                        size_weight: 1000,
                        overhead: 1,
                    },
                    BudgetSpec {
                        name: "split".into(),
                        max_tokens: 8,
                        target_tokens: 5,
                        size_weight: 1000,
                        overhead: 1,
                    },
                ],
            })
            .await
            .expect("ProjectChunks succeeds")
            .into_inner();

        let proj = proj_resp.projection.expect("projection present");
        assert_eq!(proj.pca_dims, pca_dims);
        assert_eq!(proj.explained_variance.len(), pca_dims as usize);
        assert_eq!(proj.points.len(), 2);
        assert!(proj.points[0].adjacent_cosine.is_some());
        assert!(proj.points[1].adjacent_cosine.is_none());
        // With 2 distinct points, the 1st principal component captures 100% of variance.
        assert!((proj.explained_variance[0] - 1.0).abs() < 1e-5);

        let drift = proj_resp.drift_curve.expect("drift_curve present");
        assert_eq!(drift.gaps.len(), 2);
        assert!(drift.gaps[0].is_chunk_cut);

        assert_eq!(proj_resp.chunk_term_stacks.len(), 2);
        for stack in &proj_resp.chunk_term_stacks {
            assert!(stack.segments.len() <= 3);
            let kept_share: f32 = stack.segments.iter().map(|s| s.share).sum();
            let other_share = if stack.total_weight > 0.0 { stack.other_weight / stack.total_weight } else { 0.0 };
            assert!((kept_share + other_share - 1.0).abs() < 1e-5);
        }
        let doc_stack = proj_resp.document_term_stack.expect("document_term_stack present");
        assert!(doc_stack.total_terms >= 10);

        let matrix = proj_resp.strategy_matrix.expect("strategy_matrix present");
        assert_eq!(matrix.strategies.len(), 2);
        assert_eq!(matrix.strategies[0].name, "single");
        assert_eq!(matrix.strategies[0].chunk_count, 1);
        assert_eq!(matrix.strategies[1].name, "split");
        assert_eq!(matrix.strategies[1].chunk_count, 2);
        assert_eq!(matrix.strategies[1].disagreements_with_baseline, 0);
    }

    // 3. ListOutputFormats and FormatDocument across all 7 formats.
    let formats_resp =
        client.list_output_formats(ListOutputFormatsRequest {}).await.expect("ListOutputFormats").into_inner();
    let format_ids: Vec<&str> = formats_resp.formats.iter().map(|f| f.format_id.as_str()).collect();
    assert_eq!(format_ids, vec!["brat", "conllu", "csv", "jsonld", "markdown", "tsv", "warc"]);

    for fmt_id in format_ids {
        let exported = client
            .format_document(FormatDocumentRequest {
                doc_id: "doc-1220".into(),
                graph: graph_bytes.clone(),
                format_id: fmt_id.into(),
                offsets: OffsetMode::Utf16 as i32,
            })
            .await
            .unwrap_or_else(|e| panic!("FormatDocument({fmt_id}) failed: {e:?}"))
            .into_inner();
        assert_eq!(exported.format_id, fmt_id);
        assert!(!exported.media_type.is_empty());
        assert!(!exported.file_extension.is_empty());
        assert!(!exported.content.is_empty(), "{fmt_id} content must be non-empty");
    }

    drop(client);
    srv.stop().await;
}
