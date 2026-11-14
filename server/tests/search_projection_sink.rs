//! End-to-end integration tests for `SearchDocumentProjection` (`#2345`):
//! single-pass projection from raw bytes (`turbo-encoding`) and UTF-8 text,
//! `ViewId::SOURCE` bidirectional offset round-tripping across `Utf8`, `Utf16`,
//! and `CodePoint`, syntactic negation and modality entity bucketing,
//! assessed-clean vs unassessed quality scalars, union-not-sum overlapping
//! noise counting, and `TurboSinkService` streaming (`DocumentSinkItem.search_projection`).

mod common;
#[path = "search_projection/fixture.rs"]
mod fixture;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::tiny;
use fixture::{RAW_TEXT, build_full_search_doc};
use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::offsets::{OffsetMode, Offsetter};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{
    InputBinding, OffsetMode as ProtoOffsetMode, ProjectSearchDocumentRequest, SearchAssessmentState,
    SearchDocumentProjection,
};
use turbo_kserve::proto::sink::DocumentSinkItem;
use turbo_kserve::proto::sink::turbo_sink_service_client::TurboSinkServiceClient;
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::search_projection::{
    columns::union_span_chars, document_to_search_projection, document_to_search_projection_bound,
    validate_search_projection,
};
use turbo_kserve::sink::{SinkConfig, SinkReceiver, SinkTyped};
use turbo_kserve::{Serve, Server};
use turbo_normalize_layers::{Artifacts, Noise};

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

#[test]
fn single_pass_projection_preserves_source_offsets_polarity_and_quality_across_modes() {
    let doc = build_full_search_doc();
    let config_digest = [0x42u8; 32];
    let binding = InputBinding {
        source_revision: "rev-2345".into(),
        parse_artifact_id: "raw/bom-utf8".into(),
        parse_artifact_digest: vec![0x11; 32],
    };

    for (mode, proto_mode) in [
        (OffsetMode::Utf8, ProtoOffsetMode::Utf8),
        (OffsetMode::Utf16, ProtoOffsetMode::Utf16),
        (OffsetMode::CodePoint, ProtoOffsetMode::CodePoint),
    ] {
        let proj = document_to_search_projection_bound(
            "doc-2345",
            &doc,
            ViewId::SOURCE,
            mode,
            Some(&binding),
            &config_digest,
            &turbo_kserve::results::disclosure::Disclosure::FULL,
        )
        .unwrap();
        validate_search_projection(&proj).unwrap();
        assert_eq!(proj.offset_mode, proto_mode as i32);

        let off = Offsetter::new(RAW_TEXT, mode);
        let s0 = &proj.sentences[0];
        let s0_bytes = off.from_wire(RAW_TEXT, "s0", s0.span_start, s0.span_end).unwrap();
        assert!(RAW_TEXT[s0_bytes.range()].starts_with("Café 😀 Berlin"));

        let cafe_tok = proj.tokens.iter().find(|t| t.term == "cafe").unwrap();
        let cafe_bytes = off.from_wire(RAW_TEXT, "cafe", cafe_tok.span_start, cafe_tok.span_end).unwrap();
        assert_eq!(&RAW_TEXT[cafe_bytes.range()], "Café");

        let syn = proj.tokens.iter().find(|t| t.term == "acme_corporation").unwrap();
        assert_eq!((syn.pos, syn.pos_len, syn.kind.as_str()), (2, 2, "synonym"));
        let syn_bytes = off.from_wire(RAW_TEXT, "syn", syn.span_start, syn.span_end).unwrap();
        assert_eq!(&RAW_TEXT[syn_bytes.range()], "Acme Corp");

        assert_eq!(proj.chunks.len(), 2);
        assert_eq!(proj.chunks[1].cut_margin, 42);
        assert!(proj.chunks[0].sketch.as_ref().unwrap().words > 0);
        assert!(proj.document_sketch.as_ref().unwrap().minhash.len() >= 16);

        let nt = proj.numeric_temporal.as_ref().unwrap();
        assert_eq!(nt.money.len(), 2);
        assert!(
            nt.money.iter().any(|m| m.currency == "EUR" && !m.ecb_converted && (m.amount_f64 - 100.0).abs() < 1e-6)
        );
        assert!(nt.money.iter().any(|m| m.currency == "USD" && m.ecb_converted && (m.amount_f64 - 108.5).abs() < 1e-6));
        assert_eq!(nt.quantities[0].unit, "kg");
        assert!((nt.quantities[0].value_f64 - 0.5).abs() < 1e-6);
        assert_eq!(nt.document_date.as_ref().unwrap().iso, "2026-05-01");
        assert!(nt.temporals[0].end_epoch_seconds > nt.temporals[0].start_epoch_seconds);

        assert_eq!(proj.geo_locations.len(), 1);
        assert_eq!(proj.geo_locations[0].canonical_name, "Berlin");
        assert_eq!(proj.geo_locations[0].country_code, "DE");
        assert_eq!(proj.geo_locations[0].spine_path, "Germany/Berlin/Berlin");

        let cs = proj.categories_sentiment.as_ref().unwrap();
        assert_eq!(cs.sentiment_label.as_deref(), Some("negative"));
        assert!((cs.sentiment_score.unwrap() - 0.82).abs() < 1e-6);

        let q = proj.quality.as_ref().unwrap();
        assert_eq!(q.noise_state, SearchAssessmentState::Assessed as i32);
        assert_eq!(q.artifacts_state, SearchAssessmentState::Assessed as i32);
        assert_eq!(q.text_quality_state, SearchAssessmentState::Unassessed as i32);
        assert!((q.worst_noise_score - 0.75).abs() < 1e-6);
        assert_eq!(q.union_noise_chars, 5);
        assert_eq!(q.artifact_count, 1);

        let ents = proj.entities.as_ref().unwrap();
        assert_eq!(ents.affirmed.len(), 1);
        assert_eq!(ents.affirmed[0].kb_id.as_deref(), Some("Q_ACME"));
        assert_eq!(ents.negated.len(), 1);
        assert_eq!(ents.negated[0].kb_id.as_deref(), Some("Q_FRAUD"));
        assert_eq!(ents.negated[0].cue.as_deref(), Some("no"));
        assert_eq!(ents.uncertain.len(), 1);
        assert_eq!(ents.uncertain[0].kb_id.as_deref(), Some("Q_GLOBEX"));
        assert_eq!(ents.uncertain[0].epistemic, "reported");
        assert_eq!(ents.uncertain[0].cue.as_deref(), Some("allegedly"));

        let prov = proj.provenance.as_ref().unwrap();
        assert_eq!(prov.analyzer_config_digest, config_digest);
        let inp = prov.input.as_ref().unwrap();
        assert_eq!(inp.raw_bytes_len, (RAW_TEXT.len() + 3) as u64);
        assert_eq!(inp.source_text_len, RAW_TEXT.len() as u32);
        assert_ne!(inp.raw_bytes_digest, inp.source_text_digest);
        assert_eq!(prov.views.len(), 2);
    }
}

#[test]
fn quality_distinguishes_assessed_clean_from_unassessed_and_unions_overlaps() {
    let unassessed = Document::new("Clean plain text.");
    let p_un = document_to_search_projection("u", &unassessed, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let q_un = p_un.quality.unwrap();
    assert_eq!(q_un.noise_state, SearchAssessmentState::Unassessed as i32);
    assert_eq!(q_un.artifacts_state, SearchAssessmentState::Unassessed as i32);
    assert_eq!((q_un.worst_noise_score, q_un.union_noise_chars, q_un.artifact_count), (0.0, 0, 0));

    let mut clean = Document::new("Clean plain text.");
    clean
        .add(
            &Key::DEFAULT,
            Noise { spans: Spans::new(ViewId::SOURCE), severity: Vec::new(), score: Vec::new() },
            Provenance::by("test:noise"),
        )
        .unwrap();
    clean
        .add(
            &Key::DEFAULT,
            Artifacts { spans: Spans::new(ViewId::SOURCE), kind: Vec::new() },
            Provenance::by("test:art"),
        )
        .unwrap();
    let p_cl = document_to_search_projection("c", &clean, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let q_cl = p_cl.quality.unwrap();
    assert_eq!(q_cl.noise_state, SearchAssessmentState::Assessed as i32);
    assert_eq!(q_cl.artifacts_state, SearchAssessmentState::Assessed as i32);
    assert_eq!((q_cl.worst_noise_score, q_cl.union_noise_chars, q_cl.artifact_count), (0.0, 0, 0));

    let s = "abcdef";
    let a = [Span::new(2, 5), Span::new(0, 3), Span::new(1, 4)];
    let b = [Span::new(0, 3), Span::new(1, 4), Span::new(2, 5)];
    assert_eq!(union_span_chars(s, &a), 5);
    assert_eq!(union_span_chars(s, &b), 5);
}

#[tokio::test]
async fn grpc_project_roundtrip_and_sink_stream_search_projection() {
    let doc = build_full_search_doc();
    let graph_bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();

    let got: Arc<Mutex<Vec<DocumentSinkItem>>> = Arc::default();
    let keep = got.clone();
    let receiver = SinkReceiver::new(move |item| keep.lock().unwrap().push(item));
    let sink = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model()],
        MAX,
        Serve { sink_receiver: Some(receiver), ..Serve::default() },
    )
    .await
    .unwrap();

    let cfg = SinkConfig {
        endpoint: format!("http://{}", sink.local_addr()),
        max_queue_items: 16,
        max_queue_bytes: 64 << 20,
        documents: false,
        typed: None,
        pii_output: Default::default(),
    };
    let sender = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model()],
        MAX,
        Serve {
            sink: Some(cfg),
            sink_search_projection: Some(SinkTyped { view: 0, offsets: ProtoOffsetMode::Utf16 }),
            ..Serve::default()
        },
    )
    .await
    .unwrap();

    let mut client = AnalysisServiceClient::connect(format!("http://{}", sender.local_addr())).await.unwrap();
    let direct = client
        .project_search_document(ProjectSearchDocumentRequest {
            doc_id: "doc-sink".into(),
            graph: graph_bytes.clone(),
            view: 0,
            offsets: ProtoOffsetMode::Utf16 as i32,
            pii_output: None,
            binding: None,
            analyzer_config_digest: Vec::new(),
        })
        .await
        .unwrap()
        .into_inner();
    let rt = client.round_trip_search_projection(direct.clone()).await.unwrap().into_inner();
    assert_eq!(rt, direct);

    let mut bad_proj: SearchDocumentProjection = direct.clone();
    bad_proj.target_text_len = 2;
    assert!(client.round_trip_search_projection(bad_proj).await.is_err());

    assert!(sender.forward_document("doc-sink", graph_bytes).await.unwrap());
    let stats = sender.close_sink(Duration::from_secs(10)).await.unwrap();
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.accepted_count, summary.rejected_count), (1, 0), "{:?}", summary.errors);

    let mut sink_client = TurboSinkServiceClient::connect(format!("http://{}", sink.local_addr())).await.unwrap();
    let standalone = DocumentSinkItem {
        sequence: 1,
        doc_id: "doc-standalone".into(),
        search_projection: Some(direct.clone()),
        ..Default::default()
    };
    let mismatch = DocumentSinkItem {
        sequence: 2,
        doc_id: "doc-mismatch".into(),
        binding: Some(InputBinding { source_revision: "other".into(), ..Default::default() }),
        search_projection: Some(direct.clone()),
        ..Default::default()
    };
    let sum2 = sink_client.stream_documents(tokio_stream::iter(vec![standalone, mismatch])).await.unwrap().into_inner();
    assert_eq!((sum2.accepted_count, sum2.rejected_count), (1, 1), "{:?}", sum2.errors);
    assert!(sum2.errors[0].contains("search projection carries another binding"), "{:?}", sum2.errors);

    sender.stop().await;
    sink.stop().await;

    let received = got.lock().unwrap().clone();
    assert_eq!(received.len(), 2);
    assert_eq!(received[0].search_projection.as_ref(), Some(&direct));
    assert_eq!(received[1].doc_id, "doc-standalone");
}
