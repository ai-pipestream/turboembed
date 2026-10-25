//! End-to-end tests for `TypedDocumentResult` projection (`InspectDocumentGraph`)
//! and wire validation (`RoundTripTypedDocument`) across `Utf8`, `Utf16`, and
//! `CodePoint` offset modes, discontinuous spans, multi-source entity voting,
//! derived views, and malformed wire refusals.

mod common;
#[path = "typed_document/fixture.rs"]
mod fixture;

use turbo_analysis::export::export;
use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, Tag, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Entities, PosTags, Sentences, Tokens};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{
    AnchorKind, InspectDocumentGraphRequest, LayerWireStatus, OffsetMode as ProtoOffsetMode, SpanFragment,
    TypedAnnotation, TypedDocumentResult, TypedLayer, typed_annotation::Value,
};
use turbo_kserve::results::Wire;
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::typed::{document_to_typed, inspect_document_graph, round_trip_typed_document};

use fixture::*;

#[test]
fn typed_document_projects_layers_discontinuous_spans_and_entity_votes_in_all_offset_modes() {
    let doc = build_rich_document();
    let (beijing, cafe, price) = (byte_span("北京"), byte_span("Café"), byte_span("$42.50"));

    for (mode, proto_mode) in [
        (OffsetMode::Utf8, ProtoOffsetMode::Utf8),
        (OffsetMode::Utf16, ProtoOffsetMode::Utf16),
        (OffsetMode::CodePoint, ProtoOffsetMode::CodePoint),
    ] {
        let typed = document_to_typed("doc-1", &doc, ViewId::SOURCE, mode).unwrap();
        assert_eq!(typed.doc_id, "doc-1");
        assert_eq!(typed.raw_text, TEXT);
        assert_eq!(typed.offsets, proto_mode as i32);
        assert_eq!(typed.views.len(), 2);
        assert_eq!(typed.views[1].instance, "clean");
        assert!(!typed.models.is_empty(), "deduplicated model table populated");

        for layer in &typed.layers {
            assert_eq!(layer.status, LayerWireStatus::Ok as i32, "layer {} status", layer.id);
        }

        let ent_layer = typed.layers.iter().find(|l| l.id == "opennlp:entities").unwrap();
        assert_eq!(ent_layer.annotations.len(), 2);
        let ent0 = &ent_layer.annotations[0];
        assert_eq!(ent0.spans.len(), 1);
        assert!((ent0.probability.unwrap() - 0.92).abs() < 1e-6);
        let Some(Value::Entity(e0)) = &ent0.value else { panic!("expected Entity value") };
        assert_eq!(e0.kind, "LOC");
        assert_eq!(e0.canonical, "北京");
        assert_eq!(e0.votes.len(), 1);
        assert_eq!(e0.votes[0].annotator, "test:ner");

        let ent1 = &ent_layer.annotations[1];
        assert_eq!(ent1.spans.len(), 2, "discontinuous entity preserves both fragments");
        assert_eq!(ent1.probability, None, "rule-based entity with 0.0 score omits probability");

        match mode {
            OffsetMode::Utf8 => {
                assert_eq!(ent0.spans[0], SpanFragment { start: beijing.start, end: beijing.end });
                assert_eq!(ent1.spans[0], SpanFragment { start: cafe.start, end: cafe.end });
                assert_eq!(ent1.spans[1], SpanFragment { start: price.start, end: price.end });
            }
            OffsetMode::Utf16 => {
                assert_eq!(
                    ent0.spans[0],
                    SpanFragment { start: java_utf16(beijing.start), end: java_utf16(beijing.end) }
                );
            }
            OffsetMode::CodePoint => {
                let cp_start = TEXT[..beijing.start as usize].chars().count() as u32;
                let cp_end = TEXT[..beijing.end as usize].chars().count() as u32;
                assert_eq!(ent0.spans[0], SpanFragment { start: cp_start, end: cp_end });
            }
        }

        let back = round_trip_typed_document(&typed).unwrap();
        assert_eq!(back, typed);
        let via_wire = TypedDocumentResult::from_msg(&typed).unwrap();
        assert_eq!(via_wire.to_msg(), typed);
    }

    let clean_typed = document_to_typed("doc-clean", &doc, ViewId(1), OffsetMode::Utf8).unwrap();
    assert_eq!(clean_typed.view, 1);
    assert_eq!(round_trip_typed_document(&clean_typed).unwrap(), clean_typed);
}

#[tokio::test]
async fn inspect_and_round_trip_typed_document_over_grpc_with_refusals() {
    let (server, mut client) = common::serve_no_admin().await;
    let whole = Span::new(0, TEXT.len() as u32);
    let (cafe, beijing, price) = (byte_span("Café"), byte_span("北京"), byte_span("$42.50"));
    let mut export_doc = Document::new(TEXT);
    add(&mut export_doc, Sentences { spans: spans(&[whole]) }, Provenance::by("test:sent"));
    add(
        &mut export_doc,
        Tokens { spans: spans(&[cafe, beijing, price]), sentence: Some(vec![Ix::new(0); 3]) },
        Provenance::by("test:tok"),
    );
    let on_tok = Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:pos") };
    add(
        &mut export_doc,
        PosTags {
            tagset: tagset("pos", &["NN", "NNP", "SYM"]),
            tags: vec![Tag(0), Tag(1), Tag(2)],
            prob: Some(vec![0.9, 0.95, 0.8]),
        },
        on_tok.clone(),
    );
    let mut ent_spans = Spans::<Entities>::new(ViewId::SOURCE);
    ent_spans.try_push_fragments(&[beijing]).unwrap();
    ent_spans.try_push_fragments(&[cafe, price]).unwrap();
    add(
        &mut export_doc,
        Entities {
            spans: ent_spans,
            tokens: vec![IxRange::new(1, 2), IxRange::new(0, 3)],
            kind: vec![Tag(0), Tag(1)],
            kinds: tagset("ner", &["LOC", "PRODUCT"]),
            score: vec![0.92, 0.0],
        },
        on_tok,
    );
    let mut chunk_spans = Spans::<Chunks>::new(ViewId::SOURCE);
    chunk_spans.try_push_fragments(&[cafe]).unwrap();
    chunk_spans.try_push_fragments(&[beijing, price]).unwrap();
    add(
        &mut export_doc,
        Chunks {
            spans: chunk_spans,
            level: vec![0, 0],
            parent: vec![None, None],
            sentences: vec![IxRange::new(0, 1), IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart, Cut::Sentence],
            cost: vec![0, 3],
            model_tokens: vec![1, 2],
            over_budget: vec![false, false],
            prompt_overhead: 0,
        },
        Provenance { reads: vec![Key::<Sentences>::DEFAULT.id()], ..Provenance::by("test:chunks") },
    );
    let graph_bytes = export(&export_doc, codecs()).unwrap().to_bytes().unwrap();

    let inspected = client
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "grpc-doc".into(),
            graph: graph_bytes,
            view: 0,
            offsets: ProtoOffsetMode::Utf16 as i32,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(inspected.raw_text, TEXT);
    assert!(inspected.layers.iter().any(|l| l.id == "opennlp:entities"));
    assert!(inspected.layers.iter().any(|l| l.id == "turbo:chunks"));

    let echoed = client.round_trip_typed_document(inspected.clone()).await.unwrap().into_inner();
    assert_eq!(echoed, inspected);

    let mut bad_surrogate = inspected.clone();
    let tok_layer = bad_surrogate.layers.iter_mut().find(|l| l.id == "opennlp:tokens").unwrap();
    tok_layer.annotations[0].spans[0].start = java_utf16(byte_span("\u{1f600}").start) + 1;
    let err = client.round_trip_typed_document(bad_surrogate).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err}");

    let mut bad_inverted = inspected.clone();
    let tok_layer = bad_inverted.layers.iter_mut().find(|l| l.id == "opennlp:tokens").unwrap();
    tok_layer.annotations[0].spans[0] = SpanFragment { start: 4, end: 2 };
    let err = round_trip_typed_document(&bad_inverted).unwrap_err();
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);

    let mut bad_prob = inspected.clone();
    let pos_layer = bad_prob.layers.iter_mut().find(|l| l.id == "opennlp:pos").unwrap();
    pos_layer.annotations[0].probability = Some(1.5);
    let err = round_trip_typed_document(&bad_prob).unwrap_err();
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);

    let mut bad_ix = inspected.clone();
    let ent_layer = bad_ix.layers.iter_mut().find(|l| l.id == "opennlp:entities").unwrap();
    if let Some(Value::Entity(ref mut e)) = ent_layer.annotations[0].value {
        e.token_end = 99;
    }
    let err = round_trip_typed_document(&bad_ix).unwrap_err();
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);

    // Duplicate model_ref in models table must be refused.
    let mut dup_model = inspected.clone();
    dup_model.models.push(dup_model.models[0].clone());
    let err = round_trip_typed_document(&dup_model).unwrap_err();
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);

    // Unknown model_ref on a layer or annotation must be refused, both when
    // models is populated and when models is empty.
    let mut unknown_model = inspected.clone();
    unknown_model.layers[0].model_refs = vec!["bogus".into()];
    let err = round_trip_typed_document(&unknown_model).unwrap_err();
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);

    let mut empty_models_with_ref = inspected.clone();
    empty_models_with_ref.models.clear();
    empty_models_with_ref.layers[0].model_refs = vec!["bogus".into()];
    empty_models_with_ref.layers[0].annotations[0].model_ref = Some("also-bogus".into());
    let err = client.round_trip_typed_document(empty_models_with_ref).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err}");

    // Out-of-bounds indexes_layer bound must be refused.
    let mut bad_bound = inspected.clone();
    let ent_layer = bad_bound.layers.iter_mut().find(|l| l.id == "opennlp:entities").unwrap();
    ent_layer.indexes_layer[0].bound = 999;
    let err = client.round_trip_typed_document(bad_bound).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err}");

    let err = inspect_document_graph(&InspectDocumentGraphRequest {
        doc_id: "bad".into(),
        graph: vec![0xff, 0x00, 0x01],
        view: 0,
        offsets: ProtoOffsetMode::Utf8 as i32,
    })
    .unwrap_err();
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);

    let _: (AnchorKind, TypedLayer, TypedAnnotation) =
        (AnchorKind::Text, inspected.layers[0].clone(), inspected.layers[0].annotations[0].clone());

    server.stop().await;
}
