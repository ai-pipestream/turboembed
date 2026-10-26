//! Retained independent review probes. No production source changes.
mod common;
#[path = "typed_document/fixture.rs"]
mod fixture;

use turbo_analysis::export::export;
use turbo_analysis::{
    Analyzer, Document, EvidenceStep, Execution, Groups, Ix, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet,
    View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{
    Block, DeonticKind, Entities, EpistemicStatus, Modality, ModalityPolarity, ModalityTag, PosTags, Structure, Tokens,
};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::proto::result::{
    DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, SpanFragment, TypedDocumentResult,
    typed_annotation::Value,
};
use turbo_kserve::results::document::codecs;

fn spans<L>(view: ViewId, pairs: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(view);
    for &(a, b) in pairs {
        out.push(Span::new(a, b));
    }
    out
}

fn named_fixture() -> Document {
    let mut doc = Document::new("A B C");
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, &[(0, 5)]), sentence: None },
        Provenance::by("review:default-tokenizer"),
    )
    .unwrap();
    doc.add(
        &Key::<Tokens>::named("gold"),
        Tokens { spans: spans(ViewId::SOURCE, &[(0, 1), (2, 3), (4, 5)]), sentence: None },
        Provenance::by("review:gold-tokenizer"),
    )
    .unwrap();
    let kind = TagSet::new("ner", &["THING"]);
    doc.add(
        &Key::<Entities>::DEFAULT,
        Entities {
            spans: spans(ViewId::SOURCE, &[(0, 1)]),
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: kind.clone(),
            score: vec![0.0],
        },
        Provenance { reads: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("review:default-ner") },
    )
    .unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .step(EvidenceStep::new(Key::<Entities>::DEFAULT))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let _ = analyzer.run(&mut docs);
    let mut doc = docs.pop().unwrap();
    doc.add(
        &Key::<Entities>::named("gold"),
        Entities {
            spans: spans(ViewId::SOURCE, &[(4, 5)]),
            tokens: vec![IxRange::new(2, 3)],
            kind: vec![Tag(0)],
            kinds: kind,
            score: vec![0.0],
        },
        Provenance { reads: vec![Key::<Tokens>::named("gold").id()], ..Provenance::by("review:gold-ner") },
    )
    .unwrap();
    doc.add(
        &Key::<PosTags>::named("gold"),
        PosTags { tagset: TagSet::new("pos", &["NN"]), tags: vec![Tag(0); 3], prob: None },
        Provenance { on: vec![Key::<Tokens>::named("gold").id()], ..Provenance::by("review:gold-pos") },
    )
    .unwrap();
    doc
}

async fn project(client: &mut common::Analysis, doc: &Document, mode: OffsetMode, view: u32) -> TypedDocumentResult {
    let graph = export(doc, codecs()).unwrap().to_bytes().unwrap();
    // Exercise the existing graph transport before the new typed transport.
    let graph = client
        .round_trip_document_graph(DocumentGraphRequest { graph: graph.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    client
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "review1253".into(),
            graph: graph.to_vec(),
            view,
            offsets: mode as i32,
        })
        .await
        .unwrap()
        .into_inner()
}

fn invalid(err: &tonic::Status) {
    assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err}");
    assert_eq!(err.metadata().get("turbo-code").unwrap().to_str().unwrap(), TURBO_E_INVALID_ARGUMENT.to_string());
    assert_eq!(err.metadata().get("turbo-field").unwrap(), "0");
}

#[tokio::test]
async fn valid_named_token_ranges_round_trip() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut errors = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = project(&mut client, &named_fixture(), mode, 0).await;
        let entity = out.layers.iter().find(|l| l.id == Entities::NAME.to_string() && l.instance == "gold").unwrap();
        assert_eq!(entity.indexes_layer[0].instance, "gold");
        assert_eq!(entity.indexes_layer[0].bound, 3);
        let result = client.round_trip_typed_document(out).await;
        if let Err(err) = result {
            errors.push((mode, err));
        }
    }
    server.stop().await;
    assert!(errors.is_empty(), "valid named token ranges rejected: {errors:?}");
}

#[tokio::test]
async fn named_entities_do_not_inherit_other_instance_evidence() {
    let (server, mut client) = common::serve_no_admin().await;
    let out = project(&mut client, &named_fixture(), OffsetMode::Utf8, 0).await;
    let default = out.layers.iter().find(|l| l.id == Entities::NAME.to_string() && l.instance.is_empty()).unwrap();
    let Some(Value::Entity(default)) = &default.annotations[0].value else { panic!("entity") };
    assert_eq!(default.votes.len(), 1);
    let gold = out.layers.iter().find(|l| l.id == Entities::NAME.to_string() && l.instance == "gold").unwrap();
    let Some(Value::Entity(gold)) = &gold.annotations[0].value else { panic!("entity") };
    assert!(gold.votes.is_empty(), "gold entity C inherited default entity A evidence: {:?}", gold.votes);
    server.stop().await;
}

#[tokio::test]
async fn missing_aligned_anchor_is_refused() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut out = project(&mut client, &named_fixture(), OffsetMode::Utf8, 0).await;
    // Remove entities to keep this independent of token-range validation.
    out.layers.retain(|l| l.id != Entities::NAME.to_string());
    out.layers.iter_mut().find(|l| l.id == PosTags::NAME.to_string()).unwrap().anchor_layer =
        Some("opennlp:tokens@absent".into());
    let result = client.round_trip_typed_document(out).await;
    assert!(result.is_err(), "missing aligned anchor was accepted");
    invalid(&result.unwrap_err());
    server.stop().await;
}

#[tokio::test]
async fn aligned_target_index_is_refused() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut out = project(&mut client, &named_fixture(), OffsetMode::Utf8, 0).await;
    out.layers.retain(|l| l.id != Entities::NAME.to_string());
    let pos = out.layers.iter_mut().find(|l| l.id == PosTags::NAME.to_string()).unwrap();
    let Some(Value::PosTag(tag)) = &mut pos.annotations[0].value else { panic!("POS") };
    tag.target_index = 99;
    let result = client.round_trip_typed_document(out).await;
    assert!(result.is_err(), "POS target_index=99 was accepted against three tokens");
    invalid(&result.unwrap_err());
    server.stop().await;
}

#[tokio::test]
async fn missing_default_index_target_is_not_replaced_by_named_instance() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut out = project(&mut client, &named_fixture(), OffsetMode::Utf8, 0).await;
    out.layers
        .retain(|l| l.id != Entities::NAME.to_string() && !(l.id == Tokens::NAME.to_string() && l.instance.is_empty()));
    let pos = out.layers.iter_mut().find(|l| l.id == PosTags::NAME.to_string()).unwrap();
    pos.indexes_layer.push(turbo_kserve::proto::result::LayerIndexBound {
        layer: Tokens::NAME.to_string(),
        instance: String::new(),
        bound: 1,
    });
    let result = client.round_trip_typed_document(out).await;
    assert!(result.is_err(), "missing default index target was replaced by named gold instance");
    invalid(&result.unwrap_err());
    server.stop().await;
}

#[tokio::test]
async fn owner_rich_document_survives_real_graph_and_typed_transport() {
    let (server, mut client) = common::serve_no_admin().await;
    let doc = fixture::build_rich_document();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        for view in [0, 1] {
            let out = project(&mut client, &doc, mode, view).await;
            assert!(out.layers.len() >= 35);
            assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
        }
    }
    server.stop().await;
}

#[tokio::test]
async fn nested_structure_group_round_trips() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("A B C");
    doc.add(
        &Key::<Structure>::DEFAULT,
        Structure {
            spans: spans(ViewId::SOURCE, &[(0, 5), (0, 1)]),
            kind: vec![Block::Section, Block::Paragraph],
            parent: vec![None, Some(Ix::new(0))],
            depth: vec![0, 1],
        },
        Provenance::by("review:structure"),
    )
    .unwrap();
    doc.add(
        &Key::<Groups<Structure>>::DEFAULT,
        Groups::new([[Ix::new(0), Ix::new(1)]]),
        Provenance { reads: vec![Key::<Structure>::DEFAULT.id()], ..Provenance::by("review:groups") },
    )
    .unwrap();
    let out = project(&mut client, &doc, OffsetMode::Utf8, 0).await;
    let result = client.round_trip_typed_document(out).await;
    assert!(result.is_ok(), "valid nested Structure group rejected: {result:?}");
    server.stop().await;
}

#[tokio::test]
async fn shared_modality_payload_is_preserved() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("May not go");
    let mut modality = Modality::new(ViewId::SOURCE);
    modality.push(
        Span::new(0, 10),
        ModalityTag {
            polarity: ModalityPolarity::Negated,
            epistemic: EpistemicStatus::Hedged,
            deontic: DeonticKind::Prohibition,
            cue: Some(Span::new(4, 7)),
            confidence: 0.9,
        },
    );
    doc.add(&Key::<Modality>::DEFAULT, modality, Provenance::by("review:modality")).unwrap();
    let mut sizes = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = project(&mut client, &doc, mode, 0).await;
        let layer = out.layers.iter().find(|l| l.id == Modality::NAME.to_string()).unwrap();
        sizes.push(layer.annotations.len());
    }
    server.stop().await;
    assert_eq!(sizes, [1, 1, 1], "nonempty shared Modality projected as empty successful document layer in all modes");
}

#[tokio::test]
async fn genuine_derived_views_offsets_provenance_and_structured_errors() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("Café 😀 北京");
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.replace(5, "Cafe");
    builder.keep(source.text().len());
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("plain"), View::derived(&source, text, alignment), Provenance::by("review:view"))
        .unwrap();
    let mut prov = Provenance::by("review:tokenizer");
    prov.config = 42;
    prov.execution = Some(Execution { backend: "cpu".into(), device: "cpu".into(), precision: "f32".into() });
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: spans(ViewId::SOURCE, &[(11, 17)]), sentence: None }, prov)
        .unwrap();
    doc.add(
        &Key::<Tokens>::named("plain"),
        Tokens { spans: spans(ViewId(1), &[(0, 4), (5, 9), (10, 16)]), sentence: None },
        Provenance::by("review:plain-tokens"),
    )
    .unwrap();
    doc.add(
        &Key::<PosTags>::named("plain"),
        PosTags { tagset: TagSet::new("pos", &["NN"]), tags: vec![Tag(0); 3], prob: Some(vec![0.9; 3]) },
        Provenance { on: vec![Key::<Tokens>::named("plain").id()], ..Provenance::by("review:plain-pos") },
    )
    .unwrap();
    for view in [0, 1] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = project(&mut client, &doc, mode, view).await;
            let text = if view == 0 { "Café 😀 北京" } else { "Cafe 😀 北京" };
            assert_eq!(out.raw_text, text);
            let start = text.find("北京").unwrap();
            let expected = match mode {
                OffsetMode::Utf8 => SpanFragment { start: start as u32, end: text.len() as u32 },
                OffsetMode::Utf16 => SpanFragment {
                    start: text[..start].encode_utf16().count() as u32,
                    end: text.encode_utf16().count() as u32,
                },
                OffsetMode::CodePoint => {
                    SpanFragment { start: text[..start].chars().count() as u32, end: text.chars().count() as u32 }
                }
            };
            let tok = out.layers.iter().find(|l| l.id == Tokens::NAME.to_string()).unwrap();
            assert_eq!(tok.annotations[0].spans, vec![expected]);
            let model = out.models.iter().find(|m| m.model_ref == tok.model_refs[0]).unwrap();
            assert_eq!(model.config_hash, 42);
            assert_eq!(model.execution.as_ref().unwrap().backend, "cpu");
            assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
            let mut bad = out.clone();
            bad.layers[0].annotations[0].spans[0].end = 1000;
            invalid(&client.round_trip_typed_document(bad).await.unwrap_err());
            let mut bad = out.clone();
            // UTF8 interior of é; UTF16 interior of 😀; code point outside the text.
            bad.layers[0].annotations[0].spans[0] = match mode {
                OffsetMode::Utf8 => SpanFragment { start: if view == 0 { 4 } else { 6 }, end: 8 },
                OffsetMode::Utf16 => SpanFragment { start: 6, end: 7 },
                OffsetMode::CodePoint => SpanFragment { start: 100, end: 101 },
            };
            invalid(&client.round_trip_typed_document(bad).await.unwrap_err());
            let mut bad = out;
            bad.layers[0].annotations[0].probability = Some(f64::NAN);
            invalid(&client.round_trip_typed_document(bad).await.unwrap_err());
        }
    }
    for (graph, offsets, view) in [
        (vec![255, 0, 1], 0, 0),
        (export(&doc, codecs()).unwrap().to_bytes().unwrap(), 99, 0),
        (export(&doc, codecs()).unwrap().to_bytes().unwrap(), 0, 999),
    ] {
        invalid(
            &client
                .inspect_document_graph(InspectDocumentGraphRequest { doc_id: "bad".into(), graph, view, offsets })
                .await
                .unwrap_err(),
        );
    }
    server.stop().await;
}
