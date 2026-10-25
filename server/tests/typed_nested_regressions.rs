//! Additional real graph/RPC controls for transitive anchors and nested references.

mod common;
#[path = "typed_nested/context.rs"]
mod context;

use turbo_analysis::export::export;
use turbo_analysis::{
    Document, Ix, IxRange, Key, Layer, ModelRef, NBest, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{
    ChunkTableHeaders, ChunkTags, Chunks, ColumnAlign, ConsolidatedMention, Cut, DocumentEntities, DocumentEntity,
    Entities, EntityRelation, EntitySalience, EntitySalienceScore, PosTags, Sentences, SparseTerms,
    SyntacticRoleCounts, TableCell, Tables, Tokens,
};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::proto::result::{
    DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, SpanFragment, TypedDocumentResult,
    typed_annotation::Value,
};
use turbo_kserve::results::document::codecs;

fn spans<L>(view: ViewId, pairs: &[(u32, u32)]) -> Spans<L> {
    let mut spans = Spans::new(view);
    for &(start, end) in pairs {
        spans.push(Span::new(start, end));
    }
    spans
}

async fn project(client: &mut common::Analysis, doc: &Document, mode: OffsetMode, view: u32) -> TypedDocumentResult {
    let graph = export(doc, codecs()).unwrap().to_bytes().unwrap();
    let graph = client
        .round_trip_document_graph(DocumentGraphRequest { graph: graph.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    client
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "nested".into(),
            graph: graph.to_vec(),
            view,
            offsets: mode as i32,
        })
        .await
        .unwrap()
        .into_inner()
}

fn invalid(error: tonic::Status) {
    assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error}");
    assert_eq!(error.metadata().get("turbo-code").unwrap().to_str().unwrap(), TURBO_E_INVALID_ARGUMENT.to_string());
    assert_eq!(error.metadata().get("turbo-field").unwrap(), "0");
}

#[tokio::test]
async fn named_pos_and_chunk_nbest_keep_both_views_in_all_offset_modes() {
    let (server, mut client) = common::serve_no_admin().await;
    for with_default in [false, true] {
        let mut doc = Document::new("xxé");
        let source = doc.get::<View>().unwrap().clone();
        let mut builder = ViewBuilder::new(source.text());
        builder.replace(2, "");
        let (text, alignment) = builder.finish();
        doc.add(&Key::<View>::named("trim"), View::derived(&source, text, alignment), Provenance::by("nested:view"))
            .unwrap();
        if with_default {
            doc.add(
                &Key::<Tokens>::DEFAULT,
                Tokens { spans: spans(ViewId::SOURCE, &[(0, 1)]), sentence: None },
                Provenance::by("nested:default"),
            )
            .unwrap();
        }
        let tokens = Key::<Tokens>::named("selected");
        doc.add(
            &tokens,
            Tokens { spans: spans(ViewId(1), &[(0, 2)]), sentence: None },
            Provenance::by("nested:tokens"),
        )
        .unwrap();
        let tags = TagSet::new("tags", &["X"]);
        let pos = Key::<PosTags>::named("pos-selected");
        doc.add(
            &pos,
            PosTags { tagset: tags.clone(), tags: vec![Tag(0)], prob: None },
            Provenance { on: vec![tokens.id()], ..Provenance::by("nested:pos") },
        )
        .unwrap();
        let chunks = Key::<ChunkTags>::named("chunk-selected");
        doc.add(
            &chunks,
            ChunkTags { tagset: tags.clone(), tags: vec![Tag(0)], prob: None },
            Provenance { on: vec![tokens.id()], ..Provenance::by("nested:chunks") },
        )
        .unwrap();
        let mut nbest = NBest::<PosTags>::new(tags.clone(), 1);
        nbest.push(f64::NEG_INFINITY, &[Tag(0)], Some(&[0.8])).unwrap();
        doc.add(
            &Key::<NBest<PosTags>>::DEFAULT,
            nbest,
            Provenance { on: vec![pos.id()], ..Provenance::by("nested:nbest-pos") },
        )
        .unwrap();
        let mut nbest = NBest::<ChunkTags>::new(tags, 1);
        nbest.push(-0.2, &[Tag(0)], None).unwrap();
        doc.add(
            &Key::<NBest<ChunkTags>>::named("candidates"),
            nbest,
            Provenance { on: vec![chunks.id()], ..Provenance::by("nested:nbest-chunks") },
        )
        .unwrap();
        for view in [0, 1] {
            for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
                let out = project(&mut client, &doc, mode, view).await;
                let start = if view == 0 { 2 } else { 0 };
                let end = start + if mode == OffsetMode::Utf8 { 2 } else { 1 };
                let mut count = 0;
                for layer in out.layers.iter().filter(|l| l.id.starts_with("turbo:nbest")) {
                    assert_eq!(layer.annotations[0].spans, [SpanFragment { start, end }]);
                    let anchor = layer.anchor_layer.as_deref().unwrap();
                    assert!(anchor.ends_with("@pos-selected") || anchor.ends_with("@chunk-selected"));
                    count += 1;
                }
                assert_eq!(count, 2);
                assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
            }
        }
    }
    server.stop().await;
}

fn consolidated() -> Document {
    let mut doc = Document::new("é😀 A");
    let pairs = [(0, 2), (7, 8)];
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, &pairs), sentence: None },
        Provenance::by("nested:tokens"),
    )
    .unwrap();
    for (key, count) in [(Key::<Entities>::DEFAULT, 1), (Key::<Entities>::named("gold"), 2)] {
        doc.add(
            &key,
            Entities {
                spans: spans(ViewId::SOURCE, &pairs[..count]),
                tokens: (0..count).map(|i| IxRange::new(i as u32, i as u32 + 1)).collect(),
                kind: vec![Tag(0); count],
                kinds: TagSet::new("ner", &["THING"]),
                score: vec![0.8; count],
            },
            Provenance { reads: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("nested:ner") },
        )
        .unwrap();
    }
    let mut entities = DocumentEntities::new(ViewId::SOURCE);
    for (i, &(start, end)) in pairs.iter().enumerate() {
        entities.entities.push(DocumentEntity {
            canonical_name: format!("entity{i}").into(),
            kind: "THING".into(),
            kb_id: None,
            geo_coord: None,
            country_code: None,
            mentions: vec![ConsolidatedMention {
                span: Span::new(start, end),
                entity: Some(Ix::new(i as u32)),
                is_representative: true,
                role: None,
            }]
            .into_boxed_slice(),
            role_counts: SyntacticRoleCounts::default(),
            mean_sentiment: 0.0,
            sentiment_mentions: 0,
            confidence: 0.8,
        });
    }
    entities.relations.push(EntityRelation { subject: 0, object: 1, kind: "related".into(), score: -0.5, support: 1 });
    doc.add(
        &Key::<DocumentEntities>::named("rollup"),
        entities,
        Provenance { on: vec![Key::<Entities>::named("gold").id()], ..Provenance::by("nested:rollup") },
    )
    .unwrap();
    doc
}

#[tokio::test]
async fn consolidated_mentions_use_exact_entity_targets_and_refuse_malformed_nested_fields() {
    let (server, mut client) = common::serve_no_admin().await;
    let doc = consolidated();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let good = project(&mut client, &doc, mode, 0).await;
        let layer = good.layers.iter().find(|l| l.id == DocumentEntities::NAME.to_string()).unwrap();
        assert_eq!(layer.indexes_layer[0].instance, "gold");
        assert_eq!(client.round_trip_typed_document(good.clone()).await.unwrap().into_inner(), good);
        for field in ["index", "target", "span", "confidence", "sentiment", "relation-score", "relation-subject"] {
            let mut bad = good.clone();
            let layer = bad.layers.iter_mut().find(|l| l.id == DocumentEntities::NAME.to_string()).unwrap();
            if field == "target" {
                layer.indexes_layer.clear();
            }
            let Some(Value::DocumentEntity(entity)) = &mut layer.annotations[0].value else { panic!("entity") };
            match field {
                "index" => entity.mentions[0].entity_index = Some(99),
                "span" => entity.mentions[0].span = None,
                "confidence" => entity.confidence = 1.1,
                "sentiment" => entity.mean_sentiment = f32::NAN,
                "relation-score" => entity.relations[0].score = f32::NAN,
                "relation-subject" => entity.relations[0].subject = 1,
                _ => {}
            }
            invalid(client.round_trip_typed_document(bad).await.unwrap_err());
        }
    }
    server.stop().await;
}

#[tokio::test]
async fn consolidated_mentions_reject_ambiguous_targets_in_either_order() {
    let (server, mut client) = common::serve_no_admin().await;
    let doc = consolidated();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let good = project(&mut client, &doc, mode, 0).await;
        assert_eq!(client.round_trip_typed_document(good.clone()).await.unwrap().into_inner(), good);
        for default_first in [false, true] {
            let mut bad = good.clone();
            let layer = bad.layers.iter_mut().find(|l| l.id == DocumentEntities::NAME.to_string()).unwrap();
            let gold = layer
                .indexes_layer
                .iter()
                .find(|bound| bound.layer == "opennlp:entities" && bound.instance == "gold")
                .unwrap()
                .clone();
            let mut default = gold.clone();
            default.instance.clear();
            default.bound = 1;
            layer.indexes_layer.retain(|bound| bound.layer != "opennlp:entities");
            if default_first {
                layer.indexes_layer.extend([default, gold]);
            } else {
                layer.indexes_layer.extend([gold, default]);
            }
            let error = client.round_trip_typed_document(bad).await.unwrap_err();
            assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error}");
            assert!(error.message().contains("ambiguous opennlp:entities index targets"), "{error}");
        }
    }
    server.stop().await;
}

#[tokio::test]
async fn candidate_and_sparse_nested_values_return_structured_invalid_arguments() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("é😀");
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, &[(0, 2)]), sentence: None },
        Provenance::by("nested:tokens"),
    )
    .unwrap();
    let tags = TagSet::new("pos", &["NN"]);
    doc.add(
        &Key::<PosTags>::DEFAULT,
        PosTags { tagset: tags.clone(), tags: vec![Tag(0)], prob: None },
        Provenance::by("nested:pos"),
    )
    .unwrap();
    let mut nbest = NBest::<PosTags>::new(tags, 1);
    nbest.push(-0.5, &[Tag(0)], Some(&[0.8])).unwrap();
    doc.add(&Key::<NBest<PosTags>>::DEFAULT, nbest, Provenance::by("nested:nbest")).unwrap();
    let mut sparse =
        SparseTerms::<View>::new(ModelRef { id: "fixture".into(), revision: "1".into(), sha256: "fixture".into() });
    sparse.push(1, 1.0, Some(Span::new(0, 2)));
    sparse.push_occurrences(&[Span::new(0, 2)]);
    sparse.unit();
    doc.add(&Key::<SparseTerms<View>>::DEFAULT, sparse, Provenance::by("nested:sparse")).unwrap();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let good = project(&mut client, &doc, mode, 0).await;
        assert_eq!(client.round_trip_typed_document(good.clone()).await.unwrap().into_inner(), good);
        for field in ["probability", "rank", "sequence-score", "weight", "span", "occurrence"] {
            let mut bad = good.clone();
            if matches!(field, "probability" | "rank" | "sequence-score") {
                let layer = bad.layers.iter_mut().find(|l| l.id.starts_with("turbo:nbest")).unwrap();
                let Some(Value::Nbest(nbest)) = &mut layer.annotations[0].value else { panic!("nbest") };
                match field {
                    "probability" => nbest.candidates[0].probability = Some(f64::NAN),
                    "rank" => nbest.candidates[0].rank = 99,
                    _ => nbest.candidates[0].sequence_score = f64::NAN,
                }
            } else {
                let layer = bad.layers.iter_mut().find(|l| l.id.starts_with("turbo:sparse")).unwrap();
                let Some(Value::SparseTerms(sparse)) = &mut layer.annotations[0].value else { panic!("sparse") };
                let invalid = match mode {
                    OffsetMode::Utf8 => SpanFragment { start: 1, end: 2 },
                    OffsetMode::Utf16 => SpanFragment { start: 2, end: 3 },
                    _ => SpanFragment { start: 2, end: 3 },
                };
                match field {
                    "weight" => sparse.entries[0].weight = f32::NAN,
                    "span" => sparse.entries[0].span = Some(invalid),
                    _ => sparse.entries[0].occurrences[0] = invalid,
                }
            }
            invalid(client.round_trip_typed_document(bad).await.unwrap_err());
        }
    }
    server.stop().await;
}
