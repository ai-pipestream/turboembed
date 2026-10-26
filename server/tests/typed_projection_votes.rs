//! Projected nested entity evidence must remain valid typed wire input.
mod common;

use turbo_analysis::export::export;
use turbo_analysis::{
    Document, EvidenceStep, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, OffsetMode, typed_annotation::Value};
use turbo_kserve::results::document::codecs;

fn fixture(name: &'static str, fragments: &[Span], collapse: bool) -> Document {
    let mut doc = Document::new("é-😀-z");
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    if collapse {
        builder.replace(9, "Q");
    } else {
        builder.keep(2);
        builder.replace(3, "");
        builder.keep(7);
        builder.replace(8, "");
        builder.keep(9);
    }
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("plain"), View::derived(&source, text, alignment), Provenance::by("test:view"))
        .unwrap();
    let mut tokens = Spans::new(ViewId::SOURCE);
    tokens.push(Span::new(0, 9));
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test:tokens")).unwrap();
    let key = Key::<Entities>::named(name);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push_fragments(fragments);
    doc.add(
        &key,
        Entities {
            spans,
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("ner", &["THING"]),
            score: vec![0.75],
        },
        Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:entity") },
    )
    .unwrap();
    let step = EvidenceStep::new(key);
    let evidence = step.build(&doc).unwrap();
    doc.add(&step.output(), evidence, Provenance { on: vec![key.id()], ..Provenance::by("test:evidence") }).unwrap();
    doc
}

async fn verify(fragments: &[Span], collapse: bool, byte_expected: (u32, u32), unit_expected: (u32, u32)) {
    let (server, mut client) = common::serve_no_admin().await;
    let mut failures = Vec::new();
    for name in ["", "gold"] {
        let graph = export(&fixture(name, fragments, collapse), codecs()).unwrap().to_bytes().unwrap();
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            // Same-view projections are a positive control for the original fragments.
            for view in [0, 1] {
                let out = client
                    .inspect_document_graph(InspectDocumentGraphRequest {
                        doc_id: "entity-votes".into(),
                        graph: graph.clone(),
                        view,
                        offsets: mode as i32,
                    })
                    .await
                    .unwrap()
                    .into_inner();
                let layer =
                    out.layers.iter().find(|l| l.id == Entities::NAME.to_string() && l.instance == name).unwrap();
                let annotation = &layer.annotations[0];
                let Some(Value::Entity(entity)) = &annotation.value else { panic!("missing entity") };
                assert_eq!(entity.votes.len(), 1);
                let vote = &entity.votes[0];
                assert_eq!(vote.annotator, "test:entity");
                // This fixture has no model provenance, so EvidenceStep does
                // not claim a model probability. The entity retains its score.
                assert_eq!(vote.probability, None);
                assert_eq!(annotation.probability, Some(0.75));
                assert!(out.models.iter().any(|m| Some(&m.model_ref) == vote.model_ref.as_ref()));
                if view == 1 {
                    let expected = if mode == OffsetMode::Utf8 { byte_expected } else { unit_expected };
                    assert_eq!(annotation.spans.iter().map(|f| (f.start, f.end)).collect::<Vec<_>>(), [expected]);
                } else {
                    assert_eq!(annotation.spans.len(), fragments.len());
                    assert_eq!(annotation.spans, vote.spans);
                }
                let back = client.round_trip_typed_document(out.clone()).await;
                if view == 0 {
                    assert_eq!(back.unwrap().into_inner(), out);
                } else {
                    match back {
                        Ok(back) => {
                            assert_eq!(back.into_inner(), out);
                            assert_eq!(annotation.spans, vote.spans);
                        }
                        Err(err) => failures.push(format!("instance={name:?}, {mode:?}: {err}")),
                    }
                }
            }
        }
    }
    server.stop().await;
    assert!(failures.is_empty(), "server refused projected entity votes: {failures:#?}");
}

#[tokio::test]
async fn entity_votes_drop_only_the_fragment_on_deleted_text() {
    verify(&[Span::new(0, 2), Span::new(7, 8)], false, (0, 2), (0, 1)).await;
}

#[tokio::test]
async fn entity_votes_preserve_one_location_when_all_fragments_are_deleted() {
    verify(&[Span::new(2, 3), Span::new(7, 8)], false, (2, 2), (1, 1)).await;
}

#[tokio::test]
async fn entity_votes_merge_fragments_that_project_onto_the_same_replacement() {
    verify(&[Span::new(0, 2), Span::new(8, 9)], true, (0, 1), (0, 1)).await;
}
