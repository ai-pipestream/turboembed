mod common;
#[path = "typed_document/fixture.rs"]
mod fixture;

use turbo_analysis::export::export;
use turbo_analysis::{Document, Evidence, EvidenceStep, Evidenced, Key, Layer, Provenance, Tag, View};
use turbo_analysis_layers::{Categories, CategoryScope, Chunks, Sentences, Structure};
use turbo_kserve::proto::result::{DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode};
use turbo_kserve::results::document::codecs;

fn add<S: CategoryScope>(doc: &mut Document) -> String
where
    Categories<S>: Evidenced,
{
    let key = Key::<Categories<S>>::named("evidence-selected");
    let mut categories = Categories::<S>::new(fixture::tagset("inventory", &["a", "b"]));
    for _ in 0..doc.get::<S>().unwrap().len() {
        categories.push(Tag(0), &[0.8, 0.2]);
    }
    doc.add(
        &key,
        categories,
        Provenance { reads: vec![Key::<S>::DEFAULT.id()], ..Provenance::by("inventory:category") },
    )
    .unwrap();
    let evidence = EvidenceStep::new(key).build(doc).unwrap();
    assert!(evidence.records() > 0);
    let evidence_key = Key::<Evidence<Categories<S>>>::named("evidence-selected");
    doc.add(&evidence_key, evidence, Provenance { on: vec![key.id()], ..Provenance::by("inventory:evidence") })
        .unwrap();
    evidence_key.id().layer.to_string()
}

#[tokio::test]
async fn nonempty_standard_category_evidence_is_not_projected() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = fixture::build_rich_document();
    let ids = [add::<View>(&mut doc), add::<Sentences>(&mut doc), add::<Structure>(&mut doc), add::<Chunks>(&mut doc)];
    let bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let bytes = client
        .round_trip_document_graph(DocumentGraphRequest { graph: bytes.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    let mut missing = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = client
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "inventory".into(),
                graph: bytes.to_vec(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        client.round_trip_typed_document(out.clone()).await.unwrap();
        for id in &ids {
            if !out.layers.iter().any(|l| l.id == *id && l.instance == "evidence-selected") {
                missing.push(format!("{id}/{mode:?}"));
            }
        }
    }
    server.stop().await;
    assert!(missing.is_empty(), "nonempty imported category evidence was skipped: {missing:?}");
}
