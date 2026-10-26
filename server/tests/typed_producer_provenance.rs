//! Evidence keeps its recorded identity without borrowing a layer configuration.
mod common;

use std::collections::HashMap;
use turbo_analysis::{
    Document, Evidence, EvidenceStep, Evidenced, Execution, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans,
    Tag, TagSet, View, ViewId, export::export,
};
use turbo_analysis_layers::{Categories, Entities, Tokens};
use turbo_kserve::{
    proto::result::{DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, typed_annotation::Value},
    results::document::codecs,
};

const TEXT: &str = "é😀";
const NAMES: [&str; 6] = ["", "second", "model-a", "model-a-second", "model-b", "model-b-second"];

fn spans<L>() -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, TEXT.len() as u32));
    spans
}

fn model(i: usize) -> Option<ModelRef> {
    (i >= 2).then(|| ModelRef {
        id: "same-model".into(),
        revision: format!("revision-{}", i / 2),
        sha256: format!("digest-{}", i / 2),
    })
}

fn add<L: Layer + Evidenced>(doc: &mut Document, i: usize, layer: L, anchor: turbo_analysis::KeyId) {
    let key = Key::<L>::named(NAMES[i]);
    doc.add(
        &key,
        layer,
        Provenance {
            version: "recorded-version",
            config: 101 + i as u64,
            model: model(i),
            execution: Some(Execution {
                backend: "cpu".into(),
                device: format!("device-{i}"),
                precision: "exact".into(),
            }),
            on: vec![anchor],
            ..Provenance::by("shared-producer")
        },
    )
    .unwrap();
    let evidence = EvidenceStep::new(key).build(doc).unwrap().clone();
    doc.add(
        &Key::<Evidence<L>>::named(NAMES[i]),
        evidence,
        Provenance { on: vec![key.id()], ..Provenance::by("evidence-wrapper") },
    )
    .unwrap();
}

async fn check(entities: bool) {
    let (server, mut client) = common::serve_no_admin().await;
    for reverse in [false, true] {
        let mut doc = Document::new(TEXT);
        doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: spans(), sentence: None }, Provenance::by("tokens")).unwrap();
        for pos in 0..6 {
            let i = if reverse { 5 - pos } else { pos };
            if entities {
                add(
                    &mut doc,
                    i,
                    Entities {
                        spans: spans(),
                        tokens: vec![IxRange::new(0, 1)],
                        kind: vec![Tag(0)],
                        kinds: TagSet::new("entity", &["PERSON"]),
                        score: vec![0.8],
                    },
                    Key::<Tokens>::DEFAULT.id(),
                );
            } else {
                let mut cats = Categories::<View>::new(TagSet::new("category", &["a", "b"]));
                cats.push(Tag(0), &[0.8, 0.2]);
                add(&mut doc, i, cats, Key::<View>::DEFAULT.id());
            }
        }
        let graph = export(&doc, codecs()).unwrap().to_bytes().unwrap();
        let graph = client
            .round_trip_document_graph(DocumentGraphRequest { graph: graph.into() })
            .await
            .unwrap()
            .into_inner()
            .graph;
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = client
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "producer-provenance".into(),
                    graph: graph.to_vec(),
                    view: 0,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            let mut refs = HashMap::new();
            let mut producers = 0;
            let mut full_records = 0;
            for layer in &out.layers {
                let i = NAMES.iter().position(|n| *n == layer.instance).unwrap();
                for ann in &layer.annotations {
                    if matches!(ann.value, Some(Value::Category(_)) | Some(Value::Entity(_))) {
                        let full = out.models.iter().find(|m| Some(&m.model_ref) == ann.model_ref.as_ref()).unwrap();
                        assert_eq!(full.config_hash, 101 + i as u64);
                        assert_eq!(full.version, "recorded-version");
                        assert_eq!(full.execution.as_ref().unwrap().device, format!("device-{i}"));
                        full_records += 1;
                    }
                    let producer_ref = match &ann.value {
                        Some(Value::Evidence(_)) => ann.model_ref.as_ref().unwrap(),
                        Some(Value::Entity(entity)) => {
                            assert_eq!(entity.votes.len(), 1);
                            entity.votes[0].model_ref.as_ref().unwrap()
                        }
                        _ => continue,
                    };
                    let producer = out.models.iter().find(|m| &m.model_ref == producer_ref).unwrap();
                    assert_eq!(producer.annotator, "shared-producer");
                    assert_eq!(producer.config_hash, 0, "unrecorded configuration: {reverse}/{mode:?}/{i}");
                    assert_eq!(producer.version, "0");
                    assert!(producer.execution.is_none());
                    let expected = model(i);
                    assert_eq!(
                        producer.bundle.as_ref().map(|m| (&m.id, &m.revision, &m.sha256)),
                        expected.as_ref().map(|m| (&m.id, &m.revision, &m.sha256))
                    );
                    if let Some(previous) = refs.insert(i / 2, producer_ref.clone()) {
                        assert_eq!(previous, *producer_ref, "identical producer identities should deduplicate");
                    }
                    producers += 1;
                }
            }
            assert_eq!((producers, full_records, refs.len()), (6, 6, 3));
            assert_eq!(refs.values().collect::<std::collections::HashSet<_>>().len(), 3);
            assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
        }
    }
    server.stop().await;
}

#[tokio::test]
async fn category_evidence_preserves_only_recorded_producer_provenance() {
    check(false).await;
}

#[tokio::test]
async fn entity_votes_preserve_only_recorded_producer_provenance() {
    check(true).await;
}
