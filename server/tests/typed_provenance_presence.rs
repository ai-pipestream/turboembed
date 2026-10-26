//! Optional provenance presence must survive model-table deduplication.
mod common;

use turbo_analysis::{
    Document, Evidence, EvidenceStep, Execution, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, View, ViewId,
    export::export,
};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_kserve::{
    proto::result::{
        DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, TypedDocumentResult, typed_annotation::Value,
    },
    results::document::codecs,
};

const TEXT: &str = "é😀";

fn empty_model() -> ModelRef {
    ModelRef { id: String::new(), revision: String::new(), sha256: String::new() }
}

async fn project(c: &mut common::Analysis, doc: &Document, mode: OffsetMode) -> TypedDocumentResult {
    let bytes = export(doc, codecs()).unwrap().to_bytes().unwrap();
    let graph = c
        .round_trip_document_graph(DocumentGraphRequest { graph: bytes.clone().into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    assert_eq!(&graph[..], &bytes[..], "native optional-record presence changed");
    let msg = c
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "presence".into(),
            graph: graph.to_vec(),
            view: 0,
            offsets: mode as i32,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(c.round_trip_typed_document(msg.clone()).await.unwrap().into_inner(), msg);
    msg
}

#[tokio::test]
async fn optional_model_and_execution_records_retain_presence() {
    let (server, mut client) = common::serve_no_admin().await;
    let cases = [("none", false, false), ("bundle", true, false), ("execution", false, true), ("both", true, true)];
    for reverse in [false, true] {
        let mut doc = Document::new(TEXT);
        for pos in 0..4 {
            let (name, bundle, execution) = cases[if reverse { 3 - pos } else { pos }];
            let mut spans = Spans::new(ViewId::SOURCE);
            spans.push(Span::new(0, TEXT.len() as u32));
            doc.add(
                &Key::<Tokens>::named(name),
                Tokens { spans, sentence: None },
                Provenance {
                    config: 123,
                    model: bundle.then(empty_model),
                    execution: execution.then(|| Execution {
                        backend: String::new(),
                        device: String::new(),
                        precision: String::new(),
                    }),
                    ..Provenance::by("shared")
                },
            )
            .unwrap();
        }
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let msg = project(&mut client, &doc, mode).await;
            let mut refs = std::collections::HashSet::new();
            for (name, bundle, execution) in cases {
                let layer = msg.layers.iter().find(|l| l.id == "opennlp:tokens" && l.instance == name).unwrap();
                let p = msg.models.iter().find(|p| p.model_ref == layer.model_refs[0]).unwrap();
                assert_eq!(
                    (p.bundle.is_some(), p.execution.is_some()),
                    (bundle, execution),
                    "{name}/{reverse}/{mode:?}"
                );
                assert_eq!(p.config_hash, 123);
                if let Some(m) = &p.bundle {
                    assert!(m.id.is_empty() && m.revision.is_empty() && m.sha256.is_empty());
                }
                if let Some(e) = &p.execution {
                    assert!(e.backend.is_empty() && e.device.is_empty() && e.precision.is_empty());
                }
                refs.insert(&p.model_ref);
            }
            assert_eq!(refs.len(), 4);
        }
    }
    server.stop().await;
}

#[tokio::test]
async fn producer_model_absence_differs_from_present_empty_identity() {
    let (server, mut client) = common::serve_no_admin().await;
    for reverse in [false, true] {
        let mut doc = Document::new(TEXT);
        for pos in 0..2 {
            let present = (if reverse { 1 - pos } else { pos }) == 1;
            let name = if present { "present" } else { "absent" };
            let key = Key::<Categories<View>>::named(name);
            let mut cats = Categories::<View>::new(TagSet::new("category", &["a", "b"]));
            cats.push(Tag(0), &[0.8, 0.2]);
            doc.add(
                &key,
                cats,
                Provenance {
                    config: 123,
                    model: present.then(empty_model),
                    on: vec![Key::<View>::DEFAULT.id()],
                    ..Provenance::by("shared")
                },
            )
            .unwrap();
            let evidence = EvidenceStep::new(key).build(&doc).unwrap().clone();
            doc.add(
                &Key::<Evidence<Categories<View>>>::named(name),
                evidence,
                Provenance { on: vec![key.id()], ..Provenance::by("evidence") },
            )
            .unwrap();
        }
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let msg = project(&mut client, &doc, mode).await;
            let mut refs = std::collections::HashSet::new();
            for layer in &msg.layers {
                for a in &layer.annotations {
                    if !matches!(a.value, Some(Value::Evidence(_))) {
                        continue;
                    }
                    let p = msg.models.iter().find(|p| Some(&p.model_ref) == a.model_ref.as_ref()).unwrap();
                    assert_eq!(p.bundle.is_some(), layer.instance == "present", "{reverse}/{mode:?}");
                    assert_eq!(p.config_hash, 0);
                    assert_eq!(p.version, "0");
                    assert!(p.execution.is_none());
                    refs.insert(&p.model_ref);
                }
            }
            assert_eq!(refs.len(), 2);
        }
    }
    server.stop().await;
}
