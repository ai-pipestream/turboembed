mod common;

use turbo_analysis::export::export;
use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{
    AnchoredLayerKind, ArgumentRole, Entities, EventArgument, EventFrame, Events, Sentences, Tokens,
};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::proto::result::{
    DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, typed_annotation::Value,
};
use turbo_kserve::results::document::codecs;

fn spans<L>(end: u32) -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, end));
    spans
}

#[tokio::test]
async fn nested_event_references_and_spans_are_validated() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("é😀");
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: spans(6) }, Provenance::by("event:sentences")).unwrap();
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: spans(2), sentence: None }, Provenance::by("event:tokens"))
        .unwrap();
    doc.add(
        &Key::<Entities>::DEFAULT,
        Entities {
            spans: spans(2),
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("ner", &["THING"]),
            score: vec![0.8],
        },
        Provenance::by("event:entities"),
    )
    .unwrap();
    let mut events = Events::new(ViewId::SOURCE);
    events.push(EventFrame {
        trigger: Span::new(0, 2),
        trigger_lemma: Some("é"),
        sentence_ix: 0,
        confidence: 0.8,
        arguments: &[EventArgument {
            span: Span::new(0, 2),
            role: ArgumentRole::Agent,
            anchored_layer: AnchoredLayerKind::Names,
            anchored_index: Some(0),
        }],
    });
    doc.add(&Key::<Events>::DEFAULT, events, Provenance::by("event:events")).unwrap();
    let graph = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let graph = client
        .round_trip_document_graph(DocumentGraphRequest { graph: graph.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    let mut accepted = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let good = client
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "event".into(),
                graph: graph.to_vec(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(client.round_trip_typed_document(good.clone()).await.unwrap().into_inner(), good);
        for field in ["sentence", "argument-index", "argument-layer", "argument-span"] {
            let mut bad = good.clone();
            let value = bad
                .layers
                .iter_mut()
                .flat_map(|l| &mut l.annotations)
                .find_map(|a| match &mut a.value {
                    Some(Value::Event(event)) => Some(event),
                    _ => None,
                })
                .unwrap();
            match field {
                "sentence" => value.sentence_index = 99,
                "argument-index" => value.arguments[0].anchored_index = Some(99),
                "argument-layer" => value.arguments[0].anchored_layer = "none".into(),
                _ => value.arguments[0].span = None,
            }
            match client.round_trip_typed_document(bad).await {
                Ok(_) => accepted.push(format!("{mode:?}:{field}")),
                Err(error) => {
                    assert_eq!(error.code(), tonic::Code::InvalidArgument);
                    assert_eq!(
                        error.metadata().get("turbo-code").unwrap().to_str().unwrap(),
                        TURBO_E_INVALID_ARGUMENT.to_string()
                    );
                    assert_eq!(error.metadata().get("turbo-field").unwrap(), "0");
                }
            }
        }
    }
    assert!(accepted.is_empty(), "malformed nested event references accepted: {accepted:?}");
    server.stop().await;
}
