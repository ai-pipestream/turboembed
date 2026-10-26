mod common;

use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Provenance, Span, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{KeyValuePair, KeyValuePairs};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::proto::result::{
    DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, SpanFragment, typed_annotation::Value,
};
use turbo_kserve::results::document::codecs;

#[tokio::test]
async fn key_value_component_spans_obey_wire_boundaries_in_both_views() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("é:😀");
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.replace(2, "e");
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("ascii"), View::derived(&source, text, alignment), Provenance::by("pairs:view"))
        .unwrap();
    let mut pairs = KeyValuePairs::new(ViewId::SOURCE);
    pairs.spans.push(Span::new(0, 7));
    pairs.pairs.push(KeyValuePair { key: Span::new(0, 2), separator: Span::new(2, 3), value: Span::new(3, 7) });
    doc.add(&Key::<KeyValuePairs>::DEFAULT, pairs, Provenance::by("pairs:source")).unwrap();
    let graph = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let graph = client
        .round_trip_document_graph(DocumentGraphRequest { graph: graph.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    let mut accepted = Vec::new();
    for view in [0, 1] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let good = client
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "pairs".into(),
                    graph: graph.to_vec(),
                    view,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            assert_eq!(client.round_trip_typed_document(good.clone()).await.unwrap().into_inner(), good);
            let pair = good
                .layers
                .iter()
                .flat_map(|l| &l.annotations)
                .find_map(|a| match &a.value {
                    Some(Value::KeyValuePair(pair)) => Some(pair),
                    _ => None,
                })
                .unwrap();
            assert_eq!(pair.key_text, if view == 0 { "é" } else { "e" });
            assert_eq!(pair.value_text, "😀");
            for (field, missing) in
                ["key", "separator", "value"].into_iter().flat_map(|field| [(field, false), (field, true)])
            {
                let mut bad = good.clone();
                let pair = bad
                    .layers
                    .iter_mut()
                    .flat_map(|l| &mut l.annotations)
                    .find_map(|a| match &mut a.value {
                        Some(Value::KeyValuePair(pair)) => Some(pair),
                        _ => None,
                    })
                    .unwrap();
                let span = match field {
                    "key" => &mut pair.key_span,
                    "separator" => &mut pair.separator_span,
                    _ => &mut pair.value_span,
                };
                *span = if missing {
                    None
                } else {
                    Some(match mode {
                        OffsetMode::Utf8 if view == 0 => SpanFragment { start: 1, end: 2 },
                        _ => SpanFragment { start: 3, end: 4 },
                    })
                };
                match client.round_trip_typed_document(bad).await {
                    Ok(_) => accepted.push(format!("{view}:{mode:?}:{field}:{missing}")),
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
    }
    assert!(accepted.is_empty(), "malformed key/value component spans accepted: {accepted:?}");
    server.stop().await;
}
