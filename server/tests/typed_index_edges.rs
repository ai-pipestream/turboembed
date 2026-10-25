mod common;
#[path = "typed_document/fixture.rs"]
mod fixture;

use turbo_analysis::export::export;
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, OffsetMode, typed_annotation::Value};
use turbo_kserve::results::document::codecs;

#[tokio::test]
async fn optional_token_ranges_require_both_endpoints() {
    let (server, mut client) = common::serve_no_admin().await;
    let graph = export(&fixture::build_rich_document(), codecs()).unwrap().to_bytes().unwrap();
    let mut accepted = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = client
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "indices".into(),
                graph: graph.clone(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
        for variant in 0..6 {
            let mut bad = out.clone();
            let annotation = bad
                .layers
                .iter_mut()
                .flat_map(|l| &mut l.annotations)
                .find_map(|a| match a.value.as_mut()? {
                    Value::Aspect(a) if variant < 2 => Some((&mut a.token_start, &mut a.token_end)),
                    Value::Pragmatics(p) if variant >= 2 => {
                        if variant < 4 {
                            Some((&mut p.positive_token_start, &mut p.positive_token_end))
                        } else {
                            Some((&mut p.negative_token_start, &mut p.negative_token_end))
                        }
                    }
                    _ => None,
                })
                .unwrap();
            assert!(annotation.0.is_some() && annotation.1.is_some());
            if variant % 2 == 0 {
                *annotation.0 = None;
            } else {
                *annotation.1 = None;
            }
            match client.round_trip_typed_document(bad).await {
                Ok(_) => accepted.push((mode, variant)),
                Err(e) => assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e}"),
            }
        }
        // Optional pairs may both be absent: implicit aspect and no evidence.
        let mut absent = out;
        for a in absent.layers.iter_mut().flat_map(|l| &mut l.annotations) {
            match a.value.as_mut() {
                Some(Value::Aspect(a)) => {
                    a.token_start = None;
                    a.token_end = None;
                }
                Some(Value::Pragmatics(p)) => {
                    p.positive_token_start = None;
                    p.positive_token_end = None;
                    p.negative_token_start = None;
                    p.negative_token_end = None;
                }
                _ => {}
            }
        }
        assert_eq!(client.round_trip_typed_document(absent.clone()).await.unwrap().into_inner(), absent);
    }
    server.stop().await;
    assert!(accepted.is_empty(), "accepted incomplete token ranges: {accepted:?}");
}

#[tokio::test]
async fn group_and_edge_indices_require_a_target_bound() {
    let (server, mut client) = common::serve_no_admin().await;
    let graph = export(&fixture::build_rich_document(), codecs()).unwrap().to_bytes().unwrap();
    let mut accepted = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = client
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "indices".into(),
                graph: graph.clone(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
        for variant in 0..3 {
            for excessive in [false, true] {
                let mut bad = out.clone();
                let layer = bad
                    .layers
                    .iter_mut()
                    .find(|l| {
                        l.annotations.iter().any(|a| {
                            matches!((&a.value, variant), (Some(Value::Group(_)), 0) | (Some(Value::Edge(_)), 1..=2))
                        })
                    })
                    .unwrap();
                assert!(!layer.indexes_layer.is_empty());
                layer.indexes_layer.clear();
                if excessive {
                    match layer.annotations[0].value.as_mut().unwrap() {
                        Value::Group(g) => g.members.push(u32::MAX),
                        Value::Edge(e) if variant == 1 => e.from_index = u32::MAX,
                        Value::Edge(e) => e.to_index = u32::MAX,
                        _ => unreachable!(),
                    }
                }
                match client.round_trip_typed_document(bad).await {
                    Ok(_) => accepted.push((mode, variant, excessive)),
                    Err(e) => assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e}"),
                }
            }
        }
    }
    server.stop().await;
    assert!(accepted.is_empty(), "accepted indices without bounds: {accepted:?}");
}

#[tokio::test]
async fn sentence_and_opinion_references_follow_the_declared_bounds() {
    let (server, mut client) = common::serve_no_admin().await;
    let graph = export(&fixture::build_rich_document(), codecs()).unwrap().to_bytes().unwrap();
    let mut accepted = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = client
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "indices".into(),
                graph: graph.clone(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
        for variant in 0..4 {
            let mut bad = out.clone();
            let value = bad
                .layers
                .iter_mut()
                .flat_map(|l| &mut l.annotations)
                .find_map(|a| match a.value.as_mut()? {
                    Value::Token(t) if variant == 0 => Some(t.sentence_index.as_mut().unwrap()),
                    Value::Chunk(c) if variant == 1 => Some(&mut c.sentence_end),
                    Value::Aspect(a) if variant >= 2 => {
                        if variant == 2 {
                            Some(a.sentence_index.as_mut().unwrap())
                        } else {
                            Some(&mut a.opinion_indices[0])
                        }
                    }
                    _ => None,
                })
                .unwrap();
            *value = u32::MAX;
            match client.round_trip_typed_document(bad).await {
                Ok(_) => accepted.push((mode, variant)),
                Err(e) => assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e}"),
            }
        }
    }
    server.stop().await;
    assert!(accepted.is_empty(), "accepted out-of-bounds references: {accepted:?}");
}
