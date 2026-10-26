mod common;
#[path = "typed_document/fixture.rs"]
mod fixture;

use turbo_analysis::{Groups, Ix, IxRange, Provenance, Span, Tag, ViewId, export::export};
use turbo_coref::{MentionKind, Mentions};
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, OffsetMode, typed_annotation::Value};
use turbo_kserve::results::document::codecs;
use turbo_relation::Relations;

#[tokio::test]
async fn relations_mentions_and_levels_keep_native_reference_bounds() {
    let mut doc = fixture::build_rich_document();
    fixture::add(
        &mut doc,
        Relations {
            spans: fixture::spans(&[Span::new(0, fixture::TEXT.len() as u32)]),
            subject: vec![Ix::new(0)],
            object: vec![Ix::new(1)],
            kind: vec![Tag(0)],
            kinds: fixture::tagset("relation", &["related"]),
            score: vec![0.8],
        },
        Provenance::by("test:relations"),
    );
    let mut mentions = Mentions::new(ViewId::SOURCE);
    mentions.push(fixture::byte_span("北京"), Ix::new(0), IxRange::new(1, 2), Ix::new(1), MentionKind::Proper);
    mentions.entity = Some(vec![Some(Ix::new(0))]);
    fixture::add(&mut doc, mentions, Provenance::by("test:mentions"));
    fixture::add(&mut doc, Groups::<Mentions>::new([[Ix::new(0)]]), Provenance::by("test:chains"));
    let graph = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let (server, mut client) = common::serve_no_admin().await;
    let mut accepted = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = client
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "references".into(),
                graph: graph.clone(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
        for variant in 0..11 {
            let mut bad = out.clone();
            let mut changed = false;
            for layer in &mut bad.layers {
                for annotation in &mut layer.annotations {
                    match annotation.value.as_mut().unwrap() {
                        Value::Relation(r) if variant < 3 => match variant {
                            0 => r.subject_entity = u32::MAX,
                            1 => r.object_entity = u32::MAX,
                            _ => r.object_entity = r.subject_entity,
                        },
                        Value::CorefMention(m) if (3..5).contains(&variant) => {
                            if variant == 3 {
                                m.sentence_index = u32::MAX;
                            } else {
                                m.entity_index = Some(u32::MAX);
                            }
                        }
                        Value::LevelUnit(l) if variant == 5 && l.kind == "chunk" => l.unit_index = u32::MAX,
                        Value::LevelUnit(l) if variant == 6 && l.kind == "document" => l.unit_index = 1,
                        Value::LevelUnit(l) if variant == 7 => l.kind = "unknown".into(),
                        Value::LevelUnit(l) if variant == 8 && l.kind == "chunk" => {
                            l.kind = "section".into();
                            l.unit_index = u32::MAX;
                        }
                        Value::CorefChain(c) if variant == 9 => c.mention_indices[0] = u32::MAX,
                        Value::CorefChain(_) if variant == 10 => layer.indexes_layer.clear(),
                        _ => continue,
                    }
                    changed = true;
                    break;
                }
                if changed {
                    break;
                }
            }
            assert!(changed, "variant {variant} did not find its native annotation");
            match client.round_trip_typed_document(bad).await {
                Ok(_) => accepted.push((mode, variant)),
                Err(e) => assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e}"),
            }
        }
        let mut optional = out;
        for a in optional.layers.iter_mut().flat_map(|l| &mut l.annotations) {
            if let Some(Value::CorefMention(m)) = &mut a.value {
                m.entity_index = None;
            }
        }
        assert_eq!(client.round_trip_typed_document(optional.clone()).await.unwrap().into_inner(), optional);
    }
    server.stop().await;
    assert!(accepted.is_empty(), "accepted malformed references: {accepted:?}");
}
