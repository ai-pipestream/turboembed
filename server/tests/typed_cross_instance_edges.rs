//! Edges between two instances of one layer type, as `Edges::between`
//! makes them, round-trip through the typed projection: a layer that names
//! two instances of a type is refused only where it reads that type's
//! length.
mod common;
#[path = "typed_document/fixture.rs"]
mod fixture;

use turbo_analysis::{Edge, Edges, Ix, IxRange, Key, Provenance, Span, Tag, export::export};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, OffsetMode, typed_annotation::Value};
use turbo_kserve::results::document::codecs;

const EDGES: &str = "turbo:edges<opennlp:entities, opennlp:entities>";

/// The rich fixture plus a two-row `opennlp:entities@gold` and one edge
/// from the default entities to `to`.
fn graph(to: Key<Entities>) -> Vec<u8> {
    let mut doc = fixture::build_rich_document();
    let gold = Key::<Entities>::named("gold");
    let ents = Entities {
        spans: fixture::spans(&[fixture::byte_span("北京"), Span::new(0, fixture::byte_span("$42.50").end)]),
        tokens: vec![IxRange::new(1, 2), IxRange::new(0, 3)],
        kind: vec![Tag(0), Tag(1)],
        kinds: fixture::tagset("ner", &["LOC", "PRODUCT"]),
        score: vec![0.9, 0.5],
    };
    doc.add(&gold, ents, Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:gold-ner") })
        .unwrap();
    let edges = Edges::between(
        Key::<Entities>::DEFAULT,
        to,
        fixture::tagset("align", &["same"]),
        [Edge::new(Ix::new(0), Ix::new(1), Tag(0), 0.5)],
    );
    let prov =
        Provenance { reads: vec![Key::<Entities>::DEFAULT.id(), gold.id()], ..Provenance::by("test:entity-align") };
    doc.add(&Key::DEFAULT, edges, prov).unwrap();
    export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

#[tokio::test]
async fn cross_instance_entity_edges_round_trip_and_stay_bounded() {
    let (server, mut client) = common::serve_no_admin().await;
    for (to, cross) in [(Key::<Entities>::DEFAULT, false), (Key::<Entities>::named("gold"), true)] {
        let graph = graph(to);
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = client
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "edges".into(),
                    graph: graph.clone(),
                    view: 0,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            let edges = out.layers.iter().position(|l| l.id == EDGES).unwrap();
            let bounds: Vec<(&str, &str)> =
                out.layers[edges].indexes_layer.iter().map(|b| (b.layer.as_str(), b.instance.as_str())).collect();
            let want: &[(&str, &str)] = if cross {
                &[("opennlp:entities", ""), ("opennlp:entities", "gold")]
            } else {
                &[("opennlp:entities", ""), ("opennlp:entities", "")]
            };
            assert_eq!(bounds, want, "cross={cross} {mode:?}");
            let echoed = client.round_trip_typed_document(out.clone()).await;
            assert_eq!(echoed.map(|r| r.into_inner()).as_ref().map_err(|e| e.message()), Ok(&out), "cross={cross}");

            // The edge's to_index is still judged against its own target:
            // gold has two rows, so index 2 is out of range.
            let mut bad = out.clone();
            let Some(Value::Edge(edge)) = &mut bad.layers[edges].annotations[0].value else {
                panic!("{EDGES}[0] is not an edge");
            };
            edge.to_index = 2;
            let err = client.round_trip_typed_document(bad).await.unwrap_err();
            assert!(err.message().contains("edge to_index 2 exceeds target"), "cross={cross}: {}", err.message());
        }
    }
    server.stop().await;
}
