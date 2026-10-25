//! Registered quotation records must survive real binary and typed RPCs.
mod common;
#[path = "typed_quotations/fixture.rs"]
mod fixture;
use turbo_analysis::{
    Key, Layer, ViewId,
    export::{DocumentGraph, export, import},
};
use turbo_analysis_layers::Quotations;
use turbo_kserve::{
    proto::result::{AnchorKind, DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode},
    results::document::codecs,
};

#[tokio::test]
async fn registered_quotations_do_not_disappear_in_typed_projection() {
    let doc = fixture::document();
    let bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let (server, mut client) = common::serve_no_admin().await;
    let graph = client
        .round_trip_document_graph(DocumentGraphRequest { graph: bytes.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    let transported = import(&DocumentGraph::from_bytes(&graph).unwrap(), codecs()).unwrap();
    for name in ["", "gold"] {
        let key = Key::<Quotations>::named(name);
        let a = doc.get_key(&key).unwrap();
        let b = transported.get_key(&key).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            a.confidence.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.confidence.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    let mut missing = Vec::new();
    for view in [0, 1, 2] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = client
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "quotes".into(),
                    graph: graph.to_vec(),
                    view,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            assert_eq!(out.raw_text, doc.text(ViewId(view as u16)));
            let echo = client.round_trip_typed_document(out.clone()).await.unwrap().into_inner();
            assert_eq!(echo, out);
            for name in ["", "gold"] {
                let key = Key::<Quotations>::named(name);
                let native = doc.get_key(&key).unwrap();
                let layer = out.layers.iter().find(|l| l.id == Quotations::ID && l.instance == name).unwrap();
                if layer.annotations.len() != native.len() {
                    missing.push((view, mode, name, layer.annotations.len()));
                    continue;
                }
                assert_eq!(layer.scope, AnchorKind::Text as i32);
                assert!(layer.indexes_layer.is_empty(), "speaker_chain is raw context, not a checked reference");
                let provenance = doc.provenance(key.id()).unwrap();
                assert_eq!(layer.model_refs.len(), 1);
                let producer = out.models.iter().find(|m| m.model_ref == layer.model_refs[0]).unwrap();
                assert_eq!(producer.annotator, provenance.annotator);
                assert_eq!(producer.version, provenance.version);
                assert_eq!(producer.config_hash, provenance.config);
                for (i, a) in layer.annotations.iter().enumerate() {
                    let q = native.get(i).unwrap();
                    assert_eq!(
                        a.spans,
                        vec![fixture::projected(&doc, native.spans.view(), ViewId(view as u16), q.content, mode)]
                    );
                    assert_eq!(a.probability.unwrap().to_bits(), f64::from(q.confidence).to_bits());
                    assert_eq!(a.model_ref, Some(producer.model_ref.clone()));
                }
            }
        }
    }
    server.stop().await;
    assert!(missing.is_empty(), "registered nonempty quotations disappeared: {missing:?}");
}
