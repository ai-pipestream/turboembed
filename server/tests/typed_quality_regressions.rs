//! Nonempty standard quality layers crossing the real graph and typed RPCs.

mod common;

use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Layer, Provenance, Span, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{QualityFlags, QualityMetrics, SpanQuality, TextQuality};
use turbo_kserve::proto::result::{
    AnchorKind, DocumentGraphRequest, InspectDocumentGraphRequest, LayerWireStatus, OffsetMode, SpanFragment,
    TypedDocumentResult,
};
use turbo_kserve::results::document::codecs;

fn metrics(derived: bool) -> QualityMetrics {
    QualityMetrics {
        composite_score: if derived { 0.75 } else { 0.25 },
        mtld_lexical_diversity: if derived { 23.5 } else { 8.5 },
        flesch_kincaid_grade: -1.45,
        automated_readability_index: -0.0,
        coleman_liau_index: -4.25,
        information_density: 0.4,
        alpha_ratio: 0.7,
        digit_ratio: 0.1,
        symbol_ratio: 0.1,
        whitespace_ratio: 0.1,
        replacement_ratio: 0.01,
        flags: if derived { QualityFlags::ALL } else { QualityFlags::OCR_BROKEN_HYPHENS | QualityFlags::TOO_SHORT },
    }
}

fn fixture() -> Document {
    let mut doc = Document::new("xxé😀 A\nB");
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.replace(2, "");
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("trim"), View::derived(&source, text, alignment), Provenance::by("quality:view"))
        .unwrap();
    for (view, name, pairs) in [(ViewId::SOURCE, "", [(2, 8), (9, 10)]), (ViewId(1), "selected", [(0, 6), (7, 8)])] {
        doc.add(
            &Key::<TextQuality>::named(name),
            TextQuality::new(view, doc.text(view).len() as u32, metrics(view.0 != 0)),
            Provenance::by("quality:summary"),
        )
        .unwrap();
        let mut layer = SpanQuality::new(view);
        for (i, (start, end)) in pairs.into_iter().enumerate() {
            layer.push(Span::new(start, end), metrics((view.0 as usize + i) % 2 != 0));
        }
        doc.add(&Key::<SpanQuality>::named(name), layer, Provenance::by("quality:regions")).unwrap();
    }
    doc
}

async fn project(client: &mut common::Analysis, graph: &[u8], mode: OffsetMode, view: u32) -> TypedDocumentResult {
    client
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "quality".into(),
            graph: graph.to_vec(),
            view,
            offsets: mode as i32,
        })
        .await
        .unwrap()
        .into_inner()
}

fn wire_span(text: &str, mode: OffsetMode, start: usize, end: usize) -> SpanFragment {
    let offset = |n| match mode {
        OffsetMode::Utf8 => n as u32,
        OffsetMode::Utf16 => text[..n].encode_utf16().count() as u32,
        OffsetMode::CodePoint => text[..n].chars().count() as u32,
    };
    SpanFragment { start: offset(start), end: offset(end) }
}

async fn quality_round_trip(layer_id: &str) {
    let (server, mut client) = common::serve_no_admin().await;
    let bytes = export(&fixture(), codecs()).unwrap().to_bytes().unwrap();
    let graph = client
        .round_trip_document_graph(DocumentGraphRequest { graph: bytes.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    for view in [0, 1] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = project(&mut client, &graph, mode, view).await;
            let layers: Vec<_> = out.layers.iter().filter(|l| l.id == layer_id).collect();
            assert_eq!(layers.len(), 2);
            for layer in layers {
                assert_eq!(layer.status, LayerWireStatus::Ok as i32);
                let count = if layer_id == TextQuality::ID { 1 } else { 2 };
                assert_eq!(
                    layer.annotations.len(),
                    count,
                    "{layer_id}@{} became an empty successful layer",
                    layer.instance
                );
                assert_eq!(layer.scope, AnchorKind::Text as i32);
                assert_eq!(layer.model_refs.len(), 1);
                let pairs = if layer_id == TextQuality::ID {
                    vec![(if view == 0 && !layer.instance.is_empty() { 2 } else { 0 }, out.raw_text.len())]
                } else if view == 0 {
                    vec![(2, 8), (9, 10)]
                } else {
                    vec![(0, 6), (7, 8)]
                };
                for (ann, (start, end)) in layer.annotations.iter().zip(pairs) {
                    assert_eq!(ann.spans, [wire_span(&out.raw_text, mode, start, end)]);
                    assert!(ann.value.is_some(), "quality metrics are missing");
                    assert_eq!(ann.probability, None, "quality heuristics are not probabilities");
                    assert_eq!(ann.model_ref.as_ref(), layer.model_refs.first());
                }
            }
            assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
        }
    }
    server.stop().await;
}

#[tokio::test]
async fn text_quality_metrics_survive_named_views_and_every_offset_mode() {
    quality_round_trip(TextQuality::ID).await;
}

#[tokio::test]
async fn span_quality_metrics_survive_named_views_and_every_offset_mode() {
    quality_round_trip(SpanQuality::ID).await;
}
