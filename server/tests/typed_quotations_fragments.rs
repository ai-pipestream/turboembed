//! Native quotations may have legal discontinuous content, including raw context.
mod common;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId, export::export};
use turbo_analysis_layers::{Quotation, Quotations, QuoteKind};
use turbo_kserve::{
    proto::result::{DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, typed_annotation::Value},
    results::document::codecs,
};
#[tokio::test]
async fn discontinuous_content_survives_partial_total_and_overlapping_projection() {
    let mut d = Document::new("é-x-😀");
    let source = d.get::<View>().unwrap().clone();
    for (name, partial, replacement) in [("partial", true, ""), ("overlap", false, "Q"), ("deleted", false, "")] {
        let mut b = ViewBuilder::new(source.text());
        if partial {
            b.replace(2, "");
            b.keep(9);
        } else {
            b.replace(9, replacement);
        }
        let (text, align) = b.finish();
        d.add(&Key::named(name), View::derived(&source, text, align), Provenance::by("test:view")).unwrap();
    }
    for name in ["", "gold"] {
        let mut quotes = Quotations::new(ViewId::SOURCE);
        quotes.push(Quotation {
            content: Span::new(0, 9),
            speaker: Some(Span::new(3, 4)),
            cue_verb: None,
            speaker_chain: Some(u32::MAX),
            kind: QuoteKind::Direct,
            confidence: -0.0,
        });
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push_fragments(&[Span::new(0, 2), Span::new(5, 9)]);
        quotes.spans = spans;
        // Native rows remain in source start order. Deleting the first
        // fragment moves row 0's visible start after row 1's visible start;
        // projection must preserve record order without imposing a new sort.
        quotes.push(Quotation {
            content: Span::new(3, 4),
            speaker: None,
            cue_verb: None,
            speaker_chain: None,
            kind: QuoteKind::Indirect,
            confidence: 0.75,
        });
        d.add(&Key::named(name), quotes, Provenance::by("test:discontinuous-quote")).unwrap();
    }
    let graph = export(&d, codecs()).unwrap().to_bytes().unwrap();
    let (server, mut c) = common::serve_no_admin().await;
    let graph =
        c.round_trip_document_graph(DocumentGraphRequest { graph: graph.into() }).await.unwrap().into_inner().graph;
    for view in [0, 1, 2, 3] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = c
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "quote-fragments".into(),
                    graph: graph.to_vec(),
                    view,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            let echoed = c.round_trip_typed_document(out.clone()).await.unwrap().into_inner();
            assert_eq!(echoed, out);
            let expected = match (view, mode) {
                (0, OffsetMode::Utf8) => vec![(0, 2), (5, 9)],
                (0, OffsetMode::Utf16) => vec![(0, 1), (4, 6)],
                (0, OffsetMode::CodePoint) => vec![(0, 1), (4, 5)],
                (1, OffsetMode::Utf8) => vec![(3, 7)],
                (1, OffsetMode::Utf16) => vec![(3, 5)],
                (1, OffsetMode::CodePoint) => vec![(3, 4)],
                (2, _) => vec![(0, 1)],
                (3, _) => vec![(0, 0)],
                _ => unreachable!(),
            };
            for name in ["", "gold"] {
                let layer = out.layers.iter().find(|l| l.id == "turbo:quotations" && l.instance == name).unwrap();
                assert_eq!(layer.annotations.len(), 2);
                assert!(layer.indexes_layer.is_empty());
                assert_eq!(layer.annotations[0].spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(), expected);
                let second = match (view, mode) {
                    (0, OffsetMode::Utf8) => (3, 4),
                    (0, _) => (2, 3),
                    (1, _) => (1, 2),
                    (2, _) => (0, 1),
                    (3, _) => (0, 0),
                    _ => unreachable!(),
                };
                assert_eq!(layer.annotations[1].spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(), [second]);
                if view == 1 {
                    assert!(layer.annotations[0].spans[0].start > layer.annotations[1].spans[0].start);
                }
                let Some(Value::Quotation(q)) = &layer.annotations[0].value else { panic!("missing quotation") };
                assert_eq!(q.speaker_chain, Some(u32::MAX));
                assert_eq!(q.confidence.unwrap().to_bits(), (-0.0_f32).to_bits());
                assert!(q.speaker.is_some());
                assert!(q.cue_verb.is_none());
                let layer = echoed.layers.iter().find(|l| l.id == "turbo:quotations" && l.instance == name).unwrap();
                let Some(Value::Quotation(q)) = &layer.annotations[0].value else { panic!("missing echo") };
                assert_eq!(q.confidence.unwrap().to_bits(), (-0.0_f32).to_bits());
            }
        }
    }
    server.stop().await;
}
