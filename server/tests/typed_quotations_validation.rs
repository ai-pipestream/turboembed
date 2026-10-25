//! Native quotation domains and nested offsets use structured refusals.
mod common;
#[path = "typed_quotations/fixture.rs"]
#[allow(dead_code)]
mod fixture;
use turbo_analysis::{Layer, export::export};
use turbo_analysis_layers::Quotations;
use turbo_kserve::{
    proto::result::{
        AnchorKind, InspectDocumentGraphRequest, OffsetMode, QuotationAnnotation, SentenceAnnotation, SpanFragment,
        TypedDocumentResult, typed_annotation::Value,
    },
    results::document::codecs,
};
fn quote(msg: &mut TypedDocumentResult) -> &mut QuotationAnnotation {
    let layer = msg.layers.iter_mut().find(|l| l.id == Quotations::ID && l.instance.is_empty()).unwrap();
    let Some(Value::Quotation(q)) = layer.annotations[0].value.as_mut() else { panic!("missing quotation") };
    q
}
fn interior(text: &str, mode: OffsetMode) -> SpanFragment {
    let at = match mode {
        OffsetMode::Utf8 => 1,
        OffsetMode::Utf16 => text[..text.find('😀').unwrap()].encode_utf16().count() as u32 + 1,
        OffsetMode::CodePoint => text.chars().count() as u32 + 1,
    };
    SpanFragment { start: at, end: at }
}
fn refused(e: tonic::Status) {
    assert_eq!(e.code(), tonic::Code::InvalidArgument);
    assert_eq!(e.metadata().get("turbo-code").unwrap(), "256");
    assert_eq!(e.metadata().get("turbo-field").unwrap(), "0");
}
#[tokio::test]
async fn malformed_quotation_payloads_and_nested_offsets_are_refused() {
    let graph = export(&fixture::document(), codecs()).unwrap().to_bytes().unwrap();
    let (server, mut c) = common::serve_no_admin().await;
    let mut accepted = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = c
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "bad-quotes".into(),
                graph: graph.clone(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(c.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
        for variant in 0..24 {
            let mut bad = out.clone();
            let inside = interior(&bad.raw_text, mode);
            if variant < 6 {
                quote(&mut bad).confidence =
                    [None, Some(f32::NAN), Some(f32::INFINITY), Some(f32::NEG_INFINITY), Some(-0.1), Some(1.1)]
                        [variant];
            } else if variant < 11 {
                quote(&mut bad).kind = [None, Some(-1), Some(3), Some(255), Some(i32::MAX)][variant - 6];
            } else if variant >= 17 && variant < 23 {
                let span = match (variant - 17) % 3 {
                    0 => SpanFragment { start: 2, end: 1 },
                    1 => SpanFragment { start: 0, end: u32::MAX },
                    _ => inside,
                };
                if variant < 20 {
                    quote(&mut bad).speaker = Some(span);
                } else {
                    quote(&mut bad).cue_verb = Some(span);
                }
            } else {
                let layer = bad.layers.iter_mut().find(|l| l.id == Quotations::ID && l.instance.is_empty()).unwrap();
                match variant {
                    11 => layer.annotations[0].spans.clear(),
                    12 => layer.id = "test:not-quotations".into(),
                    13 => layer.scope = AnchorKind::Document as i32,
                    14 => layer.annotations[0].value = Some(Value::Sentence(SentenceAnnotation::default())),
                    15 => layer.annotations[0].value = None,
                    16 => layer.annotations[0].spans = vec![SpanFragment { start: 0, end: 0 }; 2],
                    23 => layer.annotations[0].spans = vec![inside],
                    _ => unreachable!(),
                }
            }
            match c.round_trip_typed_document(bad).await {
                Ok(_) => accepted.push((mode, variant)),
                Err(e) => refused(e),
            }
        }
    }
    server.stop().await;
    assert!(accepted.is_empty(), "accepted malformed quotations: {accepted:?}");
}
#[tokio::test]
async fn confidence_bits_optional_presence_and_raw_chain_context_round_trip() {
    let graph = export(&fixture::document(), codecs()).unwrap().to_bytes().unwrap();
    let (server, mut c) = common::serve_no_admin().await;
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = c
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "quote-presence".into(),
                graph: graph.clone(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        for confidence in [0.0, -0.0, 1.0, f32::from_bits(1), f32::MIN_POSITIVE] {
            for chain in [None, Some(0), Some(u32::MAX)] {
                for present in [false, true] {
                    let mut valid = out.clone();
                    let q = quote(&mut valid);
                    q.confidence = Some(confidence);
                    q.speaker_chain = chain;
                    q.speaker = present.then_some(SpanFragment { start: 0, end: 0 });
                    q.cue_verb = q.speaker;
                    let mut echoed = c.round_trip_typed_document(valid.clone()).await.unwrap().into_inner();
                    assert_eq!(echoed, valid);
                    let q = quote(&mut echoed);
                    assert_eq!(q.confidence.unwrap().to_bits(), confidence.to_bits());
                    assert_eq!(q.speaker_chain, chain);
                    assert_eq!(q.speaker.is_some(), present);
                    assert_eq!(q.cue_verb.is_some(), present);
                }
            }
        }
    }
    server.stop().await;
}
