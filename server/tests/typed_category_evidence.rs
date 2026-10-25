mod common;
#[path = "typed_document/fixture.rs"]
mod fixture;

use std::sync::Arc;
use turbo_analysis::export::{LayerRef, export};
use turbo_analysis::{
    Document, Evidence, EvidenceStep, Evidenced, Key, Layer, Provenance, Span, Support, Tag, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{Categories, CategoryScope, Chunks, Sentences, Structure, Tokens};
use turbo_kserve::proto::result::{
    AnchorKind, DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, SpanFragment, TypedDocumentResult,
    typed_annotation::Value,
};
use turbo_kserve::results::document::codecs;

fn add<S: CategoryScope>(doc: &mut Document) -> String
where
    Categories<S>: Evidenced,
{
    let key = Key::<Categories<S>>::named("evidence-selected");
    let mut categories = Categories::<S>::new(fixture::tagset("inventory", &["a", "b"]));
    for _ in 0..doc.get::<S>().unwrap().len() {
        categories.push(Tag(0), &[0.8, 0.2]);
    }
    doc.add(
        &key,
        categories,
        Provenance { reads: vec![Key::<S>::DEFAULT.id()], ..Provenance::by("inventory:category") },
    )
    .unwrap();
    let mut evidence = EvidenceStep::new(key).build(doc).unwrap().clone();
    assert!(evidence.records() > 0);
    evidence.raw[0].1 = -2.5;
    evidence.calibrated = vec![(0, 0.75)];
    evidence.abstention = vec![(0, "policy:review".into())];
    evidence.layers = Arc::new(vec![LayerRef { layer: Tokens::ID.into(), instance: "support".into() }]);
    evidence.support_row.clear();
    evidence.support = vec![Support { layer: 0, start: 0, end: 1 }; evidence.records()];
    evidence.verify(doc).unwrap();
    let evidence_key = Key::<Evidence<Categories<S>>>::named("evidence-selected");
    doc.add(&evidence_key, evidence, Provenance { on: vec![key.id()], ..Provenance::by("inventory:evidence") })
        .unwrap();
    evidence_key.id().layer.to_string()
}

#[tokio::test]
async fn category_evidence_keeps_payloads_producers_support_and_projected_spans() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = fixture::build_rich_document();
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.keep(3);
    builder.replace(5, "e");
    builder.keep(source.text().len());
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("ascii"), View::derived(&source, text, alignment), Provenance::by("inventory:view"))
        .unwrap();
    doc.add(
        &Key::<Tokens>::named("support"),
        Tokens { spans: fixture::spans(&[Span::new(0, 5)]), sentence: None },
        Provenance::by("inventory:tokens"),
    )
    .unwrap();
    let ids = [add::<View>(&mut doc), add::<Sentences>(&mut doc), add::<Structure>(&mut doc), add::<Chunks>(&mut doc)];
    let bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let bytes = client
        .round_trip_document_graph(DocumentGraphRequest { graph: bytes.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    let source_spans = [
        doc.get_key(&Key::<Evidence<Categories<View>>>::named("evidence-selected")).unwrap().spans.clone(),
        doc.get_key(&Key::<Evidence<Categories<Sentences>>>::named("evidence-selected")).unwrap().spans.clone(),
        doc.get_key(&Key::<Evidence<Categories<Structure>>>::named("evidence-selected")).unwrap().spans.clone(),
        doc.get_key(&Key::<Evidence<Categories<Chunks>>>::named("evidence-selected")).unwrap().spans.clone(),
    ];
    let mut malformed_accepted = 0;
    let mut malformed_cases = 0;
    for view in [0, 1, 2] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = client
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "inventory".into(),
                    graph: bytes.to_vec(),
                    view,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
            for (scope, id) in ids.iter().enumerate() {
                let layer_ix = out
                    .layers
                    .iter()
                    .position(|l| l.id == *id && l.instance == "evidence-selected")
                    .expect("nonempty evidence layer disappeared");
                let layer = &out.layers[layer_ix];
                assert_eq!(layer.scope, AnchorKind::Aligned as i32);
                assert!(layer.anchor_layer.as_ref().unwrap().ends_with("@evidence-selected"));
                assert!(!layer.annotations.is_empty());
                let flattened: Vec<_> = layer.annotations.iter().flat_map(|a| a.spans.iter().cloned()).collect();
                let expected: Vec<_> = source_spans[scope].iter().map(|&s| wire_span(&doc, view, mode, s)).collect();
                assert_eq!(flattened, expected);
                for (i, ann) in layer.annotations.iter().enumerate() {
                    assert_eq!(ann.probability, None);
                    let producer = out.models.iter().find(|m| Some(&m.model_ref) == ann.model_ref.as_ref()).unwrap();
                    assert_eq!(producer.annotator, "inventory:category");
                    let Some(Value::Evidence(e)) = &ann.value else { panic!("missing evidence payload") };
                    assert_eq!(e.target_index, i as u32);
                    assert_eq!(e.raw_score, Some(if i == 0 { -2.5 } else { 0.8 }));
                    assert_eq!(e.calibrated_probability, (i == 0).then_some(0.75));
                    assert_eq!(e.abstention.as_deref(), (i == 0).then_some("policy:review"));
                    assert_eq!(e.support.len(), 1);
                    let s = &e.support[0];
                    assert_eq!((s.layer.as_str(), s.instance.as_str(), s.start, s.end), (Tokens::ID, "support", 0, 1));
                }
                for msg in malformed(&out, layer_ix) {
                    malformed_cases += 1;
                    match client.round_trip_typed_document(msg).await {
                        Ok(_) => malformed_accepted += 1,
                        Err(error) => {
                            assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error}");
                            assert_eq!(error.metadata().get("turbo-code").unwrap(), "256");
                            assert_eq!(error.metadata().get("turbo-field").unwrap(), "0");
                        }
                    }
                }
            }
        }
    }
    server.stop().await;
    eprintln!("category evidence controls: {malformed_cases} malformed, {malformed_accepted} accepted");
    assert_eq!(malformed_accepted, 0);
}

fn wire_span(doc: &Document, view: u32, mode: OffsetMode, span: Span) -> SpanFragment {
    let view = ViewId(u16::try_from(view).unwrap());
    let mapped = doc.from_source(view, span);
    let text = doc.text(view);
    let offset = |n: u32| match mode {
        OffsetMode::Utf8 => n,
        OffsetMode::Utf16 => text[..n as usize].encode_utf16().count() as u32,
        OffsetMode::CodePoint => text[..n as usize].chars().count() as u32,
    };
    SpanFragment { start: offset(mapped.start), end: offset(mapped.end) }
}

fn malformed(good: &TypedDocumentResult, layer: usize) -> Vec<TypedDocumentResult> {
    let mut cases = Vec::new();
    for index in 0..16 {
        let mut msg = good.clone();
        let ann = &mut msg.layers[layer].annotations[0];
        let Some(Value::Evidence(e)) = &mut ann.value else { unreachable!() };
        match index {
            0..=2 => e.raw_score = Some([f64::NAN, f64::INFINITY, f64::NEG_INFINITY][index]),
            3..=7 => {
                e.calibrated_probability = Some([-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY][index - 3])
            }
            8 => e.target_index = u32::MAX,
            9 => e.support[0].instance = "absent".into(),
            10 => e.support[0].layer = "absent".into(),
            11 => e.support[0].end = 2,
            12 => e.support[0].start = 2,
            13 => ann.model_ref = None,
            14 => ann.model_ref = Some("absent".into()),
            15 => msg.layers[layer].scope = AnchorKind::Document as i32,
            _ => unreachable!(),
        }
        cases.push(msg);
    }
    cases
}
