//! Nonempty standard quality layers crossing the real graph and typed RPCs.

mod common;

use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Layer, Provenance, Span, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{QualityFlags, QualityMetrics, SpanQuality, TextQuality};
use turbo_kserve::proto::result::{
    AnchorKind, DocumentGraphRequest, InspectDocumentGraphRequest, LayerWireStatus, OffsetMode,
    QualityMetricsAnnotation, SpanFragment, TypedDocumentResult, typed_annotation::Value,
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
            layer.push(Span::new(start, end), metrics(!(view.0 as usize + i).is_multiple_of(2)));
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
                for (i, (ann, (start, end))) in layer.annotations.iter().zip(pairs).enumerate() {
                    assert_eq!(ann.spans, [wire_span(&out.raw_text, mode, start, end)]);
                    let actual = match &ann.value {
                        Some(Value::TextQuality(m)) if layer_id == TextQuality::ID => m,
                        Some(Value::SpanQuality(m)) if layer_id == SpanQuality::ID => m,
                        other => panic!("wrong quality payload: {other:?}"),
                    };
                    let expected = metrics(!(usize::from(!layer.instance.is_empty()) + i).is_multiple_of(2));
                    check_metrics(actual, expected);
                    assert_eq!(ann.probability, None, "quality heuristics are not probabilities");
                    assert_eq!(ann.model_ref.as_ref(), layer.model_refs.first());
                }
            }
            let echoed = client.round_trip_typed_document(out.clone()).await.unwrap().into_inner();
            assert_eq!(echoed, out);
            for layer in echoed.layers.iter().filter(|l| l.id == layer_id) {
                for ann in &layer.annotations {
                    let m = match ann.value.as_ref().unwrap() {
                        Value::TextQuality(m) | Value::SpanQuality(m) => m,
                        _ => unreachable!(),
                    };
                    assert_eq!(m.automated_readability_index.unwrap().to_bits(), (-0.0f32).to_bits());
                }
            }
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

fn fields(m: &QualityMetricsAnnotation) -> [Option<f32>; 11] {
    [
        m.composite_score,
        m.mtld_lexical_diversity,
        m.flesch_kincaid_grade,
        m.automated_readability_index,
        m.coleman_liau_index,
        m.information_density,
        m.alpha_ratio,
        m.digit_ratio,
        m.symbol_ratio,
        m.whitespace_ratio,
        m.replacement_ratio,
    ]
}

fn fields_mut(m: &mut QualityMetricsAnnotation) -> [&mut Option<f32>; 11] {
    [
        &mut m.composite_score,
        &mut m.mtld_lexical_diversity,
        &mut m.flesch_kincaid_grade,
        &mut m.automated_readability_index,
        &mut m.coleman_liau_index,
        &mut m.information_density,
        &mut m.alpha_ratio,
        &mut m.digit_ratio,
        &mut m.symbol_ratio,
        &mut m.whitespace_ratio,
        &mut m.replacement_ratio,
    ]
}

fn check_metrics(actual: &QualityMetricsAnnotation, expected: QualityMetrics) {
    let want = [
        expected.composite_score,
        expected.mtld_lexical_diversity,
        expected.flesch_kincaid_grade,
        expected.automated_readability_index,
        expected.coleman_liau_index,
        expected.information_density,
        expected.alpha_ratio,
        expected.digit_ratio,
        expected.symbol_ratio,
        expected.whitespace_ratio,
        expected.replacement_ratio,
    ];
    assert_eq!(fields(actual).map(|x| x.unwrap().to_bits()), want.map(f32::to_bits));
    assert_eq!(actual.flags, u32::from(expected.flags.0));
}

fn malformed_metrics(good: &QualityMetricsAnnotation) -> Vec<QualityMetricsAnnotation> {
    let mut bad = Vec::new();
    for field in 0..11 {
        let mut m = *good;
        *fields_mut(&mut m)[field] = None;
        bad.push(m);
        let values: &[f32] = match field {
            1 => &[-0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            2..=4 => &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            _ => &[-0.1, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        };
        for &value in values {
            let mut m = *good;
            *fields_mut(&mut m)[field] = Some(value);
            bad.push(m);
        }
    }
    for flags in [1 << 7, 1 << 16, u32::MAX] {
        bad.push(QualityMetricsAnnotation { flags, ..*good });
    }
    bad
}

fn invalid(error: tonic::Status) {
    assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error}");
    assert_eq!(error.metadata().get("turbo-code").unwrap(), "256");
    assert_eq!(error.metadata().get("turbo-field").unwrap(), "0");
}

#[tokio::test]
async fn typed_quality_refuses_malformed_metrics_payloads_and_spans() {
    let (server, mut client) = common::serve_no_admin().await;
    let graph = export(&fixture(), codecs()).unwrap().to_bytes().unwrap();
    let mut accepted = 0;
    let mut cases = 0;
    for view in [0, 1] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let good = project(&mut client, &graph, mode, view).await;
            assert_eq!(client.round_trip_typed_document(good.clone()).await.unwrap().into_inner(), good);
            for id in [TextQuality::ID, SpanQuality::ID] {
                let layer_ix = good.layers.iter().position(|l| l.id == id && l.instance == "selected").unwrap();
                let m = match good.layers[layer_ix].annotations[0].value.as_ref().unwrap() {
                    Value::TextQuality(m) | Value::SpanQuality(m) => m,
                    _ => unreachable!(),
                };
                let mut mutations = Vec::new();
                for bad in malformed_metrics(m) {
                    let mut msg = good.clone();
                    msg.layers[layer_ix].annotations[0].value =
                        Some(if id == TextQuality::ID { Value::TextQuality(bad) } else { Value::SpanQuality(bad) });
                    mutations.push(msg);
                }
                for shape in 0..10 {
                    let mut msg = good.clone();
                    let l = &mut msg.layers[layer_ix];
                    match shape {
                        0 => l.annotations[0].spans.clear(),
                        1 => l.scope = AnchorKind::Document as i32,
                        2 => {
                            l.annotations[0].value = Some(if id == TextQuality::ID {
                                Value::SpanQuality(*m)
                            } else {
                                Value::TextQuality(*m)
                            })
                        }
                        3 => l.annotations[0].spans[0].end = u32::MAX,
                        4 => l.annotations[0].spans[0] = SpanFragment { start: 2, end: 1 },
                        5 if id == TextQuality::ID => l.annotations.clear(),
                        6 if id == TextQuality::ID => l.annotations.push(l.annotations[0].clone()),
                        7 if mode != OffsetMode::CodePoint => {
                            l.annotations[0].spans[0].start = match (view, mode) {
                                (0, OffsetMode::Utf8) => 3,
                                (0, OffsetMode::Utf16) => 4,
                                (_, OffsetMode::Utf8) => 1,
                                _ => 2,
                            };
                        }
                        8 => l.id = "test:wrong-quality-layer".into(),
                        9 => {
                            let end = l.annotations[0].spans[0].end;
                            l.annotations[0].spans.push(SpanFragment { start: end, end });
                        }
                        _ => continue,
                    }
                    mutations.push(msg);
                }
                for msg in mutations {
                    cases += 1;
                    match client.round_trip_typed_document(msg).await {
                        Ok(_) => accepted += 1,
                        Err(error) => invalid(error),
                    }
                }
            }
        }
    }
    server.stop().await;
    eprintln!("quality malformed-message controls: {cases} cases, {accepted} accepted");
    assert_eq!(accepted, 0, "accepted {accepted}/{cases} malformed quality messages");
}

#[tokio::test]
async fn quality_preserves_metric_endpoints_and_empty_projected_regions() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = fixture();
    for (name, ratio) in [("zero", -0.0), ("one", 1.0)] {
        let m = QualityMetrics {
            composite_score: ratio,
            mtld_lexical_diversity: 0.0,
            flesch_kincaid_grade: -f32::MAX,
            automated_readability_index: -0.0,
            coleman_liau_index: f32::MAX,
            information_density: ratio,
            alpha_ratio: ratio,
            digit_ratio: ratio,
            symbol_ratio: ratio,
            whitespace_ratio: ratio,
            replacement_ratio: ratio,
            flags: QualityFlags::CLEAN,
        };
        m.check().unwrap();
        doc.add(
            &Key::<TextQuality>::named(name),
            TextQuality::new(ViewId::SOURCE, 12, m),
            Provenance::by("quality:endpoint"),
        )
        .unwrap();
        let mut regions = SpanQuality::new(ViewId::SOURCE);
        regions.push(Span::new(0, 2), m);
        doc.add(&Key::<SpanQuality>::named(name), regions, Provenance::by("quality:removed-prefix")).unwrap();
    }
    let graph = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = project(&mut client, &graph, mode, 1).await;
        let out = client.round_trip_typed_document(out).await.unwrap().into_inner();
        for name in ["zero", "one"] {
            for layer in out.layers.iter().filter(|l| l.instance == name) {
                let ann = &layer.annotations[0];
                let m = match ann.value.as_ref().unwrap() {
                    Value::TextQuality(m) => m,
                    Value::SpanQuality(m) => {
                        assert_eq!(ann.spans, [SpanFragment { start: 0, end: 0 }]);
                        m
                    }
                    _ => unreachable!(),
                };
                let ratio: f32 = if name == "zero" { -0.0 } else { 1.0 };
                for index in [0, 5, 6, 7, 8, 9, 10] {
                    assert_eq!(fields(m)[index].unwrap().to_bits(), ratio.to_bits());
                }
                assert_eq!(m.mtld_lexical_diversity, Some(0.0));
                assert_eq!(m.flesch_kincaid_grade, Some(-f32::MAX));
                assert_eq!(m.automated_readability_index.unwrap().to_bits(), (-0.0f32).to_bits());
                assert_eq!(m.coleman_liau_index, Some(f32::MAX));
                assert_eq!(m.flags, 0);
            }
        }
    }
    server.stop().await;
}
