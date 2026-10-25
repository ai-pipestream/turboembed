use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Ix, Key, Layer, Provenance, Span, ViewId};
use turbo_analysis_layers::{QualityFlags, QualityMetrics, SpanQuality, TextQuality};

fn fixture_graph() -> DocumentGraph {
    let mut doc = Document::new("été x");
    let metrics = QualityMetrics {
        composite_score: 0.7345678,
        mtld_lexical_diversity: 243.0 / 53.0,
        flesch_kincaid_grade: -1.45,
        automated_readability_index: -0.0,
        coleman_liau_index: 8.731,
        information_density: 0.4,
        alpha_ratio: 0.7,
        digit_ratio: 0.1,
        symbol_ratio: 0.1,
        whitespace_ratio: 0.1,
        replacement_ratio: 0.01,
        flags: QualityFlags::OCR_BROKEN_HYPHENS | QualityFlags::TOO_SHORT,
    };
    doc.add(&Key::DEFAULT, TextQuality::new(ViewId::SOURCE, 7, metrics), Provenance::by("test")).unwrap();
    let mut spans = SpanQuality::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5), metrics);
    doc.add(&Key::DEFAULT, spans, Provenance::by("test")).unwrap();
    export(&doc, &turbo_analysis_layers::export::codecs()).unwrap()
}

#[test]
fn standard_codecs_preserve_every_float_bit_and_flag() {
    let graph = fixture_graph();
    let codecs = turbo_analysis_layers::export::codecs();
    let bytes = graph.to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(export(&restored, &codecs).unwrap().to_bytes().unwrap(), bytes);
    assert_eq!(
        restored.get::<TextQuality>().unwrap().metrics.automated_readability_index.to_bits(),
        (-0.0f32).to_bits()
    );
}

#[test]
fn wire_refuses_nonfinite_metrics_invalid_ratios_flags_and_trailing_bytes() {
    let codecs = turbo_analysis_layers::export::codecs();
    for layer in [TextQuality::NAME, SpanQuality::NAME] {
        for (offset, value) in [
            (0, f32::NAN.to_bits()),
            (0, 2.0f32.to_bits()),
            (4, f32::INFINITY.to_bits()),
            (8, f32::NEG_INFINITY.to_bits()),
        ] {
            let mut graph = fixture_graph();
            let record = graph.layers.iter_mut().find(|r| r.layer == layer.to_string()).unwrap();
            let body = record.body.as_mut().unwrap();
            let start = body.len() - 46;
            body[start + offset..start + offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(import(&graph, &codecs).is_err(), "{layer} offset {offset}");
        }
        let mut graph = fixture_graph();
        let body = graph.layers.iter_mut().find(|r| r.layer == layer.to_string()).unwrap().body.as_mut().unwrap();
        let end = body.len();
        body[end - 2..].copy_from_slice(&0x8000u16.to_le_bytes());
        assert!(import(&graph, &codecs).is_err());
        let mut graph = fixture_graph();
        graph.layers.iter_mut().find(|r| r.layer == layer.to_string()).unwrap().body.as_mut().unwrap().push(0);
        assert!(import(&graph, &codecs).is_err());
    }
}

#[test]
fn layer_checks_refuse_partial_summary_mismatched_columns_and_split_utf8() {
    let codecs = turbo_analysis_layers::export::codecs();
    for (start, end) in [(0, 5), (1, 7), (0, 999)] {
        let mut doc = Document::new("été x");
        let mut q = TextQuality::new(ViewId::SOURCE, 7, QualityMetrics::default());
        q.spans = turbo_analysis::Spans::new(ViewId::SOURCE);
        q.spans.push(Span::new(start, end));
        assert!(doc.add(&Key::DEFAULT, q, Provenance::by("test")).is_err());
    }
    let mut graph = fixture_graph();
    let body =
        graph.layers.iter_mut().find(|r| r.layer == SpanQuality::NAME.to_string()).unwrap().body.as_mut().unwrap();
    // View string length + empty view + item count + fragment count.
    body[12..16].copy_from_slice(&1u32.to_le_bytes());
    assert!(import(&graph, &codecs).is_err());
    let mut spans = SpanQuality::new(ViewId::SOURCE);
    spans.spans.push(Span::new(0, 1));
    assert!(spans.check().is_err());
    assert_eq!(TextQuality::new(ViewId::SOURCE, 0, QualityMetrics::default()).spans.get(Ix::new(0)), Span::new(0, 0));
}
