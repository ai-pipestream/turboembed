//! Regression test verifying that malformed `GrammarDiagnostics` binary imports
//! cannot poison or exhaust custom `rule_id` handling for subsequent valid documents.

use turbo_analysis::export::{export, import};
use turbo_analysis::{Document, Key, Layer, Provenance, Span, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{GrammarCategory, GrammarDiagnostic, GrammarDiagnostics};

#[test]
fn malformed_imports_must_not_exhaust_future_custom_rule_ids() {
    let mut doc = Document::new("bad");
    let mut layer = GrammarDiagnostics::new(ViewId::SOURCE);
    layer.push(GrammarDiagnostic::new(
        Span::new(0, 3),
        "SVA-001",
        GrammarCategory::SubjectVerbAgreement,
        "Bad",
        ["good"],
        0.9,
    ));
    doc.add(&Key::<GrammarDiagnostics>::DEFAULT, layer, Provenance::by("review:wire")).unwrap();
    let registry = codecs();
    let mut graph = export(&doc, &registry).unwrap();
    let layer_index = graph.layers.iter().position(|r| r.layer == GrammarDiagnostics::NAME.to_string()).unwrap();
    let bytes = graph.layers[layer_index].body.as_mut().unwrap();
    let rule_start = bytes.windows(7).position(|b| b == b"SVA-001").unwrap();
    let confidence_start = bytes.len() - 4;
    bytes[confidence_start..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    for i in 0..256 {
        graph.layers[layer_index].body.as_mut().unwrap()[rule_start..rule_start + 7]
            .copy_from_slice(format!("BAD-{i:03}").as_bytes());
        assert!(import(&graph, &registry).is_err());
    }
    let bytes = graph.layers[layer_index].body.as_mut().unwrap();
    bytes[rule_start..rule_start + 7].copy_from_slice(b"NEW-001");
    bytes[confidence_start..].copy_from_slice(&0.9f32.to_bits().to_le_bytes());
    let result = import(&graph, &registry);
    assert!(result.is_ok(), "Rejected valid document after invalid imports: {}", result.err().unwrap());
}
