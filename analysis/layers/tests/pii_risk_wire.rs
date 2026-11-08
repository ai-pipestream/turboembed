//! The `turbo:pii-risk` codec (#1889): a round trip keeps every bit, and a
//! payload that does not hold together is refused.

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Key, Layer, Provenance, Span, ViewId};
use turbo_analysis_layers::{CustomKinds, Pii, PiiEvidence, PiiKind, PiiMention, PiiRisk, PiiSeverityTable};

const TEXT: &str = "card 4111 and badge B-77";

fn fixture_graph() -> DocumentGraph {
    let mut doc = Document::new(TEXT);
    let custom = CustomKinds::from_names(["badge"]).unwrap();
    let badge = custom.find("badge").unwrap();
    let mut pii = Pii::with_custom_kinds(ViewId::SOURCE, custom);
    let d = pii.detector("test", "1").unwrap();
    for (span, kind) in [(Span::new(5, 9), PiiKind::Card), (Span::new(20, 24), badge)] {
        pii.push(PiiMention { span, kind, detector: d, evidence: &[PiiEvidence::Shape], score: None, normalized: "x" });
    }
    let risk = PiiRisk::compute(&pii, TEXT, &PiiSeverityTable::new());
    doc.add(&Key::DEFAULT, pii, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, risk, Provenance::by("test")).unwrap();
    export(&doc, &turbo_analysis_layers::export::codecs()).unwrap()
}

fn body(graph: &mut DocumentGraph) -> &mut Vec<u8> {
    graph.layers.iter_mut().find(|r| r.layer == PiiRisk::NAME.to_string()).unwrap().body.as_mut().unwrap()
}

#[test]
fn a_round_trip_keeps_every_bit() {
    let graph = fixture_graph();
    let codecs = turbo_analysis_layers::export::codecs();
    let bytes = graph.to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(export(&restored, &codecs).unwrap().to_bytes().unwrap(), bytes);
    let risk = restored.get::<PiiRisk>().unwrap();
    assert_eq!(risk.mentions, 2);
    assert_eq!(risk.kind_name(risk.kinds[1].kind), Some("badge"));
}

#[test]
fn the_payload_holds_no_value() {
    let mut graph = fixture_graph();
    let b = body(&mut graph);
    assert!(!b.windows(4).any(|w| w == b"4111"));
}

#[test]
fn a_bad_level_a_nonfinite_score_or_trailing_bytes_are_refused() {
    let codecs = turbo_analysis_layers::export::codecs();
    // The level byte follows the 4-byte score.
    let mut graph = fixture_graph();
    body(&mut graph)[4] = 9;
    assert!(import(&graph, &codecs).is_err(), "unknown level");
    let mut graph = fixture_graph();
    body(&mut graph)[4] = 1;
    assert!(import(&graph, &codecs).is_err(), "level none with a positive score");
    let mut graph = fixture_graph();
    body(&mut graph)[0..4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    assert!(import(&graph, &codecs).is_err(), "nan score");
    let mut graph = fixture_graph();
    body(&mut graph).push(0);
    assert!(import(&graph, &codecs).is_err(), "trailing byte");
}
