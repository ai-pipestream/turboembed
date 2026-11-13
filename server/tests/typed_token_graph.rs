//! Wire projection and validation tests for `turbo:token-graph` (#2340)
//! across `Utf8`, `Utf16`, and `CodePoint` offset modes and derived views.

use turbo_analysis::export::{export, import};
use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance, Span, ViewId};
use turbo_analysis_layers::{TokenArcKind, TokenGraph};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{
    LayerWireStatus, OffsetMode as ProtoOffsetMode, SpanFragment, typed_annotation::Value,
};
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::typed::{document_to_typed_with, round_trip_typed_document};
use turbo_normalize::{CaseFold, Chain, Dehyphenate, NormalizeLatticeAnnotator, NormalizeStep, UnicodeForm};
use turbo_tokenize::WhitespaceTokenizer;

#[test]
fn token_graph_projects_to_source_offsets_in_utf8_utf16_and_code_point_modes() {
    let source = "🚀 CAFE\u{0301}-\n   TERIA state-of-the-art";
    let chain = Chain::new().then(Dehyphenate::new()).then(UnicodeForm::nfc()).then(CaseFold::lowercase());

    let analyzer = Analyzer::builder()
        .step(Bound::new(NormalizeStep::new(chain)).write(0, "clean"))
        .step(Bound::new(WhitespaceTokenizer::default()).read(0, "clean").write(0, "clean"))
        .step(Bound::new(NormalizeLatticeAnnotator).read(0, "clean").read(1, "clean").write(0, "clean"))
        .build()
        .unwrap();

    let mut doc = Document::new(source);
    analyzer.run(std::slice::from_mut(&mut doc)).remove(0).unwrap();

    // Binary DocumentGraph export/import round-trip preserves TokenGraph.
    let graph = export(&doc, codecs()).unwrap();
    let reimported = import(&graph, codecs()).unwrap();
    let tg = reimported.get_key(&Key::<TokenGraph>::named("clean")).unwrap();
    assert_eq!(tg.matches_phrase(&["café", "teria"]).len(), 1);
    assert_eq!(tg.matches_phrase(&["state", "of", "the", "art"]).len(), 1);

    let cafe_src = Span::new(5, 11); // "CAFE\u{0301}" in UTF-8
    let teria_src = Span::new(16, 21); // "TERIA" in UTF-8
    assert_eq!(&source[cafe_src.range()], "CAFE\u{0301}");
    assert_eq!(&source[teria_src.range()], "TERIA");

    for (mode, proto_mode) in [
        (OffsetMode::Utf8, ProtoOffsetMode::Utf8),
        (OffsetMode::Utf16, ProtoOffsetMode::Utf16),
        (OffsetMode::CodePoint, ProtoOffsetMode::CodePoint),
    ] {
        let typed = document_to_typed_with("doc-tg", &doc, ViewId::SOURCE, mode, &Disclosure::FULL).unwrap();
        assert_eq!(typed.offsets, proto_mode as i32);
        let layer = typed.layers.iter().find(|l| l.id == "turbo:token-graph").expect("token-graph layer");
        assert_eq!(layer.status, LayerWireStatus::Ok as i32);

        let cafe_ann = layer
            .annotations
            .iter()
            .find(|a| matches!(&a.value, Some(Value::TokenArc(t)) if t.term == "café"))
            .expect("café sub-arc");
        let teria_ann = layer
            .annotations
            .iter()
            .find(|a| matches!(&a.value, Some(Value::TokenArc(t)) if t.term == "teria"))
            .expect("teria sub-arc");

        match mode {
            OffsetMode::Utf8 => {
                assert_eq!(cafe_ann.spans, vec![SpanFragment::exact(5, 11)]);
                assert_eq!(teria_ann.spans, vec![SpanFragment::exact(16, 21)]);
            }
            OffsetMode::Utf16 => {
                assert_eq!(cafe_ann.spans, vec![SpanFragment::exact(3, 8)]);
                assert_eq!(teria_ann.spans, vec![SpanFragment::exact(13, 18)]);
            }
            OffsetMode::CodePoint => {
                assert_eq!(cafe_ann.spans, vec![SpanFragment::exact(2, 7)]);
                assert_eq!(teria_ann.spans, vec![SpanFragment::exact(12, 17)]);
            }
        }

        let echoed = round_trip_typed_document(&typed).unwrap();
        assert_eq!(echoed, typed);
    }

    // Malformed position_length = 0 is rejected by wire validation.
    let mut bad_doc = Document::new("hello");
    let mut bad_tg = TokenGraph::new(ViewId::SOURCE);
    bad_tg.num_positions = 1;
    bad_tg.push(Span::new(0, 5), "hello", 0, 1, TokenArcKind::Primary);
    bad_doc.add(&Key::DEFAULT, bad_tg, Provenance::by("test:tg")).unwrap();
    let mut typed =
        document_to_typed_with("doc-bad", &bad_doc, ViewId::SOURCE, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    let layer = typed.layers.iter_mut().find(|l| l.id == "turbo:token-graph").unwrap();
    if let Some(Value::TokenArc(ref mut arc)) = layer.annotations[0].value {
        arc.position_length = 0;
    }
    assert!(round_trip_typed_document(&typed).is_err());
}
