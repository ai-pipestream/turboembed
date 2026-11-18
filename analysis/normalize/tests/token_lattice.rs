use turbo_analysis::{Analyzer, Bound, Document, Key, NormalizationXray, Span, View, ViewId, XrayRunKind};
use turbo_analysis_layers::{TokenGraph, TokenGraphBuilder, Tokens};
use turbo_normalize::{
    CaseFold, Chain, Dehyphenate, NormalizeLatticeAnnotator, NormalizeStep, UnicodeForm, emit_normalize_lattice,
};
use turbo_tokenize::{WhitespaceTokenizer, WordTokenizer};

#[test]
fn dehyphenate_nfc_casefold_chain_preserves_bidirectional_source_offsets_and_utf16() {
    // Source holds astral emoji '🚀' (4 UTF-8 bytes, 2 UTF-16 code units),
    // decomposed NFD "CAFE\u{0301}-\n   TERIA", and line-broken "DATA-\nBASE".
    let source = "🚀 CAFE\u{0301}-\n   TERIA and DATA-\nBASE search";
    let chain = Chain::new().then(Dehyphenate::new()).then(UnicodeForm::nfc()).then(CaseFold::lowercase());

    let analyzer = Analyzer::builder()
        .step(Bound::new(NormalizeStep::new(chain)).write(0, "norm"))
        .step(Bound::new(WordTokenizer::new()).read(0, "norm").write(0, "norm").write(1, "norm"))
        .step(Bound::new(NormalizeLatticeAnnotator).read(0, "norm").read(1, "norm").write(0, "norm"))
        .build()
        .unwrap();

    let mut doc = Document::new(source);
    analyzer.run(std::slice::from_mut(&mut doc)).remove(0).unwrap();

    let norm_view = doc.get_key(&Key::<View>::named("norm")).unwrap();
    assert_eq!(norm_view.text(), "🚀 caféteria and database search");

    // Verify NormalizationXray between SOURCE and norm view records the exact deletions.
    let xray = NormalizationXray::between(&doc, ViewId::SOURCE, norm_view.id()).unwrap();
    assert!(!xray.is_empty());
    assert!(xray.iter().any(|r| r.kind == XrayRunKind::Deleted));

    let graph = doc.get_key(&Key::<TokenGraph>::named("norm")).unwrap();
    // Both whole compounds and split parts match slop=0 without shifting downstream ordinals!
    assert_eq!(graph.matches_phrase(&["🚀", "caféteria", "and", "database", "search"]).len(), 1);
    assert_eq!(graph.matches_phrase(&["🚀", "café", "teria", "and", "data", "base", "search"]).len(), 1);
    assert_eq!(graph.matches_phrase(&["🚀", "café", "teria", "and", "database", "search"]).len(), 1);
    assert!(graph.matches_phrase(&["caféteria", "database"]).is_empty());

    // Project every arc back to ViewId::SOURCE (UTF-8 and UTF-16) and back to norm view!
    for (_, arc) in graph.iter() {
        let src_span = doc.to_source(norm_view.id(), arc.span);
        let src_slice = &source[src_span.range()];
        match arc.term {
            "🚀" => assert_eq!(src_slice, "🚀"),
            "café" => assert_eq!(src_slice, "CAFE\u{0301}"),
            "teria" => assert_eq!(src_slice, "TERIA"),
            "caféteria" => assert_eq!(src_slice, "CAFE\u{0301}-\n   TERIA"),
            "and" => assert_eq!(src_slice, "and"),
            "data" => assert_eq!(src_slice, "DATA"),
            "base" => assert_eq!(src_slice, "BASE"),
            "database" => assert_eq!(src_slice, "DATA-\nBASE"),
            "search" => assert_eq!(src_slice, "search"),
            other => panic!("unexpected term {other}"),
        }

        // Verify UTF-16 source round-trip through OffsetTable.
        let src_u16 = doc.view_to_source_utf16(norm_view.id(), arc.span).unwrap();
        let back_src_bytes = doc.view_span_from_utf16(ViewId::SOURCE, src_u16).unwrap();
        assert_eq!(back_src_bytes, src_span);
        let round_trip_view = doc.source_utf16_to_view(norm_view.id(), src_u16).unwrap();
        assert_eq!(round_trip_view, arc.span);
    }
}

#[test]
fn inline_hyphenated_compounds_emit_both_parts_and_joined_across_tokenizers() {
    let text = "fast data-base engine";

    // Case 1: WordTokenizer splits "data-base" into ["data", "base"].
    let a1 = Analyzer::builder().add(WordTokenizer::new()).add(NormalizeLatticeAnnotator).build().unwrap();
    let mut doc1 = Document::new(text);
    a1.run(std::slice::from_mut(&mut doc1)).remove(0).unwrap();
    let g1 = doc1.get::<TokenGraph>().unwrap();
    assert_eq!(g1.matches_phrase(&["fast", "database", "engine"]).len(), 1);
    assert_eq!(g1.matches_phrase(&["fast", "data", "base", "engine"]).len(), 1);
    let db_arc = g1.iter().find(|(_, a)| a.term == "database").unwrap().1;
    assert_eq!(db_arc.span, Span::new(5, 14));
    assert_eq!(&text[db_arc.span.range()], "data-base");

    // Case 2: WhitespaceTokenizer keeps "data-base" as one token.
    let a2 = Analyzer::builder().add(WhitespaceTokenizer::default()).build().unwrap();
    let mut doc2 = Document::new(text);
    a2.run(std::slice::from_mut(&mut doc2)).remove(0).unwrap();
    let v = doc2.get::<View>().unwrap();
    let toks = doc2.get::<Tokens>().unwrap();
    let mut builder = TokenGraphBuilder::from_tokens(v.text(), toks, None).unwrap();
    emit_normalize_lattice(v.text(), v.alignment(), toks, &mut builder);
    let g2 = builder.finish().unwrap();
    assert_eq!(g2.matches_phrase(&["fast", "database", "engine"]).len(), 1);
    assert_eq!(g2.matches_phrase(&["fast", "data", "base", "engine"]).len(), 1);
    let data_arc = g2.iter().find(|(_, a)| a.term == "data").unwrap().1;
    let base_arc = g2.iter().find(|(_, a)| a.term == "base").unwrap().1;
    assert_eq!(&text[data_arc.span.range()], "data");
    assert_eq!(&text[base_arc.span.range()], "base");
}
