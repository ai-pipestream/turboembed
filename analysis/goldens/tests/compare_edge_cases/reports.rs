//! The reports: the layer id and the row index, in every variant.

use super::*;

// ---------------------------------------------------------------------
// Reports: the layer id and the row index, in every variant

#[test]
fn every_item_report_names_its_layer_and_row() {
    let g = golden();
    let cases: Vec<(&str, Mismatch)> = {
        let mut cases = Vec::new();
        let mut p = Parts::matching();
        p.sentences[1] = (27, 36);
        cases.push(("turbo:sentences", fail::<Sentences>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.sentence_conf[0] = 0.0;
        cases.push(("turbo:confidence<turbo:sentences>", fail::<Confidence<Sentences>>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.tokens[7] = (30, 35);
        cases.push(("turbo:tokens", fail::<Tokens>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.token_conf[7] = 0.0;
        cases.push(("turbo:confidence<turbo:tokens>", fail::<Confidence<Tokens>>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.tags[7] = "NOUN";
        cases.push(("turbo:pos", fail::<PosTags>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.lemmas[7] = "x";
        cases.push(("turbo:lemmas", fail::<Lemmas>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.entities[1].5 = 0.0;
        cases.push(("turbo:entities", fail::<Entities>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.language[0].0 = "xx";
        cases.push(("turbo:language", fail::<Language>(&p, &g, Tolerance::exact())));
        cases
    };
    let index = [1, 0, 7, 7, 7, 7, 1, 0];
    for ((id, m), index) in cases.iter().zip(index) {
        assert_eq!(m.layer(), *id);
        let shown = m.to_string();
        assert!(shown.starts_with(&format!("{id} item {index}: the ")), "{shown}");
        assert!(shown.contains("\n  rust:   ") && shown.contains("\n  golden: "), "{shown}");
        assert!(matches!(m, Mismatch::Item { index: i, .. } if *i == index), "{m:?}");
    }
}

#[test]
fn every_other_report_starts_with_the_layer_and_the_accessor_agrees() {
    let g = golden();
    let mut reports: Vec<Mismatch> = Vec::new();
    // Input.
    let other = Document::new("another text");
    reports.push(matches_golden::<Tokens>(&other, &g, Tolerance::exact()).unwrap_err());
    // MissingGolden.
    let mut no_lemmas = golden();
    no_lemmas.layers.retain(|l| l.id != "turbo:lemmas");
    reports.push(matches_golden::<Lemmas>(&Parts::matching().doc(), &no_lemmas, Tolerance::exact()).unwrap_err());
    // MissingLayer, plain and for a named instance.
    let doc = Document::new(TEXT);
    reports.push(matches_golden::<Tokens>(&doc, &g, Tolerance::exact()).unwrap_err());
    reports.push(matches_golden_key(&doc, &Key::<Tokens>::named("x"), &g, Tolerance::exact()).unwrap_err());
    // Anchor.
    let mut anchored = golden();
    anchored.layers.iter_mut().find(|l| l.id == "turbo:tokens").unwrap().anchor = GoldenAnchor::Document;
    reports.push(matches_golden::<Tokens>(&Parts::matching().doc(), &anchored, Tolerance::exact()).unwrap_err());
    // Columns.
    let mut renamed = golden();
    renamed.layers.iter_mut().find(|l| l.id == "turbo:lemmas").unwrap().columns[0].name = "x".into();
    reports.push(matches_golden::<Lemmas>(&Parts::matching().doc(), &renamed, Tolerance::exact()).unwrap_err());
    // NoValues.
    let mut p = Parts::matching();
    p.prob = None;
    reports.push(fail::<PosTags>(&p, &g, Tolerance::exact()));
    // Count.
    let mut p = Parts::matching();
    p.sentences.pop();
    p.sentence_conf.pop();
    reports.push(fail::<Sentences>(&p, &g, Tolerance::exact()));

    let expected = [
        "turbo:tokens",
        "turbo:lemmas",
        "turbo:tokens",
        "turbo:tokens@x",
        "turbo:tokens",
        "turbo:lemmas",
        "turbo:pos",
        "turbo:sentences",
    ];
    assert_eq!(reports.len(), expected.len());
    for (m, id) in reports.iter().zip(expected) {
        assert_eq!(m.layer(), id, "{m:?}");
        assert!(m.to_string().starts_with(&format!("{id}: ")), "{m}");
    }
    assert!(matches!(reports[0], Mismatch::Input { .. }));
    assert!(matches!(reports[3], Mismatch::MissingLayer { .. }));
    assert!(matches!(reports[4], Mismatch::Anchor { .. }));
    assert!(matches!(reports[5], Mismatch::Columns { .. }));
    says(
        &reports[0].to_string(),
        &["12 bytes", "expected 37 bytes", "f03e4a09022fa42c68b4fbab2730c195ceb468707eb0b2598a7b1e91abc4918f"],
    );
    says(&reports[1].to_string(), &["[turbo:sentences, turbo:confidence<turbo:sentences>, turbo:tokens"]);
    says(&reports[4].to_string(), &["anchored to document, expected text"]);
    says(&reports[5].to_string(), &["the columns [x:str], expected [lemma:str]"]);
    says(&reports[7].to_string(), &["1 items in Rust, 2 in the golden"]);
}

#[test]
fn a_report_for_a_named_instance_carries_the_name_and_the_row() {
    let g = golden();
    let mut doc = Document::new(TEXT);
    let s = spans(&[(0, 3), (4, 9), (10, 14), (15, 17), (18, 25), (25, 26), (27, 29), (30, 36), (35, 37)]);
    doc.add(&Key::named("mine"), Tokens { spans: s, sentence: None }, Provenance::by("t")).unwrap();
    let m = matches_golden_key(&doc, &Key::<Tokens>::named("mine"), &g, Tolerance::exact()).unwrap_err();
    assert_eq!(m.layer(), "turbo:tokens@mine");
    says(
        &m.to_string(),
        &["turbo:tokens@mine item 8: the span differs", "rust:   bytes 35..37 \"d!\"", "golden: bytes 36..37 \"!\""],
    );
}
