use super::*;

#[test]
fn a_matching_document_passes_every_layer_exactly() {
    let g = golden();
    let doc = Parts::matching().doc();
    assert_matches_golden::<Sentences>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Tokens>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Confidence<Sentences>>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Confidence<Tokens>>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<PosTags>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Lemmas>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Entities>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Language>(&doc, &g, Tolerance::exact());
}

fn mismatch<L: GoldenCompare>(p: &Parts, tol: Tolerance) -> (Mismatch, String) {
    let m = matches_golden::<L>(&p.doc(), &golden(), tol).unwrap_err();
    let shown = m.to_string();
    (m, shown)
}

pub(super) fn has(shown: &str, parts: &[&str]) {
    for p in parts {
        assert!(shown.contains(p), "{p:?} is not in the report:\n{shown}");
    }
}

#[test]
fn a_span_shifted_by_one_byte() {
    let mut p = Parts::matching();
    p.tokens[1] = (4, 10);
    let (m, shown) = mismatch::<Tokens>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 1, what: "span", .. }), "{m:?}");
    has(
        &shown,
        &["opennlp:tokens item 1: the span differs", "rust:   bytes 4..10 \"Smith \"", "golden: bytes 4..9 \"Smith\""],
    );
}

#[test]
fn a_changed_tag() {
    let mut p = Parts::matching();
    p.tags[2] = "NOUN";
    let (m, shown) = mismatch::<PosTags>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 2, what: "tag", .. }), "{m:?}");
    // The report shows the token the tag is for, on both sides.
    has(
        &shown,
        &[
            "opennlp:pos item 2: the tag differs",
            "rust:   bytes 10..14 \"went\", tag \"NOUN\", prob 3fef95810624dd2f (0.987)",
            "golden: bytes 10..14 \"went\", tag \"VERB\", prob 3fef95810624dd2f (0.987)",
        ],
    );
}

#[test]
fn a_value_just_outside_the_tolerance() {
    let mut p = Parts::matching();
    let golden_value: f64 = 0.98;
    // 2e-9 above the golden value, or the nearest double below that whose
    // distance from it, as f64 subtraction gives it, is at most 2e-9.
    let mut v = golden_value + 2e-9;
    while v - golden_value > 2e-9 {
        v = f64::from_bits(v.to_bits() - 1);
    }
    p.token_conf[3] = v;
    let diff = (p.token_conf[3] - golden_value).abs();
    assert!(diff > 1e-9 + 1e-10 && diff <= 2e-9, "the test's own distance: {diff:e}");
    let (m, shown) = mismatch::<Confidence<Tokens>>(&p, Tolerance::abs(1e-9));
    assert!(matches!(m, Mismatch::Item { index: 3, what: "value", .. }), "{m:?}");
    let rust_bits = format!("{:016x}", p.token_conf[3].to_bits());
    has(
        &shown,
        &[
            "turbo:confidence<opennlp:tokens> item 3: the value differs",
            "tolerance 1e-9",
            &format!("rust:   bytes 15..17 \"to\", value {rust_bits} ({:?})", p.token_conf[3]),
            "golden: bytes 15..17 \"to\", value 3fef5c28f5c28f5c (0.98)",
        ],
    );
    assert_eq!(matches_golden::<Confidence<Tokens>>(&p.doc(), &golden(), Tolerance::abs(2e-9)), Ok(()));
    // At exact bits a value one unit in the last place off differs.
    p.token_conf[3] = f64::from_bits(golden_value.to_bits() + 1);
    let (m, _) = mismatch::<Confidence<Tokens>>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 3, what: "value", .. }), "{m:?}");
}

#[test]
fn a_missing_last_row() {
    let mut p = Parts::matching();
    p.sentences.pop();
    p.sentence_conf.pop();
    let (m, shown) = mismatch::<Sentences>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 1, golden: 2, .. }), "{m:?}");
    has(
        &shown,
        &["opennlp:sentences: 1 items in Rust, 2 in the golden", "golden item 1", "bytes 27..37 \"It rained!\""],
    );
    let (m, shown) = mismatch::<Confidence<Sentences>>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 1, golden: 2, .. }), "{m:?}");
    has(&shown, &["golden item 1", "bytes 27..37 \"It rained!\", value 3fee666666666666 (0.95)"]);
}

#[test]
fn a_golden_layer_with_the_wrong_columns() {
    let mut g = golden();
    let conf = g.layers.iter_mut().find(|l| l.id == "turbo:confidence<opennlp:tokens>").unwrap();
    conf.columns[0].name = "prob".into();
    let m = matches_golden::<Confidence<Tokens>>(&Parts::matching().doc(), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Columns { .. }), "{m:?}");
    has(&m.to_string(), &["turbo:confidence<opennlp:tokens>", "the columns [prob:f64], expected [value:f64]"]);

    let mut g = golden();
    let pos = g.layers.iter_mut().find(|l| l.id == "opennlp:pos").unwrap();
    pos.columns[1].data = ColumnData::U32(vec![0; 9]);
    let m = matches_golden::<PosTags>(&Parts::matching().doc(), &g, Tolerance::exact()).unwrap_err();
    has(&m.to_string(), &["the columns [tag:str, prob:u32], expected [tag:str, prob:f64]"]);
}

#[test]
fn a_language_code_difference() {
    let mut p = Parts::matching();
    p.language[0].0 = "deu";
    let (m, shown) = mismatch::<Language>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 0, what: "code", .. }), "{m:?}");
    has(
        &shown,
        &[
            "opennlp:language item 0: the code differs",
            "rust:   code \"deu\", confidence 3fedccac6fbfc4c9",
            "golden: code \"eng\", confidence 3fedccac6fbfc4c9",
        ],
    );
    // With the code equal, the confidence is compared next.
    let mut p = Parts::matching();
    p.language[0].1 = 0.0412;
    let (m, _) = mismatch::<Language>(&p, Tolerance::abs(1e-9));
    assert!(matches!(m, Mismatch::Item { index: 0, what: "confidence", .. }), "{m:?}");
}

#[test]
fn tags_without_probabilities_do_not_pass() {
    let mut p = Parts::matching();
    p.prob = None;
    let (m, shown) = mismatch::<PosTags>(&p, Tolerance::exact());
    assert_eq!(m, Mismatch::NoValues { layer: "opennlp:pos".into(), column: "prob" });
    has(&shown, &["opennlp:pos: the Rust layer has no probabilities", "prob column"]);
}

#[test]
fn entity_tokens_kinds_and_lemmas_are_compared() {
    let mut p = Parts::matching();
    p.entities[1].3 = 6;
    let (m, shown) = mismatch::<Entities>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 1, what: "tokens", .. }), "{m:?}");
    has(
        &shown,
        &[
            "rust:   bytes 18..25 \"Zürich\", tokens 4..6, kind \"location\"",
            "golden: bytes 18..25 \"Zürich\", tokens 4..5",
        ],
    );

    let mut p = Parts::matching();
    p.entities[0].4 = "organization";
    let (m, shown) = mismatch::<Entities>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 0, what: "kind", .. }), "{m:?}");
    has(&shown, &["kind \"organization\"", "kind \"person\"", "bytes 4..9 \"Smith\""]);

    let mut p = Parts::matching();
    p.lemmas[7] = "rained";
    let (m, shown) = mismatch::<Lemmas>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 7, what: "lemma", .. }), "{m:?}");
    has(&shown, &["bytes 30..36 \"rained\", lemma \"rained\"", "lemma \"rain\""]);
}

#[test]
fn a_document_with_another_text_is_refused() {
    let g = golden();
    let doc = Document::new("Dr. Smith went to Zurich. It rained!");
    let m = matches_golden::<Sentences>(&doc, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Input { expected_len: 37, found_len: 36, .. }), "{m:?}");
}

#[test]
fn a_missing_or_skipped_layer_is_reported() {
    let g = golden();
    let mut doc = Document::new(TEXT);
    let m = matches_golden::<Tokens>(&doc, &g, Tolerance::exact()).unwrap_err();
    assert_eq!(m.to_string(), "opennlp:tokens: the document has no such layer");
    doc.skip(Key::<Tokens>::DEFAULT.id(), "no model".into(), Provenance::by("test")).unwrap();
    let m = matches_golden::<Tokens>(&doc, &g, Tolerance::exact()).unwrap_err();
    assert_eq!(m.to_string(), "opennlp:tokens: the document's layer is skipped: no model");
}

#[test]
fn a_golden_without_the_layer_lists_what_it_has() {
    let mut g = golden();
    g.layers.retain(|l| l.id != "opennlp:lemmas");
    let m = matches_golden::<Lemmas>(&Parts::matching().doc(), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(&m, Mismatch::MissingGolden { available, .. } if available.len() == 7), "{m:?}");
}

/// A named instance is compared when its key is given, and tokens in a
/// derived view are compared in source bytes, through the view's
/// alignment.
#[test]
fn a_named_instance_on_a_derived_view() {
    let g = golden();
    let mut doc = Document::new(TEXT);
    // "Zürich" becomes "Zurich": every token from there on is a byte earlier
    // in the view, and maps back to the source's offsets.
    let mut b = ViewBuilder::new(TEXT);
    b.keep(19);
    b.replace(21, "u");
    let (folded, alignment) = b.finish();
    assert_eq!(folded, "Dr. Smith went to Zurich. It rained!");
    let view = View::derived(doc.get::<View>().unwrap(), folded, alignment);
    doc.add(&Key::named("folded"), view, Provenance::by("test")).unwrap();
    let id = doc.get_key(&Key::<View>::named("folded")).unwrap().id();
    let spans = [(0, 3), (4, 9), (10, 14), (15, 17), (18, 24), (24, 25), (26, 28), (29, 35), (35, 36)];
    doc.add(&Key::named("gold"), tokens(id, &spans), Provenance::by("test")).unwrap();
    assert_matches_golden_key(&doc, &Key::<Tokens>::named("gold"), &g, Tolerance::exact());

    let mut shifted = spans;
    shifted[4] = (18, 23);
    let mut doc2 = Document::new(TEXT);
    let mut b = ViewBuilder::new(TEXT);
    b.keep(19);
    b.replace(21, "u");
    let (folded, alignment) = b.finish();
    let view = View::derived(doc2.get::<View>().unwrap(), folded, alignment);
    doc2.add(&Key::named("folded"), view, Provenance::by("test")).unwrap();
    let id2 = doc2.get_key(&Key::<View>::named("folded")).unwrap().id();
    doc2.add(&Key::named("gold"), tokens(id2, &shifted), Provenance::by("test")).unwrap();
    let m = matches_golden_key(&doc2, &Key::<Tokens>::named("gold"), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 4, what: "span", .. }), "{m:?}");
    has(
        &m.to_string(),
        &["opennlp:tokens@gold item 4", "rust:   bytes 18..24 \"Züric\"", "golden: bytes 18..25 \"Zürich\""],
    );
}

// ---------------------------------------------------------------------
// Malformed files: each refused with its line.
