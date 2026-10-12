//! Edge cases of the comparison against `tests/fixtures/zurich.golden`:
//! absolute tolerances at their boundary, reports for aligned layers,
//! the language layer's several rows, empty golden layers and the shape
//! of every report.

use std::path::PathBuf;

use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Confidence, Entities, Language, Lemmas, PosTags, Sentences, Strs, TagSet, Tokens};
use turbo_goldens::*;

const TEXT: &str = "Dr. Smith went to Zürich. It rained!";

fn fixture_text() -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zurich.golden")).unwrap()
}

fn golden() -> Golden {
    Golden::parse(&fixture_text()).unwrap()
}

/// The fixture with `from` replaced by `to` (which must occur).
fn golden_with(from: &str, to: &str) -> Golden {
    let text = fixture_text();
    assert!(text.contains(from), "{from:?} is not in the fixture");
    Golden::parse(&text.replacen(from, to, 1)).unwrap()
}

/// The fixture's layers as plain values, so a test can change one.
struct Parts {
    sentences: Vec<(u32, u32)>,
    sentence_conf: Vec<f64>,
    tokens: Vec<(u32, u32)>,
    token_conf: Vec<f64>,
    tags: Vec<&'static str>,
    prob: Option<Vec<f64>>,
    lemmas: Vec<&'static str>,
    entities: Vec<(u32, u32, u32, u32, &'static str, f64)>,
    language: (&'static str, f64),
}

impl Parts {
    fn matching() -> Parts {
        Parts {
            sentences: vec![(0, 26), (27, 37)],
            sentence_conf: vec![0.9876543210987654, 0.95],
            tokens: vec![(0, 3), (4, 9), (10, 14), (15, 17), (18, 25), (25, 26), (27, 29), (30, 36), (36, 37)],
            token_conf: vec![0.9991, 0.99842, 0.9999, 0.98, 0.97321, 2.0 / 3.0, 0.999, 0.9935, 1.0],
            tags: vec!["PROPN", "PROPN", "VERB", "ADP", "PROPN", "PUNCT", "PRON", "VERB", "PUNCT"],
            prob: Some(vec![0.9712, 0.9554, 0.987, 0.9933, 0.8123, 0.99, 0.991, 0.9601, 0.99]),
            lemmas: vec!["Dr.", "Smith", "go", "to", "Zürich", ".", "it", "rain", "!"],
            entities: vec![(4, 9, 1, 2, "person", 0.8765432), (18, 25, 4, 5, "location", 0.7123)],
            language: ("eng", 0.9312345678901234),
        }
    }

    fn doc(&self) -> Document {
        let mut doc = Document::new(TEXT);
        let by = || Provenance::by("test");
        doc.add(&Key::DEFAULT, Sentences { spans: spans(&self.sentences) }, by()).unwrap();
        doc.add(&Key::DEFAULT, Confidence::<Sentences>::new(self.sentence_conf.clone()), by()).unwrap();
        doc.add(&Key::DEFAULT, Tokens { spans: spans(&self.tokens), sentence: None }, by()).unwrap();
        doc.add(&Key::DEFAULT, Confidence::<Tokens>::new(self.token_conf.clone()), by()).unwrap();
        let tagset = TagSet::new("ud", &["PROPN", "VERB", "ADP", "PUNCT", "PRON", "NOUN"]);
        let tags = self.tags.iter().map(|t| tagset.tag(t).unwrap()).collect();
        doc.add(&Key::DEFAULT, PosTags { tagset, tags, prob: self.prob.clone() }, by()).unwrap();
        let mut lemma = Strs::default();
        for l in &self.lemmas {
            lemma.push(l);
        }
        doc.add(&Key::DEFAULT, Lemmas { lemma }, by()).unwrap();
        let kinds = TagSet::new("kinds", &["person", "location", "organization"]);
        let mut sp = Spans::new(ViewId::SOURCE);
        let (mut toks, mut kind, mut score) = (Vec::new(), Vec::new(), Vec::new());
        for &(s, e, ts, te, k, p) in &self.entities {
            sp.push(Span::new(s, e));
            toks.push(IxRange::new(ts, te));
            kind.push(kinds.tag(k).unwrap());
            score.push(p);
        }
        doc.add(&Key::DEFAULT, Entities { spans: sp, tokens: toks, kind, kinds, score }, by()).unwrap();
        let (code, score) = self.language;
        doc.add(&Key::DEFAULT, Language { code: code.into(), score }, by()).unwrap();
        doc
    }
}

fn spans<L>(items: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in items {
        s.push(Span::new(a, b));
    }
    s
}

fn fail<L: GoldenCompare>(p: &Parts, g: &Golden, tol: Tolerance) -> Mismatch {
    matches_golden::<L>(&p.doc(), g, tol).expect_err("the layer should not match")
}

fn says(shown: &str, parts: &[&str]) {
    for p in parts {
        assert!(shown.contains(p), "{p:?} is not in the report:\n{shown}");
    }
}

fn next_down(x: f64) -> f64 {
    f64::from_bits(x.to_bits() - 1)
}

// ---------------------------------------------------------------------
// Absolute tolerance on the POS probabilities

#[test]
fn pos_probabilities_within_and_outside_an_absolute_distance() {
    let g = golden();
    let tol = Tolerance::abs(1e-9);
    // Row 4's probability is 0.8123 in the golden.
    for delta in [0.0, 5e-10, -5e-10, 9e-10, -9e-10] {
        let mut p = Parts::matching();
        p.prob.as_mut().unwrap()[4] = 0.8123 + delta;
        assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, tol), Ok(()), "delta {delta:e}");
    }
    for delta in [2e-9, -2e-9, 1e-3, -0.5, 1.0] {
        let mut p = Parts::matching();
        p.prob.as_mut().unwrap()[4] = 0.8123 + delta;
        let m = fail::<PosTags>(&p, &g, tol);
        assert!(matches!(m, Mismatch::Item { index: 4, what: "prob", .. }), "delta {delta:e}: {m:?}");
        says(&m.to_string(), &["opennlp:pos item 4: the prob differs", "tolerance 1e-9", "(by "]);
    }
    // Exact fails on the smallest change that the absolute tolerance accepts.
    let mut p = Parts::matching();
    p.prob.as_mut().unwrap()[4] = f64::from_bits(0.8123f64.to_bits() + 1);
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, tol), Ok(()));
    let m = fail::<PosTags>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 4, what: "prob", .. }), "{m:?}");
}

#[test]
fn the_distance_itself_is_accepted_and_one_unit_less_is_not() {
    let g = golden();
    let mut p = Parts::matching();
    let rust = 0.8123 + 3e-7;
    p.prob.as_mut().unwrap()[4] = rust;
    let distance = (rust - 0.8123f64).abs();
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::abs(distance)), Ok(()));
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(next_down(distance)));
    assert!(matches!(m, Mismatch::Item { index: 4, what: "prob", .. }), "{m:?}");
    // Zero distance is the same as exact for finite values, except that it
    // does not tell 0.0 from -0.0.
    assert_eq!(matches_golden::<PosTags>(&Parts::matching().doc(), &g, Tolerance::abs(0.0)), Ok(()));
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(0.0));
    assert!(matches!(m, Mismatch::Item { what: "prob", .. }), "{m:?}");
}

#[test]
fn a_tolerance_does_not_loosen_tags_or_spans() {
    let g = golden();
    let mut p = Parts::matching();
    p.tags[2] = "NOUN";
    p.prob.as_mut().unwrap()[2] += 0.5;
    // Both the tag and the probability differ; the tag is reported, since
    // labels are compared before values.
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(1.0));
    assert!(matches!(m, Mismatch::Item { index: 2, what: "tag", .. }), "{m:?}");
    // With the same tag the probability is within a distance of 1.
    let mut p = Parts::matching();
    p.prob.as_mut().unwrap()[2] -= 0.5;
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::abs(1.0)), Ok(()));
    // A span differs by one byte: no tolerance accepts it.
    let mut p = Parts::matching();
    p.tokens[1] = (4, 10);
    let m = fail::<Tokens>(&p, &g, Tolerance::abs(f64::MAX));
    assert!(matches!(m, Mismatch::Item { index: 1, what: "span", .. }), "{m:?}");
}

#[test]
fn only_the_first_difference_is_reported() {
    let g = golden();
    let mut p = Parts::matching();
    p.tokens[5] = (25, 27);
    p.tokens[2] = (10, 15);
    p.tokens[7] = (30, 35);
    let m = fail::<Tokens>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 2, .. }), "{m:?}");
    assert_eq!(m.to_string().matches("item").count(), 1, "{m}");
    // Within one item the span comes before the label: an entity whose span
    // and kind both differ reports the span.
    let mut p = Parts::matching();
    p.entities[0] = (4, 8, 1, 2, "organization", 0.1);
    let m = fail::<Entities>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 0, what: "span", .. }), "{m:?}");
    // Then the token range, then the kind, then the probability.
    p.entities[0] = (4, 9, 1, 3, "organization", 0.1);
    assert!(matches!(fail::<Entities>(&p, &g, Tolerance::exact()), Mismatch::Item { what: "tokens", .. }));
    p.entities[0] = (4, 9, 1, 2, "organization", 0.1);
    assert!(matches!(fail::<Entities>(&p, &g, Tolerance::exact()), Mismatch::Item { what: "kind", .. }));
    p.entities[0] = (4, 9, 1, 2, "person", 0.1);
    assert!(matches!(fail::<Entities>(&p, &g, Tolerance::exact()), Mismatch::Item { what: "prob", .. }));
}

// ---------------------------------------------------------------------
// Aligned layers report the token they are for

#[test]
fn a_mismatch_in_an_aligned_layer_shows_the_anchor_token_on_both_sides() {
    let g = golden();
    let mut p = Parts::matching();
    p.prob.as_mut().unwrap()[4] = 0.5;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(1e-9));
    says(
        &m.to_string(),
        &[
            "rust:   bytes 18..25 \"Zürich\", tag \"PROPN\", prob 3fe0000000000000 (0.5)",
            "golden: bytes 18..25 \"Zürich\", tag \"PROPN\", prob 3fe9fe5c91d14e3c (0.8123)",
        ],
    );
    // The confidence of a sentence shows the sentence.
    let mut p = Parts::matching();
    p.sentence_conf[1] = 0.5;
    let m = fail::<Confidence<Sentences>>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 1, what: "value", .. }), "{m:?}");
    says(
        &m.to_string(),
        &[
            "turbo:confidence<opennlp:sentences> item 1: the value differs",
            "rust:   bytes 27..37 \"It rained!\", value 3fe0000000000000 (0.5)",
            "golden: bytes 27..37 \"It rained!\", value 3fee666666666666 (0.95)",
        ],
    );
    // The lemma of the first token, which has a period in it.
    let mut p = Parts::matching();
    p.lemmas[0] = "Dr";
    let m = fail::<Lemmas>(&p, &g, Tolerance::exact());
    says(&m.to_string(), &["opennlp:lemmas item 0", "bytes 0..3 \"Dr.\", lemma \"Dr\"", "lemma \"Dr.\""]);
    // The last one, at the end of the text.
    let mut p = Parts::matching();
    p.lemmas[8] = "?";
    let m = fail::<Lemmas>(&p, &g, Tolerance::exact());
    says(&m.to_string(), &["item 8", "bytes 36..37 \"!\""]);
}

#[test]
fn an_aligned_report_takes_its_token_from_the_rust_side_of_the_document() {
    // The Rust tokens differ from the golden's, but the tags are compared
    // by index: the tag differs, and each side shows its own token.
    let g = golden();
    let mut p = Parts::matching();
    p.tokens[4] = (18, 24);
    p.tags[4] = "NOUN";
    let m = fail::<PosTags>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 4, what: "tag", .. }), "{m:?}");
    says(&m.to_string(), &["rust:   bytes 18..24 \"Züric\"", "golden: bytes 18..25 \"Zürich\""]);
}

/// The fixture's token layers cut to `n` tokens (and no entities, which
/// refer to tokens).
fn cut_tokens(p: &mut Parts, n: usize) {
    p.tokens.truncate(n);
    p.token_conf.truncate(n);
    p.tags.truncate(n);
    p.prob.as_mut().unwrap().truncate(n);
    p.lemmas.truncate(n);
    p.entities.clear();
}

#[test]
fn counts_report_the_first_item_past_the_shorter_side() {
    let g = golden();
    // Rust has one token too many.
    let mut p = Parts::matching();
    p.tokens.push((37, 37));
    p.token_conf.push(0.5);
    p.tags.push("PUNCT");
    p.prob.as_mut().unwrap().push(0.5);
    p.lemmas.push("");
    let m = fail::<Tokens>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 10, golden: 9, .. }), "{m:?}");
    says(
        &m.to_string(),
        &["opennlp:tokens: 10 items in Rust, 9 in the golden", "Rust item 9:", "Rust: bytes 37..37 \"\""],
    );
    // The aligned layers report their counts the same way.
    let m = fail::<PosTags>(&p, &g, Tolerance::exact());
    says(
        &m.to_string(),
        &["opennlp:pos: 10 items in Rust, 9 in the golden", "Rust item 9:", "bytes 37..37 \"\", tag \"PUNCT\""],
    );
    // Rust has fewer; the first golden row past it is shown.
    let mut p = Parts::matching();
    cut_tokens(&mut p, 3);
    let m = fail::<Tokens>(&p, &g, Tolerance::exact());
    says(&m.to_string(), &["3 items in Rust, 9 in the golden", "golden item 3:", "golden: bytes 15..17 \"to\""]);
    let m = fail::<Lemmas>(&p, &g, Tolerance::exact());
    says(
        &m.to_string(),
        &["opennlp:lemmas: 3 items in Rust, 9 in the golden", "golden: bytes 15..17 \"to\", lemma \"to\""],
    );
    // Rust has none at all.
    let mut p = Parts::matching();
    p.sentences.clear();
    p.sentence_conf.clear();
    let m = fail::<Sentences>(&p, &g, Tolerance::exact());
    says(&m.to_string(), &["0 items in Rust, 2 in the golden", "golden item 0:", "golden: bytes 0..26"]);
    // A difference before the end wins over a count difference.
    let mut p = Parts::matching();
    cut_tokens(&mut p, 3);
    p.tokens[1] = (4, 8);
    assert!(matches!(fail::<Tokens>(&p, &g, Tolerance::exact()), Mismatch::Item { index: 1, .. }));
}

#[test]
fn a_golden_span_outside_character_boundaries_is_reported_without_a_panic() {
    // The golden's token 4 cuts the two bytes of "ü" (bytes 19..21) in the middle.
    let g = golden_with("18\t25\n25\t26", "18\t20\n25\t26");
    let m = fail::<Tokens>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 4, what: "span", .. }), "{m:?}");
    says(
        &m.to_string(),
        &["rust:   bytes 18..25 \"Zürich\"", "golden: bytes 18..20 (outside the text or not on character boundaries)"],
    );
}

// ---------------------------------------------------------------------
// Language

fn language_golden(rows: &[&str]) -> Golden {
    let text = fixture_text();
    let start = text.find("layer\topennlp:language").unwrap();
    let mut out = text[..start].to_string();
    out.push_str(&format!("layer\topennlp:language\tdocument\t{}\tcode:str\tconfidence:f64\n", rows.len()));
    for r in rows {
        out.push_str(r);
        out.push('\n');
    }
    out.push_str("end\topennlp:language\n");
    Golden::parse(&out).unwrap()
}

#[test]
fn language_compares_row_zero_only() {
    let mut p = Parts::matching();
    let eng = "eng\t3fedccac6fbfc4c9 0.9312345678901234";
    // The fixture has four rows; the best one is the Rust layer's.
    assert_eq!(golden().layer("opennlp:language").unwrap().len(), 4);
    assert_eq!(matches_golden::<Language>(&p.doc(), &golden(), Tolerance::exact()), Ok(()));
    // Rows after row 0 may hold anything, including the Rust layer's own code again,
    // other codes, NaN and rows out of order of confidence.
    let g = language_golden(&[
        eng,
        "eng\t3fedccac6fbfc4c9 0.9312345678901234",
        "xx\t7ff8000000000000 NaN",
        "yy\tffffffffffffffff NaN",
    ]);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    // A golden with a single row.
    let g = language_golden(&[eng]);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    // Row 0 differs while the Rust code is found in a later row: not a match.
    let g = language_golden(&["deu\t3fedccac6fbfc4c9 0.93", "eng\t3fedccac6fbfc4c9 0.9312345678901234"]);
    let m = fail::<Language>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 0, what: "code", .. }), "{m:?}");
    says(&m.to_string(), &["opennlp:language item 0", "rust:   code \"eng\"", "golden: code \"deu\""]);
    // Row 0's confidence is compared, with the tolerance.
    p.language.1 += 1e-12;
    let g = language_golden(&[eng, "deu\t3fa5182a9930be0e 0.0412"]);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()));
    let m = fail::<Language>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 0, what: "confidence", .. }), "{m:?}");
    // A golden with no rows has no best language: a count difference.
    let g = language_golden(&[]);
    let m = fail::<Language>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 1, golden: 0, .. }), "{m:?}");
    says(&m.to_string(), &["opennlp:language: 1 items in Rust, 0 in the golden", "Rust item 0:", "Rust: code \"eng\""]);
}

#[test]
fn language_with_unusual_values() {
    let eng = |bits: &str| format!("eng\t{bits} x");
    let mut p = Parts::matching();
    // A NaN confidence matches only the same bits at exact, and any NaN at a tolerance.
    p.language.1 = f64::from_bits(0x7ff8000000000000);
    let g = language_golden(&[&eng("7ff8000000000000")]);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    let g = language_golden(&[&eng("fff8000000000000")]);
    assert!(matches!(fail::<Language>(&p, &g, Tolerance::exact()), Mismatch::Item { what: "confidence", .. }));
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()));
    // Infinities: the same sign matches, the other does not, and a finite value is not near one.
    p.language.1 = f64::INFINITY;
    let g = language_golden(&[&eng("7ff0000000000000")]);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()));
    let g = language_golden(&[&eng("fff0000000000000")]);
    assert!(matches!(fail::<Language>(&p, &g, Tolerance::abs(f64::MAX)), Mismatch::Item { what: "confidence", .. }));
    // -0.0 and 0.0.
    p.language.1 = 0.0;
    let g = language_golden(&[&eng("8000000000000000")]);
    assert!(matches!(fail::<Language>(&p, &g, Tolerance::exact()), Mismatch::Item { what: "confidence", .. }));
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::abs(0.0)), Ok(()));
    // An empty code is a code.
    let mut p = Parts::matching();
    p.language.0 = "";
    let g = language_golden(&["\t3fedccac6fbfc4c9 x"]);
    p.language.1 = f64::from_bits(0x3fedccac6fbfc4c9);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::exact()), Ok(()));
}

// ---------------------------------------------------------------------
// Reports: the layer id and the row index, in every variant

#[test]
fn every_item_report_names_its_layer_and_row() {
    let g = golden();
    let cases: Vec<(&str, Mismatch)> = {
        let mut cases = Vec::new();
        let mut p = Parts::matching();
        p.sentences[1] = (27, 36);
        cases.push(("opennlp:sentences", fail::<Sentences>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.sentence_conf[0] = 0.0;
        cases.push(("turbo:confidence<opennlp:sentences>", fail::<Confidence<Sentences>>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.tokens[7] = (30, 35);
        cases.push(("opennlp:tokens", fail::<Tokens>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.token_conf[7] = 0.0;
        cases.push(("turbo:confidence<opennlp:tokens>", fail::<Confidence<Tokens>>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.tags[7] = "NOUN";
        cases.push(("opennlp:pos", fail::<PosTags>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.lemmas[7] = "x";
        cases.push(("opennlp:lemmas", fail::<Lemmas>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.entities[1].5 = 0.0;
        cases.push(("opennlp:entities", fail::<Entities>(&p, &g, Tolerance::exact())));
        let mut p = Parts::matching();
        p.language.0 = "xx";
        cases.push(("opennlp:language", fail::<Language>(&p, &g, Tolerance::exact())));
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
    no_lemmas.layers.retain(|l| l.id != "opennlp:lemmas");
    reports.push(matches_golden::<Lemmas>(&Parts::matching().doc(), &no_lemmas, Tolerance::exact()).unwrap_err());
    // MissingLayer, plain and for a named instance.
    let doc = Document::new(TEXT);
    reports.push(matches_golden::<Tokens>(&doc, &g, Tolerance::exact()).unwrap_err());
    reports.push(matches_golden_key(&doc, &Key::<Tokens>::named("x"), &g, Tolerance::exact()).unwrap_err());
    // Anchor.
    let mut anchored = golden();
    anchored.layers.iter_mut().find(|l| l.id == "opennlp:tokens").unwrap().anchor = GoldenAnchor::Document;
    reports.push(matches_golden::<Tokens>(&Parts::matching().doc(), &anchored, Tolerance::exact()).unwrap_err());
    // Columns.
    let mut renamed = golden();
    renamed.layers.iter_mut().find(|l| l.id == "opennlp:lemmas").unwrap().columns[0].name = "x".into();
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
        "opennlp:tokens",
        "opennlp:lemmas",
        "opennlp:tokens",
        "opennlp:tokens@x",
        "opennlp:tokens",
        "opennlp:lemmas",
        "opennlp:pos",
        "opennlp:sentences",
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
    says(&reports[1].to_string(), &["[opennlp:sentences, turbo:confidence<opennlp:sentences>, opennlp:tokens"]);
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
    assert_eq!(m.layer(), "opennlp:tokens@mine");
    says(
        &m.to_string(),
        &["opennlp:tokens@mine item 8: the span differs", "rust:   bytes 35..37 \"d!\"", "golden: bytes 36..37 \"!\""],
    );
}

// ---------------------------------------------------------------------
// Golden layers that are empty or inconsistent

#[test]
fn an_empty_golden_layer_is_a_count_difference_not_a_pass() {
    let text = fixture_text();
    let sentences = "layer\topennlp:sentences\ttext\t2\tstart:u32\tend:u32\n0\t26\n27\t37\nend\topennlp:sentences\n";
    assert!(text.contains(sentences));
    // The confidence layer is aligned to it, so empty both.
    let conf = "layer\tturbo:confidence<opennlp:sentences>\taligned:opennlp:sentences\t2\tvalue:f64\n3fef9add3c1be9b8 0.9876543210987654\n3fee666666666666 0.95\nend\tturbo:confidence<opennlp:sentences>\n";
    assert!(text.contains(conf));
    let emptied = text
        .replace(sentences, "layer\topennlp:sentences\ttext\t0\tstart:u32\tend:u32\nend\topennlp:sentences\n")
        .replace(
            conf,
            "layer\tturbo:confidence<opennlp:sentences>\taligned:opennlp:sentences\t0\tvalue:f64\nend\tturbo:confidence<opennlp:sentences>\n",
        );
    let g = Golden::parse(&emptied).unwrap();
    let m = fail::<Sentences>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 2, golden: 0, .. }), "{m:?}");
    says(
        &m.to_string(),
        &["2 items in Rust, 0 in the golden", "Rust item 0:", "Rust: bytes 0..26 \"Dr. Smith went to Zürich.\""],
    );
    let m = fail::<Confidence<Sentences>>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 2, golden: 0, .. }), "{m:?}");
    // A Rust layer that is empty too matches the empty golden.
    let mut p = Parts::matching();
    p.sentences.clear();
    p.sentence_conf.clear();
    assert_eq!(matches_golden::<Sentences>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    assert_eq!(matches_golden::<Confidence<Sentences>>(&p.doc(), &g, Tolerance::exact()), Ok(()));
}

#[test]
fn a_golden_whose_columns_have_different_lengths_is_a_columns_mismatch() {
    let mut g = golden();
    let pos = g.layers.iter_mut().find(|l| l.id == "opennlp:pos").unwrap();
    pos.columns[1].data = ColumnData::F64(vec![0.5; 8]);
    let m = fail::<PosTags>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Columns { .. }), "{m:?}");
    says(
        &m.to_string(),
        &["opennlp:pos: the golden layer has the columns [tag:str, prob:f64], expected [tag:str, prob:f64]"],
    );
}

#[test]
fn a_golden_layer_with_extra_or_missing_columns() {
    let mut g = golden();
    let toks = g.layers.iter_mut().find(|l| l.id == "opennlp:tokens").unwrap();
    toks.columns.push(Column { name: "extra".into(), data: ColumnData::U32(vec![0; 9]) });
    let m = fail::<Tokens>(&Parts::matching(), &g, Tolerance::exact());
    says(&m.to_string(), &["[start:u32, end:u32, extra:u32], expected [start:u32, end:u32]"]);
    let mut g = golden();
    let toks = g.layers.iter_mut().find(|l| l.id == "opennlp:tokens").unwrap();
    toks.columns.pop();
    let m = fail::<Tokens>(&Parts::matching(), &g, Tolerance::exact());
    says(&m.to_string(), &["[start:u32], expected [start:u32, end:u32]"]);
    // Swapped types.
    let mut g = golden();
    let toks = g.layers.iter_mut().find(|l| l.id == "opennlp:tokens").unwrap();
    toks.columns[0].data = ColumnData::F64(vec![0.0; 9]);
    let m = fail::<Tokens>(&Parts::matching(), &g, Tolerance::exact());
    says(&m.to_string(), &["[start:f64, end:u32]"]);
}

const POS0: &str = "3fef141205bc01a3 0.9712";

#[test]
fn a_golden_with_a_nan_value_compares_by_bits_unless_given_a_tolerance() {
    let g = golden_with(POS0, "7ff8000000000000 NaN");
    let mut p = Parts::matching();
    p.prob.as_mut().unwrap()[0] = f64::NAN;
    assert_eq!(f64::NAN.to_bits(), 0x7ff8000000000000);
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    // A NaN with another sign or payload differs at exact bits.
    for bits in [0xfff8000000000000u64, 0x7ff8000000000001, 0x7ff0000000000001] {
        p.prob.as_mut().unwrap()[0] = f64::from_bits(bits);
        let m = fail::<PosTags>(&p, &g, Tolerance::exact());
        assert!(matches!(m, Mismatch::Item { index: 0, what: "prob", .. }), "{bits:x}: {m:?}");
        assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()), "{bits:x}");
    }
    // A number is not a NaN, within any distance, and the report shows the golden's NaN.
    p.prob.as_mut().unwrap()[0] = 0.9712;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(f64::MAX));
    assert!(matches!(m, Mismatch::Item { index: 0, what: "prob", .. }), "{m:?}");
    says(&m.to_string(), &["golden: bytes 0..3 \"Dr.\", tag \"PROPN\", prob 7ff8000000000000 (NaN)"]);
    // A golden number against a Rust NaN.
    let g = golden();
    p.prob.as_mut().unwrap()[0] = f64::NAN;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(f64::MAX));
    assert!(matches!(m, Mismatch::Item { index: 0, what: "prob", .. }), "{m:?}");
}

#[test]
fn a_golden_with_an_infinite_value() {
    let g = golden_with(POS0, "7ff0000000000000 Infinity");
    let mut p = Parts::matching();
    p.prob.as_mut().unwrap()[0] = f64::INFINITY;
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()));
    p.prob.as_mut().unwrap()[0] = f64::MAX;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(f64::MAX));
    assert!(matches!(m, Mismatch::Item { index: 0, .. }), "{m:?}");
    p.prob.as_mut().unwrap()[0] = f64::NEG_INFINITY;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(1e300));
    assert!(matches!(m, Mismatch::Item { index: 0, .. }), "{m:?}");
}

// ---------------------------------------------------------------------
// Tolerance

#[test]
fn tolerance_constructors_and_display() {
    assert_eq!(Tolerance::exact(), Tolerance::Exact);
    assert_eq!(Tolerance::abs(0.5), Tolerance::Abs(0.5));
    assert_eq!(Tolerance::exact().to_string(), "exact bits");
    assert_eq!(Tolerance::abs(1e-9).to_string(), "1e-9");
    assert_eq!(Tolerance::abs(0.0).to_string(), "0e0");
    assert_eq!(Tolerance::abs(0.5).to_string(), "5e-1");
    // The largest finite distance is allowed; nothing finite is farther than it unless
    // the subtraction overflows, which is then not within it.
    let t = Tolerance::abs(f64::MAX);
    assert!(t.accepts(f64::MAX, 0.0));
    assert!(!t.accepts(f64::MAX, -f64::MAX));
    assert!(t.accepts(1.0, -1.0));
}

#[test]
#[should_panic(expected = "finite and not negative")]
fn a_nan_tolerance_is_refused() {
    Tolerance::abs(f64::NAN);
}

#[test]
#[should_panic(expected = "finite and not negative")]
fn an_infinite_tolerance_is_refused() {
    Tolerance::abs(f64::INFINITY);
}

#[test]
fn negative_zero_is_a_valid_distance() {
    // -0.0 is not negative by the `>=` test, and acts as 0.
    let t = Tolerance::abs(-0.0);
    assert!(t.accepts(1.0, 1.0));
    assert!(!t.accepts(1.0, f64::from_bits(1.0f64.to_bits() + 1)));
}

#[test]
#[should_panic(expected = "layer does not match its golden")]
fn assert_matches_golden_panics_with_the_report() {
    let mut p = Parts::matching();
    p.tokens[1] = (4, 10);
    assert_matches_golden::<Tokens>(&p.doc(), &golden(), Tolerance::exact());
}

#[test]
#[should_panic(expected = "opennlp:tokens@x: the document has no such layer")]
fn assert_matches_golden_key_names_the_instance() {
    assert_matches_golden_key(&Document::new(TEXT), &Key::<Tokens>::named("x"), &golden(), Tolerance::exact());
}
