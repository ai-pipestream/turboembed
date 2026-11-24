//! Edge cases of the comparison against `tests/fixtures/zurich.golden`:
//! absolute tolerances at their boundary, reports for aligned layers,
//! the language layer's several rows, empty golden layers and the shape
//! of every report.

#[path = "compare_edge_cases/layers.rs"]
mod layers;
#[path = "compare_edge_cases/reports.rs"]
mod reports;

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
    language: Vec<(&'static str, f64)>,
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
            language: vec![("eng", 0.9312345678901234), ("deu", 0.0412), ("nld", 0.012), ("fra", 1.0e-5)],
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
        let mut code = Strs::default();
        for &(c, _) in &self.language {
            code.push(c);
        }
        let score = self.language.iter().map(|&(_, s)| s).collect();
        doc.add(&Key::DEFAULT, Language { code, score }, by()).unwrap();
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
        says(&m.to_string(), &["turbo:pos item 4: the prob differs", "tolerance 1e-9", "(by "]);
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
#[should_panic(expected = "turbo:tokens@x: the document has no such layer")]
fn assert_matches_golden_key_names_the_instance() {
    assert_matches_golden_key(&Document::new(TEXT), &Key::<Tokens>::named("x"), &golden(), Tolerance::exact());
}
