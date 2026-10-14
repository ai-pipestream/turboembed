//! The detector on a small GIS model written here: ranking and ties,
//! texts without evidence, the annotator, and opening model files.

mod common;

use turbo_analysis::{Analyzer, Annotator, Document, Key, Status};
use turbo_analysis_layers::Language;
use turbo_langdetect::{ContextGenerator, LanguageDetector, LanguageModel, ModelError, NO_EVIDENCE, Ranking, Scratch};
use turbo_maxent::ModelFile;

use common::{MANIFEST, gis_model, zip};

const OUTCOMES: [&str; 3] = ["aaa", "bbb", "ccc"];

/// `x` is evidence for bbb (weight 2), `y` for aaa and ccc (weight 1
/// each), and a lone high surrogate for ccc (weight 5).
fn classifier() -> Vec<u8> {
    let units = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
    gis_model(
        &OUTCOMES,
        &[(units("x"), vec![(1, 2.0)]), (units("y"), vec![(0, 1.0), (2, 1.0)]), (vec![0xD83D], vec![(2, 5.0)])],
    )
}

fn model_file(manifest: &str, entries: &[(&str, &[u8])]) -> ModelFile {
    let mut all: Vec<(&str, &[u8])> = vec![("manifest.properties", manifest.as_bytes())];
    all.extend_from_slice(entries);
    ModelFile::open(zip(&all)).unwrap()
}

fn open(manifest: &str) -> Result<LanguageModel, ModelError> {
    let classifier = classifier();
    LanguageModel::open(model_file(manifest, &[("langdetect.model", &classifier)]))
}

fn detector() -> LanguageDetector {
    LanguageDetector::new(open(MANIFEST).unwrap())
}

/// The ranking of a text as (code, probability), best first.
fn ranking(d: &LanguageDetector, text: &str) -> (Vec<(&'static str, f64)>, bool) {
    let mut ranking = Ranking::new();
    d.predict_languages(text, &mut Scratch::new(), &mut ranking);
    let rows = ranking.outcomes().iter().zip(ranking.probabilities()).map(|(&o, &p)| (OUTCOMES[o as usize], p));
    (rows.collect(), ranking.has_evidence())
}

#[test]
fn the_best_language_comes_first_and_ties_keep_the_outcome_order() {
    let d = detector();
    let e2 = 2f64.exp();
    let (rows, evidence) = ranking(&d, "x");
    assert!(evidence);
    assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<_>>(), ["bbb", "aaa", "ccc"]);
    assert!((rows[0].1 - e2 / (e2 + 2.0)).abs() < 1e-15, "{rows:?}");
    // aaa and ccc have the same sums, so the same bits.
    assert_eq!(rows[1].1.to_bits(), rows[2].1.to_bits());
    assert!((rows.iter().map(|r| r.1).sum::<f64>() - 1.0).abs() < 1e-15);

    // y alone: aaa and ccc tie for first, in outcome order.
    let (rows, _) = ranking(&d, "y");
    assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<_>>(), ["aaa", "ccc", "bbb"]);
    let best = d.predict_language("y", &mut Scratch::new());
    assert_eq!((best.outcome, best.code, best.probability), (0, "aaa", rows[0].1));
    assert!(best.evidence);

    // "x y" has x, y, "x ", " y", " " and "x y": aaa and ccc get 1, bbb 2.
    let (rows, _) = ranking(&d, "x y");
    assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<_>>(), ["bbb", "aaa", "ccc"]);
}

#[test]
fn predict_language_is_the_first_language_predict_languages_ranks() {
    let d = detector();
    let mut scratch = Scratch::new();
    for text in ["x", "y", "xy", "yyy x", "", "q", "\u{1F600}"] {
        let (rows, evidence) = ranking(&d, text);
        let best = d.predict_language(text, &mut scratch);
        assert_eq!((best.code, best.probability.to_bits(), best.evidence), (rows[0].0, rows[0].1.to_bits(), evidence));
    }
}

#[test]
fn without_evidence_every_language_has_the_prior_in_outcome_order() {
    let d = detector();
    // Empty, whitespace, digits, a URL, emoji: the normalizers leave
    // nothing. "q" and "qq z" are n-grams the model does not know.
    for text in ["", "  \n\t ", "1234 5678", "https://example.com/x", "\u{1F600}\u{1F600}", "q", "qq z"] {
        let (rows, evidence) = ranking(&d, text);
        assert!(!evidence, "{text:?}");
        assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<_>>(), OUTCOMES, "{text:?}");
        assert!(rows.iter().all(|r| r.1.to_bits() == rows[0].1.to_bits()), "{text:?}");
        assert!((rows[0].1 - 1.0 / 3.0).abs() < 1e-15);
        let best = d.predict_language(text, &mut Scratch::new());
        assert_eq!((best.code, best.evidence), ("aaa", false));
    }
}

#[test]
fn a_lone_surrogate_n_gram_finds_its_predicate() {
    // The normalizers remove every character outside the BMP before the
    // n-grams are taken, so the detector never sees half of one. The
    // generator alone, on unprepared text, makes the lone halves the JVM
    // would make, and they find the predicate stored for one.
    let d = detector();
    let mut g = ContextGenerator::default();
    g.generate_prepared("\u{1F600}");
    let maxent = d.model().maxent();
    assert!(g.features().any(|f| maxent.has_predicate_modified_utf8(f)));
    let mut p = [0.0; 3];
    maxent.eval_features(g.features(), &mut p);
    assert!(p[2] > p[0] && p[0] == p[1], "{p:?}");
    let (_, evidence) = ranking(&d, "\u{1F600}");
    assert!(!evidence);
}

// The annotator

#[test]
fn the_annotator_writes_the_ranking_or_declines_without_evidence() {
    let analyzer = Analyzer::builder().add(detector()).build().unwrap();
    let mut docs = vec![
        Document::new("x y"),
        Document::new(""),
        Document::new("  \n "),
        Document::new("12 34"),
        Document::new("http://example.com \u{1F600}"),
        Document::new("qqq"),
        Document::new("y"),
    ];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");

    let rows = |d: &Document| {
        let l = d.get::<Language>().unwrap();
        l.iter().map(|(c, p)| (c.to_string(), p)).collect::<Vec<_>>()
    };
    let d = detector();
    for (doc, text) in [(&docs[0], "x y"), (&docs[6], "y")] {
        let (expected, _) = ranking(&d, text);
        let expected: Vec<(String, f64)> = expected.iter().map(|&(c, p)| (c.to_string(), p)).collect();
        assert_eq!(rows(doc), expected);
        assert_eq!(doc.status(Key::<Language>::DEFAULT.id()), Some(&Status::Complete));
    }
    for doc in &docs[1..6] {
        assert!(doc.get::<Language>().is_none());
        assert_eq!(doc.status(Key::<Language>::DEFAULT.id()), Some(&Status::Skipped(NO_EVIDENCE.into())));
    }
    assert_eq!(NO_EVIDENCE, "no language evidence");
}

#[test]
fn the_annotator_names_its_model() {
    let classifier = classifier();
    let file = model_file(MANIFEST, &[("langdetect.model", &classifier)]);
    let sha256 = file.sha256().to_hex();
    let d = LanguageDetector::new(LanguageModel::open(file).unwrap());
    let model = d.model().model_ref();
    assert_eq!(model.id, "opennlp:LanguageDetectorME:und");
    assert_eq!(model.revision, "3.0.0-SNAPSHOT");
    assert_eq!(model.sha256, sha256);
    assert_eq!(Annotator::model(&d).as_ref(), Some(model));
    assert_eq!(d.model().supported_languages().collect::<Vec<_>>(), OUTCOMES);
}

// Opening models

#[test]
fn a_model_without_a_factory_gets_the_default_one() {
    let manifest = MANIFEST.replace("factory=opennlp.tools.langdetect.LanguageDetectorFactory\n", "");
    assert!(!manifest.contains("factory"));
    assert!(open(&manifest).is_ok());
}

#[test]
fn other_components_and_factories_are_refused() {
    let manifest = MANIFEST.replace("Component-Name=LanguageDetectorME", "Component-Name=TokenizerME");
    let e = open(&manifest).unwrap_err();
    assert!(matches!(&e, ModelError::Component(c) if c == "TokenizerME"), "{e:?}");
    assert!(e.to_string().contains("\"TokenizerME\" is not LanguageDetectorME"), "{e}");

    let manifest = MANIFEST.replace("langdetect.LanguageDetectorFactory", "langdetect.MyFactory");
    let e = open(&manifest).unwrap_err();
    assert!(matches!(&e, ModelError::Factory(c) if c == "opennlp.tools.langdetect.MyFactory"), "{e:?}");
}

#[test]
fn a_model_with_an_outcome_holding_half_a_surrogate_pair_is_refused() {
    // The outcome "ccc" rewritten in place to a lone high surrogate, as
    // writeUTF writes it: three bytes, so the length stays 3.
    let mut classifier = classifier();
    let at = classifier.windows(5).position(|w| w == b"\0\x03ccc").unwrap();
    classifier[at + 2..at + 5].copy_from_slice(b"\xED\xA0\xBD");
    let e = LanguageModel::open(model_file(MANIFEST, &[("langdetect.model", &classifier)])).unwrap_err();
    let lone = turbo_maxent::JavaString::from_utf16(&[0xD83D]);
    assert!(matches!(&e, ModelError::OutcomeNotText(o) if *o == lone), "{e:?}");
    assert!(e.to_string().contains("holds half of a surrogate pair"), "{e}");
}

#[test]
fn a_model_without_its_classifier_is_refused() {
    let e = LanguageModel::open(model_file(MANIFEST, &[])).unwrap_err();
    assert!(matches!(e, ModelError::File(_)), "{e:?}");
    assert!(e.to_string().contains("langdetect.model"), "{e}");
}
