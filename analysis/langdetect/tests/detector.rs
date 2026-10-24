//! The detector on a small GIS model written here: ranking and ties,
//! texts without evidence, the annotator, and opening model files.

mod common;

use turbo_analysis::{Analyzer, Annotator, Document, Key, Status};
use turbo_analysis_layers::Language;
use turbo_langdetect::{
    ContextGenerator, LanguageDetector, LanguageModel, ModelError, NO_EVIDENCE, ProbingConfig, ProbingContext, Ranking,
    Scratch,
};
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
        assert_eq!(doc.provenance(Key::<Language>::DEFAULT.id()).unwrap().version, "2");
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

#[test]
fn models_with_empty_or_duplicate_outcomes_are_refused() {
    let units = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
    let empty_cls = gis_model(&["aaa", ""], &[(units("x"), vec![(0, 1.0)])]);
    let e = LanguageModel::open(model_file(MANIFEST, &[("langdetect.model", &empty_cls)])).unwrap_err();
    assert!(matches!(e, ModelError::EmptyOutcome), "{e:?}");
    assert!(e.to_string().contains("empty string"), "{e}");

    let dup_cls = gis_model(&["aaa", "bbb", "aaa"], &[(units("x"), vec![(0, 1.0)])]);
    let e = LanguageModel::open(model_file(MANIFEST, &[("langdetect.model", &dup_cls)])).unwrap_err();
    assert!(matches!(&e, ModelError::DuplicateOutcome(c) if c == "aaa"), "{e:?}");
    assert!(e.to_string().contains("duplicate outcome \"aaa\""), "{e}");
}

#[test]
#[should_panic(expected = "outcome index 99 out of bounds for 3 outcomes")]
fn code_out_of_bounds_reports_bounds_panic_and_try_code_returns_none() {
    let d = detector();
    assert_eq!(d.model().try_code(0), Some("aaa"));
    assert_eq!(d.model().try_code(3), None);
    let _ = d.model().code(99);
}

#[test]
fn shrink_excess_reclaims_large_scratch_buffers() {
    let d = detector();
    let mut scratch = Scratch::new();
    let big: String = (0..25_000).map(|i| char::from_u32(0x0100 + (i % 512)).unwrap()).collect();
    let _ = d.predict_language(&big, &mut scratch);
    scratch.shrink_excess();
    let best = d.predict_language("x", &mut scratch);
    assert_eq!(best.code, "bbb");
}

#[test]
fn cleanup_keeps_feature_iterators_valid_and_supports_reuse() {
    let snapshots = |g: &ContextGenerator| {
        (
            g.features().map(<[u8]>::to_vec).collect::<Vec<_>>(),
            g.training_features().map(<[u8]>::to_vec).collect::<Vec<_>>(),
        )
    };
    let mut g = ContextGenerator::new(1, 3);
    for text in [
        (0..4000).map(|i| char::from_u32(0x4e00 + i).unwrap()).collect(),
        "a".repeat(20_000),
        (0..6000).map(|i| char::from_u32(0x4e00 + i).unwrap()).collect(),
    ] {
        g.generate_prepared(&text);
        g.shrink_excess();
        let (features, training) = snapshots(&g);
        assert!(features.is_empty() && training.is_empty());
        assert!(g.is_empty());
        assert!(g.exact_order() && g.training_exact_order());
        g.shrink_excess();
        g.generate_prepared("x y");
        let mut fresh = ContextGenerator::new(1, 3);
        fresh.generate_prepared("x y");
        assert_eq!(snapshots(&g), snapshots(&fresh));
        g.shrink_excess();
        assert_eq!(snapshots(&g), snapshots(&fresh));
    }
}

#[test]
fn probing_stops_early_when_top_language_and_margin_stabilize() {
    let d = detector();
    let mut scratch = Scratch::new();
    let mut out = Ranking::new();
    let cfg = ProbingConfig { max_length: 100, chunk_size: 2, min_consec_improvements: 2, min_diff: 0.20 };
    // Three 2-char chunks ("x ", "x ", "x ") give 3 consecutive evaluations
    // with "bbb" first, non-decreasing confidence, and margin >= 0.20, so
    // probing stops at 6 code points out of 14.
    let len = d.probing_predict_languages_with("x x x x x x x ", cfg, &mut scratch, &mut out);
    assert_eq!(len, 6);
    assert_eq!(d.model().code(out.outcomes()[0] as usize), "bbb");
    assert!(out.has_evidence());
    assert!(!scratch.probing_context().is_empty());
}

#[test]
fn probing_respects_min_diff_max_length_empty_text_and_code_point_counts() {
    let d = detector();
    let mut scratch = Scratch::new();
    let mut out = Ranking::new();

    // Empty input evaluates 0 features (uniform prior) and returns length 0.
    let len = d.probing_predict_languages("", &mut scratch, &mut out);
    assert_eq!(len, 0);
    assert!(!out.has_evidence());
    assert_eq!(out.outcomes(), &[0, 1, 2]);

    // max_length = 0 also stops immediately with length 0 and uniform prior.
    let zero_max = ProbingConfig { max_length: 0, ..ProbingConfig::default() };
    let len = d.probing_predict_languages_with("x x x", zero_max, &mut scratch, &mut out);
    assert_eq!(len, 0);
    assert!(!out.has_evidence());

    // On "y y y y y ", "aaa" and "ccc" tie (diff == 0.0): with min_diff = 0.20
    // seenEnough never fires, so probing stops at max_length (7 code points).
    let tie_cfg = ProbingConfig { max_length: 7, chunk_size: 2, min_consec_improvements: 2, min_diff: 0.20 };
    let len = d.probing_predict_languages_with("y y y y y ", tie_cfg, &mut scratch, &mut out);
    assert_eq!(len, 7);
    assert_eq!(out.outcomes(), &[0, 2, 1]);

    // With min_diff = 0.0, the margin check is disabled and probing stops at 3 chunks (6 code points).
    let no_diff = ProbingConfig { min_diff: 0.0, ..tie_cfg };
    let len = d.probing_predict_languages_with("y y y y y ", no_diff, &mut scratch, &mut out);
    assert_eq!(len, 6);

    // Supplementary code points (4 UTF-8 bytes, 2 UTF-16 units each) count as 1 code point each.
    let emoji_cfg = ProbingConfig { max_length: 5, chunk_size: 2, min_consec_improvements: 10, min_diff: 0.99 };
    let len = d.probing_predict_languages_with("\u{1F600}x\u{1F600}x\u{1F600}x", emoji_cfg, &mut scratch, &mut out);
    assert_eq!(len, 5);
}

#[test]
fn probing_resets_window_when_confidence_dips_or_top_language_changes() {
    let d = detector();
    let mut scratch = Scratch::new();
    let mut out = Ranking::new();
    let cfg = ProbingConfig { max_length: 100, chunk_size: 2, min_consec_improvements: 2, min_diff: 0.10 };
    // Chunk 1 ("x "): bbb leads with e^2/(e^2+2). Chunk 2 ("y "): y is added so
    // bbb's confidence dips to e^2/(e^2+2e), preventing early stop on chunk 3
    // ("x ", len 6) until the dip leaves the 3-chunk window on chunk 4 (len 8).
    let len = d.probing_predict_languages_with("x y x x x ", cfg, &mut scratch, &mut out);
    assert_eq!(len, 8);
    assert_eq!(d.model().code(out.outcomes()[0] as usize), "bbb");

    // Changing the top language after chunk 1 ("y " -> aaa, then "x " -> bbb)
    // similarly requires 3 consecutive "bbb" chunks (chunks 2..=4, len 8).
    let len = d.probing_predict_languages_with("y x x x x ", cfg, &mut scratch, &mut out);
    assert_eq!(len, 8);
    assert_eq!(d.model().code(out.outcomes()[0] as usize), "bbb");

    // ProbingContext shrink_excess reclaims oversized buffers and supports reuse.
    let mut pctx = ProbingContext::default();
    let big: String = (0..20_000).map(|i| char::from_u32(0x4e00 + (i % 4096)).unwrap()).collect();
    pctx.add_chunk(&big);
    assert!(!pctx.is_empty());
    pctx.shrink_excess();
    assert!(pctx.is_empty());
    pctx.add_chunk("x");
    assert_eq!(pctx.len(), 1);
}
