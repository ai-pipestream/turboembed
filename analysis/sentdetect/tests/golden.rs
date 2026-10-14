//! The detectors against OpenNLP on the JVM (analysis/goldens/data/sentdetect).
//!
//! - The newline detector against `NewlineSentenceDetector`: every golden
//!   of the `newline` and `newline+legacy` runs.
//! - The maxent detector against `SentenceDetectorME` with the published
//!   German, English and Portuguese models, which `turbo-goldens
//!   fetch-models` downloads (`common::published_model`); without them
//!   these tests say so and pass.
//! - The maxent detector with the two fixture models (one with an
//!   abbreviation dictionary), in both whitespace modes.
//!
//! For every maxent run the dumps are checked too: every candidate the
//! detector asked the model about, feature by feature, with the model's
//! probabilities and the decision, where a split the model wanted and the
//! abbreviation dictionary refused is the dump's `acceptable 0`.
//!
//! Spans are compared exactly in UTF-8 bytes and in UTF-16 code units,
//! and probabilities exactly: the goldens hold the bits of Java's
//! doubles, and the model's arithmetic matches the JVM's bit for bit.

mod common;

use std::path::Path;

use common::{Case, GoldenSpan};
use turbo_analysis::Span;
use turbo_goldens::Unit;
use turbo_maxent::ModelFile;
use turbo_sentdetect::{Decision, Scratch, SentenceDetector, SentenceModel, WhitespaceMode, newline_spans};
use turbo_unicode::modified_utf8::push_modified_utf8;
use turbo_unicode::utf16::byte_to_utf16;

const ME: &str = "opennlp.tools.sentdetect.SentenceDetectorME";

/// The golden span of a span this crate found.
fn golden_span(text: &str, s: Span, prob: Option<f64>) -> GoldenSpan {
    GoldenSpan {
        utf8: (s.start, s.end),
        utf16: (byte_to_utf16(text, s.start as usize).unwrap(), byte_to_utf16(text, s.end as usize).unwrap()),
        prob,
    }
}

/// Spans and probabilities equal.
fn assert_spans(got: &[GoldenSpan], want: &[GoldenSpan], name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: {got:?} against {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert_eq!((g.utf8, g.utf16), (w.utf8, w.utf16), "{name}: {got:?} against {want:?}");
        let (gp, wp) = (g.prob.unwrap(), w.prob.expect("a probability in the golden"));
        assert_eq!(gp.to_bits(), wp.to_bits(), "{name}: probability {gp} against {wp}");
    }
}

fn check_newline_run(name: &str, mode: WhitespaceMode) -> usize {
    let cases = common::run(name);
    let mut out = Vec::new();
    for case in &cases {
        let what = case.path.display().to_string();
        case.assert_annotator("opennlp.tools.sentdetect.NewlineSentenceDetector");
        case.assert_model(None);
        assert_eq!(case.whitespace(), mode, "{what}");
        assert!(case.golden.layer("turbo:confidence<opennlp:sentences>").is_none(), "{what}");
        out.clear();
        newline_spans(&case.text, mode, &mut out);
        let got: Vec<GoldenSpan> = out.iter().map(|&s| golden_span(&case.text, s, None)).collect();
        assert_eq!(got, case.spans(), "{what}");
    }
    cases.len()
}

#[test]
fn newline_unicode() {
    assert_eq!(check_newline_run("newline", WhitespaceMode::Unicode), 34);
}

#[test]
fn newline_legacy() {
    assert_eq!(check_newline_run("newline+legacy", WhitespaceMode::Legacy), 34);
}

fn open(path: &Path) -> SentenceModel {
    let file = ModelFile::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    SentenceModel::open(file).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// What one maxent run's goldens and dumps held.
#[derive(Debug, Default)]
struct Counts {
    /// Goldens, per input set.
    files: Vec<(String, usize)>,
    candidates: usize,
    refused: usize,
}

/// Runs the detector over every golden of `sentdetect/<name>` and checks
/// every dump against what the detector saw.
fn check_maxent_run(name: &str, detector: &SentenceDetector, model: &Path) -> Counts {
    let split = detector.sentence_model().split_outcome();
    let maxent = detector.sentence_model().maxent();
    let (s, n) = (maxent.outcome(split).to_string(), maxent.outcome(1 - split).to_string());
    let mut scratch = Scratch::new();
    let mut out = Vec::new();
    let mut counts = Counts::default();
    for case in common::run(name) {
        let what = case.path.display().to_string();
        case.assert_annotator(ME);
        case.assert_model(Some(model));
        assert_eq!(case.whitespace(), detector.whitespace, "{what}");
        out.clear();
        detector.detect_with(&case.text, &mut scratch, &mut out);
        let got: Vec<GoldenSpan> = out.iter().map(|&(s, p)| golden_span(&case.text, s, Some(p))).collect();
        assert_spans(&got, &case.spans(), &what);
        let (c, r) = check_dump(&case, detector, &s, &n);
        counts.candidates += c;
        counts.refused += r;
        match counts.files.iter_mut().find(|(set, _)| set == case.set()) {
            Some((_, k)) => *k += 1,
            None => counts.files.push((case.set().to_string(), 1)),
        }
    }
    counts
}

/// Every candidate of the dump against the detector's trace, and returns
/// the number of candidates and of refused splits.
fn check_dump(case: &Case, detector: &SentenceDetector, s: &str, n: &str) -> (usize, usize) {
    let dump = case.dump();
    let mut seen = Vec::new();
    let (mut scratch, mut out) = (Scratch::new(), Vec::new());
    detector.detect_traced(&case.text, &mut scratch, &mut out, |c| {
        let features: Vec<Vec<u8>> = c.features.iter().map(<[u8]>::to_vec).collect();
        seen.push((c.at, features, c.probs.to_vec(), c.decision));
    });
    let what = case.path.display().to_string();
    assert_eq!(seen.len(), dump.points.len(), "{what}: candidates");
    let split = detector.sentence_model().split_outcome();
    let mut refused = 0;
    for ((at, features, probs, decision), p) in seen.iter().zip(&dump.points) {
        assert_eq!((at.byte, p.unit), (p.position, Unit::Start), "{what}");
        assert_eq!(byte_to_utf16(&case.text, at.byte as usize), Some(at.unit as usize), "{what}");
        let want: Vec<Vec<u8>> = p
            .contexts
            .iter()
            .map(|f| {
                let mut bytes = Vec::new();
                for &u in f {
                    push_modified_utf8(&mut bytes, u32::from(u));
                }
                bytes
            })
            .collect();
        let show = |f: &[Vec<u8>]| f.iter().map(|x| String::from_utf8_lossy(x).into_owned()).collect::<Vec<_>>();
        assert_eq!(features, &want, "{what} at {}: {:?} against {:?}", p.position, show(features), show(&want));
        let (ps, pn) = (p.prob(s).unwrap(), p.prob(n).unwrap());
        assert_eq!(probs[split].to_bits(), ps.to_bits(), "{what} at {}: P(s)", p.position);
        assert_eq!(probs[1 - split].to_bits(), pn.to_bits(), "{what} at {}: P(n)", p.position);
        let want_decision = match (p.best().label == s, p.acceptable) {
            (false, None) => Decision::NoSplit,
            (true, Some(true)) => Decision::Split,
            (true, Some(false)) => Decision::Vetoed,
            other => panic!("{what} at {}: best outcome and acceptable {other:?}", p.position),
        };
        assert_eq!(*decision, want_decision, "{what} at {}", p.position);
        refused += usize::from(want_decision == Decision::Vetoed);
    }
    (seen.len(), refused)
}

/// Runs a published model's goldens, or returns `None` when the model is
/// not downloaded (`common::published_model`).
fn check_published(language: &str, model_file: &str) -> Option<Counts> {
    let path = common::published_model(model_file)?;
    let model = open(&path);
    assert_eq!(model.language(), language);
    let detector = SentenceDetector::new(model, WhitespaceMode::Unicode);
    let counts = check_maxent_run(model_file.strip_suffix(".bin").unwrap(), &detector, &path);
    // The 34 edge inputs for each model.
    assert!(counts.files.contains(&("edge".to_string(), 34)), "{counts:?}");
    assert_eq!(counts.refused, 0, "no published model has an abbreviation dictionary");
    Some(counts)
}

#[test]
fn maxent_english() {
    let Some(counts) = check_published("en", "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin") else { return };
    assert_eq!(counts.files, [("edge".into(), 34), ("en".into(), 11), ("multi".into(), 11)]);
    assert_eq!(counts.candidates, 185);
}

#[test]
fn maxent_portuguese() {
    let Some(counts) = check_published("pt", "opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin") else { return };
    assert_eq!(counts.files, [("edge".into(), 34)]);
    assert_eq!(counts.candidates, 104);
}

#[test]
fn maxent_german() {
    let Some(counts) = check_published("de", "opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin") else { return };
    assert_eq!(counts.files, [("edge".into(), 34)]);
    assert_eq!(counts.candidates, 67);
}

/// The fixture model's spans and dumps in one whitespace mode, with as
/// many candidates as the earlier generator's contexts files had.
fn check_fixture(name: &str, mode: WhitespaceMode, candidates: usize) -> Counts {
    let path = common::fixtures_dir().join(format!("{name}.bin"));
    let detector = SentenceDetector::new(open(&path), mode);
    let run = if mode == WhitespaceMode::Legacy { format!("{name}+legacy") } else { name.to_string() };
    let counts = check_maxent_run(&run, &detector, &path);
    assert_eq!(counts.files, [("sentence-fixture".into(), 17)], "{run}");
    assert_eq!(counts.candidates, candidates, "{run}");
    counts
}

#[test]
fn fixture_with_abbreviations() {
    for mode in [WhitespaceMode::Unicode, WhitespaceMode::Legacy] {
        let counts = check_fixture("en-abbrev", mode, 76);
        // The dictionary refuses three splits the model wanted.
        assert_eq!(counts.refused, 3, "{mode:?}");
    }
}

#[test]
fn fixture_without_abbreviations() {
    for mode in [WhitespaceMode::Unicode, WhitespaceMode::Legacy] {
        assert_eq!(check_fixture("en-plain", mode, 77).refused, 0, "{mode:?}");
    }
}
