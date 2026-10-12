//! The detectors against OpenNLP on the JVM.
//!
//! - The newline detector against `NewlineSentenceDetector`: every file of
//!   the `newline` and `newline-legacy` golden directories.
//! - The maxent detector against `SentenceDetectorME` with the published
//!   German, English and Portuguese models: every file of
//!   `sentdetect-me-de`, `sentdetect-me-en` and `sentdetect-me-pt`.
//! - The maxent detector with the two fixture models (one with an
//!   abbreviation dictionary), in both whitespace modes: the spans, and
//!   every context the detector built, feature by feature, with the
//!   model's probabilities and the decision.
//!
//! Spans are compared exactly in UTF-8 bytes and in UTF-16 code units,
//! and probabilities exactly: the golden files hold Java's
//! `Double.toString`, which reads back to the same `f64`, and the model's
//! arithmetic matches the JVM's bit for bit.

mod common;

use std::path::Path;

use turbo_analysis::Span;
use turbo_maxent::ModelFile;
use turbo_sentdetect::{Decision, Scratch, SentenceDetector, SentenceModel, WhitespaceMode, newline_spans};
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

/// The golden span of a span this crate found.
fn golden_span(text: &str, s: Span, prob: Option<f64>) -> common::GoldenSpan {
    common::GoldenSpan {
        utf8: (s.start, s.end),
        utf16: (byte_to_utf16(text, s.start as usize).unwrap(), byte_to_utf16(text, s.end as usize).unwrap()),
        prob,
    }
}

/// The golden offsets agree with each other on the decoded input, so the
/// reader got the text right.
fn check_offsets(g: &common::Golden, name: &str) {
    for s in &g.spans {
        assert_eq!(utf16_to_byte(&g.input, s.utf16.0), Some(s.utf8.0 as usize), "{name}");
        assert_eq!(utf16_to_byte(&g.input, s.utf16.1), Some(s.utf8.1 as usize), "{name}");
    }
}

/// Spans and probabilities equal.
fn assert_spans(got: &[common::GoldenSpan], want: &[common::GoldenSpan], name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: {got:?} against {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert_eq!((g.utf8, g.utf16), (w.utf8, w.utf16), "{name}: {got:?} against {want:?}");
        let (gp, wp) = (g.prob.unwrap(), w.prob.expect("a probability in the golden file"));
        assert_eq!(gp.to_bits(), wp.to_bits(), "{name}: probability {gp} against {wp}");
    }
}

fn check_newline_dir(dir: &str, mode: WhitespaceMode) -> usize {
    let files = common::files(dir);
    let mut out = Vec::new();
    for path in &files {
        let g = common::read(path);
        let name = path.display().to_string();
        assert_eq!(g.component, "newline", "{name}");
        assert_eq!(g.whitespace, mode, "{name}");
        check_offsets(&g, &name);
        out.clear();
        newline_spans(&g.input, mode, &mut out);
        let got: Vec<common::GoldenSpan> = out.iter().map(|&s| golden_span(&g.input, s, None)).collect();
        assert_eq!(got, g.spans, "{name}");
    }
    files.len()
}

#[test]
fn newline_unicode() {
    assert_eq!(check_newline_dir("newline", WhitespaceMode::Unicode), 34);
}

#[test]
fn newline_legacy() {
    assert_eq!(check_newline_dir("newline-legacy", WhitespaceMode::Legacy), 34);
}

fn open(path: &Path) -> SentenceModel {
    let file = ModelFile::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    SentenceModel::open(file).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Runs the detector over every file of a golden directory and returns
/// how many there were.
fn check_maxent_dir(dir: &Path, component: &str, detector: &SentenceDetector, model_file: &str) -> usize {
    let files = common::files_in(dir);
    let mut scratch = Scratch::new();
    let mut out = Vec::new();
    for path in &files {
        let g = common::read(path);
        let name = path.display().to_string();
        assert_eq!(g.component, component, "{name}");
        assert_eq!(g.whitespace, detector.whitespace, "{name}");
        assert_eq!(g.model.as_deref(), Some(model_file), "{name}");
        assert_eq!(g.model_sha256.as_deref(), Some(detector.sentence_model().model_ref().sha256.as_str()), "{name}");
        check_offsets(&g, &name);
        out.clear();
        detector.detect_with(&g.input, &mut scratch, &mut out);
        let got: Vec<common::GoldenSpan> = out.iter().map(|&(s, p)| golden_span(&g.input, s, Some(p))).collect();
        assert_spans(&got, &g.spans, &name);
    }
    files.len()
}

fn check_published(language: &str, model_file: &str) {
    let model = open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models").join(model_file));
    assert_eq!(model.language(), language);
    let detector = SentenceDetector::new(model, WhitespaceMode::Unicode);
    let component = format!("sentdetect-me-{language}");
    assert_eq!(check_maxent_dir(&common::golden_root().join(&component), &component, &detector, model_file), 34);
}

#[test]
fn maxent_english() {
    check_published("en", "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin");
}

#[test]
fn maxent_portuguese() {
    check_published("pt", "opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin");
}

#[test]
fn maxent_german() {
    check_published("de", "opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin");
}

/// The fixture models' spans and contexts in one whitespace mode.
fn check_fixture(name: &str, mode: WhitespaceMode) {
    let model_file = format!("{name}.bin");
    let detector = SentenceDetector::new(open(&common::fixtures().join("models").join(&model_file)), mode);
    let split = detector.sentence_model().split_outcome();
    let component = format!("sentdetect-me-fixture-{name}");
    let suffix = if mode == WhitespaceMode::Legacy { "-legacy" } else { "" };
    let golden = common::fixtures().join("golden");
    let dir = golden.join(format!("{component}{suffix}"));
    assert_eq!(check_maxent_dir(&dir, &component, &detector, &model_file), 17);

    let contexts = common::read_contexts(&golden.join("contexts").join(format!("{component}{suffix}.txt")));
    assert_eq!(contexts.component, format!("{component}-contexts"));
    assert_eq!(contexts.whitespace, mode);
    assert_eq!(contexts.model, model_file);
    assert_eq!(contexts.inputs.len(), 17);
    let (mut scratch, mut out) = (Scratch::new(), Vec::new());
    let mut candidates = 0;
    for input in &contexts.inputs {
        let text = &input.input;
        let mut seen = Vec::new();
        out.clear();
        detector.detect_traced(text, &mut scratch, &mut out, |c| {
            let features: Vec<Vec<u8>> = c.features.iter().map(<[u8]>::to_vec).collect();
            seen.push((c.at, features, (c.probs[split], c.probs[1 - split]), c.decision));
        });
        let what = format!("{name}{suffix} on {text:?}");
        assert_eq!(seen.len(), input.candidates.len(), "{what}: candidates");
        for (got, want) in seen.iter().zip(&input.candidates) {
            let (at, features, probs, decision) = got;
            assert_eq!(at.unit as usize, want.at, "{what}");
            assert_eq!(utf16_to_byte(text, want.at), Some(at.byte as usize), "{what}");
            let show = |f: &[Vec<u8>]| f.iter().map(|x| String::from_utf8_lossy(x).into_owned()).collect::<Vec<_>>();
            assert_eq!(
                features,
                &want.features,
                "{what} at {}: {:?} against {:?}",
                want.at,
                show(features),
                show(&want.features)
            );
            assert_eq!(probs.0.to_bits(), want.probs.0.to_bits(), "{what} at {}: P(s)", want.at);
            assert_eq!(probs.1.to_bits(), want.probs.1.to_bits(), "{what} at {}: P(n)", want.at);
            let want_decision = match want.decision.as_str() {
                "split" => Decision::Split,
                "no-split" => Decision::NoSplit,
                "vetoed" => Decision::Vetoed,
                other => panic!("decision {other:?}"),
            };
            assert_eq!(*decision, want_decision, "{what} at {}", want.at);
            candidates += 1;
        }
    }
    assert!(candidates > 40, "{name}{suffix}: only {candidates} candidates");
}

#[test]
fn fixture_with_abbreviations() {
    check_fixture("en-abbrev", WhitespaceMode::Unicode);
    check_fixture("en-abbrev", WhitespaceMode::Legacy);
}

#[test]
fn fixture_without_abbreviations() {
    check_fixture("en-plain", WhitespaceMode::Unicode);
    check_fixture("en-plain", WhitespaceMode::Legacy);
}
