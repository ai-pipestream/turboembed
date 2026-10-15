//! The sentence detectors against the conformance suites in
//! `analysis/goldens/data/conformance/sentences`: the Pragmatic Segmenter
//! golden rules and specs, the pySBD tests, and Unicode's
//! `SentenceBreakTest.txt` (UAX #29, Unicode 18.0).
//!
//! Three detectors run:
//!
//! - `rules`: the rule splitter with every tailoring, for the case's
//!   language;
//! - `uax29`: the rule splitter with no tailoring, which is plain UAX #29;
//! - `maxent-de`, `maxent-en`, `maxent-pt`: OpenNLP's sentence detector with
//!   the published UD model of that language, on the cases in its
//!   language (and `maxent-en` on `SentenceBreakTest.txt`). Without the
//!   models (`turbo-goldens fetch-models`) these say so and pass.
//!
//! A text case passes when the detector's sentences, with every run of
//! whitespace made one space and the ends trimmed, are the expected
//! sentences treated the same way. A `SentenceBreakTest.txt` line passes
//! when the detector's spans are the UAX #29 segments trimmed of
//! whitespace, the empty ones dropped; `uax29` must also place every break
//! of every line exactly.
//!
//! Every case a detector gets wrong is listed, with what differs, in
//! `tests/conformance/<detector>.<suite>.failures`. The test fails when a
//! case fails that the list does not hold, or passes although the list
//! holds it, or the difference changed; so a change to a detector shows
//! here as a change to its list. To rewrite the lists after such a change:
//!
//! ```text
//! TURBO_WRITE_EXPECTED_FAILURES=1 cargo test -p turbo-sentdetect --test conformance
//! ```

mod common;

use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_goldens::sentences::{
    BreakTestCase, SentenceCase, read_break_test, read_expected_failures, sentence_suite, sentences_dir,
};
use turbo_maxent::ModelFile;
use turbo_sentdetect::{LineBreaks, RuleSplitter, Rules, SentenceDetector, SentenceModel, WhitespaceMode};
use turbo_unicode::sentence_break::sentence_break;

/// The published models, by language.
const MODELS: [(&str, &str); 3] = [
    ("de", "opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin"),
    ("en", "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin"),
    ("pt", "opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin"),
];

enum Detector {
    Rules,
    Uax29,
    Maxent(&'static str, SentenceDetector),
}

impl Detector {
    fn name(&self) -> String {
        match self {
            Detector::Rules => "rules".into(),
            Detector::Uax29 => "uax29".into(),
            Detector::Maxent(lang, _) => format!("maxent-{lang}"),
        }
    }

    /// Whether it runs on a case in language `tag`.
    fn covers(&self, tag: &str) -> bool {
        match self {
            Detector::Maxent(lang, _) => tag.split(['-', '_']).next() == Some(lang),
            _ => true,
        }
    }

    fn spans(&self, tag: &str, text: &str) -> Vec<Span> {
        let mut out = Vec::new();
        match self {
            Detector::Rules => RuleSplitter::new(tag).split(text, &mut out),
            Detector::Uax29 => RuleSplitter::uax29().split(text, &mut out),
            Detector::Maxent(_, d) => {
                let mut with_probs = Vec::new();
                d.detect(text, &mut with_probs);
                out.extend(with_probs.iter().map(|&(s, _)| s));
            }
        }
        out
    }
}

/// The rule splitter's tailorings, each as a change that turns it off.
const TAILORINGS: [(&str, fn(RuleSplitter) -> RuleSplitter); 8] = [
    ("terminators", |s| s.with_rules(Rules { terminators: false, ..s.rules() })),
    ("references", |s| s.with_rules(Rules { references: false, ..s.rules() })),
    ("continuation", |s| s.with_rules(Rules { continuation: false, ..s.rules() })),
    ("abbreviations", |s| s.with_rules(Rules { abbreviations: false, ..s.rules() })),
    ("initials", |s| s.with_rules(Rules { initials: false, ..s.rules() })),
    ("lists", |s| s.with_rules(Rules { lists: false, ..s.rules() })),
    ("brackets", |s| s.with_rules(Rules { brackets: false, ..s.rules() })),
    ("paragraph line breaks", |s| s.with_line_breaks(LineBreaks::Sentence)),
];

/// For a case the rule splitter fails, the tailorings without which it
/// passes, as a note for its reason.
fn causes(detector: &Detector, tag: &str, text: &str, passes: impl Fn(&[Span]) -> bool) -> String {
    if !matches!(detector, Detector::Rules) {
        return String::new();
    }
    let names: Vec<&str> = TAILORINGS
        .iter()
        .filter(|(_, off)| {
            let mut out = Vec::new();
            off(RuleSplitter::new(tag)).split(text, &mut out);
            passes(&out)
        })
        .map(|(name, _)| *name)
        .collect();
    if names.is_empty() { String::new() } else { format!("; passes without the {} rule", names.join(", ")) }
}

/// The maxent detector of `lang`, or `None` (having said so) without the
/// model.
fn maxent(lang: &'static str) -> Option<Detector> {
    let file = MODELS.iter().find(|(l, _)| *l == lang).unwrap().1;
    let path = common::published_model(file)?;
    let model = SentenceModel::open(ModelFile::read(&path).unwrap()).unwrap();
    Some(Detector::Maxent(lang, SentenceDetector::new(model, WhitespaceMode::Unicode)))
}

/// Whitespace runs made one space, the ends trimmed.
fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalized(sentences: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    sentences.into_iter().map(|s| normalize(s.as_ref())).filter(|s| !s.is_empty()).collect()
}

/// A short, one-line view of the end of a sentence.
fn tail(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let keep = 24;
    if chars.len() <= keep {
        format!("\"{s}\"")
    } else {
        format!("\"...{}\"", chars[chars.len() - keep..].iter().collect::<String>())
    }
}

/// Where each sentence ends, counted in non-whitespace characters, and
/// the sentence, for every sentence but the last.
fn ends(sentences: &[String]) -> Vec<(usize, &str)> {
    let mut at = 0;
    let mut out = Vec::new();
    for s in sentences {
        at += s.chars().filter(|c| !c.is_whitespace()).count();
        out.push((at, s.as_str()));
    }
    out.pop();
    out
}

/// What differs between the sentences found and the ones expected.
fn describe(case: &SentenceCase, want: &[String], got: &[String]) -> String {
    let upstream = case
        .options
        .iter()
        .filter(|o| o.ends_with("-upstream") || o.starts_with("clean=true") || o.starts_with("doc_type"))
        .map(String::as_str)
        .collect::<Vec<_>>();
    let note = if upstream.is_empty() { String::new() } else { format!(" [{}]", upstream.join(", ")) };
    if !case.is_extractive() {
        return format!("the expected sentences rewrite the text{note}");
    }
    let (w, g) = (ends(want), ends(got));
    let missing: Vec<&str> = w.iter().filter(|(p, _)| !g.iter().any(|(q, _)| q == p)).map(|(_, s)| *s).collect();
    let extra: Vec<&str> = g.iter().filter(|(p, _)| !w.iter().any(|(q, _)| q == p)).map(|(_, s)| *s).collect();
    let mut parts = Vec::new();
    if let Some(first) = missing.first() {
        parts.push(format!("{} missing, first after {}", missing.len(), tail(first)));
    }
    if let Some(first) = extra.first() {
        parts.push(format!("{} extra, first after {}", extra.len(), tail(first)));
    }
    if parts.is_empty() {
        parts.push("the same breaks, different text".into());
    }
    format!("{}{note}", parts.join("; "))
}

fn failures_path(detector: &str, suite: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance").join(format!("{detector}.{suite}.failures"))
}

/// Compares the failures of a run with its list, or writes the list.
fn check_failures(detector: &str, suite: &str, run: usize, failed: &[(String, String)]) {
    let path = failures_path(detector, suite);
    if run == 0 {
        assert!(!path.exists(), "{}: no case ran", path.display());
        eprintln!("{detector} on {suite}: no case in its language");
        return;
    }
    if std::env::var_os("TURBO_WRITE_EXPECTED_FAILURES").is_some() {
        let mut text = format!(
            "# The cases of the {suite} suite that the {detector} detector gets wrong, and what\n\
             # differs. tests/conformance.rs writes this file and fails when a run differs\n\
             # from it. {} of {run} cases fail.\n",
            failed.len()
        );
        for (id, reason) in failed {
            text.push_str(&format!("{id}\t{reason}\n"));
        }
        std::fs::write(&path, text).unwrap();
    }
    let listed: Vec<(String, String)> =
        read_expected_failures(&path).unwrap_or_else(|e| panic!("{e}")).into_iter().map(|f| (f.id, f.reason)).collect();
    let new: Vec<_> = failed.iter().filter(|f| !listed.contains(f)).collect();
    let gone: Vec<_> = listed.iter().filter(|f| !failed.contains(f)).collect();
    assert!(
        new.is_empty() && gone.is_empty(),
        "{detector} on {suite}: failing but not listed (or listed with another reason): {new:#?}\n\
         listed but not failing (or failing differently): {gone:#?}"
    );
    eprintln!("{detector} on {suite}: {} of {run} pass", run - failed.len());
}

/// Runs a detector over a text suite; returns the number of cases run.
fn text_suite(detector: &Detector, suite: &str) -> (usize, usize) {
    let mut failed = Vec::new();
    let mut run = 0;
    for case in sentence_suite(suite) {
        if !detector.covers(&case.lang) {
            continue;
        }
        run += 1;
        let sentences = |spans: &[Span]| normalized(spans.iter().map(|s| &case.text[s.start as usize..s.end as usize]));
        let got = sentences(&detector.spans(&case.lang, &case.text));
        let want = normalized(&case.sentences);
        if got != want {
            let why = causes(detector, &case.lang, &case.text, |spans| sentences(spans) == want);
            failed.push((case.id.clone(), format!("{}{why}", describe(&case, &want, &got))));
        }
    }
    check_failures(&detector.name(), suite, run, &failed);
    (run, run - failed.len())
}

fn break_test() -> Vec<BreakTestCase> {
    read_break_test(&sentences_dir().join("uax29/SentenceBreakTest-18.0.0.txt")).unwrap_or_else(|e| panic!("{e}"))
}

/// The UAX #29 segments of a line, trimmed of whitespace, empty ones
/// dropped: what a detector that writes trimmed spans can match.
fn projected(case: &BreakTestCase) -> Vec<Span> {
    let mut cuts = vec![0];
    cuts.extend(case.breaks.iter().copied());
    cuts.push(case.text.len() as u32);
    let mut out = Vec::new();
    for w in cuts.windows(2) {
        let s = &case.text[w[0] as usize..w[1] as usize];
        let lead = s.len() - s.trim_start_matches(char::is_whitespace).len();
        let trimmed = s.trim_matches(char::is_whitespace);
        if !trimmed.is_empty() {
            let start = w[0] + lead as u32;
            out.push(Span::new(start, start + trimmed.len() as u32));
        }
    }
    out
}

/// A code point with its `Sentence_Break` value.
fn show(c: Option<char>) -> String {
    match c {
        Some(c) => format!("U+{:04X} ({})", c as u32, sentence_break(c as u32).property_name()),
        None => "the end".into(),
    }
}

fn describe_breaks(case: &BreakTestCase, want: &[Span], got: &[Span]) -> String {
    let starts = |spans: &[Span]| spans.iter().skip(1).map(|s| s.start).collect::<Vec<_>>();
    let (w, g) = (starts(want), starts(got));
    let around = |at: u32| {
        let before = case.text[..at as usize].chars().next_back();
        let after = case.text[at as usize..].chars().next();
        format!("{} and {}", show(before), show(after))
    };
    if let Some(&at) = w.iter().find(|p| !g.contains(p)) {
        return format!("misses the break rule {} makes between {}", case.rule_at(at), around(at));
    }
    if let Some(&at) = g.iter().find(|p| !w.contains(p)) {
        return format!("breaks between {}, where UAX #29 does not", around(at));
    }
    "the same starts, different ends".into()
}

fn uax_suite(detector: &Detector) {
    let mut failed = Vec::new();
    let cases = break_test();
    for case in &cases {
        let want = projected(case);
        let got = detector.spans("und", &case.text);
        if got != want {
            let why = causes(detector, "und", &case.text, |spans| spans == want);
            let id = format!("uax29/SentenceBreakTest-18.0.0/line-{}", case.line);
            failed.push((id, format!("{}{why}", describe_breaks(case, &want, &got))));
        }
    }
    check_failures(&detector.name(), "uax29", cases.len(), &failed);
}

#[test]
fn uax29_places_every_break_of_the_unicode_test() {
    let splitter = RuleSplitter::uax29();
    let cases = break_test();
    for case in &cases {
        let mut got = Vec::new();
        splitter.boundaries(&case.text, &mut got);
        assert_eq!(got, case.breaks, "line {}: {:?}", case.line, case.text);
    }
    assert_eq!(cases.len(), 512);
}

#[test]
fn rules_on_pragmatic_segmenter() {
    text_suite(&Detector::Rules, "pragmatic-segmenter");
}

#[test]
fn rules_on_pysbd() {
    text_suite(&Detector::Rules, "pysbd");
}

#[test]
fn rules_on_uax29() {
    uax_suite(&Detector::Rules);
}

#[test]
fn uax29_on_pragmatic_segmenter() {
    text_suite(&Detector::Uax29, "pragmatic-segmenter");
}

#[test]
fn uax29_on_pysbd() {
    text_suite(&Detector::Uax29, "pysbd");
}

#[test]
fn uax29_on_uax29() {
    uax_suite(&Detector::Uax29);
}

#[test]
fn maxent_on_pragmatic_segmenter() {
    for (lang, _) in MODELS {
        if let Some(d) = maxent(lang) {
            text_suite(&d, "pragmatic-segmenter");
        }
    }
}

#[test]
fn maxent_on_pysbd() {
    for (lang, _) in MODELS {
        if let Some(d) = maxent(lang) {
            text_suite(&d, "pysbd");
        }
    }
}

#[test]
fn maxent_en_on_uax29() {
    if let Some(d) = maxent("en") {
        uax_suite(&d);
    }
}
