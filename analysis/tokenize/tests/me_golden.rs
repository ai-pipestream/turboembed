//! TokenizerMe against the files the OpenNLP JVM wrote: the tokens and
//! probabilities of every golden input for the five UD models, the
//! features and probabilities of every position in the contexts files,
//! and the one documented difference, a split between surrogate halves.

mod me_common;

use std::collections::BTreeMap;

use me_common::*;
use turbo_tokenize::me::{Features, Scratch, TokenizerMe, TokenizerMeOptions, UnitFacts, token_context};
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};
use turbo_unicode::{Utf16Cursor, WhitespaceMode};

/// The tokenizer for a golden file's header, checking that the model is
/// the one the file was made with.
fn tokenizer_for(headers: &dyn Fn(&str) -> String, cache: &mut BTreeMap<String, TokenizerMe>) -> TokenizerMe {
    let name = headers("model");
    let mode = match headers("whitespace").as_str() {
        "unicode" => WhitespaceMode::Unicode,
        "legacy" => WhitespaceMode::Legacy,
        other => panic!("whitespace mode {other}"),
    };
    let t = cache.entry(name.clone()).or_insert_with(|| {
        let model = model(&name);
        assert_eq!(model.sha256().to_hex(), headers("model-sha256"), "{name}");
        assert_eq!(
            model.alphanumeric_pattern(),
            String::from_utf16(&unescape(&headers("alpha-numeric-pattern"))).unwrap()
        );
        assert_eq!(model.alphanumeric().is_some(), headers("use-alpha-numeric-optimization") == "true");
        TokenizerMe::new(model, TokenizerMeOptions::default())
    });
    TokenizerMe::new(t.tokenizer_model().clone(), TokenizerMeOptions { whitespace: mode, keep_newlines: false })
}

/// Tokenizes `input` and compares with `expected`, on both offset kinds.
fn check(t: &TokenizerMe, input: &str, expected: &[(u32, u32, f64)], what: &str) {
    let mut got = Vec::new();
    t.tokenize_with(input, &mut Scratch::new(), &mut got);
    let got_spans: Vec<(u32, u32)> = got.iter().map(|(s, _)| (s.start, s.end)).collect();
    let want_spans: Vec<(u32, u32)> = expected.iter().map(|&(a, b, _)| (a, b)).collect();
    assert_eq!(got_spans, want_spans, "{what}");
    for ((s, p), &(_, _, want)) in got.iter().zip(expected) {
        assert_close(*p, want, &format_args!("{what} at {}..{}", s.start, s.end));
        // UTF-16 offsets follow from the UTF-8 ones on character boundaries.
        let a = byte_to_utf16(input, s.start as usize).expect("a character boundary");
        let b = byte_to_utf16(input, s.end as usize).expect("a character boundary");
        assert_eq!(utf16_to_byte(input, a), Some(s.start as usize));
        assert_eq!(utf16_to_byte(input, b), Some(s.end as usize));
    }
}

#[test]
fn every_golden_input_for_every_model() {
    let mut cache = BTreeMap::new();
    let mut files = 0;
    let mut spans = 0;
    for language in LANGUAGES {
        let dir = golden_dir().join(format!("tokenizer-me-{language}"));
        let mut paths: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        paths.sort();
        assert_eq!(paths.len(), 34, "{}", dir.display());
        for path in paths {
            let g = Golden::read(&path);
            assert_eq!(g.header("component"), format!("tokenizer-me-{language}"));
            let t = tokenizer_for(&|k| g.header(k).to_string(), &mut cache);
            // The UTF-16 offsets of every valid span match the UTF-8 ones.
            for s in &g.spans {
                if let Some((a, b)) = s.utf8 {
                    assert_eq!(byte_to_utf16(&g.input, a as usize), Some(s.utf16.0));
                    assert_eq!(byte_to_utf16(&g.input, b as usize), Some(s.utf16.1));
                }
            }
            let expected = expected_spans(&g.input, &g.spans);
            check(&t, &g.input, &expected, &path.display().to_string());
            files += 1;
            spans += expected.len();
        }
    }
    assert_eq!(files, 170);
    assert_eq!(spans, 2283, "every span line of every file");
}

#[test]
fn every_context_and_its_probabilities() {
    let mut positions = 0;
    for language in LANGUAGES {
        let path = golden_dir().join(format!("contexts/tokenizer-me-{language}.txt"));
        let c = Contexts::read(&path);
        assert_eq!(c.header("component"), format!("tokenizer-me-{language}-contexts"));
        assert_eq!(c.header("whitespace"), "unicode");
        let model = model(c.header("model"));
        assert_eq!(model.sha256().to_hex(), c.header("model-sha256"));
        let maxent = model.maxent();
        let order: Vec<usize> = c.outcomes.iter().map(|o| maxent.outcome_index(o).unwrap()).collect();
        let mut features = Features::new();
        let mut probs = vec![0.0; maxent.num_outcomes()];
        assert_eq!(c.inputs.len(), 5);
        for (input, units) in &c.inputs {
            for unit in units {
                let a = utf16_to_byte(input, unit.utf16.0).unwrap();
                let b = utf16_to_byte(input, unit.utf16.1).unwrap();
                let text = &input[a..b];
                let facts = UnitFacts::of(text, false);
                for (index, want, want_probs) in &unit.positions {
                    let what = format!("{} unit {text:?} at {index}", path.display());
                    let mut at = Utf16Cursor::new(text);
                    for _ in 0..*index {
                        at.step_forward();
                    }
                    token_context(&at, facts, WhitespaceMode::Unicode, &mut features);
                    assert_eq!(features.len(), want.len(), "{what}");
                    for (i, w) in want.iter().enumerate() {
                        assert_eq!(features.get(i), modified_utf8(w), "{what}, feature {i}");
                    }
                    maxent.eval_features(features.iter(), &mut probs);
                    for (k, &o) in order.iter().enumerate() {
                        assert_close(probs[o], want_probs[k], &what);
                    }
                    positions += 1;
                }
            }
        }
    }
    assert_eq!(positions, 2445, "every position of every contexts file");
}

/// A golden span by its UTF-16 offsets, as a `span-invalid` line gives it.
fn java(start: usize, end: usize, prob: f64) -> GoldenSpan {
    GoldenSpan { utf8: None, utf16: (start, end), prob }
}

/// A language, an input, the spans Java returned and the UTF-8 spans the
/// port gives.
type Deviation = (&'static str, &'static str, Vec<GoldenSpan>, Vec<(u32, u32)>);

#[test]
fn a_split_between_surrogate_halves_is_not_taken() {
    // What TokenizerME.tokenizePos and probs() returned on the JVM (the
    // generator's program, run on these inputs): each model splits inside
    // U+1F600. The port keeps the character whole and multiplies on.
    let cases: [Deviation; 6] = [
        (
            "en",
            "'\u{1f600}.",
            vec![
                java(0, 1, 0.9993166982223048),
                java(1, 2, 0.509129942823796),
                java(2, 3, 0.9862742638094529),
                java(3, 4, 1.0),
            ],
            vec![(0, 1), (1, 5), (5, 6)],
        ),
        (
            "en",
            ".\u{1f600}\"",
            vec![java(0, 2, 0.3908256036328703), java(2, 3, 0.9997042239666407), java(3, 4, 1.0)],
            vec![(0, 5), (5, 6)],
        ),
        (
            "de",
            "a\u{1f600}'",
            vec![java(0, 2, 0.7617282902778715), java(2, 3, 0.5961195915334687), java(3, 4, 1.0)],
            vec![(0, 5), (5, 6)],
        ),
        (
            "es",
            ")\u{1f600}'",
            vec![java(0, 2, 0.3177479026875649), java(2, 3, 0.9699473386493942), java(3, 4, 1.0)],
            vec![(0, 5), (5, 6)],
        ),
        ("fr", ",\u{1f600}$", vec![java(0, 2, 0.40253130669461706), java(2, 4, 0.5390366580516481)], vec![(0, 6)]),
        (
            "pt",
            "a\u{1f600}?",
            vec![java(0, 2, 0.598725004631704), java(2, 3, 0.9992097649387779), java(3, 4, 1.0)],
            vec![(0, 5), (5, 6)],
        ),
    ];
    let mut models = BTreeMap::new();
    for (language, input, java_spans, want) in cases {
        let t = models.entry(language).or_insert_with(|| {
            let name = std::fs::read_dir(crate_dir().join("tests/models"))
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .find(|n| n.starts_with(&format!("opennlp-{language}-")))
                .unwrap();
            TokenizerMe::new(model(&name), TokenizerMeOptions::default())
        });
        let expected = expected_spans(input, &java_spans);
        assert_eq!(expected.iter().map(|&(a, b, _)| (a, b)).collect::<Vec<_>>(), want, "{language} {input:?}");
        check(t, input, &expected, &format!("{language} {input:?}"));
    }
}
