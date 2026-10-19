//! TokenizerMe against what `TokenizerME` returned on the JVM
//! (analysis/goldens/data/tokenize): the tokens and probabilities of every
//! input of the five UD models, every decision of their dumps, the one
//! documented difference, a split between surrogate halves, and the
//! alphanumeric optimization. The UD models are downloaded by
//! `turbo-goldens fetch-models` (`common::published_model`); without them
//! these tests say so and pass.

mod common;

use common::*;
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::TokenizerMe;
use turbo_tokenize::me::alnum::language_pattern;
use turbo_tokenize::me::{Scratch, TokenizerModel};
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

const CLASS: &str = "opennlp.tools.tokenize.TokenizerME";

/// Tokenizes `case` and compares with what the JVM returned: on UTF-8
/// and UTF-16 offsets and probabilities, after joining the halves of a
/// split pair, and through `assert_matches_golden` where nothing is cut.
/// Returns the number of spans.
fn check(t: &TokenizerMe, case: &Case) -> usize {
    let what = case.path.display().to_string();
    let expected = expected_me_spans(&case.text, &case.jvm_spans());
    let mut got = Vec::new();
    t.tokenize_with(&case.text, &mut Scratch::new(), &mut got);
    let got_spans: Vec<(u32, u32)> = got.iter().map(|(s, _)| (s.start, s.end)).collect();
    let want_spans: Vec<(u32, u32)> = expected.iter().map(|&(a, b, _)| (a, b)).collect();
    assert_eq!(got_spans, want_spans, "{what}");
    for ((s, p), &(_, _, want)) in got.iter().zip(&expected) {
        assert_close(*p, want, &format_args!("{what} at {}..{}", s.start, s.end));
        // UTF-16 offsets follow from the UTF-8 ones on character boundaries.
        let a = byte_to_utf16(&case.text, s.start as usize).expect("a character boundary");
        let b = byte_to_utf16(&case.text, s.end as usize).expect("a character boundary");
        assert_eq!(utf16_to_byte(&case.text, a), Some(s.start as usize));
        assert_eq!(utf16_to_byte(&case.text, b), Some(s.end as usize));
    }
    compare_layers(case, &got_spans, Some(got.iter().map(|(_, p)| *p).collect()));
    expected.len()
}

/// The UD model for `language`, checked to be the one `case` was made
/// with, with the optimization off as every UD model ships.
fn checked_model(language: &str, path: &std::path::Path, case: &Case) -> TokenizerModel {
    case.assert_models(&[path.to_path_buf()]);
    let m = read_model(path);
    assert!(m.alphanumeric().is_none());
    assert_eq!(m.alphanumeric_pattern(), language_pattern(language));
    m
}

#[test]
fn every_golden_input_for_every_model() {
    let Some(models) = ud_models() else { return };
    let mut files = 0;
    let mut spans = 0;
    let mut edge = (0, 0);
    for (language, path) in models {
        let mut t = None;
        for case in run("tokenize", run_name(ud_model_name(language))) {
            if case.set() == "surrogate" {
                continue;
            }
            case.assert_annotator(CLASS, "tokenizePos");
            assert!(case.golden.header.config.is_empty(), "{}", case.path.display());
            let t: &TokenizerMe = t.get_or_insert_with(|| {
                TokenizerMe::new(checked_model(language, &path, &case), TokenizerOptions::default())
            });
            case.assert_models(std::slice::from_ref(&path));
            let n = check(t, &case);
            files += 1;
            spans += n;
            if case.set() == "edge" {
                edge.0 += 1;
                edge.1 += n;
            }
        }
    }
    // The 34 edge inputs for each model, as the earlier generator wrote
    // them, and the 22 en and multi inputs for English.
    assert_eq!(edge, (170, 2298), "every span of every edge input");
    assert_eq!(files, 192);
    assert_eq!(spans, 3150);
}

#[test]
fn every_decision_of_every_dump() {
    // The five inputs the earlier generator wrote contexts files for.
    const CONTEXT_INPUTS: [&str; 5] = ["spanish", "french", "numbers", "urls-emails", "supplementary"];
    let Some(models) = ud_models() else { return };
    let mut positions = 0;
    let mut in_context_inputs = 0;
    let mut files = 0;
    for (language, path) in models {
        let m = read_model(&path);
        for case in run("tokenize", run_name(ud_model_name(language))) {
            let n = check_dump(&case, &m);
            if case.set() == "edge" && CONTEXT_INPUTS.contains(&case.golden.header.input.id.as_str()) {
                in_context_inputs += n;
            }
            positions += n;
            files += 1;
        }
    }
    assert_eq!(files, 222);
    assert_eq!(in_context_inputs, 2445, "the positions of the earlier contexts files");
    assert_eq!(positions, 12860, "every decision of every dump");
}

/// A language, a surrogate input and the UTF-8 spans the port gives for it.
type OwnFile = (&'static str, &'static str, &'static [(u32, u32)]);

#[test]
fn a_split_between_surrogate_halves_is_not_taken() {
    // Every model runs every input of the surrogate set; the file name
    // starts with the language whose model splits inside U+1F600 there.
    // The port keeps the character whole and multiplies on. The UTF-8
    // spans it gives for the files of each model's own language:
    let own: [OwnFile; 6] = [
        ("en", "en-quote-emoji-dot", &[(0, 1), (1, 5), (5, 6)]),
        ("en", "en-dot-emoji-quote", &[(0, 5), (5, 6)]),
        ("de", "de-a-emoji-apostrophe", &[(0, 5), (5, 6)]),
        ("es", "es-paren-emoji-apostrophe", &[(0, 5), (5, 6)]),
        ("fr", "fr-comma-emoji-dollar", &[(0, 6)]),
        ("pt", "pt-a-emoji-question", &[(0, 5), (5, 6)]),
    ];
    let Some(models) = ud_models() else { return };
    let mut files = 0;
    let mut joined = 0;
    for (language, path) in models {
        let t = TokenizerMe::new(read_model(&path), TokenizerOptions::default());
        for case in run("tokenize", run_name(ud_model_name(language))).into_iter().filter(|c| c.set() == "surrogate") {
            case.assert_annotator(CLASS, "tokenizePos");
            assert!(case.golden.header.config.is_empty());
            case.assert_models(std::slice::from_ref(&path));
            let id = case.golden.header.input.id.clone();
            let jvm = case.jvm_spans();
            let expected = expected_me_spans(&case.text, &jvm);
            if let Some((_, _, want)) = own.iter().find(|(l, n, _)| *l == language && *n == id) {
                assert!(jvm.iter().any(|s| s.cut), "{id}: the JVM splits inside the pair");
                assert_eq!(expected.iter().map(|&(a, b, _)| (a, b)).collect::<Vec<_>>(), *want, "{id}");
            }
            joined += jvm.len() - expected.len();
            check(&t, &case);
            files += 1;
        }
    }
    assert_eq!(files, 30);
    // The halves the JVM split, over all thirty runs, that came out whole.
    assert_eq!(joined, 8);
}

#[test]
fn the_alphanumeric_optimization_with_each_language_pattern() {
    // The UD models rebuilt with the optimization on and their language's
    // pattern: a unit the pattern accepts is a token of probability 1, any
    // other goes to the model as without it.
    let Some(models) = ud_models() else { return };
    let mut files = 0;
    let mut positions = 0;
    for (language, path) in models {
        let mut tokenizer = None;
        for case in run("tokenize", &format!("{}+alphanumeric", run_name(ud_model_name(language)))) {
            case.assert_annotator(CLASS, "tokenizePos");
            case.assert_models(std::slice::from_ref(&path));
            assert_eq!(case.config("useAlphaNumericOptimization"), Some("true"));
            let pattern = case.config("alphaNumericPattern").expect("the pattern the tokenizer used").to_string();
            assert_eq!(pattern, language_pattern(language));
            assert_eq!(case.golden.header.config.len(), 2);
            let t: &TokenizerMe = tokenizer.get_or_insert_with(|| {
                let m = read_model(&path).with_alphanumeric_optimization(Some(&pattern)).unwrap();
                TokenizerMe::new(m, TokenizerOptions::default())
            });
            assert_eq!(t.tokenizer_model().alphanumeric_pattern(), pattern);
            assert!(t.tokenizer_model().alphanumeric().is_some());
            check(t, &case);
            positions += check_dump(&case, t.tokenizer_model());
            files += 1;
        }
        // French lists the accented letters, English does not.
        let t = tokenizer.unwrap();
        let mut out = Vec::new();
        t.tokenize("caf\u{e9}", &mut out);
        assert_eq!(out.len() == 1 && out[0].1 == 1.0, matches!(language, "fr" | "es" | "pt" | "de"), "{language}");
    }
    assert_eq!(files, 5 * 35);
    assert_eq!(positions, 6834, "every decision of every dump");
}
