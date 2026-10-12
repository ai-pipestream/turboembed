//! TokenizerMe against the files the OpenNLP JVM wrote: the tokens and
//! probabilities of every golden input for the five UD models, the
//! features and probabilities of every position in the contexts files,
//! and the one documented difference, a split between surrogate halves.

mod me_common;

use std::collections::BTreeMap;

use me_common::*;
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::TokenizerMe;
use turbo_tokenize::me::alnum::language_pattern;
use turbo_unicode::WhitespaceMode;
use turbo_unicode::utf16::byte_to_utf16;

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
        TokenizerMe::new(model, TokenizerOptions::default())
    });
    TokenizerMe::new(t.tokenizer_model().clone(), TokenizerOptions { whitespace: mode, keep_newlines: false })
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
    assert_eq!(spans, 2298, "every span line of every file");
}

#[test]
fn every_context_and_its_probabilities() {
    let mut positions = 0;
    for language in LANGUAGES {
        let path = golden_dir().join(format!("contexts/tokenizer-me-{language}.txt"));
        let c = Contexts::read(&path);
        assert_eq!(c.header("component"), format!("tokenizer-me-{language}-contexts"));
        assert_eq!(c.inputs.len(), 5);
        positions += check_contexts(&path, &model(c.header("model")));
    }
    assert_eq!(positions, 2445, "every position of every contexts file");
}

/// A language, a file of its surrogate run and the UTF-8 spans the port
/// gives for it.
type OwnFile = (&'static str, &'static str, &'static [(u32, u32)]);

#[test]
fn a_split_between_surrogate_halves_is_not_taken() {
    // Every model runs every input of variants/surrogate; the file name
    // starts with the language whose model splits inside U+1F600 there.
    // The port keeps the character whole and multiplies on. The UTF-8
    // spans it gives for the files of each model's own language:
    let own: [OwnFile; 6] = [
        ("en", "en-quote-emoji-dot.txt", &[(0, 1), (1, 5), (5, 6)]),
        ("en", "en-dot-emoji-quote.txt", &[(0, 5), (5, 6)]),
        ("de", "de-a-emoji-apostrophe.txt", &[(0, 5), (5, 6)]),
        ("es", "es-paren-emoji-apostrophe.txt", &[(0, 5), (5, 6)]),
        ("fr", "fr-comma-emoji-dollar.txt", &[(0, 6)]),
        ("pt", "pt-a-emoji-question.txt", &[(0, 5), (5, 6)]),
    ];
    let mut files = 0;
    let mut joined = 0;
    for language in LANGUAGES {
        let t = TokenizerMe::new(model(&ud_model_name(language)), TokenizerOptions::default());
        for path in golden_files(&golden_dir().join(format!("variants/surrogate-{language}"))) {
            let g = Golden::read(&path);
            assert_eq!(g.header("component"), format!("tokenizer-me-{language}"));
            assert_eq!(g.header("whitespace"), "unicode");
            check_model_headers(&|k| g.header(k).to_string(), t.tokenizer_model());
            let name = path.file_name().unwrap().to_str().unwrap();
            let expected = expected_spans(&g.input, &g.spans);
            if let Some((_, _, want)) = own.iter().find(|(l, n, _)| *l == language && *n == name) {
                assert!(g.spans.iter().any(|s| s.utf8.is_none()), "{name}: the JVM splits inside the pair");
                assert_eq!(expected.iter().map(|&(a, b, _)| (a, b)).collect::<Vec<_>>(), *want, "{name}");
            }
            joined += g.spans.len() - expected.len();
            check(&t, &g.input, &expected, &path.display().to_string());
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
    let mut files = 0;
    let mut positions = 0;
    for language in LANGUAGES {
        let dir = golden_dir().join(format!("variants/alphanumeric-{language}"));
        let mut tokenizer = None;
        for path in golden_files(&dir) {
            let g = Golden::read(&path);
            assert_eq!(g.header("component"), format!("tokenizer-me-{language}"));
            assert_eq!(g.header("whitespace"), "unicode");
            let pattern = unescape_str(g.header("effective-alpha-numeric-pattern"));
            assert_eq!(pattern, unescape_str(g.header("alpha-numeric-pattern")));
            assert_eq!(pattern, language_pattern(language));
            let t: &TokenizerMe = tokenizer.get_or_insert_with(|| {
                let m = model(&ud_model_name(language)).with_alphanumeric_optimization(Some(&pattern)).unwrap();
                TokenizerMe::new(m, TokenizerOptions::default())
            });
            assert_eq!(t.tokenizer_model().alphanumeric_pattern(), pattern);
            check_model_headers(&|k| g.header(k).to_string(), t.tokenizer_model());
            check(t, &g.input, &expected_spans(&g.input, &g.spans), &path.display().to_string());
            files += 1;
        }
        let t = tokenizer.unwrap();
        let path = golden_dir().join(format!("variants/contexts/tokenizer-me-alphanumeric-{language}.txt"));
        positions += check_contexts(&path, t.tokenizer_model());
        // French lists the accented letters, English does not.
        let mut out = Vec::new();
        t.tokenize("caf\u{e9}", &mut out);
        assert_eq!(out.len() == 1 && out[0].1 == 1.0, matches!(language, "fr" | "es" | "pt" | "de"), "{language}");
    }
    assert_eq!(files, 5 * 35);
    assert!(positions > 0);
}
