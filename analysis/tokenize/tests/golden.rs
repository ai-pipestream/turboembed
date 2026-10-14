//! The tokenizers against what OpenNLP's `SimpleTokenizer`,
//! `WhitespaceTokenizer` and UAX #29 `WordTokenizer` returned on the JVM
//! for the same inputs (analysis/goldens/data/tokenize). Spans compare
//! exactly, on UTF-8 and on UTF-16 offsets.

mod common;

use std::num::NonZeroU32;

use common::{Case, JvmSpan, compare_layers, run};
use turbo_analysis::Span;
use turbo_goldens::expect::{simple_newline_spans, whole_character_spans};
use turbo_tokenize::{
    TokenizerOptions, WhitespaceMode, WordTokenizer, WordType, simple_spans, whitespace_spans, word_tokens,
};
use turbo_unicode::utf16::byte_to_utf16;

/// One span with both offsets and its type, if any: what a golden row
/// says and what the Rust side is compared on.
#[derive(Clone, Debug, PartialEq)]
struct Expect {
    utf8: (u32, u32),
    utf16: (u32, u32),
    ty: Option<WordType>,
}

/// The spans a golden expects. Two rows holding the halves of one
/// character outside the BMP become one span covering the character:
/// the documented deviation of the simple tokenizer
/// (`turbo_goldens::expect::whole_character_spans`).
fn expected(case: &Case) -> Vec<Expect> {
    let ty = |s: &JvmSpan| s.kind.as_deref().map(|t| WordType::from_java_name(t).expect("a word type name"));
    whole_character_spans(&case.text, &case.jvm_spans())
        .unwrap_or_else(|e| panic!("{}: {e}", case.path.display()))
        .iter()
        .map(|s| Expect { utf8: s.utf8, utf16: s.utf16, ty: ty(s) })
        .collect()
}

/// Rust spans of `text` with their UTF-16 offsets worked out.
fn actual(text: &str, spans: &[Span], types: Option<&[WordType]>) -> Vec<Expect> {
    let unit = |b: u32| byte_to_utf16(text, b as usize).expect("a span on a character boundary") as u32;
    spans
        .iter()
        .enumerate()
        .map(|(i, s)| Expect { utf8: (s.start, s.end), utf16: (unit(s.start), unit(s.end)), ty: types.map(|t| t[i]) })
        .collect()
}

/// The run's goldens, checked to be of `class` with `keepNewLines` as
/// given and in `mode`.
fn cases(name: &str, class: &str, keep_newlines: bool, mode: WhitespaceMode) -> Vec<Case> {
    let cases = run("tokenize", name);
    for c in &cases {
        c.assert_annotator(class, if class.ends_with("WordTokenizer") { "tokenizeTyped" } else { "tokenizePos" });
        assert!(c.golden.header.models.is_empty(), "{}", c.path.display());
        assert_eq!(c.config("keepNewLines"), keep_newlines.then_some("true"), "{}", c.path.display());
        assert_eq!(c.whitespace(), mode, "{}", c.path.display());
        assert_eq!(c.golden.header.input.set, "edge");
    }
    assert_eq!(cases.len(), 34, "{name}");
    cases
}

const SIMPLE: &str = "opennlp.tools.tokenize.SimpleTokenizer";
const WHITESPACE: &str = "opennlp.tools.tokenize.WhitespaceTokenizer";

/// Compares a spans-only tokenizer with every golden of `name`.
fn spans_match(name: &str, class: &str, run: fn(&str, TokenizerOptions, &mut Vec<Span>), keep_newlines: bool) {
    let mode = if name.ends_with("+legacy") { WhitespaceMode::Legacy } else { WhitespaceMode::Unicode };
    let options = TokenizerOptions { whitespace: mode, keep_newlines };
    for case in cases(name, class, keep_newlines, mode) {
        assert_eq!(case.golden.header.error, None, "{}", case.path.display());
        let mut out = Vec::new();
        run(&case.text, options, &mut out);
        assert_eq!(actual(&case.text, &out, None), expected(&case), "{}", case.path.display());
        let spans: Vec<(u32, u32)> = out.iter().map(|s| (s.start, s.end)).collect();
        compare_layers(&case, &spans, None);
    }
}

#[test]
fn simple() {
    spans_match("simple", SIMPLE, simple_spans, false);
}

#[test]
fn simple_legacy() {
    spans_match("simple+legacy", SIMPLE, simple_spans, false);
}

#[test]
fn whitespace() {
    spans_match("whitespace", WHITESPACE, whitespace_spans, false);
}

#[test]
fn whitespace_legacy() {
    spans_match("whitespace+legacy", WHITESPACE, whitespace_spans, false);
}

#[test]
fn whitespace_newlines() {
    spans_match("whitespace+keep-new-lines", WHITESPACE, whitespace_spans, true);
}

#[test]
fn whitespace_newlines_legacy() {
    spans_match("whitespace+keep-new-lines+legacy", WHITESPACE, whitespace_spans, true);
}

/// The simple tokenizer with new lines kept. The Java has two defects in
/// this mode (it throws when a line break comes before the first token,
/// and gives a line break after whitespace the span of the whitespace),
/// which the port fixes. So the expectation for each input is worked out
/// independently: the plain simple tokens from the golden without new
/// lines (a line break is whitespace, so never inside a token), plus a
/// one-byte token at every line feed and carriage return. Where the Java
/// output equals that, the Rust output is compared with the Java output;
/// where it does not, the Rust output is compared with the corrected
/// expectation, and the inputs where that happened must be exactly the
/// ones listed, so a new difference cannot pass unnoticed.
fn simple_newlines(name: &str, plain_name: &str, affected: &[&str]) {
    let mode = if name.ends_with("+legacy") { WhitespaceMode::Legacy } else { WhitespaceMode::Unicode };
    let options = TokenizerOptions { whitespace: mode, keep_newlines: true };
    let plain = cases(plain_name, SIMPLE, false, mode);
    let mut differs = Vec::new();
    for case in cases(name, SIMPLE, true, mode) {
        let p = plain.iter().find(|p| p.name() == case.name()).expect("the same input without new lines");
        let text = &case.text;
        let plain_spans: Vec<(u32, u32)> = expected(p).iter().map(|e| e.utf8).collect();
        let unit = |b: u32| byte_to_utf16(text, b as usize).unwrap() as u32;
        let corrected: Vec<Expect> = simple_newline_spans(text, &plain_spans)
            .into_iter()
            .map(|(a, b)| Expect { utf8: (a, b), utf16: (unit(a), unit(b)), ty: None })
            .collect();
        let mut out = Vec::new();
        simple_spans(text, options, &mut out);
        let rust = actual(text, &out, None);
        if case.golden.header.error.is_none() && expected(&case) == corrected {
            assert_eq!(rust, expected(&case), "{}", case.path.display());
        } else {
            differs.push(case.golden.header.input.id.clone());
            assert_eq!(rust, corrected, "{} (corrected)", case.path.display());
        }
    }
    assert_eq!(differs, affected, "{name}: the inputs the Java defects affect");
}

#[test]
fn simple_newlines_unicode() {
    simple_newlines("simple+keep-new-lines", "simple", &["leading-trailing", "whitespace-only"]);
}

#[test]
fn simple_newlines_legacy() {
    simple_newlines("simple+keep-new-lines+legacy", "simple+legacy", &["leading-trailing", "whitespace-only"]);
}

#[test]
fn uax29() {
    let max = WordTokenizer::DEFAULT_MAX_TOKEN_LENGTH;
    let mut tokens = 0;
    for case in cases("uax29", "opennlp.tools.tokenize.uax29.WordTokenizer", false, WhitespaceMode::Unicode) {
        assert_eq!(case.golden.header.error, None, "{}", case.path.display());
        assert!(case.golden.layer("turbo:word-types").is_some(), "{}", case.path.display());
        let (mut spans, mut types) = (Vec::new(), Vec::new());
        word_tokens(&case.text, max, &mut spans, &mut types);
        let want = expected(&case);
        assert!(want.iter().all(|e| e.ty.is_some()), "{}", case.path.display());
        assert_eq!(actual(&case.text, &spans, Some(&types)), want, "{}", case.path.display());
        tokens += want.len();
    }
    assert_eq!(tokens, 450);
    assert_eq!(max, NonZeroU32::new(255).unwrap());
}
