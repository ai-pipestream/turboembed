//! The tokenizers against the files the OpenNLP JVM wrote on the same
//! inputs. Spans compare exactly, on UTF-8 and on UTF-16 offsets.

mod common;

use std::num::NonZeroU32;

use common::{Golden, GoldenSpan, read_dir, unescape};
use turbo_analysis::Span;
use turbo_tokenize::{
    TokenizerOptions, WhitespaceMode, WordTokenizer, WordType, simple_spans, whitespace_spans, word_tokens,
};
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

/// One span with both offsets and its type, if any: what a golden line
/// says and what the Rust side is compared on.
#[derive(Clone, Debug, PartialEq)]
struct Expect {
    utf8: (u32, u32),
    utf16: (u32, u32),
    ty: Option<WordType>,
}

/// The spans a golden file expects. Two `span-invalid:` halves of one
/// character outside the BMP become one span covering the character:
/// the documented deviation of the simple tokenizer.
fn expected(g: &Golden) -> Vec<Expect> {
    let text = &g.input;
    let ty = |extra: &[String]| extra.first().map(|t| WordType::from_java_name(t).expect("a word type name"));
    let mut out = Vec::new();
    let mut spans = g.spans.iter();
    while let Some(span) = spans.next() {
        match span {
            GoldenSpan::Valid { utf8, utf16, extra } => out.push(Expect { utf8: *utf8, utf16: *utf16, ty: ty(extra) }),
            GoldenSpan::Invalid { utf16: (high, mid), extra } => {
                let Some(GoldenSpan::Invalid { utf16: (mid2, low), .. }) = spans.next() else {
                    panic!("{}: a half span without its other half", g.path.display());
                };
                assert!(mid == mid2 && *mid == high + 1 && *low == mid + 1, "{}: halves of one character", g.name());
                let start = utf16_to_byte(text, *high as usize).expect("the pair starts a character");
                let end = utf16_to_byte(text, *low as usize).expect("the pair ends a character");
                assert_eq!(text[start..end].chars().count(), 1, "{}: one character", g.name());
                out.push(Expect { utf8: (start as u32, end as u32), utf16: (*high, *low), ty: ty(extra) });
            }
        }
    }
    out
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

fn mode(dir: &str) -> WhitespaceMode {
    if dir.ends_with("-legacy") { WhitespaceMode::Legacy } else { WhitespaceMode::Unicode }
}

fn check_header(g: &Golden, dir: &str) {
    assert_eq!(g.component, dir.trim_end_matches("-legacy"), "{}", g.path.display());
    let ws = if mode(dir) == WhitespaceMode::Legacy { "legacy" } else { "unicode" };
    assert_eq!(g.whitespace, ws, "{}", g.path.display());
}

/// Compares a spans-only tokenizer with every file of `dir`.
fn spans_match(dir: &str, run: fn(&str, TokenizerOptions, &mut Vec<Span>), keep_newlines: bool) {
    let options = TokenizerOptions { whitespace: mode(dir), keep_newlines };
    for g in read_dir(dir) {
        check_header(&g, dir);
        assert_eq!(g.error, None, "{}", g.path.display());
        let mut out = Vec::new();
        run(&g.input, options, &mut out);
        assert_eq!(actual(&g.input, &out, None), expected(&g), "{}", g.path.display());
    }
}

#[test]
fn simple() {
    spans_match("simple", simple_spans, false);
}

#[test]
fn simple_legacy() {
    spans_match("simple-legacy", simple_spans, false);
}

#[test]
fn whitespace() {
    spans_match("whitespace", whitespace_spans, false);
}

#[test]
fn whitespace_legacy() {
    spans_match("whitespace-legacy", whitespace_spans, false);
}

#[test]
fn whitespace_newlines() {
    spans_match("whitespace-newlines", whitespace_spans, true);
}

#[test]
fn whitespace_newlines_legacy() {
    spans_match("whitespace-newlines-legacy", whitespace_spans, true);
}

/// The simple tokenizer with new lines kept. The Java has two defects in
/// this mode (it throws when a line break comes before the first token,
/// and gives a line break after whitespace the span of the whitespace),
/// which the port fixes. So the expectation for each input is worked out
/// independently: the plain simple tokens from the golden file without
/// new lines (a line break is whitespace, so never inside a token), plus
/// a one-byte token at every line feed and carriage return. Where the
/// Java output equals that, the Rust output is compared with the Java
/// output; where it does not, the Rust output is compared with the
/// corrected expectation, and the inputs where that happened must be
/// exactly the ones listed, so a new difference cannot pass unnoticed.
fn simple_newlines(dir: &str, plain_dir: &str, affected: &[&str]) {
    let options = TokenizerOptions { whitespace: mode(dir), keep_newlines: true };
    let plain = read_dir(plain_dir);
    let mut differs = Vec::new();
    for g in read_dir(dir) {
        check_header(&g, dir);
        let p = plain.iter().find(|p| p.name() == g.name()).expect("the same input without new lines");
        assert_eq!(p.input, g.input);
        let text = &g.input;
        let mut corrected = expected(p);
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' || b == b'\r' {
                let unit = byte_to_utf16(text, i).unwrap() as u32;
                corrected.push(Expect { utf8: (i as u32, i as u32 + 1), utf16: (unit, unit + 1), ty: None });
            }
        }
        corrected.sort_by_key(|e| e.utf8);
        let mut out = Vec::new();
        simple_spans(text, options, &mut out);
        let rust = actual(text, &out, None);
        if g.error.is_none() && expected(&g) == corrected {
            assert_eq!(rust, expected(&g), "{}", g.path.display());
        } else {
            differs.push(g.name().to_string());
            assert_eq!(rust, corrected, "{} (corrected)", g.path.display());
        }
    }
    assert_eq!(differs, affected, "{dir}: the inputs the Java defects affect");
}

#[test]
fn simple_newlines_unicode() {
    simple_newlines("simple-newlines", "simple", &["leading-trailing", "whitespace-only"]);
}

#[test]
fn simple_newlines_legacy() {
    simple_newlines("simple-newlines-legacy", "simple-legacy", &["leading-trailing", "whitespace-only"]);
}

#[test]
fn uax29() {
    let max = WordTokenizer::DEFAULT_MAX_TOKEN_LENGTH;
    for g in read_dir("uax29") {
        check_header(&g, "uax29");
        assert_eq!(g.error, None, "{}", g.path.display());
        let (mut spans, mut types) = (Vec::new(), Vec::new());
        word_tokens(&g.input, max, &mut spans, &mut types);
        assert_eq!(actual(&g.input, &spans, Some(&types)), expected(&g), "{}", g.path.display());
    }
    assert_eq!(max, NonZeroU32::new(255).unwrap());
}

#[test]
fn unescapes_the_format() {
    assert_eq!(unescape(r"a\\b\n\r\t 😀"), "a\\b\n\r\t\u{a0}\u{1f600}");
    assert_eq!(unescape(""), "");
}
