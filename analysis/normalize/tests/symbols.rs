//! `SymbolWords` and `SymbolJoiner`: the expectations of OpenNLP's
//! SymbolJoinerCharSequenceNormalizerTest and SymbolJoinerUsageExampleTest.

mod scanners_support;

use std::borrow::Cow;

use scanners_support::{check_alignment, run};
use turbo_analysis::{Annotator, Document, Refs, Span, Spans, View};
use turbo_analysis_layers::Tokens;
use turbo_normalize::symbols::symbol_word;
use turbo_normalize::{SymbolJoiner, SymbolWords, apply};

const MAP: [(&str, &str); 10] = [
    ("&", "and"),
    ("+", "plus"),
    ("@", "at"),
    ("%", "percent"),
    ("\u{A7}", "section"),
    ("\u{B6}", "paragraph"),
    ("\u{B0}", "degree"),
    ("\u{A9}", "copyright"),
    ("\u{AE}", "registered"),
    ("\u{2122}", "trademark"),
];

/// Inputs OpenNLP leaves unchanged when each is the whole input.
const UNCHANGED: [&str; 18] = [
    "R&D",
    "AT&T",
    "TSR\u{AE}",
    "&&",
    "& ",
    " &",
    "(R)",
    "(TM)",
    "(C)",
    "\u{A9}\u{FE0F}",
    "\u{AE}\u{FE0F}",
    "\u{FF06}",
    "\u{20AC}",
    "$",
    "\u{A3}",
    "&\u{301}",
    "\u{2122} ",
    "\u{A0}&",
];

#[test]
fn whole_symbols_spell_out() {
    for (symbol, word) in MAP {
        assert_eq!(run(&SymbolWords, symbol), word);
        assert_eq!(symbol_word(symbol.chars().next().unwrap()), Some(word));
        assert!(matches!(apply(&SymbolWords, word), Cow::Borrowed(_)));
    }
}

/// OpenNLP leaves `"& "`, `" &"`, `"™ "` and `"\u{A0}&"` alone because its
/// input must be exactly the symbol. In running text a symbol between
/// whitespace and the edge of the text stands alone, so `SymbolWords`
/// replaces it there; the rest are unchanged in both.
#[test]
fn embedded_symbols_are_left_alone() {
    let standalone = ["& ", " &", "\u{2122} ", "\u{A0}&"];
    for input in UNCHANGED {
        let out = run(&SymbolWords, input);
        if standalone.contains(&input) {
            assert_ne!(out, input, "{input:?}");
        } else {
            assert_eq!(out, input, "{input:?}");
        }
    }
    assert_eq!(run(&SymbolWords, "& "), "and ");
    assert_eq!(run(&SymbolWords, "\u{A0}&"), "\u{A0}and");
    for input in ["and", "court", "", "\u{1F600}"] {
        assert!(matches!(apply(&SymbolWords, input), Cow::Borrowed(_)));
    }
}

#[test]
fn symbols_in_running_text() {
    assert_eq!(run(&SymbolWords, "Dungeons & Dragons"), "Dungeons and Dragons");
    assert_eq!(run(&SymbolWords, "50 % off, R&D + QA \u{A9} 2026"), "50 percent off, R&D plus QA copyright 2026");
    assert_eq!(run(&SymbolWords, "a\t\u{B0}\nb"), "a\tdegree\nb");
    assert_eq!(run(&SymbolWords, "&&&"), "&&&");
}

fn joined(text: &str) -> String {
    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    let mut at = 0;
    for word in text.split(' ') {
        if !word.is_empty() {
            spans.push(Span::new(at as u32, (at + word.len()) as u32));
        }
        at += word.len() + 1;
    }
    let tokens = Tokens { spans, sentence: None };
    let input: Refs<'_, SymbolJoiner> = (view, &tokens);
    let (out,) = SymbolJoiner.annotate(input).unwrap();
    check_alignment(text, out.text(), out.alignment());
    out.text().to_string()
}

#[test]
fn joiner_replaces_one_symbol_tokens() {
    assert_eq!(joined("Dungeons & Dragons"), "Dungeons and Dragons");
    for (symbol, word) in MAP {
        assert_eq!(joined(symbol), word);
    }
    for input in ["R&D", "AT&T", "TSR\u{AE}", "&&", "(TM)", "\u{A9}\u{FE0F}", "\u{FF06}", "\u{20AC}", "&\u{301}"] {
        assert_eq!(joined(input), input);
    }
}

#[test]
fn joiner_token_maps_back_to_the_symbol() {
    let doc = Document::new("Dungeons & Dragons");
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    for (s, e) in [(0, 8), (9, 10), (11, 18)] {
        spans.push(Span::new(s, e));
    }
    let tokens = Tokens { spans, sentence: None };
    let (out,) = SymbolJoiner.annotate((view, &tokens)).unwrap();
    assert_eq!(out.alignment().span_to_parent(Span::new(9, 12)), Span::new(9, 10));
    assert_eq!(out.alignment().span_to_child(Span::new(11, 18)), Span::new(13, 20));
}
