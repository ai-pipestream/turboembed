//! The attributes and flags patterns read from the input's layers, and
//! the input's checks.

mod common;

use common::Doc;
use turbo_analysis::{Ix, IxRange, Span, Spans, ViewId};
use turbo_analysis_layers::{Multiplier, NumberMention, NumberNotation, Numbers, ScaleMarker, Sentences, Strs, Tokens};
use turbo_matcher::{Attr, Field, Flag, Input, MatchError, Matcher, PhraseMatcher, Tok};

fn one(label: &str, pattern: &[Tok]) -> Matcher {
    let mut m = Matcher::new();
    m.add(label, pattern).unwrap();
    m
}

#[test]
fn shape_prefix_and_suffix() {
    let doc = Doc::split("Call 555-1234 or Mr. Smith at 9");
    assert_eq!(doc.texts(&one("PHONE", &[Tok::attr(Attr::Shape, "ddd-dddd")])), ["555-1234"]);
    assert_eq!(
        doc.texts(&one("TITLE", &[Tok::attr(Attr::Shape, "Xx."), Tok::attr(Attr::Shape, "Xxxxx")])),
        ["Mr. Smith"]
    );
    assert_eq!(doc.texts(&one("S", &[Tok::attr(Attr::Prefix, "S")])), ["Smith"]);
    assert_eq!(doc.texts(&one("ITH", &[Tok::attr(Attr::Suffix, "ith")])), ["Smith"]);
    let mut m = PhraseMatcher::new(Field::Attr(Attr::Shape));
    m.add("NAME", &["Dr.", "Jones"]).unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 1);
}

#[test]
fn entities_lemmas_and_norms() {
    let doc = Doc::split("Ada Lovelace wrote notes")
        .with_ents(&[(0, 2, "PERSON")])
        .with_lemmas(&["Ada", "Lovelace", "write", "note"])
        .with_norms(&["ada", "lovelace", "wrote", "note"]);
    assert_eq!(doc.texts(&one("P", &[Tok::attr(Attr::Entity, "PERSON").plus()])), ["Ada", "Ada Lovelace", "Lovelace"]);
    // A token no entity covers has no entity value: `NotIn` passes it.
    let not_person = one("N", &[Tok::attr_not_in(Attr::Entity, ["PERSON"])]);
    assert_eq!(doc.texts(&not_person), ["wrote", "notes"]);
    assert_eq!(
        doc.texts(&one("W", &[Tok::attr(Attr::Lemma, "write"), Tok::attr(Attr::Norm, "note")])),
        ["wrote notes"]
    );
    let mut m = PhraseMatcher::new(Field::Attr(Attr::Lemma));
    m.add("WRITE_NOTE", &["write", "note"]).unwrap();
    assert_eq!(doc.run_phrases(&m).len(), 1);
}

#[test]
fn sentence_starts_numbers_and_spaces() {
    let text = "It cost 1,000 dollars. Then 5 more.";
    let words = ["It", "cost", "1", ",", "000", "dollars", ".", "Then", "5", "more", "."];
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0;
    for w in words {
        let start = at + text[at..].find(w).unwrap();
        spans.push(Span::new(start as u32, (start + w.len()) as u32));
        at = start + w.len();
    }
    let sentence: Vec<Ix<Sentences>> = (0..words.len()).map(|i| Ix::new(u32::from(i >= 7))).collect();
    let tokens = Tokens { spans, sentence: Some(sentence) };
    let mut numbers = Numbers::new(ViewId::SOURCE);
    for (span, digits) in [(Span::new(8, 13), "1000"), (Span::new(28, 29), "5")] {
        numbers.push(NumberMention {
            span,
            numeral_end: span.end,
            negative: false,
            coefficient: digits,
            scale: 0,
            multiplier: Multiplier::One,
            marker: ScaleMarker::None,
            notation: NumberNotation::LatinUs,
        });
    }
    let input = Input::new(text, &tokens).unwrap().with_numbers(&numbers);
    let texts = |m: &Matcher| -> Vec<&str> {
        m.find(&input).unwrap().iter().map(|x| &text[input.span(x.tokens).range()]).collect()
    };
    assert_eq!(
        texts(&one("N", &[Tok::flag(Flag::Number, true).plus(), Tok::lower("dollars")])),
        ["1,000 dollars", ",000 dollars", "000 dollars"]
    );
    assert_eq!(texts(&one("S", &[Tok::flag(Flag::SentStart, true)])), ["It", "Then"]);
    let glued = one("G", &[Tok::flag(Flag::SpaceAfter, false), Tok::any()]);
    assert_eq!(texts(&glued), ["1,", ",000", "dollars.", "more."]);
    // Without the layers those flags read, the call says which is missing.
    let bare = Tokens { spans: tokens.spans.clone(), sentence: None };
    let input = Input::new(text, &bare).unwrap();
    assert_eq!(
        one("S", &[Tok::flag(Flag::SentStart, true)]).find(&input),
        Err(MatchError::Missing("opennlp:sentences"))
    );
    assert_eq!(one("N", &[Tok::flag(Flag::Number, true)]).find(&input), Err(MatchError::Missing("turbo:numbers")));
    assert_eq!(one("T", &[Tok::flag(Flag::Stop, true)]).find(&input), Err(MatchError::Missing("turbo:stopwords")));
}

#[test]
fn the_input_checks_what_it_is_given() {
    let doc = Doc::split("a b c");
    let mut two = Strs::default();
    two.push("a");
    two.push("b");
    let input = Input::new(&doc.text, &doc.tokens).unwrap();
    assert_eq!(
        input.with_lemmas(&two).err(),
        Some(MatchError::Misaligned { layer: "opennlp:lemmas", len: 2, tokens: 3 })
    );
    assert_eq!(
        input.with_stopwords(&[true]).err(),
        Some(MatchError::Misaligned { layer: "turbo:stopwords", len: 1, tokens: 3 })
    );
    // A token past the text, or cutting a character.
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 9));
    let tokens = Tokens { spans, sentence: None };
    assert_eq!(Input::new("short", &tokens).err(), Some(MatchError::Token { index: 0, start: 0, end: 9 }));
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 1));
    let tokens = Tokens { spans, sentence: None };
    assert_eq!(Input::new("é", &tokens).err(), Some(MatchError::Token { index: 0, start: 0, end: 1 }));
    // A range past the tokens.
    let m = one("A", &[Tok::text("a")]);
    assert_eq!(m.find_in(&input, IxRange::new(1, 4)), Err(MatchError::Range { start: 1, end: 4, tokens: 3 }));
}
