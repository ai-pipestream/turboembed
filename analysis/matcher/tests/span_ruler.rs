//! The span ruler: spaCy's `SpanRuler` tests (spaCy, MIT:
//! `spacy/tests/pipeline/test_span_ruler.py`) ported with the same
//! expected spans, and the ruler as a pipeline step writing a named
//! `turbo:matches` instance whose offsets are checked against the source.

mod common;

use common::Doc;
use turbo_analysis::{Analyzer, Document, FilterRule, Key, Layer, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{Matches, PosTags, Tokens};
use turbo_matcher::{Attr, Field, Flag, MatchError, Policy, SpanRuler, Tok};
use turbo_stopword_layers::{StopwordLanguage, Stopwords};

fn spans(ruler: &SpanRuler, text: &str) -> Vec<(String, String)> {
    let doc = Doc::split(text);
    let m = ruler.matches(&doc.input(), ViewId::SOURCE).unwrap();
    assert_eq!(m.check(), Ok(()));
    m.spans
        .iter()
        .map(|(i, s)| {
            assert_eq!(doc.text_of(m.tokens[i.get()]), &doc.text[s.range()]);
            (m.label_of(i).to_string(), doc.text[s.range()].to_string())
        })
        .collect()
}

fn owned(v: &[(&str, &str)]) -> Vec<(String, String)> {
    v.iter().map(|&(a, b)| (a.to_string(), b.to_string())).collect()
}

fn ruler(policy: Policy) -> SpanRuler {
    SpanRuler::new(Field::Attr(Attr::Text), policy)
}

/// spaCy's `person_org_date_patterns`.
fn person_org_date() -> SpanRuler {
    let mut r = ruler(Policy::Keep);
    r.add_phrase("PERSON", &["Dina"]).unwrap();
    r.add_phrase("ORG", &["ACME"]).unwrap();
    r.add_phrase("ORG", &["ACM"]).unwrap();
    r.add_phrase("DATE", &["June", "14th"]).unwrap();
    r
}

#[test]
fn span_ruler_init() {
    let mut r = ruler(Policy::Keep);
    r.add_phrase("HELLO", &["hello", "world"]).unwrap();
    r.add_pattern("BYE", &[Tok::lower("bye"), Tok::lower("bye")]).unwrap();
    r.add_pattern("HELLO", &[Tok::text("HELLO")]).unwrap();
    r.add_pattern("COMPLEX", &[Tok::text("foo").star()]).unwrap();
    r.add_phrase("TECH_ORG", &["Apple"]).unwrap();
    r.add_phrase("TECH_ORG", &["Microsoft"]).unwrap();
    assert_eq!(r.len(), 6);
    assert_eq!(r.labels().len(), 4);
    assert!(r.labels().tag("HELLO").is_some() && r.labels().tag("BYE").is_some());
    assert_eq!(spans(&r, "hello world bye bye"), owned(&[("HELLO", "hello world"), ("BYE", "bye bye")]));
}

#[test]
fn span_ruler_overlapping_spans() {
    let mut r = ruler(Policy::Keep);
    r.add_phrase("FOOBAR", &["foo", "bar"]).unwrap();
    r.add_phrase("BARBAZ", &["bar", "baz"]).unwrap();
    assert_eq!(spans(&r, "foo bar baz"), owned(&[("FOOBAR", "foo bar"), ("BARBAZ", "bar baz")]));
}

#[test]
fn span_ruler_spans_filter() {
    // spaCy's `first_longest_spans_filter`.
    let mut r = ruler(Policy::Filter(FilterRule::LongestFirst));
    r.add_phrase("FOOBAR", &["foo", "bar"]).unwrap();
    r.add_phrase("BARBAZ", &["bar", "baz"]).unwrap();
    assert_eq!(spans(&r, "foo bar baz"), owned(&[("FOOBAR", "foo bar")]));
}

#[test]
fn longest_in_tokens_as_spacy_filter_spans() {
    // Byte lengths and token lengths disagree: "Internationalization" is
    // one long token, "New York City" three short ones. spaCy's
    // `filter_spans` counts tokens and keeps "New York City";
    // `FilterRule::LongestFirst` counts bytes and keeps the long token.
    let text = "Internationalization New York City";
    let add = |r: &mut SpanRuler| {
        r.add_pattern("WORD", &[Tok::text("Internationalization"), Tok::text("New")]).unwrap();
        r.add_phrase("CITY", &["New", "York", "City"]).unwrap();
    };
    let mut tokens = ruler(Policy::LongestTokens);
    add(&mut tokens);
    assert_eq!(spans(&tokens, text), owned(&[("CITY", "New York City")]));
    let mut bytes = ruler(Policy::Filter(FilterRule::LongestFirst));
    add(&mut bytes);
    assert_eq!(spans(&bytes, text), owned(&[("WORD", "Internationalization New")]));
    // Equally long: the earliest; over the same tokens: the label added first.
    let mut r = ruler(Policy::LongestTokens);
    r.add_phrase("B", &["bar", "baz"]).unwrap();
    r.add_phrase("A", &["foo", "bar"]).unwrap();
    r.add_phrase("A", &["bar", "baz"]).unwrap();
    assert_eq!(spans(&r, "foo bar baz"), owned(&[("A", "foo bar")]));
    assert_eq!(spans(&r, "bar baz"), owned(&[("B", "bar baz")]));
}

#[test]
fn phrases_over_several_fields() {
    let doc = Doc::split("Ada wrote NOTES on engines").with_lemmas(&["Ada", "write", "note", "on", "engine"]);
    let mut r = SpanRuler::new(Field::Attr(Attr::Lower), Policy::Keep);
    let notes = r.add_phrase("TEXT", &["notes"]).unwrap();
    let write = r.add_phrase_on(Field::Attr(Attr::Lemma), "ACT", &["write", "note"]).unwrap();
    assert_eq!(r.add_phrase_on(Field::Attr(Attr::Lower), "TEXT", &["NOTES"]).unwrap(), notes);
    assert_eq!(r.add_phrase_on(Field::Attr(Attr::Lemma), "ACT", &["write", "note"]).unwrap(), write);
    let engine = r.add_phrase_on(Field::Attr(Attr::Lemma), "TEXT", &["engine"]).unwrap();
    assert_eq!(r.phrases().len(), 2);
    let m = r.matches(&doc.input(), ViewId::SOURCE).unwrap();
    assert_eq!(m.check(), Ok(()));
    let got: Vec<(&str, &str, u32)> =
        m.spans.iter().map(|(i, s)| (m.label_of(i), &doc.text[s.range()], m.rule[i.get()])).collect();
    assert_eq!(got, [("ACT", "wrote NOTES", write), ("TEXT", "NOTES", notes), ("TEXT", "engines", engine)]);
    // Removing a label removes it from every phrase matcher.
    r.remove("TEXT").unwrap();
    let m = r.matches(&doc.input(), ViewId::SOURCE).unwrap();
    assert_eq!(m.label.len(), 1);
    assert_eq!(m.label_of(turbo_analysis::Ix::new(0)), "ACT");
}

#[test]
fn span_ruler_remove_basic() {
    let mut r = person_org_date();
    r.remove("DATE").unwrap();
    assert_eq!(r.len(), 3);
    assert_eq!(spans(&r, "Dina went to school"), owned(&[("PERSON", "Dina")]));
    r.remove("PERSON").unwrap();
    assert!(spans(&r, "Dina went to school").is_empty());
    assert_eq!(r.len(), 2);
}

#[test]
fn span_ruler_remove_nonexisting_pattern() {
    let mut r = person_org_date();
    assert_eq!(r.remove("NE"), Err(MatchError::NoLabel("NE".into())));
}

#[test]
fn span_ruler_remove_several_patterns() {
    let mut r = person_org_date();
    r.remove("DATE").unwrap();
    assert_eq!(spans(&r, "Dina founded the company ACME ."), owned(&[("PERSON", "Dina"), ("ORG", "ACME")]));
    r.remove("PERSON").unwrap();
    assert_eq!(r.len(), 2);
    assert_eq!(spans(&r, "Dina founded the company ACME"), owned(&[("ORG", "ACME")]));
    r.remove("ORG").unwrap();
    assert_eq!(r.len(), 0);
    assert!(spans(&r, "Dina founded the company ACME").is_empty());
}

#[test]
fn span_ruler_remove_patterns_in_a_row() {
    let mut r = person_org_date();
    assert_eq!(
        spans(&r, "Dina founded the company ACME on June 14th"),
        owned(&[("PERSON", "Dina"), ("ORG", "ACME"), ("DATE", "June 14th")])
    );
    r.remove("ORG").unwrap();
    r.remove("DATE").unwrap();
    assert_eq!(spans(&r, "Dina went to school").len(), 1);
}

#[test]
fn span_ruler_remove_all_patterns() {
    let mut r = person_org_date();
    assert_eq!(r.len(), 4);
    r.remove("PERSON").unwrap();
    assert_eq!(r.len(), 3);
    r.remove("ORG").unwrap();
    assert_eq!(r.len(), 1);
    r.remove("DATE").unwrap();
    assert_eq!(r.len(), 0);
    assert!(spans(&r, "Dina founded the company ACME on June 14th").is_empty());
}

#[test]
fn span_ruler_remove_and_add() {
    let text = "I saw him last time we met , this time he brought some flowers";
    let mut r = ruler(Policy::Keep);
    r.add_phrase("DATE1", &["last", "time"]).unwrap();
    assert_eq!(spans(&r, text), owned(&[("DATE1", "last time")]));
    r.add_phrase("DATE2", &["this", "time"]).unwrap();
    assert_eq!(r.len(), 2);
    assert_eq!(spans(&r, text), owned(&[("DATE1", "last time"), ("DATE2", "this time")]));
    r.remove("DATE1").unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(spans(&r, text), owned(&[("DATE2", "this time")]));
    r.add_phrase("DATE1", &["last", "time"]).unwrap();
    assert_eq!((r.len(), spans(&r, text).len()), (2, 2));
    let longer = "I saw him last time we met , this time he brought some flowers , another time some chocolate .";
    r.add_phrase("DATE3", &["another", "time"]).unwrap();
    assert_eq!((r.len(), spans(&r, longer).len()), (3, 3));
    r.remove("DATE3").unwrap();
    assert_eq!((r.len(), spans(&r, longer).len()), (2, 2));
}

#[test]
fn one_label_over_the_same_tokens_is_one_match() {
    let mut r = ruler(Policy::Keep);
    let phrase = r.add_phrase("ORG", &["ACME"]).unwrap();
    let pattern = r.add_pattern("ORG", &[Tok::flag(Flag::Upper, true)]).unwrap();
    assert_eq!(r.add_phrase("ORG", &["ACME"]), Ok(phrase));
    let doc = Doc::split("ACME and IBM");
    let m = r.matches(&doc.input(), ViewId::SOURCE).unwrap();
    assert_eq!(m.rule, [phrase, pattern]);
    assert_eq!(r.rule_label(pattern), "ORG");
}

/// A pipeline: the ruler bound to write `Key::named("rules")`, reading the
/// source view, the tokens, part-of-speech tags and stopword flags.
#[test]
fn the_step_writes_a_named_instance_with_source_offsets() {
    let text = "Résumé: the café sells fresh  bread and the bakery sells cakes.";
    let words: Vec<(usize, &str)> = text
        .split(' ')
        .scan(0, |at, w| {
            let start = *at;
            *at += w.len() + 1;
            Some((start, w))
        })
        .filter(|(_, w)| !w.is_empty())
        .collect();
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(start, w) in &words {
        spans.push(Span::new(start as u32, (start + w.len()) as u32));
    }
    let tokens = Tokens { spans, sentence: None };
    let tags = ["NN", "DT", "NN", "VBZ", "JJ", "NN", "CC", "DT", "NN", "VBZ", "NNS"];
    let tagset = TagSet::new("ptb", &["CC", "DT", "JJ", "NN", "NNS", "VBZ"]);
    let pos = PosTags { tags: tags.iter().map(|t| tagset.tag(t).unwrap()).collect(), tagset, prob: None };
    let stop = Stopwords {
        language: StopwordLanguage::English,
        stop: words.iter().map(|(_, w)| matches!(*w, "the" | "and")).collect(),
    };
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, pos, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, stop, Provenance::by("test")).unwrap();

    let mut r = SpanRuler::new(Field::Attr(Attr::Lower), Policy::Keep);
    // A determiner that is a stopword, an optional adjective, a noun.
    let np = [
        Tok::attr(Attr::Pos, "DT").and(Tok::flag(Flag::Stop, true)),
        Tok::attr(Attr::Pos, "JJ").opt(),
        Tok::attr_in(Attr::Pos, ["NN", "NNS"]),
    ];
    let np_rule = r.add_pattern("NP", &np).unwrap();
    let shop = r.add_phrase("SHOP", &["Café"]).unwrap();
    let shop2 = r.add_phrase("SHOP", &["the", "BAKERY"]).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .given(Key::<Stopwords>::DEFAULT)
        .step(r.bound("rules"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let doc = &docs[0];
    assert!(doc.get::<Matches>().is_none(), "only the named instance is written");
    let key = Key::<Matches>::named("rules");
    let m = doc.get_key(&key).unwrap();
    let found: Vec<(&str, &str, u32)> =
        m.spans.iter().map(|(i, s)| (m.label_of(i), &doc.source()[s.range()], m.rule[i.get()])).collect();
    assert_eq!(
        found,
        [
            ("NP", "the café", np_rule),
            ("SHOP", "café", shop),
            ("NP", "the bakery", np_rule),
            ("SHOP", "the bakery", shop2),
        ]
    );
    let p = doc.provenance(key.id()).unwrap();
    assert_eq!(p.annotator, "turbo-matcher:span-ruler");
    assert_eq!(p.reads.len(), 4, "the view, the tokens, the tags and the stopwords: {:?}", p.reads);
    assert_eq!(Matches::ID, "turbo:matches");
}

#[test]
fn the_step_says_which_layer_a_pattern_lacks() {
    let mut doc = Document::new("a b");
    let tokens = Doc::split("a b").tokens;
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
    let mut r = ruler(Policy::Keep);
    r.add_pattern("X", &[Tok::attr(Attr::Lemma, "a")]).unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).step(r.bound("rules")).build().unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    let err = results[0].as_ref().unwrap_err().to_string();
    assert!(err.contains("opennlp:lemmas"), "{err}");
}
