//! Focused tests for the MultiCoNER v2 kept-corpus reader of issue #1533.
//!
//! Every fixture here is original, written for these tests.

use turbo_formats::ner::{MulticonerError, MulticonerReader};

fn read(source: &str) -> Vec<Result<String, String>> {
    MulticonerReader::new(source).unwrap().map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string())).collect()
}

#[test]
fn reads_a_row_and_keeps_the_source_bytes() {
    let line =
        "{\"tokens\": [\"John\", \"works\", \"at\", \"Siemens\"], \"ner_tags\": [\"B-PER\", \"O\", \"O\", \"B-ORG\"]}";
    let sample = MulticonerReader::new(line).unwrap().next().unwrap().unwrap();
    assert_eq!(sample.to_string(), "<START:PER> John <END> works at <START:ORG> Siemens <END>");
    assert!(!sample.clear_adaptive_data());
    assert_eq!(&line[sample.source().range()], line);
    let john = sample.tokens()[0].span;
    assert_eq!(&line[john.range()], "John");
    for t in sample.tokens() {
        assert!(t.is_borrowed() && &line[t.span.range()] == t.as_str());
    }
    assert_eq!(sample.names().iter().map(|n| n.java_string()).collect::<Vec<_>>(), ["[0..1) PER", "[3..4) ORG"]);
}

#[test]
fn keys_may_come_in_either_order() {
    let line = "{\"ner_tags\": [\"O\", \"B-LOC\"], \"tokens\": [\"café\", \"Lisboa\"]}";
    assert_eq!(read(line), [Ok("café <START:LOC> Lisboa <END>".to_owned())]);
}

#[test]
fn an_escaped_token_holds_its_decoded_text_and_the_raw_span() {
    // "café" by a \u cut, a quote by \" and the emoji by a surrogate pair.
    let line =
        "{\"tokens\": [\"caf\\u00e9\", \"x\\\"y\", \"\\uD83D\\uDE00\"], \"ner_tags\": [\"B-PER\", \"I-PER\", \"O\"]}";
    let sample = MulticonerReader::new(line).unwrap().next().unwrap().unwrap();
    let texts: Vec<&str> = sample.tokens().iter().map(|t| t.as_str()).collect();
    assert_eq!(texts, ["café", "x\"y", "\u{1F600}"]);
    for t in sample.tokens() {
        assert!(!t.is_borrowed(), "an escaped token holds a text of its own");
    }
    // the spans name the raw content the tokens were decoded from
    assert_eq!(&line[sample.tokens()[0].span.range()], "caf\\u00e9");
    assert_eq!(&line[sample.tokens()[2].span.range()], "\\uD83D\\uDE00");
    assert_eq!(sample.to_string(), "<START:PER> café x\"y <END> \u{1F600}");
}

#[test]
fn i_tags_start_names_and_a_type_change_splits_them() {
    // IOB2 as Conll03NameSampleStream reads it: an `I-` with no name open
    // starts one, and an `I-` of another type closes the open name
    let line = "{\"tokens\": [\"a\", \"b\", \"c\", \"d\"], \"ner_tags\": [\"I-PER\", \"I-LOC\", \"B-ORG\", \"I-ORG\"]}";
    assert_eq!(read(line), [Ok("<START:PER> a <END> <START:LOC> b <END> <START:ORG> c d <END>".to_owned())]);
}

#[test]
fn a_row_that_is_not_the_pinned_form_is_refused() {
    // an unknown key
    let err = MulticonerReader::new("{\"tokens\": [\"a\"], \"ner_tags\": [\"O\"], \"lang\": \"en\"}")
        .unwrap()
        .next()
        .unwrap()
        .unwrap_err();
    assert!(matches!(err, MulticonerError::Syntax { line: 1, .. }));
    assert!(err.to_string().contains("the key tokens or ner_tags"), "{err}");
    // a duplicated key
    let err = MulticonerReader::new("{\"tokens\": [\"a\"], \"tokens\": [\"b\"], \"ner_tags\": [\"O\"]}")
        .unwrap()
        .next()
        .unwrap()
        .unwrap_err();
    assert!(err.to_string().contains("each key once"), "{err}");
    // a missing key
    let err = MulticonerReader::new("{\"tokens\": [\"a\"]}").unwrap().next().unwrap().unwrap_err();
    assert!(err.to_string().contains("the keys tokens and ner_tags"), "{err}");
    // garbage after the object, and an empty line
    let err =
        MulticonerReader::new("{\"tokens\": [\"a\"], \"ner_tags\": [\"O\"]} x").unwrap().next().unwrap().unwrap_err();
    assert!(err.to_string().contains("the end of the line"), "{err}");
    let err = MulticonerReader::new("\n").unwrap().next().unwrap().unwrap_err();
    assert!(err.to_string().contains("an object"), "{err}");
    // a non-string value
    let err = MulticonerReader::new("{\"tokens\": [1], \"ner_tags\": [\"O\"]}").unwrap().next().unwrap().unwrap_err();
    assert!(err.to_string().contains("a string"), "{err}");
}

#[test]
fn bad_strings_are_refused_with_their_cut() {
    // a bad escape, a bad \u cut, a lone surrogate half and an unescaped
    // control character
    for (token, what) in [
        ("\"a\\qb\"", "one of the escapes"),
        ("\"\\uZZZZ\"", "four hexadecimal digits"),
        ("\"\\uD83D\"", "a whole surrogate pair"),
        ("\"a\tb\"", "no unescaped control character"),
    ] {
        let line = format!("{{\"tokens\": [{token}], \"ner_tags\": [\"O\"]}}");
        let err = MulticonerReader::new(&line).unwrap().next().unwrap().unwrap_err();
        assert!(err.to_string().contains(what), "{err} for {token}");
    }
}

#[test]
fn a_tag_with_an_escape_is_not_a_plain_name_tag() {
    let line = "{\"tokens\": [\"a\"], \"ner_tags\": [\"B\\u002dPER\"]}";
    let err = MulticonerReader::new(line).unwrap().next().unwrap().unwrap_err();
    assert!(err.to_string().contains("a plain name tag"), "{err}");
}

#[test]
fn lengths_must_match_and_a_row_needs_a_token() {
    let err = MulticonerReader::new("{\"tokens\": [\"a\", \"b\"], \"ner_tags\": [\"O\"]}")
        .unwrap()
        .next()
        .unwrap()
        .unwrap_err();
    assert!(matches!(err, MulticonerError::Lengths { line: 1, tokens: 2, tags: 1 }));
    let err = MulticonerReader::new("{\"tokens\": [], \"ner_tags\": []}").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(err, MulticonerError::Empty { line: 1 }));
}

#[test]
fn a_bad_tag_fails_its_row_and_the_read_goes_on() {
    let source = concat!(
        "{\"tokens\": [\"a\"], \"ner_tags\": [\"Q\"]}\n",
        "{\"tokens\": [\"b\", \"c\"], \"ner_tags\": [\"B-LOC\", \"I-LOC\"]}\n",
    );
    assert_eq!(read(source), [Err("Invalid tag: Q".to_owned()), Ok("<START:LOC> b c <END>".to_owned()),]);
    let err =
        MulticonerReader::new("{\"tokens\": [\"a\"], \"ner_tags\": [\"B-\"]}").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(err, MulticonerError::InvalidTag { tag, .. } if tag == "B-"));
}

#[test]
fn supplementary_characters_are_one_token_of_four_bytes() {
    let line = "{\"tokens\": [\"\u{1F600} face\"], \"ner_tags\": [\"O\"]}";
    let sample = MulticonerReader::new(line).unwrap().next().unwrap().unwrap();
    assert_eq!(sample.tokens()[0].as_str(), "\u{1F600} face");
    assert_eq!(&line[sample.tokens()[0].span.range()], "\u{1F600} face");
}
