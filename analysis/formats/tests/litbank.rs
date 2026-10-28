//! Focused tests for the LitBank kept-corpus reader of issue #1533.
//!
//! Every fixture here is original, written for these tests.

use std::collections::BTreeSet;

use turbo_analysis::Span;
use turbo_formats::ner::{LITBANK_TYPES, LitBankError, LitBankReader, NewlineSentences, RuleTokenizer, SentenceSpans};
use turbo_tokenize::TokenizerOptions;

const TEXT: &str = "Ada went to London.\nBob left Paris.";

const ANN: &str =
    concat!("T1\tPER 0 3\tAda\n", "T2\tGPE 12 18\tLondon\n", "T3\tPER 20 23\tBob\n", "T4\tGPE 29 34\tParis\n",);

fn reader<'a>(text: &'a str, ann: &'a str) -> LitBankReader<'a, RuleTokenizer> {
    let sentences = NewlineSentences;
    let tokenizer = RuleTokenizer::Simple(TokenizerOptions { keep_newlines: false });
    LitBankReader::new(text, ann, sentences, tokenizer, None).unwrap()
}

fn read(text: &str, ann: &str) -> Vec<Result<String, String>> {
    reader(text, ann).map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string())).collect()
}

#[test]
fn reads_a_document_a_sentence_per_line() {
    assert_eq!(
        read(TEXT, ANN),
        [
            "\n<START:PER> Ada <END> went to <START:GPE> London <END> .",
            "<START:PER> Bob <END> left <START:GPE> Paris <END> .",
        ]
        .into_iter()
        .map(|s| Ok(s.to_owned()))
        .collect::<Vec<_>>()
    );
    let samples: Vec<_> = reader(TEXT, ANN).map(|r| r.unwrap()).collect();
    assert!(samples[0].clear_adaptive_data(), "the first sample of a document clears adaptive data");
    assert!(!samples[1].clear_adaptive_data());
    let names: Vec<_> = samples[0].names().iter().map(|n| n.java_string()).collect();
    assert_eq!(names, ["[0..1) PER", "[3..4) GPE"]);
    let london = samples[0].tokens()[3].span;
    assert_eq!(&TEXT[london.range()], "London");
    for t in samples[0].tokens() {
        assert!(t.is_borrowed() && &TEXT[t.span.range()] == t.as_str());
    }
}

#[test]
fn the_types_of_litbank_are_six() {
    assert_eq!(LITBANK_TYPES, ["PER", "FAC", "GPE", "LOC", "ORG", "VEH"]);
}

#[test]
fn offsets_count_utf16_code_units() {
    // "café" is four units, the emoji two of four bytes, so "New" starts at
    // unit 8 and byte 11, and the emoji is units 5..7, bytes 6..10.
    let text = "café \u{1F600} New";
    let ann = concat!("T1\tLOC 8 11\tNew\n", "T2\tMISC 5 7\t\u{1F600}\n");
    assert_eq!(read(text, ann), [Ok("\ncafé <START:MISC> \u{1F600} <END> <START:LOC> New <END>".to_owned())]);
    let sample = reader(text, ann).next().unwrap().unwrap();
    assert_eq!(&text[sample.tokens()[1].span.range()], "\u{1F600}");
    assert_eq!(sample.names().iter().map(|n| n.java_string()).collect::<Vec<_>>(), ["[1..2) MISC", "[2..3) LOC"]);
}

#[test]
fn a_type_left_out_is_not_a_name() {
    let sentences = NewlineSentences;
    let tokenizer = RuleTokenizer::Simple(TokenizerOptions::default());
    let types = Some(LITBANK_TYPES.into_iter().filter(|&t| t == "PER").map(str::to_owned).collect::<BTreeSet<_>>());
    let samples: Vec<_> =
        LitBankReader::new(TEXT, ANN, sentences, tokenizer, types).unwrap().map(|r| r.unwrap()).collect();
    let kinds: Vec<_> = samples[0].names().iter().map(|n| n.kind.unwrap()).collect();
    assert_eq!(kinds, ["PER"]);
}

#[test]
fn an_empty_set_of_types_is_refused() {
    let sentences = NewlineSentences;
    let tokenizer = RuleTokenizer::Simple(TokenizerOptions::default());
    let err = LitBankReader::new(TEXT, ANN, sentences, tokenizer, Some(BTreeSet::new())).unwrap_err();
    assert!(matches!(err, LitBankError::NoNameTypes));
}

#[test]
fn malformed_annotation_lines_fail_the_document() {
    // the error names the line of the annotation file
    let cases: &[(&str, &str)] = &[
        ("T1\tPER 0 3\n", "three tab-separated fields"),
        ("T1\tPER 0\tAda\n", "TYPE START END"),
        ("T1\tPER x 3\tAda\n", "For input string"),
        ("T1\tPER -1 3\tAda\n", "zero or greater"),
        ("T1\tPER 3 0\tAda\n", "larger than end index"),
        ("T1\tPER 0 99\tAda\n", "outside a text"),
        ("T1\tPER 0 3;4 5\tAda\n", "single START END pair"),
        ("T1\tPER 0 3\tAde\n", "the covered text of T1"),
        ("T1\tPER 3 8\t went\n", "leading or trailing whitespace"),
    ];
    for &(ann, what) in cases {
        let err = reader_err(TEXT, ann);
        assert!(err.to_string().contains(what), "{err} for {ann:?}");
    }
    // an offset inside a surrogate pair, on the text that holds one
    let text = "café \u{1F600} New";
    let err = reader_err(text, "T1\tMISC 5 6\t\u{1F600}\n");
    assert!(err.to_string().contains("surrogate pair"), "{err}");
}

fn reader_err(text: &str, ann: &str) -> LitBankError {
    let sentences = NewlineSentences;
    let tokenizer = RuleTokenizer::Simple(TokenizerOptions::default());
    match LitBankReader::new(text, ann, sentences, tokenizer, None) {
        Err(e) => e,
        Ok(mut reader) => reader.next().expect("a sample or its error").unwrap_err(),
    }
}

/// A sentence detector that splits a text in halves, to hold sentences
/// that no line break gives.
struct Halves;

impl SentenceSpans for Halves {
    fn sentence_spans(&self, text: &str, out: &mut Vec<Span>) {
        let half = text.len() as u32 / 2;
        out.push(Span::new(0, half));
        out.push(Span::new(half, text.len() as u32));
    }
}

#[test]
fn a_span_inside_a_token_or_across_sentences_is_refused() {
    // "and Bob" is 4..11, inside "Ada" and "Bob" but starting inside "and"
    let text = "Ada and Bob here.";
    let err = reader_err(text, "T1\tPER 1 3\tda\n");
    assert!(err.to_string().contains("inside a token"), "{err}");
    // "and Bob" is 4..11: it starts in the first half and ends in the
    // second, so a sentence of Halves ends inside it
    let sentences = Halves;
    let tokenizer = RuleTokenizer::Simple(TokenizerOptions::default());
    let mut reader = LitBankReader::new(text, "T1\tPER 4 11\tand Bob\n", sentences, tokenizer, None).unwrap();
    let err = reader.next().unwrap().unwrap_err();
    assert!(err.to_string().contains("crosses a sentence"), "{err}");
}

#[test]
fn lines_that_are_not_text_bound_are_passed_over() {
    let ann = concat!(
        "#1\tAnnotatorNotes T1\ta note\n",
        "T1\tPER 0 3\tAda\n",
        "R1\tCoreference Arg1:T1\tArg2:T1\n",
        "T2\tGPE 12 18\tLondon\n",
    );
    let sample = reader(TEXT, ann).next().unwrap().unwrap();
    let kinds: Vec<_> = sample.names().iter().map(|n| n.kind.unwrap()).collect();
    assert_eq!(kinds, ["PER", "GPE"]);
}
