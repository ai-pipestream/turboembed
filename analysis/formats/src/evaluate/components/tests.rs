use std::borrow::Cow;

use turbo_analysis::Span;
use turbo_tokenize::detok::Detokenizer;

use super::span::trim;
use super::*;
use crate::sample::{ChunkSample, NameSample, PosSample, SentenceSample, TokenSample};
use crate::source::Field;

fn field(source: &str, start: u32, end: u32) -> Field<'_> {
    Field::of(source, Span::new(start, end))
}

#[test]
fn pos_counts_up_to_the_missing_tag() {
    let src = "a_X b_Y c_Z";
    let s = PosSample::new(
        vec![field(src, 0, 1), field(src, 4, 5), field(src, 8, 9)],
        vec![field(src, 2, 3), field(src, 6, 7), field(src, 10, 11)],
        Span::new(0, 11),
    )
    .unwrap();
    let mut e = PosEvaluator::default();
    assert_eq!(e.evaluate(&s, &["X", "Y", "Z"]), Ok(Verdict::Correct));
    assert_eq!(e.evaluate(&s, &["X", "Q"]), Err(EvalError::IndexOutOfBounds { index: 2, len: 2 }));
    assert_eq!((e.word_count(), e.word_accuracy()), (5, 4.0 / 5.0));
    let err = e.evaluate(&s, &["X", "Y", "Z", "W"]).unwrap_err();
    assert_eq!(err.to_string(), "There must be exactly one tag for each token. tokens: 3, tags: 4");
    assert_eq!(e.word_count(), 8);
}

#[test]
fn names_score_the_default_type_but_judge_the_own() {
    let src = "<START:p> a <END> b <START> c <END>";
    // a `<START>` with no type takes the type of the last typed start tag
    let s = NameSample::new(
        vec![field(src, 10, 11), field(src, 18, 19), field(src, 28, 29)],
        vec![Name::new(0, 1, Some("p")).unwrap(), Name::new(2, 3, Some("p")).unwrap()],
        false,
        Span::new(0, 35),
        1,
    )
    .unwrap();
    let mut e = NameFinderEvaluator::default();
    let p = [Name::new(2, 3, Some("p")).unwrap(), Name::new(0, 1, Some("p")).unwrap()];
    assert_eq!(e.evaluate(&s, &p), Ok(Verdict::Correct));
    assert_eq!(e.fmeasure().true_positives(), 2);
    let overlap = [Name::new(0, 2, Some("p")).unwrap(), Name::new(1, 3, None).unwrap()];
    let err = Name::new(3, 2, None).unwrap_err();
    assert_eq!(err.to_string(), "start index must not be larger than end index: start=3, end=2");
    let err = e.evaluate(&s, &overlap).unwrap_err();
    assert_eq!(err.to_string(), "name spans [0..2) p and [1..3) are overlapped in file: null, sentence: [a, b, c]");
    assert_eq!(e.fmeasure().selected(), 4);
}

#[test]
fn chunks_score_phrases() {
    let src = "a x B-NP\nb x I-NP\nc x O\n";
    let s = ChunkSample::new(
        vec![field(src, 0, 1), field(src, 9, 10), field(src, 18, 19)],
        vec![field(src, 2, 3), field(src, 11, 12), field(src, 20, 21)],
        vec![field(src, 4, 8), field(src, 13, 17), field(src, 22, 23)],
        Span::new(0, 23),
    )
    .unwrap();
    let mut e = ChunkerEvaluator::default();
    assert_eq!(e.evaluate(&s, &["B-NP", "B-NP", "O"]), Ok(Verdict::Misclassified));
    assert_eq!((e.fmeasure().true_positives(), e.fmeasure().selected()), (0, 2));
    let err = e.evaluate(&s, &["B-NP", "X", "O"]).unwrap_err();
    assert!(matches!(err, EvalError::Phrases(FormatError::ShortChunkTag { .. })));
    let err = e.evaluate(&s, &["B-NP", "x\u{1F600}", "O"]).unwrap_err();
    assert!(matches!(err, EvalError::Phrases(FormatError::SplitsSurrogatePair { .. })));
    let err = e.evaluate(&s, &["O"]).unwrap_err();
    assert_eq!(err.to_string(), "All arrays must have the same length: sentenceSize: 3, tagsSize: 3, predsSize: 1!");
}

#[test]
fn tokens_and_the_detokenizer() {
    let s = TokenSample::new(
        Cow::Borrowed("a, b "),
        vec![Span::new(0, 1), Span::new(1, 2), Span::new(3, 4)],
        vec![Span::new(0, 1), Span::new(8, 9), Span::new(10, 11)],
        Span::new(0, 11),
    )
    .unwrap();
    let mut e = TokenizerEvaluator::default();
    assert_eq!(e.evaluate(&s, &[Span::new(0, 1), Span::new(1, 2), Span::new(3, 4)]), Ok(Verdict::Correct));
    let err = e.evaluate(&s, &[Span::new(0, 9)]).unwrap_err();
    assert_eq!(err.to_string(), "Span [0..9) is out of bounds, text length: 5!");
    let dictionary = turbo_tokenize::detok::DetokenizerDictionary::new();
    let mut d = DetokenizerEvaluator::default();
    assert_eq!(d.evaluate(&s, &Detokenizer::new(&dictionary)), Ok(Verdict::Misclassified));
    assert_eq!(d.fmeasure().true_positives(), 0);
}

#[test]
fn sentences_are_trimmed() {
    let s = SentenceSample::new(
        Cow::Borrowed("One. Two. "),
        vec![Span::new(0, 4), Span::new(5, 9)],
        vec![Span::new(0, 4), Span::new(5, 9)],
        Span::new(0, 9),
    )
    .unwrap();
    assert_eq!(s.document(), "One. Two. ");
    let mut e = SentenceDetectorEvaluator::new();
    assert_eq!(e.evaluate(&s, &[Span::new(0, 5), Span::new(4, 10)]), Ok(Verdict::Correct));
    assert_eq!(e.fmeasure().true_positives(), 2);
    assert_eq!(
        e.evaluate(&s, &[Span::new(8, 12)]),
        Err(EvalError::TrimOutOfText { index: 11, len: 10, index_units: 11, len_units: 10 })
    );
    assert_eq!(
        e.evaluate(&s, &[Span::new(12, 12)]).unwrap_err().to_string(),
        "Sentence span is outside of document text [len 10] and span [12..12)"
    );
    assert_eq!(trim(" \u{a0} ", Span::new(0, 4)), Ok(Span::new(0, 0)));
    assert_eq!(trim("\u{a0}", Span::new(1, 2)), Ok(Span::new(1, 2)));
}

#[test]
fn span_errors_count_utf16_units() {
    // "\u{c9}t\u{1F600}." is 8 bytes and 5 units; with its space, 9 and 6.
    let s = SentenceSample::new(
        Cow::Borrowed("\u{c9}t\u{1F600}. "),
        vec![Span::new(0, 8)],
        vec![Span::new(0, 8)],
        Span::new(0, 8),
    )
    .unwrap();
    let mut e = SentenceDetectorEvaluator::new();
    let err = e.evaluate(&s, &[Span::new(9, 11)]).unwrap_err();
    assert_eq!(err, EvalError::TrimOutOfText { index: 9, len: 9, index_units: 6, len_units: 6 });
    assert_eq!(err.to_string(), "Index 6 out of bounds for length 6");
    let err = e.evaluate(&s, &[Span::new(13, 13)]).unwrap_err();
    assert_eq!(err.to_string(), "Sentence span is outside of document text [len 6] and span [10..10)");
    let s = TokenSample::new(
        Cow::Borrowed("\u{c9}\u{1F600} "),
        vec![Span::new(0, 2), Span::new(2, 6)],
        vec![Span::new(0, 2), Span::new(9, 13)],
        Span::new(0, 13),
    )
    .unwrap();
    let err = TokenizerEvaluator::default().evaluate(&s, &[Span::new(2, 6), Span::new(7, 9)]).unwrap_err();
    assert_eq!(err.to_string(), "Span [4..6) is out of bounds, text length: 4!");
}
