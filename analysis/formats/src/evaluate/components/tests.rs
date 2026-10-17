use turbo_tokenize::detok::Detokenizer;
use turbo_unicode::WhitespaceMode;

use super::span::trim;
use super::*;
use crate::native::{ChunkReader, NameReader, SentenceReader, TokenReader, WordTagReader};
use crate::sample::Name;

#[test]
fn pos_counts_up_to_the_missing_tag() {
    let src = "a_X b_Y c_Z";
    let s = WordTagReader::new(src, WhitespaceMode::Unicode).unwrap().next().unwrap().unwrap();
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
    let s = NameReader::new(src, WhitespaceMode::Unicode).unwrap().next().unwrap().unwrap();
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
    let s = ChunkReader::new(src).unwrap().next().unwrap().unwrap();
    let mut e = ChunkerEvaluator::default();
    assert_eq!(e.evaluate(&s, &["B-NP", "B-NP", "O"]), Ok(Verdict::Misclassified));
    assert_eq!((e.fmeasure().true_positives(), e.fmeasure().selected()), (0, 2));
    let err = e.evaluate(&s, &["B-NP", "X", "O"]).unwrap_err();
    assert_eq!(err.java_class(), Some("java.lang.StringIndexOutOfBoundsException"));
    let err = e.evaluate(&s, &["B-NP", "x\u{1F600}", "O"]).unwrap_err();
    assert_eq!(err.java_class(), None);
    let err = e.evaluate(&s, &["O"]).unwrap_err();
    assert_eq!(err.to_string(), "All arrays must have the same length: sentenceSize: 3, tagsSize: 3, predsSize: 1!");
}

#[test]
fn tokens_and_the_detokenizer() {
    let src = "a<SPLIT>, b";
    let s = TokenReader::new(src, WhitespaceMode::Unicode).unwrap().next().unwrap().unwrap();
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
    let src = "One.\nTwo.\n";
    let s = SentenceReader::new(src).unwrap().next().unwrap().unwrap();
    assert_eq!(s.document(), "One. Two. ");
    let mut e = SentenceDetectorEvaluator::new(WhitespaceMode::Unicode);
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
    assert_eq!(trim(" \u{a0} ", Span::new(0, 4), WhitespaceMode::Unicode), Ok(Span::new(0, 0)));
    assert_eq!(trim("\u{a0}", Span::new(1, 2), WhitespaceMode::Unicode), Ok(Span::new(1, 2)));
}

#[test]
fn span_errors_count_utf16_units() {
    // "\u{c9}t\u{1F600}." is 8 bytes and 5 units; with its space, 9 and 6.
    let src = "\u{c9}t\u{1F600}.\n";
    let s = SentenceReader::new(src).unwrap().next().unwrap().unwrap();
    let mut e = SentenceDetectorEvaluator::new(WhitespaceMode::Unicode);
    let err = e.evaluate(&s, &[Span::new(9, 11)]).unwrap_err();
    assert_eq!(err, EvalError::TrimOutOfText { index: 9, len: 9, index_units: 6, len_units: 6 });
    assert_eq!(err.to_string(), "Index 6 out of bounds for length 6");
    let err = e.evaluate(&s, &[Span::new(13, 13)]).unwrap_err();
    assert_eq!(err.to_string(), "Sentence span is outside of document text [len 6] and span [10..10)");
    let src = "\u{c9}<SPLIT>\u{1F600}";
    let s = TokenReader::new(src, WhitespaceMode::Unicode).unwrap().next().unwrap().unwrap();
    let err = TokenizerEvaluator::default().evaluate(&s, &[Span::new(2, 6), Span::new(7, 9)]).unwrap_err();
    assert_eq!(err.to_string(), "Span [4..6) is out of bounds, text length: 4!");
}
