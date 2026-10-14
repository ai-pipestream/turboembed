//! The evaluators of OpenNLP's components: each scores what a component
//! predicted for a reference sample, as the component's `Evaluator`
//! subclass does in `processSample`, and tells whether the sample built
//! from the prediction equals the reference (what the evaluation
//! monitors hear as correctly classified or misclassified).
//!
//! The Java evaluators run the component themselves; these take its
//! prediction, so any tagger, tokenizer or detector can be scored. Only
//! the detokenizer evaluator runs its component, which this crate's
//! dependencies hold.
//!
//! The order of the work is Java's, so a prediction that makes Java
//! throw leaves the scores as Java leaves them: the part-of-speech and
//! lemma evaluators count every token up to the one Java fails on, the
//! span evaluators update their counts before the sample is built from
//! the prediction and fail after.

use std::fmt;

use turbo_analysis::Span;
use turbo_tokenize::detok::Detokenizer;
use turbo_unicode::WhitespaceMode;

use super::measure::{FMeasure, LabeledSpan, Mean};
use crate::error::FormatError;
use crate::sample::chunk::phrase_spans;
use crate::sample::name::{DEFAULT_NAME_TYPE, java_list};
use crate::sample::{
    ChunkSample, DocumentSample, LanguageSample, LemmaSample, Name, NameSample, PosSample, SentenceSample, TokenSample,
};
use crate::source::Field;

/// What the evaluation monitors hear about a sample: whether the sample
/// built from the prediction equals the reference (`Object.equals` of the
/// sample classes).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    Correct,
    Misclassified,
}

impl Verdict {
    fn of(equal: bool) -> Verdict {
        if equal { Verdict::Correct } else { Verdict::Misclassified }
    }
}

/// A prediction Java's evaluator throws on. The span errors keep the
/// byte offsets of the text and the same offsets in UTF-16 code units,
/// which their messages give, as Java counts them. An offset past the
/// end of the text counts one unit for each byte past it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalError {
    /// Fewer predicted tags or lemmas than reference tokens: the
    /// `ArrayIndexOutOfBoundsException` of reading the prediction at
    /// `index`.
    IndexOutOfBounds { index: usize, len: usize },
    /// A part-of-speech prediction with more tags than tokens, which
    /// `POSSample` refuses.
    TagCount { tokens: usize, tags: usize },
    /// A lemma or chunk prediction of another length than the tokens,
    /// which `LemmaSample` or `ChunkSample` refuses.
    ArrayLengths { sentence: usize, tags: usize, preds: usize },
    /// Predicted names that overlap after `NameSample` sorts them.
    OverlappingNames { first: String, second: String, sentence: String },
    /// A predicted token span past the end of the text, which
    /// `TokenSample` refuses.
    TokenOutOfText { span: Span, len: usize, units: Span, len_units: usize },
    /// A predicted sentence span past the end of the document, which
    /// `SentenceSample` refuses.
    SentenceOutOfText { span: Span, len: usize, units: Span, len_units: usize },
    /// `Span.trim` reading past the end of the document: the
    /// `StringIndexOutOfBoundsException` of `charAt(index)`.
    TrimOutOfText { index: usize, len: usize, index_units: usize, len_units: usize },
    /// The reference or predicted chunk tags have no phrases
    /// (`getPhrasesAsSpanList` throws).
    Phrases(FormatError),
}

impl EvalError {
    /// The class of the exception Java throws.
    pub fn java_class(&self) -> &'static str {
        match self {
            EvalError::IndexOutOfBounds { .. } => "java.lang.ArrayIndexOutOfBoundsException",
            EvalError::TagCount { .. }
            | EvalError::ArrayLengths { .. }
            | EvalError::TokenOutOfText { .. }
            | EvalError::SentenceOutOfText { .. } => "java.lang.IllegalArgumentException",
            EvalError::OverlappingNames { .. } => "java.lang.RuntimeException",
            EvalError::TrimOutOfText { .. } => "java.lang.StringIndexOutOfBoundsException",
            EvalError::Phrases(e) => e.java_class().unwrap_or("java.lang.RuntimeException"),
        }
    }
}

fn span_string(span: Span) -> String {
    format!("[{}..{})", span.start, span.end)
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::IndexOutOfBounds { index, len }
            | EvalError::TrimOutOfText { index_units: index, len_units: len, .. } => {
                write!(f, "Index {index} out of bounds for length {len}")
            }
            EvalError::TagCount { tokens, tags } => {
                write!(f, "There must be exactly one tag for each token. tokens: {tokens}, tags: {tags}")
            }
            EvalError::ArrayLengths { sentence, tags, preds } => write!(
                f,
                "All arrays must have the same length: sentenceSize: {sentence}, tagsSize: {tags}, predsSize: {preds}!"
            ),
            EvalError::OverlappingNames { first, second, sentence } => {
                write!(f, "name spans {first} and {second} are overlapped in file: null, sentence: {sentence}")
            }
            EvalError::TokenOutOfText { units, len_units, .. } => {
                write!(f, "Span {} is out of bounds, text length: {len_units}!", span_string(*units))
            }
            EvalError::SentenceOutOfText { units, len_units, .. } => {
                write!(
                    f,
                    "Sentence span is outside of document text [len {len_units}] and span {}",
                    span_string(*units)
                )
            }
            EvalError::Phrases(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for EvalError {}

/// Counts each reference value against the prediction at its place, as
/// the part-of-speech and lemma evaluators do: the error at the first
/// place the prediction lacks, after counting the places before it.
fn count_matches(accuracy: &mut Mean, references: &[Field<'_>], predicted: &[&str]) -> Result<(), EvalError> {
    for (index, reference) in references.iter().enumerate() {
        let Some(&p) = predicted.get(index) else {
            return Err(EvalError::IndexOutOfBounds { index, len: predicted.len() });
        };
        accuracy.add(if reference.as_str() == p { 1.0 } else { 0.0 });
    }
    Ok(())
}

fn same(references: &[Field<'_>], predicted: &[&str]) -> bool {
    references.len() == predicted.len() && references.iter().zip(predicted).all(|(r, &p)| r.as_str() == p)
}

/// The part-of-speech evaluator, `POSEvaluator`: the share of tokens
/// tagged as the reference tags them.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PosEvaluator {
    accuracy: Mean,
}

impl PosEvaluator {
    /// Scores the tags `predicted` for `reference`'s tokens.
    ///
    /// # Errors
    ///
    /// [`EvalError::IndexOutOfBounds`] for fewer tags than tokens, after
    /// counting the tokens that have one, and [`EvalError::TagCount`] for
    /// more, after counting every token.
    pub fn evaluate(&mut self, reference: &PosSample<'_>, predicted: &[&str]) -> Result<Verdict, EvalError> {
        count_matches(&mut self.accuracy, reference.tags(), predicted)?;
        if predicted.len() != reference.len() {
            return Err(EvalError::TagCount { tokens: reference.len(), tags: predicted.len() });
        }
        Ok(Verdict::of(same(reference.tags(), predicted)))
    }

    /// The share of tokens tagged right (`getWordAccuracy`), 0 before
    /// any.
    pub fn word_accuracy(&self) -> f64 {
        self.accuracy.mean()
    }

    /// The number of tokens counted (`getWordCount`).
    pub fn word_count(&self) -> u64 {
        self.accuracy.count()
    }
}

/// The lemmatizer evaluator, `LemmatizerEvaluator`: the share of tokens
/// with the reference lemma.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct LemmatizerEvaluator {
    accuracy: Mean,
}

impl LemmatizerEvaluator {
    /// Scores the lemmas `predicted` for `reference`'s tokens.
    ///
    /// # Errors
    ///
    /// [`EvalError::IndexOutOfBounds`] for fewer lemmas than tokens, after
    /// counting the tokens that have one, and [`EvalError::ArrayLengths`]
    /// for more, after counting every token.
    pub fn evaluate(&mut self, reference: &LemmaSample<'_>, predicted: &[&str]) -> Result<Verdict, EvalError> {
        count_matches(&mut self.accuracy, reference.lemmas(), predicted)?;
        let (sentence, tags) = (reference.tokens().len(), reference.tags().len());
        if predicted.len() != sentence {
            return Err(EvalError::ArrayLengths { sentence, tags, preds: predicted.len() });
        }
        Ok(Verdict::of(same(reference.lemmas(), predicted)))
    }

    pub fn word_accuracy(&self) -> f64 {
        self.accuracy.mean()
    }

    pub fn word_count(&self) -> u64 {
        self.accuracy.count()
    }
}

/// The name finder evaluator, `TokenNameFinderEvaluator`: precision and
/// recall of the predicted names, a name right only with the reference's
/// tokens and type.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NameFinderEvaluator {
    fmeasure: FMeasure,
}

impl NameFinderEvaluator {
    /// Scores the names `predicted` for `reference`'s tokens. A reference
    /// name without a type is scored as one of type `default`, which is
    /// what the name finder gives such names, but the verdict compares
    /// the reference's own names, so it is never correct.
    ///
    /// # Errors
    ///
    /// [`EvalError::OverlappingNames`] when predicted names overlap after
    /// sorting, after the scores are counted.
    pub fn evaluate(&mut self, reference: &NameSample<'_>, predicted: &[Name<'_>]) -> Result<Verdict, EvalError> {
        let references: Vec<LabeledSpan<'_>> = reference
            .names()
            .iter()
            .map(|n| LabeledSpan::new(n.start, n.end, Some(n.kind.unwrap_or(DEFAULT_NAME_TYPE))))
            .collect();
        let mut predictions: Vec<LabeledSpan<'_>> = predicted.iter().map(LabeledSpan::from).collect();
        self.fmeasure.update(&references, &predictions);
        let mut sorted: Vec<&Name<'_>> = predicted.iter().collect();
        sorted.sort_by(|a, b| a.java_cmp(b));
        for pair in sorted.windows(2) {
            if pair[1].start < pair[0].end {
                return Err(EvalError::OverlappingNames {
                    first: pair[0].java_string(),
                    second: pair[1].java_string(),
                    sentence: java_list(reference.tokens().iter().map(Field::as_str)),
                });
            }
        }
        predictions = sorted.into_iter().map(LabeledSpan::from).collect();
        let own: Vec<LabeledSpan<'_>> = reference.names().iter().map(LabeledSpan::from).collect();
        Ok(Verdict::of(own == predictions))
    }

    pub fn fmeasure(&self) -> &FMeasure {
        &self.fmeasure
    }
}

/// The chunker evaluator, `ChunkerEvaluator`: precision and recall of the
/// phrases the predicted chunk tags mark.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkerEvaluator {
    fmeasure: FMeasure,
}

impl ChunkerEvaluator {
    /// Scores the chunk tags `predicted` for `reference`'s tokens.
    ///
    /// # Errors
    ///
    /// [`EvalError::ArrayLengths`] for another number of tags than tokens,
    /// and [`EvalError::Phrases`] when the reference's or the predicted
    /// tags have no phrases (the reference's first); no score is counted.
    pub fn evaluate(&mut self, reference: &ChunkSample<'_>, predicted: &[&str]) -> Result<Verdict, EvalError> {
        let (sentence, tags) = (reference.tokens().len(), reference.tags().len());
        if predicted.len() != sentence || tags != sentence {
            return Err(EvalError::ArrayLengths { sentence, tags, preds: predicted.len() });
        }
        let reference_tags: Vec<&str> = reference.chunks().iter().map(Field::as_str).collect();
        let references = labeled(&reference_tags, &phrase_spans(&reference_tags).map_err(EvalError::Phrases)?);
        let predictions = labeled(predicted, &phrase_spans(predicted).map_err(EvalError::Phrases)?);
        self.fmeasure.update(&references, &predictions);
        Ok(Verdict::of(same(reference.chunks(), predicted)))
    }

    pub fn fmeasure(&self) -> &FMeasure {
        &self.fmeasure
    }
}

fn labeled<'t>(tags: &[&'t str], phrases: &[(u32, u32, usize)]) -> Vec<LabeledSpan<'t>> {
    phrases.iter().map(|&(start, end, cut)| LabeledSpan::new(start, end, Some(&tags[start as usize][cut..]))).collect()
}

fn untyped(spans: &[Span]) -> Vec<LabeledSpan<'static>> {
    spans.iter().map(|s| LabeledSpan::new(s.start, s.end, None)).collect()
}

/// The tokenizer evaluator, `TokenizerEvaluator`: precision and recall of
/// the predicted token spans.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenizerEvaluator {
    fmeasure: FMeasure,
}

impl TokenizerEvaluator {
    /// Scores the token spans `predicted`, byte spans of `reference`'s
    /// text.
    ///
    /// # Errors
    ///
    /// [`EvalError::TokenOutOfText`] for a span past the text's end, after
    /// the scores are counted.
    pub fn evaluate(&mut self, reference: &TokenSample<'_>, predicted: &[Span]) -> Result<Verdict, EvalError> {
        self.fmeasure.update(&untyped(reference.tokens()), &untyped(predicted));
        check_in_text(predicted, reference.text())?;
        Ok(Verdict::of(reference.tokens() == predicted))
    }

    pub fn fmeasure(&self) -> &FMeasure {
        &self.fmeasure
    }
}

/// The number of UTF-16 code units of `text` before byte `at`: those of
/// the characters that start before it, and one for each byte past the
/// end.
fn units(text: &str, at: usize) -> usize {
    if at > text.len() {
        return text.encode_utf16().count() + (at - text.len());
    }
    text.char_indices().take_while(|&(i, _)| i < at).map(|(_, c)| c.len_utf16()).sum()
}

fn span_units(text: &str, span: Span) -> Span {
    Span::new(units(text, span.start as usize) as u32, units(text, span.end as usize) as u32)
}

fn check_in_text(spans: &[Span], text: &str) -> Result<(), EvalError> {
    let len = text.len();
    match spans.iter().find(|s| s.start as usize > len || s.end as usize > len) {
        Some(&span) => {
            Err(EvalError::TokenOutOfText { span, len, units: span_units(text, span), len_units: units(text, len) })
        }
        None => Ok(()),
    }
}

/// The detokenizer evaluator, `DetokenizerEvaluator`: whether the
/// detokenizer joins a token sample's tokens back into its text, scored
/// as one item per sample.
///
/// A sample read from the token format never matches: its text ends with
/// a space ([`crate::native::token`]) and the detokenized text does not.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DetokenizerEvaluator {
    fmeasure: FMeasure,
}

impl DetokenizerEvaluator {
    /// Detokenizes `reference`'s tokens with `detokenizer` and scores the
    /// text against the reference's.
    ///
    /// # Errors
    ///
    /// [`EvalError::TokenOutOfText`] when a reference token span is past
    /// the end of the detokenized text, after the score is counted.
    pub fn evaluate(
        &mut self,
        reference: &TokenSample<'_>,
        detokenizer: &Detokenizer<'_>,
    ) -> Result<Verdict, EvalError> {
        let tokens: Vec<&str> = (0..reference.tokens().len()).map(|i| reference.token(i)).collect();
        let mut text = String::new();
        detokenizer.detokenize(&tokens, None, &mut text);
        self.fmeasure.update(&[reference.text()], &[text.as_str()]);
        check_in_text(reference.tokens(), &text)?;
        Ok(Verdict::of(reference.text() == text))
    }

    pub fn fmeasure(&self) -> &FMeasure {
        &self.fmeasure
    }
}

/// The sentence detector evaluator, `SentenceDetectorEvaluator`:
/// precision and recall of the predicted sentence spans, both they and
/// the reference's trimmed of whitespace first (`Span.trim`, which
/// follows the whitespace mode).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SentenceDetectorEvaluator {
    fmeasure: FMeasure,
    mode: WhitespaceMode,
}

impl SentenceDetectorEvaluator {
    /// An evaluator that trims spans of whitespace in `mode`.
    pub fn new(mode: WhitespaceMode) -> SentenceDetectorEvaluator {
        SentenceDetectorEvaluator { fmeasure: FMeasure::default(), mode }
    }

    /// Scores the sentence spans `predicted`, byte spans of `reference`'s
    /// document. The verdict compares the trimmed predictions with the
    /// reference's untrimmed spans.
    ///
    /// # Errors
    ///
    /// [`EvalError::TrimOutOfText`] when trimming a prediction reads past
    /// the document's end, and [`EvalError::SentenceOutOfText`] for an
    /// empty prediction past it, after the scores are counted.
    pub fn evaluate(&mut self, reference: &SentenceSample<'_>, predicted: &[Span]) -> Result<Verdict, EvalError> {
        let document = reference.document();
        let trim = |spans: &[Span]| spans.iter().map(|&s| trim(document, s, self.mode)).collect::<Result<Vec<_>, _>>();
        let predictions = trim(predicted)?;
        let references = trim(reference.sentences())?;
        self.fmeasure.update(&untyped(&references), &untyped(&predictions));
        if let Some(&span) = predictions.iter().find(|s| s.end as usize > document.len()) {
            return Err(EvalError::SentenceOutOfText {
                span,
                len: document.len(),
                units: span_units(document, span),
                len_units: units(document, document.len()),
            });
        }
        Ok(Verdict::of(reference.sentences() == predictions.as_slice()))
    }

    pub fn fmeasure(&self) -> &FMeasure {
        &self.fmeasure
    }
}

/// `Span.trim(text)` on bytes: the whitespace at the span's ends removed,
/// an empty span at its start when it is all whitespace. A byte inside a
/// character is not whitespace.
fn trim(text: &str, span: Span, mode: WhitespaceMode) -> Result<Span, EvalError> {
    let len = text.len();
    let (start, end) = (span.start as usize, span.end as usize);
    let past_end = |index: usize| EvalError::TrimOutOfText {
        index,
        len,
        index_units: units(text, index),
        len_units: units(text, len),
    };
    let space_at = |i: usize| -> Result<Option<usize>, EvalError> {
        if i >= len {
            return Err(past_end(i));
        }
        let c = text.get(i..).and_then(|t| t.chars().next());
        Ok(c.filter(|&c| mode.is_whitespace(c as u32)).map(char::len_utf8))
    };
    let space_before = |i: usize| -> Result<Option<usize>, EvalError> {
        if i > len {
            return Err(past_end(i - 1));
        }
        let c = text.get(..i).and_then(|t| t.chars().next_back());
        Ok(c.filter(|&c| mode.is_whitespace(c as u32)).map(char::len_utf8))
    };
    let mut new_start = start;
    while new_start < end {
        match space_at(new_start)? {
            Some(n) => new_start += n,
            None => break,
        }
    }
    let mut new_end = end;
    while new_end > start {
        match space_before(new_end)? {
            Some(n) => new_end -= n,
            None => break,
        }
    }
    Ok(if new_start == start && new_end == end {
        span
    } else if new_start > new_end {
        Span::new(span.start, span.start)
    } else {
        Span::new(new_start as u32, new_end as u32)
    })
}

/// The document categorizer evaluator, `DocumentCategorizerEvaluator`:
/// the share of documents given the reference category.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DocumentCategorizerEvaluator {
    accuracy: Mean,
}

impl DocumentCategorizerEvaluator {
    /// Scores the category `predicted` for `reference`.
    pub fn evaluate(&mut self, reference: &DocumentSample<'_>, predicted: &str) -> Verdict {
        let correct = reference.category.as_str() == predicted;
        self.accuracy.add(if correct { 1.0 } else { 0.0 });
        Verdict::of(correct)
    }

    pub fn accuracy(&self) -> f64 {
        self.accuracy.mean()
    }

    /// The number of documents scored (`getDocumentCount`).
    pub fn document_count(&self) -> u64 {
        self.accuracy.count()
    }
}

/// The language detector evaluator, `LanguageDetectorEvaluator`: the
/// share of samples given the reference language.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct LanguageDetectorEvaluator {
    accuracy: Mean,
}

impl LanguageDetectorEvaluator {
    /// Scores the language `predicted` for `reference`.
    pub fn evaluate(&mut self, reference: &LanguageSample<'_>, predicted: &str) -> Verdict {
        let correct = reference.language.as_str() == predicted;
        self.accuracy.add(if correct { 1.0 } else { 0.0 });
        Verdict::of(correct)
    }

    pub fn accuracy(&self) -> f64 {
        self.accuracy.mean()
    }

    pub fn document_count(&self) -> u64 {
        self.accuracy.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{ChunkReader, NameReader, SentenceReader, TokenReader, WordTagReader};

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
        let p = [Name::new(2, 3, Some("p")), Name::new(0, 1, Some("p"))];
        assert_eq!(e.evaluate(&s, &p), Ok(Verdict::Correct));
        assert_eq!(e.fmeasure().true_positives(), 2);
        let overlap = [Name::new(0, 2, Some("p")), Name::new(1, 3, None)];
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
        assert_eq!(err.java_class(), "java.lang.StringIndexOutOfBoundsException");
        let err = e.evaluate(&s, &["O"]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "All arrays must have the same length: sentenceSize: 3, tagsSize: 3, predsSize: 1!"
        );
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
}
