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

mod span;
#[cfg(test)]
mod tests;

use std::fmt;

use turbo_analysis::Span;

pub use self::span::{DetokenizerEvaluator, SentenceDetectorEvaluator, TokenizerEvaluator};
use super::measure::{FMeasure, LabeledSpan, Mean};
use crate::error::FormatError;
use crate::sample::chunk::phrase_spans;
use crate::sample::name::{DEFAULT_NAME_TYPE, java_list};
use crate::sample::{ChunkSample, DocumentSample, LanguageSample, LemmaSample, Name, NameSample, PosSample};
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
    pub(super) fn of(equal: bool) -> Verdict {
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
    /// The class of the exception Java throws, or `None` for a refusal
    /// of this port's own (a chunk tag that splits a surrogate pair,
    /// [`FormatError::SplitsSurrogatePair`]).
    pub fn java_class(&self) -> Option<&'static str> {
        match self {
            EvalError::IndexOutOfBounds { .. } => Some("java.lang.ArrayIndexOutOfBoundsException"),
            EvalError::TagCount { .. }
            | EvalError::ArrayLengths { .. }
            | EvalError::TokenOutOfText { .. }
            | EvalError::SentenceOutOfText { .. } => Some("java.lang.IllegalArgumentException"),
            EvalError::OverlappingNames { .. } => Some("java.lang.RuntimeException"),
            EvalError::TrimOutOfText { .. } => Some("java.lang.StringIndexOutOfBoundsException"),
            EvalError::Phrases(e) => e.java_class(),
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
