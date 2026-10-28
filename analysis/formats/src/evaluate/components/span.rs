//! Span-based evaluators (`TokenizerEvaluator`, `DetokenizerEvaluator`,
//! `SentenceDetectorEvaluator`) and UTF-16 / byte span trimming.

use turbo_analysis::Span;
use turbo_tokenize::detok::Detokenizer;
use turbo_unicode::WhitespaceMode;

use super::{EvalError, Verdict};
use crate::evaluate::measure::{FMeasure, LabeledSpan};
use crate::sample::{SentenceSample, TokenSample};

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
/// No sample of the token format with a token matches: its text ends
/// with a space ([`crate::native::token`]) and the detokenized text does
/// not. An empty line (an empty text, no tokens) matches.
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
pub(super) fn trim(text: &str, span: Span, mode: WhitespaceMode) -> Result<Span, EvalError> {
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
        Ok(c.filter(|&c| mode.is_whitespace(c as u32)).map(char::len_utf8).filter(|&n| i - n >= start))
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
