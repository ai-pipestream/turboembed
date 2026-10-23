//! Fallback sentence splitting for scripts written without sentence-ending
//! punctuation (`Thai`, `Lao`, `Khmer`).
//!
//! UAX #29 and the language tailorings have no rule for finding sentence
//! boundaries inside unpunctuated text, so a paragraph of Thai, Lao or Khmer
//! without ASCII or script punctuation comes out of [`RuleSplitter`](super::RuleSplitter)
//! as a single sentence span. When an [`UnpunctuatedSplitter`] (such as
//! `turbo-sat`'s `SatAnnotator`) is attached to a [`RuleSplitter`](super::RuleSplitter)
//! or [`RuleSentenceDetector`](super::RuleSentenceDetector), any candidate
//! sentence span that contains a character of [`UNPUNCTUATED_SCRIPTS`] is
//! handed to that fallback splitter, and its sub-spans are validated, trimmed
//! of Unicode `White_Space` and shifted back into the document's byte offsets.

use std::fmt;

use turbo_analysis::{Annotator, Error, Execution, ModelRef, Refs, Span, Spans};
use turbo_analysis_layers::Sentences;
use turbo_unicode::script::script;
use turbo_unicode::sentence_break::sentence_break;
use turbo_unicode::{Script, SentenceBreak, WhitespaceMode};

use crate::check_len;
use crate::scan::trim;

use super::{RULE_SCRATCH, RuleSentenceDetector, RuleSplitter};

/// Scripts written without sentence-ending punctuation that [`RuleSplitter`]
/// routes to its [`UnpunctuatedSplitter`] fallback when one is configured.
pub const UNPUNCTUATED_SCRIPTS: [Script; 3] = [Script::Thai, Script::Lao, Script::Khmer];

/// Whether `c` belongs to one of [`UNPUNCTUATED_SCRIPTS`] (`Thai`, `Lao`,
/// `Khmer`).
#[inline]
pub fn is_unpunctuated_script(c: char) -> bool {
    matches!(script(c), Script::Thai | Script::Lao | Script::Khmer)
}

/// Whether `text` contains at least one character of [`UNPUNCTUATED_SCRIPTS`].
#[inline]
pub fn has_unpunctuated_script(text: &str) -> bool {
    text.chars().any(is_unpunctuated_script)
}

/// Whether a candidate sentence `slice` produced by the rule splitter needs
/// refinement by an [`UnpunctuatedSplitter`]: it contains at least one
/// character of [`UNPUNCTUATED_SCRIPTS`] and is not already terminated by a
/// Khmer sentence full stop (`U+17D4` KHMER SIGN KHAN or `U+17D5` KHMER SIGN
/// BARIYOOSAN, optionally followed by closing punctuation).
#[inline]
pub fn needs_unpunctuated_fallback(slice: &str) -> bool {
    if !has_unpunctuated_script(slice) {
        return false;
    }
    let last_non_close =
        slice.chars().rev().find(|&c| !c.is_whitespace() && sentence_break(c as u32) != SentenceBreak::Close);
    !matches!(last_non_close, Some('\u{17D4}' | '\u{17D5}'))
}

/// A sentence splitter consulted by [`RuleSplitter`] and
/// [`RuleSentenceDetector`] on candidate spans that contain characters of
/// [`UNPUNCTUATED_SCRIPTS`].
pub trait UnpunctuatedSplitter: Send + Sync + fmt::Debug {
    /// Splits each slice in `texts` into sentence spans (UTF-8 byte offsets
    /// relative to that slice). Returns one result per element of `texts`.
    fn split_batch(&self, texts: &[&str]) -> Vec<Result<Vec<Span>, Error>>;

    /// Splits a single `text` slice into sentence spans appended to `out`.
    fn split(&self, text: &str, out: &mut Vec<Span>) -> Result<(), Error> {
        let mut batch = self.split_batch(&[text]);
        if batch.len() != 1 {
            return Err(Error::Annotator {
                annotator: RuleSentenceDetector::ID,
                detail: format!("unpunctuated fallback returned {} results for 1 slice", batch.len()),
            });
        }
        out.extend(batch.remove(0)?);
        Ok(())
    }

    /// Model identity of the fallback splitter, when backed by a model.
    fn model(&self) -> Option<ModelRef> {
        None
    }

    /// Execution device and precision of the fallback splitter, when backed by
    /// a model.
    fn execution(&self) -> Option<Execution> {
        None
    }

    /// Configuration hash of the fallback splitter, folded into
    /// [`RuleSentenceDetector::config_hash`](turbo_analysis::Annotator::config_hash).
    fn config_hash(&self) -> u64;
}

/// Validates sub-spans returned by an [`UnpunctuatedSplitter`] over `slice`
/// (which starts at `base` bytes in the outer text), trims each sub-span of
/// Unicode `White_Space`, and appends the shifted spans to `out`.
///
/// If the fallback returns no non-whitespace spans for a non-whitespace
/// `slice`, `base..base + slice.len()` is kept so no text is dropped.
pub(crate) fn validate_and_shift(
    annotator: &'static str,
    slice: &str,
    base: u32,
    sub_spans: &[Span],
    out: &mut Vec<Span>,
) -> Result<(), Error> {
    let len = slice.len();
    let before = out.len();
    let mut prev_end = 0usize;
    for &sp in sub_spans {
        let s = sp.start as usize;
        let e = sp.end as usize;
        if s < prev_end || s > e || e > len {
            return Err(Error::Annotator {
                annotator,
                detail: format!(
                    "unpunctuated fallback returned out-of-order or out-of-bounds span {s}..{e} in slice of {len} bytes"
                ),
            });
        }
        if !slice.is_char_boundary(s) || !slice.is_char_boundary(e) {
            return Err(Error::Annotator {
                annotator,
                detail: format!("unpunctuated fallback span {s}..{e} is not on a UTF-8 character boundary"),
            });
        }
        prev_end = e;
        if let Some((ts, te)) = trim(slice, s, e, WhitespaceMode::Unicode) {
            out.push(Span::new(base + ts as u32, base + te as u32));
        }
    }
    if out.len() == before
        && let Some((ts, te)) = trim(slice, 0, len, WhitespaceMode::Unicode)
    {
        out.push(Span::new(base + ts as u32, base + te as u32));
    }
    Ok(())
}

/// Splits `text` with `splitter`'s rules, then refines any span containing
/// [`UNPUNCTUATED_SCRIPTS`] through `fallback`.
pub(crate) fn split_with_fallback(
    splitter: &RuleSplitter,
    text: &str,
    fallback: &dyn UnpunctuatedSplitter,
    out: &mut Vec<Span>,
) -> Result<(), Error> {
    check_len(RuleSentenceDetector::ID, text)?;
    let mut rule_spans = Vec::new();
    RULE_SCRATCH.with_borrow_mut(|scratch| splitter.split_rules_only(text, scratch, &mut rule_spans));
    let mut slices: Vec<(usize, u32, &str)> = Vec::new();
    for (idx, sp) in rule_spans.iter().enumerate() {
        let slice = &text[sp.start as usize..sp.end as usize];
        if needs_unpunctuated_fallback(slice) {
            slices.push((idx, sp.start, slice));
        }
    }
    if slices.is_empty() {
        out.extend_from_slice(&rule_spans);
        return Ok(());
    }
    let texts: Vec<&str> = slices.iter().map(|&(_, _, s)| s).collect();
    let results = fallback.split_batch(&texts);
    if results.len() != texts.len() {
        return Err(Error::Annotator {
            annotator: RuleSentenceDetector::ID,
            detail: format!("unpunctuated fallback returned {} results for {} slices", results.len(), texts.len()),
        });
    }
    let mut slice_iter = slices.into_iter().zip(results).peekable();
    for (idx, &sp) in rule_spans.iter().enumerate() {
        if let Some(&((s_idx, _, _), _)) = slice_iter.peek()
            && s_idx == idx
        {
            let ((_, base, slice), res) = slice_iter.next().unwrap();
            let sub_spans = res?;
            validate_and_shift(RuleSentenceDetector::ID, slice, base, &sub_spans, out)?;
        } else {
            out.push(sp);
        }
    }
    Ok(())
}

/// Batch annotation for [`RuleSentenceDetector`] when an
/// [`UnpunctuatedSplitter`] fallback is configured: batches all unpunctuated
/// script spans across all documents in `inputs` into a single
/// [`UnpunctuatedSplitter::split_batch`] call.
pub(crate) fn annotate_batch_with_fallback<'d>(
    detector: &RuleSentenceDetector,
    fallback: &dyn UnpunctuatedSplitter,
    inputs: Vec<Refs<'d, RuleSentenceDetector>>,
) -> Vec<Result<(Sentences,), Error>> {
    struct DocPlan<'a> {
        text: &'a str,
        rule_spans: Vec<Span>,
    }

    let mut plans: Vec<Result<DocPlan<'d>, Error>> = Vec::with_capacity(inputs.len());
    let mut query_slices: Vec<&'d str> = Vec::new();
    let mut query_owners: Vec<(usize, usize, u32)> = Vec::new();

    for (doc_idx, (view,)) in inputs.iter().enumerate() {
        let text = view.text();
        if let Err(e) = check_len(RuleSentenceDetector::ID, text) {
            plans.push(Err(e));
            continue;
        }
        let mut rule_spans = Vec::new();
        RULE_SCRATCH.with_borrow_mut(|scratch| detector.splitter.split_rules_only(text, scratch, &mut rule_spans));
        for (span_idx, sp) in rule_spans.iter().enumerate() {
            let slice = &text[sp.start as usize..sp.end as usize];
            if needs_unpunctuated_fallback(slice) {
                query_slices.push(slice);
                query_owners.push((doc_idx, span_idx, sp.start));
            }
        }
        plans.push(Ok(DocPlan { text, rule_spans }));
    }

    let batch_results = if query_slices.is_empty() {
        Vec::new()
    } else {
        let res = fallback.split_batch(&query_slices);
        if res.len() != query_slices.len() {
            let err = Error::Annotator {
                annotator: RuleSentenceDetector::ID,
                detail: format!(
                    "unpunctuated fallback returned {} results for {} slices",
                    res.len(),
                    query_slices.len()
                ),
            };
            return (0..inputs.len()).map(|_| Err(err.clone())).collect();
        }
        res
    };

    let mut owner_iter = query_owners.into_iter().zip(batch_results).peekable();
    let mut out = Vec::with_capacity(inputs.len());

    for (doc_idx, ((view,), plan)) in inputs.into_iter().zip(plans).enumerate() {
        let doc_plan = match plan {
            Ok(p) => p,
            Err(e) => {
                out.push(Err(e));
                continue;
            }
        };
        let mut final_spans = Vec::with_capacity(doc_plan.rule_spans.len());
        let mut doc_err = None;
        for (span_idx, &sp) in doc_plan.rule_spans.iter().enumerate() {
            if let Some(&((d_idx, s_idx, _), _)) = owner_iter.peek()
                && d_idx == doc_idx
                && s_idx == span_idx
            {
                let ((_, _, base), res) = owner_iter.next().unwrap();
                if doc_err.is_some() {
                    continue;
                }
                let slice = &doc_plan.text[sp.start as usize..sp.end as usize];
                match res
                    .and_then(|sub| validate_and_shift(RuleSentenceDetector::ID, slice, base, &sub, &mut final_spans))
                {
                    Ok(()) => {}
                    Err(e) => doc_err = Some(e),
                }
            } else if doc_err.is_none() {
                final_spans.push(sp);
            }
        }
        if let Some(e) = doc_err {
            out.push(Err(e));
        } else {
            let mut spans = Spans::with_capacity(view.id(), final_spans.len());
            for s in final_spans {
                spans.push(s);
            }
            out.push(Ok((Sentences { spans },)));
        }
    }
    out
}
