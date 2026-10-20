use super::{Answer, DecodeOptions, SpanAnswer, WindowLogits};
use crate::status::{Error, INVALID_ARGUMENT, Result};
use std::{collections::BTreeMap, ops::Range};

fn invalid(message: impl Into<String>) -> Error {
    Error::new(INVALID_ARGUMENT, message)
}

fn validate(text: &str, window: &WindowLogits) -> Result<()> {
    let n = window.start.len();
    if n == 0 || window.end.len() != n || window.context.len() != n || window.max_context.len() != n {
        return Err(invalid("QA window columns must have the same nonzero length"));
    }
    if window.cls >= n || window.context[window.cls].is_some() || window.max_context[window.cls] {
        return Err(invalid("QA CLS must be a valid non-context position"));
    }
    if window.start.iter().chain(&window.end).any(|x| !x.is_finite()) {
        return Err(invalid("QA logits must be finite"));
    }
    let mut previous: Option<&Range<usize>> = None;
    let mut context_ended = false;
    for (span, &owns) in window.context.iter().zip(&window.max_context) {
        match span {
            None => {
                if owns {
                    return Err(invalid("QA non-context position cannot own candidates"));
                }
                context_ended |= previous.is_some();
            }
            Some(span) => {
                if context_ended {
                    return Err(invalid("QA context tokens must form one contiguous segment"));
                }
                if span.start >= span.end || text.get(span.clone()).is_none() {
                    return Err(invalid("QA context offset is empty, outside the text, or splits UTF-8"));
                }
                if previous.is_some_and(|p| p.start > span.start || p.end > span.end) {
                    return Err(invalid("QA context offsets must be in text order"));
                }
                previous = Some(span);
            }
        }
    }
    Ok(())
}

/// Select checked byte spans from raw logits. No text is sliced before all
/// windows pass validation. Repeated spans across windows retain their best
/// score. Null calibration uses the minimum CLS score across windows, matching
/// the SQuAD-v2 convention; a high null score in an irrelevant window must not
/// suppress an answer in another window.
///
/// Empty window input is invalid. A valid window without context tokens
/// produces `Answer::None`. Ties use earlier byte offsets, then shorter spans.
pub fn decode(text: &str, windows: &[WindowLogits], options: DecodeOptions) -> Result<Answer> {
    if windows.is_empty() || options.max_answer_tokens == 0 || options.top_k == 0 {
        return Err(invalid("QA requires windows, a positive answer length, and positive top_k"));
    }
    if !options.null_threshold.is_finite() {
        return Err(invalid("QA null threshold must be finite"));
    }
    for w in windows {
        validate(text, w)?;
    }
    let mut null_score = f64::INFINITY;
    let mut candidates: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for w in windows {
        null_score = null_score.min(f64::from(w.start[w.cls]) + f64::from(w.end[w.cls]));
        for (i, offset) in w.context.iter().enumerate() {
            let Some(begin) = offset.as_ref().filter(|_| w.max_context[i]) else { continue };
            let limit = i.saturating_add(options.max_answer_tokens).min(w.context.len());
            for j in i..limit {
                let Some(end) = &w.context[j] else { break };
                let score = f64::from(w.start[i]) + f64::from(w.end[j]);
                candidates.entry((begin.start, end.end)).and_modify(|s| *s = s.max(score)).or_insert(score);
            }
        }
    }
    let mut ranked: Vec<_> = candidates.into_iter().collect();
    ranked.sort_by(|(a, x), (b, y)| y.total_cmp(x).then_with(|| a.cmp(b)));
    let Some((_, best)) = ranked.first() else { return Ok(Answer::None { null_score }) };
    if *best - null_score <= options.null_threshold {
        return Ok(Answer::None { null_score });
    }
    let mut answers: Vec<SpanAnswer> = Vec::new();
    for ((start, end), score) in ranked {
        if options.non_overlapping && answers.iter().any(|a| start < a.bytes.end && a.bytes.start < end) {
            continue;
        }
        answers.push(SpanAnswer { bytes: start..end, score, probability: 0.0 });
        if answers.len() == options.top_k {
            break;
        }
    }
    let max_score = answers[0].score.max(null_score);
    let null_mass = (null_score - max_score).exp();
    let mut total = null_mass;
    for answer in &mut answers {
        answer.probability = (answer.score - max_score).exp();
        total += answer.probability;
    }
    for answer in &mut answers {
        answer.probability /= total;
    }
    Ok(Answer::Spans { answers, null_score, null_probability: null_mass / total })
}
