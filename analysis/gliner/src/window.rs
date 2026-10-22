//! Sentence-aligned packing with document-global token ranges.
use crate::{ExtractionSchema, Model, PackedInput, packing::Packer, schema::MAX_TOKENS};
use std::ops::Range;
use turbo_analysis::{Layer, View};
use turbo_analysis_layers::{Sentences, Tokens};

pub(crate) struct Window {
    pub tokens: Range<usize>,
    pub packed: PackedInput,
}

/// Validate sentence ownership before packing. With no sentence layer each
/// token is an indivisible unit. Supplied sentences are never split or skipped.
fn units(view: &View, tokens: &Tokens, sentences: Option<&Sentences>) -> Result<Vec<Range<usize>>, String> {
    let (starts, ends) = tokens.spans.columns();
    let Some(sentences) = sentences else {
        return Ok((0..starts.len()).map(|i| i..i + 1).collect());
    };
    sentences.check()?;
    if sentences.spans.view() != view.id() || sentences.spans.fragment_columns().is_some() {
        return Err("gliner needs contiguous sentences in its input view".into());
    }
    let (a, b) = sentences.spans.columns();
    let mut at = 0;
    let mut units = Vec::with_capacity(a.len());
    for (sentence, (&start, &end)) in a.iter().zip(b).enumerate() {
        if view.text().get(start as usize..end as usize).is_none() {
            return Err(format!("sentence {sentence}: UTF-8 bounds"));
        }
        let first = at;
        while at < starts.len() && starts[at] < end {
            if starts[at] < start || ends[at] > end {
                return Err(format!("token {at}: outside sentence {sentence}"));
            }
            if tokens.sentence.as_ref().is_some_and(|s| s[at].get() != sentence) {
                return Err(format!("token {at}: sentence index mismatch at sentence {sentence}"));
            }
            at += 1;
        }
        units.push(first..at);
    }
    if at != starts.len() {
        return Err(format!("token {at}: outside supplied sentences"));
    }
    Ok(units)
}

pub(crate) fn pack_windows(
    model: &Model,
    schema: &ExtractionSchema,
    view: &View,
    tokens: &Tokens,
    sentences: Option<&Sentences>,
) -> Result<Vec<Window>, String> {
    tokens.check()?;
    if tokens.spans.view() != view.id() || tokens.spans.fragment_columns().is_some() {
        return Err("gliner needs contiguous tokens in its input view".into());
    }
    let (starts, ends) = tokens.spans.columns();
    // Overlapping UTF-16 projections are supported, but both columns must be
    // monotone for a token interval's first start / last end to be its envelope.
    if !starts.windows(2).all(|x| x[0] <= x[1]) || !ends.windows(2).all(|x| x[0] <= x[1]) {
        return Err("gliner tokens must be in source order".into());
    }
    let units = units(view, tokens, sentences)?;
    if starts.is_empty() {
        return Ok(Vec::new());
    }
    let ec = model.encoder.config();
    let mut packer = Packer::new(&model.tokenizer);
    let prefix = packer.prefix(model.config(), ec.max_sequence, schema)?;
    let mut current = prefix.clone();
    let mut first = 0;
    let mut windows = Vec::new();
    let mut ids = Vec::new();
    let mut positions = Vec::new();
    for (index, unit) in units.into_iter().enumerate() {
        if unit.is_empty() {
            continue;
        }
        let kind = if sentences.is_some() { "sentence" } else { "token" };
        let fail = |message: &str| format!("{kind} {index}: {message}");
        if unit.len() > MAX_TOKENS {
            return Err(fail("max_tokens limit: 384"));
        }
        ids.clear();
        positions.clear();
        for i in unit.clone() {
            let word =
                view.text().get(starts[i] as usize..ends[i] as usize).ok_or_else(|| fail("token UTF-8 bounds"))?;
            let pieces = packer.word(word).map_err(|e| fail(&e))?;
            if prefix.ids.len() + ids.len() + pieces.len() + 1 > ec.max_sequence {
                return Err(fail("max_sequence limit exceeded"));
            }
            positions.push(ids.len());
            ids.extend_from_slice(pieces);
        }
        if current.words.len() + unit.len() > MAX_TOKENS || current.ids.len() + ids.len() + 1 > ec.max_sequence {
            current.ids.push(model.config().eos_token);
            windows.push(Window { tokens: first..unit.start, packed: current });
            current = prefix.clone();
            first = unit.start;
        }
        let offset = current.ids.len();
        current.words.extend(positions.iter().map(|p| p + offset));
        current.ids.extend_from_slice(&ids);
    }
    current.ids.push(model.config().eos_token);
    windows.push(Window { tokens: first..starts.len(), packed: current });
    Ok(windows)
}
