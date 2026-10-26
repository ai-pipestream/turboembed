//! Real source offsets in a corpus address space, without joining comments.
//! Each sentence start is the byte base of its borrowed original comment.
//! Subtracting that base maps a corpus span back to the exact source bytes.

use std::collections::HashMap;

use turbo_analysis::{Ix, IxRange, Span, Spans, ViewId};
use turbo_analysis_layers::{Incivility, IncivilityItem, IncivilityLabel, IncivilityTarget, Sentences, Tokens};
use turbo_doccat::{GoldIncivilityUnit, IncivilityAnnotator};

use super::data::Unit;

pub(super) struct Prepared<'a> {
    pub tokens: Vec<Vec<String>>,
    pub units: Vec<GoldIncivilityUnit>,
    pub parent_ids: Vec<Option<String>>,
    pub token_layer: Tokens,
    pub sentence_layer: Sentences,
    sources: Vec<&'a str>,
}

impl Prepared<'_> {
    /// Projects a corpus token range to bytes in the original comment.
    pub fn source_span(&self, row: usize, range: IxRange<Tokens>) -> Result<Span, String> {
        let (starts, ends) = self.token_layer.spans.columns();
        let start = range.start.get();
        let end = range.end.get();
        if start >= end || end > starts.len() {
            return Err("empty or out-of-bounds incivility evidence".into());
        }
        let base = self.sentence_layer.spans.columns().0[row];
        let local = starts[start].checked_sub(base).zip(ends[end - 1].checked_sub(base));
        let Some((a, b)) = local else {
            return Err("incivility evidence crosses a comment boundary".into());
        };
        let span = Span::new(a, b);
        if !self.sources[row].get(span.range()).is_some_and(|s| !s.is_empty()) {
            return Err("incivility evidence is not a non-empty source-text span".into());
        }
        Ok(span)
    }
}

pub(super) fn prepare(units: &[Unit]) -> Result<Prepared<'_>, String> {
    let mut all_tokens = Vec::with_capacity(units.len());
    let mut gold = Vec::with_capacity(units.len());
    let mut parent_ids = Vec::with_capacity(units.len());
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    let mut tok_sent = Vec::new();
    let mut byte_cursor = 0u32;
    for (i, u) in units.iter().enumerate() {
        let row_len = u32::try_from(u.text.len()).map_err(|_| "comment exceeds u32 bytes")?;
        let sent_end = byte_cursor.checked_add(row_len).ok_or("corpus exceeds u32 bytes")?;
        let row = u32::try_from(i).map_err(|_| "corpus exceeds u32 comments")?;
        let mut raw_spans = Vec::new();
        turbo_tokenize::simple_spans(&u.text, turbo_tokenize::TokenizerOptions::default(), &mut raw_spans);
        let keys: Vec<String> = raw_spans
            .iter()
            .map(|s| turbo_doccat::incivility_token_key(&u.text[s.start as usize..s.end as usize]))
            .collect();
        for span in raw_spans {
            tok_spans.push(Span::new(byte_cursor + span.start, byte_cursor + span.end));
            tok_sent.push(Ix::new(row));
        }
        sent_spans.push(Span::new(byte_cursor, sent_end));
        byte_cursor = sent_end;
        all_tokens.push(keys);
        gold.push(GoldIncivilityUnit {
            positive_labels: IncivilityLabel::ALL.into_iter().filter(|&l| u.targets.get(l) >= 0.5).collect(),
        });
        parent_ids.push(u.parent_id.clone());
    }
    Ok(Prepared {
        tokens: all_tokens,
        units: gold,
        parent_ids,
        token_layer: Tokens { spans: tok_spans, sentence: Some(tok_sent) },
        sentence_layer: Sentences { spans: sent_spans },
        sources: units.iter().map(|u| u.text.as_str()).collect(),
    })
}

pub(super) fn score_split(
    annotator: &IncivilityAnnotator,
    p: &Prepared<'_>,
    vectors: &[f32],
    parent_lookup: Option<&HashMap<String, &[f32]>>,
    dim: usize,
) -> Result<Incivility, String> {
    let mut layer = Incivility::new();
    let mut tok_offset = 0usize;
    for (i, tokens) in p.tokens.iter().enumerate() {
        let v = if dim == 0 { &[][..] } else { &vectors[i * dim..(i + 1) * dim] };
        let ctx = parent_lookup.and_then(|map| p.parent_ids[i].as_deref().and_then(|pid| map.get(pid).copied()));
        let (scores, local_ev) = annotator.score_unit_in_context(tokens, 0, tokens.len(), None, Some(v), ctx)?;
        let shifted: Vec<_> = local_ev
            .into_iter()
            .map(|(l, r)| {
                (l, IxRange::new(r.start.get() as u32 + tok_offset as u32, r.end.get() as u32 + tok_offset as u32))
            })
            .collect();
        for &(_, range) in &shifted {
            p.source_span(i, range)?;
        }
        layer.try_push(
            &p.token_layer,
            Some(&p.sentence_layer),
            None,
            None,
            IncivilityItem { target: IncivilityTarget::Sentence(Ix::new(i as u32)), scores, evidence: &shifted },
        )?;
        tok_offset += tokens.len();
    }
    Ok(layer)
}
