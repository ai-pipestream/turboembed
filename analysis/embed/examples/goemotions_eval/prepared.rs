//! Real source offsets in a corpus address space, without joining comments.
//! Each sentence start is the byte base of its borrowed original comment.
//! Subtracting that base maps a corpus span back to the exact source bytes.

use super::*;

pub(super) struct Prepared<'a> {
    pub keys: Vec<Vec<String>>,
    pub gold: Vec<GoldEmotionUnit>,
    pub tokens: Tokens,
    pub sentences: Sentences,
    sources: Vec<&'a str>,
}

impl Prepared<'_> {
    /// Projects a corpus token range to bytes in the original comment.
    pub fn source_span(&self, row: usize, range: IxRange<Tokens>) -> Result<Span, String> {
        let (starts, ends) = self.tokens.spans.columns();
        let start = range.start.get();
        let end = range.end.get();
        if start >= end || end > starts.len() {
            return Err("empty or out-of-bounds emotion evidence".into());
        }
        let base = self.sentences.spans.columns().0[row];
        let local = starts[start].checked_sub(base).zip(ends[end - 1].checked_sub(base));
        let Some((a, b)) = local else { return Err("emotion evidence crosses a comment boundary".into()) };
        let span = Span::new(a, b);
        if !self.sources[row].get(span.range()).is_some_and(|s| !s.is_empty()) {
            return Err("emotion evidence is not a non-empty source-text span".into());
        }
        Ok(span)
    }
}

pub(super) fn prepare(rows: &[Row]) -> Result<Prepared<'_>, String> {
    let mut all_keys = Vec::with_capacity(rows.len());
    let mut gold = Vec::with_capacity(rows.len());
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    let mut tok_sent = Vec::new();
    let mut byte_cursor = 0u32;
    for (i, r) in rows.iter().enumerate() {
        let row_len = u32::try_from(r.text.len()).map_err(|_| "comment exceeds u32 bytes")?;
        let sent_end = byte_cursor.checked_add(row_len).ok_or("corpus exceeds u32 bytes")?;
        let row = u32::try_from(i).map_err(|_| "corpus exceeds u32 comments")?;
        let mut raw_spans = Vec::new();
        turbo_tokenize::simple_spans(&r.text, turbo_tokenize::TokenizerOptions::default(), &mut raw_spans);
        let keys: Vec<String> =
            raw_spans.iter().map(|s| token_key(&r.text[s.start as usize..s.end as usize])).collect();
        for span in raw_spans {
            tok_spans.push(Span::new(byte_cursor + span.start, byte_cursor + span.end));
            tok_sent.push(Ix::new(row));
        }
        sent_spans.push(Span::new(byte_cursor, sent_end));
        byte_cursor = sent_end;
        all_keys.push(keys);
        gold.push(GoldEmotionUnit { labels: r.labels.clone() });
    }
    Ok(Prepared {
        keys: all_keys,
        gold,
        tokens: Tokens { spans: tok_spans, sentence: Some(tok_sent) },
        sentences: Sentences { spans: sent_spans },
        sources: rows.iter().map(|r| r.text.as_str()).collect(),
    })
}

pub(super) fn score_split(
    annotator: &EmotionAnnotator,
    p: &Prepared<'_>,
    vectors: &[f32],
    dim: usize,
) -> Result<EmotionLayer, String> {
    let mut layer = EmotionLayer::new();
    let mut tok_offset = 0usize;
    for (i, keys) in p.keys.iter().enumerate() {
        let v = &vectors[i * dim..(i + 1) * dim];
        let (scores, local_ev) = annotator.score_unit(keys, 0, keys.len(), Some(v))?;
        let shifted: Vec<_> = local_ev
            .into_iter()
            .map(|(d, r)| {
                (d, IxRange::new(r.start.get() as u32 + tok_offset as u32, r.end.get() as u32 + tok_offset as u32))
            })
            .collect();
        for &(_, range) in &shifted {
            p.source_span(i, range)?;
        }
        layer.try_push(
            &p.tokens,
            Some(&p.sentences),
            None,
            EmotionItem { target: EmotionTarget::Sentence(Ix::new(i as u32)), scores, evidence: &shifted },
        )?;
        tok_offset += keys.len();
    }
    Ok(layer)
}
