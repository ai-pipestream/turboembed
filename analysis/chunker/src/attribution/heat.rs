//! The document heat partition over attributed chunks.

use std::collections::BTreeMap;

use turbo_analysis::{Document, Error, Ix, Key, Layer, Span, ViewId};
use turbo_analysis_layers::Chunks;

use super::{ChunkAttribution, MatchedSpan};

/// One run of the document heat partition: an unscored gap (`score`
/// and `chunk` both `None`) or a stretch of one winning chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct HeatSegment {
    pub span: Span,
    pub score: Option<f32>,
    pub chunk: Option<Ix<Chunks>>,
    pub matched_spans: Vec<MatchedSpan>,
}

/// The whole view partitioned into contiguous scored and unscored
/// heat segments.
#[derive(Clone, Debug, PartialEq)]
pub struct DocumentHeat {
    pub view: ViewId,
    pub segments: Vec<HeatSegment>,
}

fn invalid(detail: String) -> Error {
    Error::Invalid { layer: Key::<Chunks>::DEFAULT.id(), detail }
}

impl DocumentHeat {
    /// Partitions `view` of `doc` by the attributed level-0 chunks:
    /// where they overlap the highest `total_score` wins (ties to the
    /// lowest chunk index); elsewhere the segment is an unscored gap. A
    /// matched span rides with the segment its start falls in, and only
    /// when its own chunk wins there: a losing chunk's spans are dropped,
    /// and so is a zero-width span at the view end. Scores compare by
    /// `total_cmp`, so `+0.0` outranks `-0.0` whatever the chunk index.
    ///
    /// Refuses an unknown view, invalid or other-view chunks, duplicate
    /// or non-level-0 attributions, a span that is not its chunk's
    /// envelope, a non-finite total, and a matched span reversed, past
    /// the view end, off a character boundary, or outside its envelope.
    pub fn build(
        doc: &Document,
        view: ViewId,
        chunks: &Chunks,
        attributions: &[ChunkAttribution],
    ) -> Result<DocumentHeat, Error> {
        if view.0 as usize >= doc.view_count() {
            return Err(Error::Bounds {
                layer: Key::<Chunks>::DEFAULT.id(),
                view,
                detail: format!("the document has {} views, view {} does not exist", doc.view_count(), view.0),
            });
        }
        chunks.check().map_err(invalid)?;
        if chunks.spans.view() != view {
            return Err(Error::Bounds {
                layer: Key::<Chunks>::DEFAULT.id(),
                view,
                detail: format!("chunks are in view {}, the heat is over view {}", chunks.spans.view().0, view.0),
            });
        }
        let text = doc.text(view);
        let len = text.len() as u32;
        let mut by_chunk: BTreeMap<u32, &ChunkAttribution> = BTreeMap::new();
        for a in attributions {
            let at = a.chunk.get();
            if at >= chunks.spans.len() {
                return Err(invalid(format!("attribution names chunk {at} of {}", chunks.spans.len())));
            }
            if chunks.level[at] != 0 {
                return Err(invalid(format!("attribution names chunk {at} at level {}, not 0", chunks.level[at])));
            }
            if by_chunk.insert(at as u32, a).is_some() {
                return Err(invalid(format!("chunk {at} is attributed twice")));
            }
            if a.span != chunks.spans.get(a.chunk) {
                return Err(invalid(format!(
                    "attribution span {}..{} is not chunk {at}'s envelope",
                    a.span.start, a.span.end
                )));
            }
            if !a.total_score.is_finite() {
                return Err(invalid(format!("chunk {at} has a non-finite total score {}", a.total_score)));
            }
            for m in &a.matched_spans {
                if m.span.start > m.span.end
                    || m.span.end > len
                    || m.span.start < a.span.start
                    || m.span.end > a.span.end
                {
                    return Err(invalid(format!(
                        "chunk {at} matched span {}..{} is outside its envelope {}..{}",
                        m.span.start, m.span.end, a.span.start, a.span.end
                    )));
                }
                if !text.is_char_boundary(m.span.start as usize) || !text.is_char_boundary(m.span.end as usize) {
                    return Err(invalid(format!(
                        "chunk {at} matched span {}..{} splits a character",
                        m.span.start, m.span.end
                    )));
                }
            }
        }
        if len == 0 {
            return Ok(DocumentHeat { view, segments: Vec::new() });
        }
        let mut cuts = vec![0u32, len];
        for a in by_chunk.values() {
            for f in chunks.spans.fragments(a.chunk) {
                cuts.push(f.start.min(len));
                cuts.push(f.end.min(len));
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        let mut segments: Vec<HeatSegment> = Vec::new();
        for w in cuts.windows(2) {
            let (s, e) = (w[0], w[1]);
            if s == e {
                continue;
            }
            let mut winner: Option<&ChunkAttribution> = None;
            for a in by_chunk.values() {
                let covers = chunks.spans.fragments(a.chunk).any(|f| f.start <= s && e <= f.end);
                if covers
                    && winner.is_none_or(|w| {
                        a.total_score.total_cmp(&w.total_score) == std::cmp::Ordering::Greater
                            || (a.total_score == w.total_score && a.chunk < w.chunk)
                    })
                {
                    winner = Some(a);
                }
            }
            let (score, chunk) = match winner {
                Some(a) => (Some(a.total_score), Some(a.chunk)),
                None => (None, None),
            };
            let matched: Vec<MatchedSpan> = match winner {
                Some(a) => a.matched_spans.iter().filter(|m| s <= m.span.start && m.span.start < e).cloned().collect(),
                None => Vec::new(),
            };
            if let Some(last) = segments.last_mut()
                && last.chunk == chunk
                && last.score == score
                && last.span.end == s
            {
                last.span.end = e;
                last.matched_spans.extend(matched);
            } else {
                segments.push(HeatSegment { span: Span { start: s, end: e }, score, chunk, matched_spans: matched });
            }
        }
        Ok(DocumentHeat { view, segments })
    }
}
