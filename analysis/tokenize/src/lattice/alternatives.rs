//! Multi-granularity [`Alternatives<Tokens>`] and [`TokenGraph`] emission for
//! [`LatticeTokenizer`] (issue #2340).

use turbo_analysis::{Alternatives, Annotator, Error, ModeSet, ModelRef, Refs, Span, SplitMode, View};
use turbo_analysis_layers::{PosTags, Sentences, TokenGraph, Tokens};

use super::dictionary::MecabDictionary;
use super::tokenizer::LatticeTokenizer;
use crate::fnv1a;

/// Finds the finest multi-part lexicon decomposition (`>= 2` parts) of `word`
/// in `dict.trie`, returning internal + terminal byte offsets `[0, ..., word.len()]`.
fn finest_lexicon_split(dict: &MecabDictionary, word: &str) -> Option<Vec<usize>> {
    let n = word.len();
    if n < 2 {
        return None;
    }
    let bytes = word.as_bytes();
    let mut best_parts: Vec<Option<u32>> = vec![None; n + 1];
    let mut prev_at: Vec<usize> = vec![0; n + 1];
    best_parts[0] = Some(0);
    for i in 0..n {
        let Some(cur) = best_parts[i] else {
            continue;
        };
        if !word.is_char_boundary(i) {
            continue;
        }
        dict.trie.prefixes(&bytes[i..], |len, _val| {
            if i == 0 && len == n {
                return;
            }
            let end = i + len;
            let cand = cur + 1;
            if best_parts[end].is_none_or(|old| cand > old) {
                best_parts[end] = Some(cand);
                prev_at[end] = i;
            }
        });
    }
    let count = best_parts[n]?;
    if count < 2 {
        return None;
    }
    let mut bounds = Vec::with_capacity(count as usize + 1);
    let mut at = n;
    while at > 0 {
        bounds.push(at);
        at = prev_at[at];
    }
    bounds.push(0);
    bounds.reverse();
    Some(bounds)
}

impl LatticeTokenizer {
    /// Builds a multi-granularity [`Alternatives<Tokens>`] lattice aligned to
    /// `tokens` (produced over `view`), exposing:
    /// - [`SplitMode::Best`]: the exact 1-best Viterbi morphemes of `tokens`,
    /// - [`SplitMode::A`]: finest dictionary sub-morphemes,
    /// - [`SplitMode::B`]: 1-best intermediate morphemes, and
    /// - [`SplitMode::C`]: maximal dictionary compounds across abutting morphemes.
    pub fn build_alternatives(&self, view: &View, tokens: &Tokens) -> Result<Alternatives<Tokens>, Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: LatticeGraphAnnotator::ID,
                detail: "tokens belong to another view".into(),
            });
        }
        let n_tok = tokens.spans.len();
        if n_tok == 0 {
            return Ok(Alternatives::new(view.id()));
        }
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let mut unit_spans: Vec<Span> = Vec::with_capacity(n_tok);
        let mut best_edges: Vec<(u32, u32, Span)> = Vec::with_capacity(n_tok);

        for i in 0..n_tok {
            let span = Span::new(starts[i], ends[i]);
            let slice = text.get(span.range()).ok_or_else(|| Error::Annotator {
                annotator: LatticeGraphAnnotator::ID,
                detail: format!("token {i} span {span:?} is outside view text"),
            })?;
            let p_start = unit_spans.len() as u32;
            if let Some(bounds) = finest_lexicon_split(&self.dictionary, slice) {
                for w in bounds.windows(2) {
                    unit_spans.push(Span::new(span.start + w[0] as u32, span.start + w[1] as u32));
                }
                let p_len = (bounds.len() - 1) as u32;
                best_edges.push((p_start, p_len, span));
            } else {
                unit_spans.push(span);
                best_edges.push((p_start, 1, span));
            }
        }

        let num_pos = unit_spans.len();
        let mut edge_map: std::collections::BTreeMap<(u32, u32), (Span, ModeSet)> = std::collections::BTreeMap::new();
        for (p, &u_span) in unit_spans.iter().enumerate() {
            edge_map.insert((p as u32, 1), (u_span, ModeSet::A));
        }
        for &(p, len, span) in &best_edges {
            edge_map
                .entry((p, len))
                .and_modify(|(_, m)| *m |= SplitMode::Best | SplitMode::B)
                .or_insert((span, SplitMode::Best | SplitMode::B));
        }

        // Discover super-morpheme lexicon entries spanning >= 2 abutting unit positions.
        let bytes = text.as_bytes();
        let mut p = 0usize;
        while p < num_pos {
            let mut run_end = p + 1;
            while run_end < num_pos && unit_spans[run_end - 1].end == unit_spans[run_end].start {
                run_end += 1;
            }
            for u in p..run_end {
                let start_byte = unit_spans[u].start as usize;
                let max_byte = unit_spans[run_end - 1].end as usize;
                self.dictionary.trie.prefixes(&bytes[start_byte..max_byte], |len, _val| {
                    let target_end = (start_byte + len) as u32;
                    if let Ok(rel) = unit_spans[u + 1..run_end].binary_search_by_key(&target_end, |s| s.end) {
                        let v = u + 1 + rel + 1;
                        let pos_len = (v - u) as u32;
                        if pos_len >= 2 {
                            edge_map
                                .entry((u as u32, pos_len))
                                .or_insert((Span::new(start_byte as u32, target_end), ModeSet::EMPTY));
                        }
                    }
                });
            }
            p = run_end;
        }

        // Greedily assign SplitMode::C along longest available spans from 0..num_pos.
        let mut cur = 0u32;
        while (cur as usize) < num_pos {
            let best_len =
                edge_map.range((cur, 1)..=(cur, num_pos as u32)).map(|(&(_, len), _)| len).max().unwrap_or(1);
            if let Some((_, mode)) = edge_map.get_mut(&(cur, best_len)) {
                *mode |= SplitMode::C;
            }
            cur += best_len;
        }

        let mut alt = Alternatives::new(view.id());
        for ((pos, len), (span, mode)) in edge_map {
            alt.push(span, pos, len, mode);
        }
        Ok(alt)
    }
}

/// Pipeline step running [`LatticeTokenizer`] and emitting 1-best [`Tokens`],
/// [`PosTags`], [`Alternatives<Tokens>`], and [`TokenGraph`] in one pass.
#[derive(Debug, Clone)]
pub struct LatticeGraphAnnotator {
    tokenizer: LatticeTokenizer,
}

impl LatticeGraphAnnotator {
    pub fn new(tokenizer: LatticeTokenizer) -> Self {
        LatticeGraphAnnotator { tokenizer }
    }

    pub fn tokenizer(&self) -> &LatticeTokenizer {
        &self.tokenizer
    }
}

impl Annotator for LatticeGraphAnnotator {
    const ID: &'static str = "turbo-tokenize:lattice-graph";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens, PosTags, Alternatives<Tokens>, TokenGraph);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let mut tags = Vec::new();
        let tokens = self
            .tokenizer
            .segment(Self::ID, view, sentences, |entry| tags.push(self.tokenizer.dictionary.tag(entry)))?;
        let pos = self.tokenizer.pos_tags(tags);
        let alt = self.tokenizer.build_alternatives(view, &tokens)?;
        let graph = TokenGraph::from_alternatives(view.text(), &alt)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((tokens, pos, alt, graph))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.tokenizer.model_ref())
    }

    fn config_hash(&self) -> u64 {
        let [a, b] = self.tokenizer.whitespace_hash();
        fnv1a(&[a, b, 0xA1])
    }
}
