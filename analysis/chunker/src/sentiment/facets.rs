//! Per-chunk sentiment, aspect, and incivility facets ([`ChunkFacets`], [`ChunkFacetBuilder`])
//! and Fournier (2013) Boundary Similarity ([`boundary_similarity`], [`chunk_boundary_gaps`]).

use std::ops::Range;

use turbo_analysis::{Anchor, Annotator, Error, Ix, IxRange, Key, KeyId, Layer, LayerName, Refs, Span, View};
use turbo_analysis_layers::{
    Aspects, Categories, Chunks, Incivility, IncivilityLabel, IncivilityScores, IncivilityTarget, Polarity, Sentences,
    Strs, Tokens,
};

use super::{MICRO, category_row_polarity_micro, label_bit, resolve_aspect_label, resolve_aspect_sentence};

/// Per-chunk sentiment, aspect, and incivility facets aligned 1-to-1 with [`Chunks`].
///
/// Follows the sparse-row layout of [`TermStats`](crate::terms::TermStats): each chunk `c`
/// owns `aspect_row[c]..aspect_row[c + 1]` in `aspect`, `aspect_label`, and `aspect_polarity`,
/// and `incivility_row[c]..incivility_row[c + 1]` in `incivility_label`.
#[derive(Clone, Debug)]
pub struct ChunkFacets {
    pub polarity: Vec<Polarity>,
    pub polarity_score: Vec<f64>,
    pub aspect_row: Vec<u32>,
    pub aspect: Vec<Ix<Aspects>>,
    pub aspect_label: Strs,
    pub aspect_polarity: Vec<Polarity>,
    pub incivility_row: Vec<u32>,
    pub incivility_label: Vec<IncivilityLabel>,
    pub incivility_max: Vec<IncivilityScores>,
}

impl ChunkFacets {
    fn aspect_range(&self, c: Ix<Chunks>) -> Range<usize> {
        let i = c.get();
        self.aspect_row[i] as usize..self.aspect_row[i + 1] as usize
    }

    /// Aggregated sentiment polarity for chunk `c`.
    #[inline]
    pub fn chunk_polarity(&self, c: Ix<Chunks>) -> Polarity {
        self.polarity[c.get()]
    }

    /// Signed polarity score in `[-1.0, 1.0]` for chunk `c`.
    #[inline]
    pub fn chunk_polarity_score(&self, c: Ix<Chunks>) -> f64 {
        self.polarity_score[c.get()]
    }

    /// Aspect indices from [`Aspects`] falling within chunk `c`.
    #[inline]
    pub fn aspect_ixs(&self, c: Ix<Chunks>) -> &[Ix<Aspects>] {
        &self.aspect[self.aspect_range(c)]
    }

    /// Aspect polarities parallel to [`Self::aspect_ixs`] for chunk `c`.
    #[inline]
    pub fn aspect_polarities(&self, c: Ix<Chunks>) -> &[Polarity] {
        &self.aspect_polarity[self.aspect_range(c)]
    }

    /// Iterates `(Ix<Aspects>, label, Polarity)` for all aspects falling within chunk `c`.
    pub fn aspects_of(&self, c: Ix<Chunks>) -> impl ExactSizeIterator<Item = (Ix<Aspects>, &str, Polarity)> + '_ {
        let r = self.aspect_range(c);
        r.map(move |k| (self.aspect[k], self.aspect_label.get(k), self.aspect_polarity[k]))
    }

    /// Flagged incivility labels present in chunk `c`, in canonical [`IncivilityLabel`] order.
    #[inline]
    pub fn incivility_labels(&self, c: Ix<Chunks>) -> &[IncivilityLabel] {
        let i = c.get();
        let r = self.incivility_row[i] as usize..self.incivility_row[i + 1] as usize;
        &self.incivility_label[r]
    }

    /// Element-wise maximum incivility dimension scores across sentences in chunk `c`.
    #[inline]
    pub fn incivility_scores(&self, c: Ix<Chunks>) -> IncivilityScores {
        self.incivility_max[c.get()]
    }
}

impl Layer for ChunkFacets {
    const ID: &'static str = "turbo:chunk-facets";
    const OF: Option<&'static LayerName> = Some(Chunks::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<Chunks>()
    }

    fn len(&self) -> usize {
        self.polarity.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let max_asp = self.aspect.iter().map(|ix| ix.get() + 1).max().unwrap_or(0);
        if max_asp > 0 { vec![(Key::<Aspects>::DEFAULT.id(), max_asp)] } else { Vec::new() }
    }

    fn check(&self) -> Result<(), String> {
        let n = self.polarity.len();
        if self.polarity_score.len() != n || self.incivility_max.len() != n {
            return Err("a polarity score and incivility scores for every chunk".into());
        }
        if self.aspect_row.len() != n + 1
            || self.aspect_row.first().copied() != Some(0)
            || self.aspect_row.last().copied().unwrap_or(0) as usize != self.aspect.len()
            || !self.aspect_row.windows(2).all(|w| w[0] <= w[1])
        {
            return Err("aspect_row must start at 0, be non-decreasing, and end at aspect.len()".into());
        }
        if self.aspect_label.len() != self.aspect.len() || self.aspect_polarity.len() != self.aspect.len() {
            return Err("an aspect label and polarity for every aspect entry".into());
        }
        if self.incivility_row.len() != n + 1
            || self.incivility_row.first().copied() != Some(0)
            || self.incivility_row.last().copied().unwrap_or(0) as usize != self.incivility_label.len()
            || !self.incivility_row.windows(2).all(|w| w[0] <= w[1])
        {
            return Err("incivility_row must start at 0, be non-decreasing, and end at incivility_label.len()".into());
        }
        for i in 0..n {
            let r = self.incivility_row[i] as usize..self.incivility_row[i + 1] as usize;
            if !self.incivility_label[r].windows(2).all(|w| w[0] < w[1]) {
                return Err(format!("chunk {i}: incivility labels must be in strictly ascending order"));
            }
        }
        if self.polarity_score.iter().any(|x| !x.is_finite() || !(-1.0..=1.0).contains(x)) {
            return Err("polarity_score must be finite and in -1.0..=1.0".into());
        }
        if self.incivility_max.iter().any(|s| !s.is_valid()) {
            return Err("incivility_max scores must be finite and in 0.0..=1.0".into());
        }
        Ok(())
    }
}

#[derive(Clone)]
struct ResolvedAspect {
    a_ix: Ix<Aspects>,
    label: String,
    polarity: Polarity,
    compound: f64,
    span: Option<Span>,
}

/// The byte envelope of a non-empty token range, when the tokens
/// are known and the range is in bounds.
fn token_span(toks: Option<&Tokens>, r: IxRange<Tokens>) -> Option<Span> {
    let toks = toks?;
    if r.start.get() >= r.end.get() || r.end.get() > toks.spans.len() {
        return None;
    }
    Some(Span::new(toks.spans.get(r.start).start, toks.spans.get(Ix::new(r.end.get() as u32 - 1)).end))
}

#[inline]
fn merge_incivility(dst: &mut (IncivilityScores, u8), src: (IncivilityScores, u8)) {
    dst.0.insult = dst.0.insult.max(src.0.insult);
    dst.0.profanity = dst.0.profanity.max(src.0.profanity);
    dst.0.threat = dst.0.threat.max(src.0.threat);
    dst.0.identity_attack = dst.0.identity_attack.max(src.0.identity_attack);
    dst.0.intolerance = dst.0.intolerance.max(src.0.intolerance);
    dst.0.severe_toxicity = dst.0.severe_toxicity.max(src.0.severe_toxicity);
    dst.1 |= src.1;
}

fn item_incivility(inc: &Incivility, item_ix: Ix<Incivility>) -> (IncivilityScores, u8) {
    let sc = inc.scores(item_ix);
    let mut mask = 0u8;
    for (lbl, _) in inc.evidence_of(item_ix) {
        mask |= label_bit(lbl);
    }
    for (lbl, val) in sc.iter() {
        if val >= 0.5 {
            mask |= label_bit(lbl);
        }
    }
    (sc, mask)
}

/// Annotator that projects sentence-level sentiment categories, [`Aspects`], and
/// [`Incivility`] onto [`Chunks`] as [`ChunkFacets`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkFacetBuilder;

impl Annotator for ChunkFacetBuilder {
    const ID: &'static str = "turbo:chunk-facet-builder";
    type Reads = (
        Chunks,
        Sentences,
        Option<View>,
        Option<Tokens>,
        Option<Categories<Sentences>>,
        Option<Aspects>,
        Option<Incivility>,
    );
    type Writes = (ChunkFacets,);

    fn annotate(
        &self,
        (chunks, sents, view, toks, cats, aspects, incivility): Refs<'_, Self>,
    ) -> Result<(ChunkFacets,), Error> {
        let view_id = sents.spans.view();
        if chunks.spans.view() != view_id
            || view.is_some_and(|v| v.id() != view_id)
            || toks.is_some_and(|t| t.spans.view() != view_id)
        {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        let n_sents = sents.spans.len();
        if let Some(c) = cats
            && c.len() != n_sents
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} sentence categories for {} sentences", c.len(), n_sents),
            });
        }

        let n_chunks = chunks.len();
        let mut polarity = Vec::with_capacity(n_chunks);
        let mut polarity_score = Vec::with_capacity(n_chunks);
        let mut aspect_row = Vec::with_capacity(n_chunks + 1);
        let mut aspect_ixs = Vec::new();
        let mut aspect_label = Strs::default();
        let mut aspect_polarity = Vec::new();
        let mut incivility_row = Vec::with_capacity(n_chunks + 1);
        let mut incivility_label = Vec::new();
        let mut incivility_max = Vec::with_capacity(n_chunks);

        aspect_row.push(0);
        incivility_row.push(0);

        // Pre-resolve aspects by sentence index once: O(n_aspects).
        let mut sent_aspects: Vec<Vec<ResolvedAspect>> = Vec::new();
        let mut aspects_in_sent_order = true;
        if let Some(asp) = aspects
            && !asp.is_empty()
        {
            sent_aspects.resize_with(n_sents, Vec::new);
            let mut last_sent = 0usize;
            for (a_ix, item) in asp.iter() {
                if let Some(s_idx) = resolve_aspect_sentence(a_ix, asp, sents, toks)
                    && s_idx < n_sents
                {
                    if s_idx < last_sent {
                        aspects_in_sent_order = false;
                    }
                    last_sent = s_idx;
                    sent_aspects[s_idx].push(ResolvedAspect {
                        a_ix,
                        label: resolve_aspect_label(a_ix, asp, toks, view),
                        polarity: item.polarity.label(),
                        compound: item.polarity.compound().clamp(-1.0, 1.0),
                        span: item.target.tokens().and_then(|r| token_span(toks, r)),
                    });
                }
            }
        }

        // Pre-compute sentence category polarity once: O(n_sents).
        let sent_cat_pol: Vec<Option<f64>> = match cats {
            Some(c) => (0..n_sents)
                .map(|s_idx| category_row_polarity_micro(c, s_idx).map(|p| (p as f64) / (MICRO as f64)))
                .collect(),
            None => Vec::new(),
        };

        // Pre-aggregate incivility by sentence and by chunk once: O(n_incivility).
        let mut sent_incivility: Vec<Vec<(IncivilityScores, u8, Option<Span>)>> = Vec::new();
        let mut chunk_incivility = Vec::new();
        if let Some(inc) = incivility
            && !inc.is_empty()
        {
            sent_incivility.resize_with(n_sents, Vec::new);
            chunk_incivility.resize(n_chunks, (IncivilityScores::default(), 0u8));
            for i in 0..inc.len() {
                let item_ix = Ix::<Incivility>::new(i as u32);
                let entry = item_incivility(inc, item_ix);
                match inc.target[i] {
                    IncivilityTarget::Sentence(s_ix) if s_ix.get() < n_sents => {
                        let mut evidence: Option<Span> = None;
                        for (_, r) in inc.evidence_of(item_ix) {
                            if let Some(s) = token_span(toks, r) {
                                evidence = Some(match evidence {
                                    None => s,
                                    Some(e) => Span::new(e.start.min(s.start), e.end.max(s.end)),
                                });
                            }
                        }
                        sent_incivility[s_ix.get()].push((entry.0, entry.1, evidence));
                    }
                    IncivilityTarget::Chunk(ch_ix) if ch_ix.get() < n_chunks => {
                        merge_incivility(&mut chunk_incivility[ch_ix.get()], entry);
                    }
                    _ => {}
                }
            }
        }

        let mut chunk_resolved: Vec<&ResolvedAspect> = Vec::new();
        for (ch_idx, &sr) in chunks.sentences.iter().enumerate() {
            let s_end = sr.end.get().min(n_sents);
            let s_start = sr.start.get().min(s_end);

            // 1. Collect aspects whose sentence falls in [s_start, s_end).
            let frags: Vec<Span> = chunks.spans.fragments(Ix::new(ch_idx as u32)).collect();
            let in_chunk = |s: Option<Span>| s.is_none_or(|sp| frags.iter().any(|f| f.overlaps(sp)));
            let chunk_aspect_start = aspect_ixs.len();
            let mut aspect_compound_sum = 0.0f64;
            let mut aspect_compound_n = 0usize;
            if !sent_aspects.is_empty() {
                chunk_resolved.clear();
                for items in &sent_aspects[s_start..s_end] {
                    chunk_resolved.extend(items.iter());
                }
                if !aspects_in_sent_order && chunk_resolved.len() > 1 {
                    chunk_resolved.sort_by_key(|r| r.a_ix.get());
                }
                for resolved in chunk_resolved.iter().filter(|r| in_chunk(r.span)) {
                    aspect_ixs.push(resolved.a_ix);
                    aspect_label.push(&resolved.label);
                    aspect_polarity.push(resolved.polarity);
                    aspect_compound_sum += resolved.compound;
                    aspect_compound_n += 1;
                }
            }
            aspect_row.push(aspect_ixs.len() as u32);

            // 2. Compute chunk polarity and signed score.
            let mut signed_sum = 0.0f64;
            let mut signed_n = 0usize;

            if !sent_cat_pol.is_empty() {
                for &p in &sent_cat_pol[s_start..s_end] {
                    if let Some(val) = p {
                        signed_sum += val;
                        signed_n += 1;
                    }
                }
            }

            if signed_n == 0 && aspect_compound_n > 0 {
                signed_sum = aspect_compound_sum;
                signed_n = aspect_compound_n;
            } else if signed_n == 0 && aspect_ixs.len() > chunk_aspect_start {
                for &pol in &aspect_polarity[chunk_aspect_start..] {
                    match pol {
                        Polarity::Positive => signed_sum += 1.0,
                        Polarity::Negative => signed_sum -= 1.0,
                        Polarity::Neutral => {}
                    }
                    signed_n += 1;
                }
            }

            let mean_score = if signed_n > 0 { (signed_sum / (signed_n as f64)).clamp(-1.0, 1.0) } else { 0.0 };
            let chunk_pol = if mean_score > 0.05 {
                Polarity::Positive
            } else if mean_score < -0.05 {
                Polarity::Negative
            } else {
                Polarity::Neutral
            };
            polarity.push(chunk_pol);
            polarity_score.push(mean_score);

            // 3. Aggregate incivility scores and distinct flagged labels over sentences in chunk `c`
            //    or direct `IncivilityTarget::Chunk(c_ix)` entries.
            // Both tables are filled together when incivility is given, the
            // sentence one with a row per sentence, which can be none.
            let mut agg = (IncivilityScores::default(), 0u8);
            if !chunk_incivility.is_empty() {
                agg = chunk_incivility[ch_idx];
                for items in &sent_incivility[s_start..s_end] {
                    for &(scores, mask, _) in items.iter().filter(|it| in_chunk(it.2)) {
                        merge_incivility(&mut agg, (scores, mask));
                    }
                }
            }
            let (max_scores, mask) = agg;
            for lbl in IncivilityLabel::ALL {
                if (mask & label_bit(lbl)) != 0 {
                    incivility_label.push(lbl);
                }
            }
            incivility_row.push(incivility_label.len() as u32);
            incivility_max.push(max_scores);
        }

        let facets = ChunkFacets {
            polarity,
            polarity_score,
            aspect_row,
            aspect: aspect_ixs,
            aspect_label,
            aspect_polarity,
            incivility_row,
            incivility_label,
            incivility_max,
        };
        Ok((facets,))
    }
}

/// Fournier (2013) Boundary Similarity ($B$) over internal sentence-gap boundary positions.
///
/// Given reference boundary gaps `gold_gaps` and predicted boundary gaps `pred_gaps`
/// (each a gap index `g` after sentence `g`, as returned by [`chunk_boundary_gaps`]),
/// pairs exact boundary matches with weight `1.0` and near-miss transpositions within
/// `near_distance` gaps (`>= 1`) with weight `0.5`, divided by the total number of matched
/// and unmatched boundary edits:
///
/// $$B = \frac{|M_{\text{exact}}| + 0.5 \cdot |M_{\text{near}}|}{|M_{\text{exact}}| + |M_{\text{near}}| + |U_{\text{gold}}| + |U_{\text{pred}}|}$$
///
/// Returns `1.0` when both `gold_gaps` and `pred_gaps` are empty.
pub fn boundary_similarity(gold_gaps: &[u32], pred_gaps: &[u32], near_distance: u32) -> f64 {
    let mut g: Vec<u32> = gold_gaps.to_vec();
    let mut p: Vec<u32> = pred_gaps.to_vec();
    g.sort_unstable();
    g.dedup();
    p.sort_unstable();
    p.dedup();

    if g.is_empty() && p.is_empty() {
        return 1.0;
    }

    let mut g_used = vec![false; g.len()];
    let mut p_used = vec![false; p.len()];
    let mut exact = 0usize;

    for (gi, &gv) in g.iter().enumerate() {
        if let Ok(pi) = p.binary_search(&gv) {
            g_used[gi] = true;
            p_used[pi] = true;
            exact += 1;
        }
    }

    let mut near = 0usize;
    if near_distance > 0 {
        for d in 1..=near_distance {
            for (gi, &gv) in g.iter().enumerate() {
                if g_used[gi] {
                    continue;
                }
                for (pi, &pv) in p.iter().enumerate() {
                    if p_used[pi] {
                        continue;
                    }
                    if gv.abs_diff(pv) == d {
                        g_used[gi] = true;
                        p_used[pi] = true;
                        near += 1;
                        break;
                    }
                }
            }
        }
    }

    let u_gold = g_used.iter().filter(|&&u| !u).count();
    let u_pred = p_used.iter().filter(|&&u| !u).count();
    let denom = exact + near + u_gold + u_pred;
    if denom == 0 { 1.0 } else { ((exact as f64) + 0.5 * (near as f64)) / (denom as f64) }
}

/// Extracts internal sentence-gap indices (`0..sents - 1`) where level-0 `chunks` place a boundary.
pub fn chunk_boundary_gaps(chunks: &Chunks) -> Vec<u32> {
    let level0: Vec<usize> = (0..chunks.len()).filter(|&i| chunks.level[i] == 0).collect();
    if level0.len() <= 1 {
        return Vec::new();
    }
    let mut gaps = Vec::with_capacity(level0.len() - 1);
    for &i in &level0[..level0.len() - 1] {
        let end = chunks.sentences[i].end.get() as u32;
        if end > 0 {
            gaps.push(end - 1);
        }
    }
    gaps
}
