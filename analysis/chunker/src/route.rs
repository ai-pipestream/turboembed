//! Query-time routing over the chunk, section and document levels
//! of [`Levels`]: one ranked list over units of every level, fusing
//! BM25 over `TermStats<Levels>` with dense cosine scores over
//! `Vectors<Levels>` when the caller has them.
//!
//! Routing is by query profile. A query's specificity is how rare
//! its terms are among the document's chunks (the mean, over the
//! query terms the document holds, of one minus the share of
//! chunks holding the term): rare-term queries are factoid and
//! weight chunks, long queries of common terms are summary queries
//! and weight the document, and the rest are topical and weight
//! sections. The weights per profile and the thresholds are fields
//! of [`LevelRouter`], with the defaults below.
//!
//! Fusion is per unit first (lexical score normalized by the best
//! unit's, dense cosine mapped to `[0, 1]`), then parents are
//! promoted when their chunk children co-activate: when at least
//! `min_siblings` of a section's (or the document's) chunk children
//! score at least `active_share` of the best sibling and make up at
//! least `sibling_share` of its children, the parent's signal rises
//! to the active children's mean times `coactivation_boost`. A
//! single hot chunk among cold siblings is dispersed, so it
//! promotes nothing. The level weight of the query's profile is
//! applied last. Ties break by score, then span start, span end,
//! then unit index, so a ranking is a function of the inputs alone.
//!
//! A hit's span is the unit's exact span in the `Levels` view; map
//! it with `Document::to_source` or `Document::map_span` for source
//! or cross-view coordinates. Chunk and section hits carry their
//! parent context: the parent unit and span plus a sibling window
//! (for a chunk) or its best child chunks (for a section), so a
//! caller can expand a winning chunk into its section without a
//! second lookup.

use turbo_analysis::{Ix, Layer, Span};
use turbo_analysis_layers::{Level, Levels, Vectors};

use crate::terms::{Bm25, Collection, Query, TermStats, Vocabulary};

/// How much each level's fused signal counts for one query profile:
/// plain multipliers, applied after co-activation promotion.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LevelWeights {
    pub chunk: f64,
    pub section: f64,
    pub document: f64,
}

impl LevelWeights {
    fn of(&self, level: Level) -> f64 {
        match level {
            Level::Chunk => self.chunk,
            Level::Section => self.section,
            Level::Document => self.document,
        }
    }

    fn check(&self, what: &str) -> Result<(), String> {
        if [self.chunk, self.section, self.document].iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(format!("{what} level weights must be finite and non-negative"));
        }
        Ok(())
    }
}

/// Which level a query is routed toward.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QueryKind {
    /// Specific, rare-term: chunks answer it.
    Factoid,
    /// About a topic: sections answer it.
    Topical,
    /// Broad and common-term: the document answers it.
    Summary,
}

/// A query measured against one document's vocabulary and
/// collection: how it routes, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryProfile {
    pub kind: QueryKind,
    /// The query terms as given, repeats included.
    pub terms: usize,
    /// The query terms the document holds (the BM25 query's terms).
    pub present: usize,
    /// Mean over the present terms of one minus the share of the
    /// document's chunk units holding the term, in `[0, 1]`; 0 when
    /// no term is present or the document has no chunks.
    pub specificity: f64,
    /// The level weights this profile routes with.
    pub weights: LevelWeights,
}

/// The parent of a hit, for expanding it into its context.
#[derive(Clone, Debug, PartialEq)]
pub struct ParentContext {
    /// The hit's parent unit in `Levels`.
    pub parent: Ix<Levels>,
    /// The parent's exact span in the `Levels` view.
    pub parent_span: Span,
    /// Whether that parent was co-activation promoted.
    pub promoted: bool,
    /// For a chunk hit, the sibling chunks around it under the same
    /// parent (`context_siblings` on each side, itself excluded);
    /// for a section hit, its best child chunks.
    pub window: Vec<Ix<Levels>>,
}

/// One ranked unit of a routed query.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub unit: Ix<Levels>,
    pub level: Level,
    /// The unit's exact span in the `Levels` view.
    pub span: Span,
    /// The final score: fused signal after promotion, times the
    /// profile's level weight.
    pub score: f64,
    /// The normalized lexical (BM25) signal in `[0, 1]`.
    pub lexical: f64,
    /// The dense signal in `[0, 1]`, when vectors were given and
    /// this unit has a usable row.
    pub dense: Option<f64>,
    /// For a section or the document, how many of its chunk
    /// children co-activated; 0 for a chunk.
    pub active_children: usize,
    pub context: Option<ParentContext>,
}

/// A routed query: its profile and its hits, best first.
#[derive(Clone, Debug, PartialEq)]
pub struct Routed {
    pub profile: QueryProfile,
    pub hits: Vec<Hit>,
}

/// One query to route: its terms (normalized as the document's
/// terms were, the `terms::Query` contract) and, for dense fusion,
/// the document's `Vectors<Levels>` paired with the query vector.
#[derive(Clone)]
pub struct RouteQuery<'a> {
    pub terms: &'a [&'a str],
    pub vectors: Option<(&'a Vectors<Levels>, &'a [f32])>,
}

/// Query-time level routing and parent-context fusion over
/// [`Levels`]. The defaults route factoid queries mostly to chunks,
/// topical ones mostly to sections, and summary ones mostly to the
/// document, fuse lexical and dense signals equally, and promote a
/// parent when at least two of its chunk children, and at least
/// half of them, reach half the best sibling's signal.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelRouter {
    pub bm25: Bm25,
    pub lexical_weight: f64,
    pub dense_weight: f64,
    pub factoid: LevelWeights,
    pub topical: LevelWeights,
    pub summary: LevelWeights,
    /// Specificity at or above which a query is factoid.
    pub factoid_specificity: f64,
    /// Specificity at or below which a long query is a summary one.
    pub summary_specificity: f64,
    /// Present query terms a summary query needs at least.
    pub summary_min_terms: usize,
    /// Co-activated chunk children a promotion needs at least.
    pub min_siblings: usize,
    /// Fraction of a parent's chunk children that must co-activate.
    pub sibling_share: f64,
    /// Fraction of the best sibling's signal a child needs to
    /// count as co-activated.
    pub active_share: f64,
    /// The promoted parent's signal: the active children's mean
    /// times this, clamped to 1, when that beats its own signal.
    pub coactivation_boost: f64,
    /// Sibling chunks on each side of a chunk hit in its context
    /// window; a section hit's window holds twice this plus one.
    pub context_siblings: usize,
}

impl Default for LevelRouter {
    fn default() -> LevelRouter {
        LevelRouter {
            bm25: Bm25::LUCENE,
            lexical_weight: 1.0,
            dense_weight: 1.0,
            factoid: LevelWeights { chunk: 1.0, section: 0.35, document: 0.1 },
            topical: LevelWeights { chunk: 0.6, section: 1.0, document: 0.3 },
            summary: LevelWeights { chunk: 0.25, section: 0.6, document: 1.0 },
            factoid_specificity: 0.7,
            summary_specificity: 0.5,
            summary_min_terms: 4,
            min_siblings: 2,
            sibling_share: 0.5,
            active_share: 0.5,
            coactivation_boost: 1.0,
            context_siblings: 1,
        }
    }
}

fn finite_non_negative(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && *v >= 0.0)
}

impl LevelRouter {
    /// How `query` routes against one document's levels and term
    /// counts, without scoring any unit for ranking.
    pub fn profile(
        &self,
        levels: &Levels,
        stats: &TermStats<Levels>,
        collection: &impl Collection,
        vocabulary: &Vocabulary,
        query: &[&str],
    ) -> QueryProfile {
        let q = Query::new(self.bm25, collection, vocabulary, query);
        self.profile_of(&q, query.len(), Self::specificity(levels, stats, &q))
    }

    /// Chunk-level rarity of the query's present terms: parent
    /// units hold every term their children hold, so rarity over
    /// all units would compress toward the middle; over chunks, a
    /// term in one chunk of many is specific and a term in every
    /// chunk is not at all.
    fn specificity(levels: &Levels, stats: &TermStats<Levels>, q: &Query) -> f64 {
        let chunks: Vec<Ix<Levels>> = levels.of(Level::Chunk).filter(|&i| stats.length[i.get()] > 0).collect();
        if chunks.is_empty() || q.terms.is_empty() {
            return 0.0;
        }
        let held = |t: Ix<Vocabulary>| chunks.iter().filter(|&&i| stats.tf(i, t) > 0).count() as f64;
        let mean =
            q.terms.iter().map(|&(t, _)| 1.0 - held(t) / chunks.len() as f64).sum::<f64>() / q.terms.len() as f64;
        mean.clamp(0.0, 1.0)
    }

    fn profile_of(&self, q: &Query, terms: usize, specificity: f64) -> QueryProfile {
        let present = q.terms.len();
        let kind = if present > 0 && specificity >= self.factoid_specificity {
            QueryKind::Factoid
        } else if present >= self.summary_min_terms.max(1) && specificity <= self.summary_specificity {
            QueryKind::Summary
        } else {
            QueryKind::Topical
        };
        let weights = match kind {
            QueryKind::Factoid => self.factoid,
            QueryKind::Topical => self.topical,
            QueryKind::Summary => self.summary,
        };
        QueryProfile { kind, terms, present, specificity, weights }
    }

    fn check_config(&self) -> Result<(), String> {
        self.factoid.check("factoid")?;
        self.topical.check("topical")?;
        self.summary.check("summary")?;
        if !finite_non_negative(&[
            self.lexical_weight,
            self.dense_weight,
            self.factoid_specificity,
            self.summary_specificity,
            self.sibling_share,
            self.active_share,
            self.coactivation_boost,
        ]) || self.lexical_weight + self.dense_weight <= 0.0
        {
            return Err(
                "router weights, thresholds and boost must be finite and non-negative, with a signal weight above zero"
                    .into(),
            );
        }
        if self.factoid_specificity > 1.0 || self.summary_specificity > 1.0 || self.active_share > 1.0 {
            return Err("router specificity and active-share thresholds must be at most 1".into());
        }
        if !(0.0 < self.sibling_share && self.sibling_share <= 1.0) {
            return Err("router sibling share must be in (0, 1]".into());
        }
        Ok(())
    }

    /// Routes one query over one document's levels. Without the
    /// query's `vectors` the ranking is lexical alone. Only units
    /// with a positive fused signal are hits, at most `top_k` of
    /// them, best first with deterministic tie-breaking (score,
    /// then span start, span end, then unit index).
    pub fn route(
        &self,
        levels: &Levels,
        stats: &TermStats<Levels>,
        collection: &impl Collection,
        vocabulary: &Vocabulary,
        query: &RouteQuery<'_>,
        top_k: usize,
    ) -> Result<Routed, String> {
        let (query_terms, vectors) = (query.terms, query.vectors);
        self.check_config()?;
        if top_k == 0 {
            return Err("top_k must be at least 1".into());
        }
        levels.check()?;
        stats.check()?;
        if stats.len() != levels.len() {
            return Err(format!("term stats for {} units over {} level units", stats.len(), levels.len()));
        }
        let query_norm = match vectors {
            Some((rows, q)) => {
                rows.check()?;
                if rows.len() != levels.len() {
                    return Err(format!("vectors for {} units over {} level units", rows.len(), levels.len()));
                }
                if q.len() != rows.dim as usize {
                    return Err(format!("a query vector of {} dimensions for rows of {}", q.len(), rows.dim));
                }
                if q.iter().any(|v| !v.is_finite()) {
                    return Err("query vector must be finite".into());
                }
                let norm = q.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
                if norm <= 0.0 {
                    return Err("query vector has no direction".into());
                }
                Some(norm)
            }
            None => None,
        };
        let q = Query::new(self.bm25, collection, vocabulary, query_terms);
        let profile = self.profile_of(&q, query_terms.len(), Self::specificity(levels, stats, &q));
        if profile.present == 0 && vectors.is_none() {
            return Ok(Routed { profile, hits: Vec::new() });
        }

        let n = levels.len();
        let raw: Vec<f64> = (0..n as u32).map(|i| q.score(stats, Ix::new(i))).collect();
        let max_raw = raw.iter().copied().fold(0.0f64, f64::max);
        let lexical: Vec<f64> =
            raw.iter().map(|&s| if max_raw > 0.0 { (s / max_raw).clamp(0.0, 1.0) } else { 0.0 }).collect();
        let dense: Vec<Option<f64>> = match (vectors, query_norm) {
            (Some((rows, qv)), Some(qnorm)) => (0..n as u32)
                .map(|i| {
                    let ix = Ix::new(i);
                    if rows.is_empty(ix) {
                        return None;
                    }
                    let row = rows.row(ix);
                    let norm = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
                    if norm <= 0.0 || !norm.is_finite() {
                        return None;
                    }
                    let cos =
                        row.iter().zip(qv).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() / (norm * qnorm);
                    cos.is_finite().then(|| (cos.clamp(-1.0, 1.0) + 1.0) / 2.0)
                })
                .collect(),
            _ => vec![None; n],
        };
        let mut signal: Vec<f64> = (0..n)
            .map(|i| match dense[i] {
                Some(d) if vectors.is_some() => {
                    let w = self.lexical_weight + self.dense_weight;
                    (self.lexical_weight * lexical[i] + self.dense_weight * d) / w
                }
                _ => lexical[i],
            })
            .collect();

        // Chunk children per parent, in unit order.
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            if levels.level(Ix::new(i as u32)) == Level::Chunk
                && let Some(p) = levels.parent[i]
            {
                children[p.get()].push(i);
            }
        }
        // Co-activation: a parent whose chunk children fire together
        // (enough of them, and enough of the whole brood, at or
        // above the active share of the best sibling) is promoted.
        let mut active_children = vec![0usize; n];
        let mut promoted = vec![false; n];
        for (p, kids) in children.iter().enumerate() {
            if kids.is_empty() || levels.level(Ix::new(p as u32)) == Level::Chunk {
                continue;
            }
            let best = kids.iter().map(|&k| signal[k]).fold(0.0f64, f64::max);
            if best <= 0.0 {
                continue;
            }
            let active: Vec<usize> = kids.iter().copied().filter(|&k| signal[k] >= self.active_share * best).collect();
            active_children[p] = active.len();
            if active.len() >= self.min_siblings.max(1) && active.len() as f64 >= self.sibling_share * kids.len() as f64
            {
                let mean = active.iter().map(|&k| signal[k]).sum::<f64>() / active.len() as f64;
                let boosted = (mean * self.coactivation_boost).clamp(0.0, 1.0);
                if boosted > signal[p] {
                    signal[p] = boosted;
                    promoted[p] = true;
                }
            }
        }

        let weights = profile.weights;
        let mut hits: Vec<Hit> = (0..n)
            .filter(|&i| signal[i] > 0.0)
            .map(|i| {
                let unit = Ix::new(i as u32);
                let level = levels.level(unit);
                Hit {
                    unit,
                    level,
                    span: levels.spans.get(unit),
                    score: signal[i] * weights.of(level),
                    lexical: lexical[i],
                    dense: dense[i],
                    active_children: active_children[i],
                    context: None,
                }
            })
            .filter(|h| h.score > 0.0)
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.span.start.cmp(&b.span.start))
                .then(a.span.end.cmp(&b.span.end))
                .then(a.unit.cmp(&b.unit))
        });
        hits.truncate(top_k);
        for hit in &mut hits {
            hit.context = self.context(levels, &children, &signal, &promoted, hit);
        }
        Ok(Routed { profile, hits })
    }

    /// The parent context of one hit: its parent unit and span,
    /// whether that parent was promoted, and the window described
    /// on [`ParentContext::window`].
    fn context(
        &self,
        levels: &Levels,
        children: &[Vec<usize>],
        signal: &[f64],
        promoted: &[bool],
        hit: &Hit,
    ) -> Option<ParentContext> {
        let parent = levels.parent[hit.unit.get()]?;
        let window = match hit.level {
            Level::Chunk => {
                let siblings = &children[parent.get()];
                let at = siblings.iter().position(|&s| s == hit.unit.get())?;
                let lo = at.saturating_sub(self.context_siblings);
                let hi = (at + self.context_siblings + 1).min(siblings.len());
                siblings[lo..hi].iter().copied().filter(|&s| s != hit.unit.get()).map(|s| Ix::new(s as u32)).collect()
            }
            Level::Section => {
                let mut kids = children[hit.unit.get()].clone();
                kids.sort_by(|&a, &b| signal[b].total_cmp(&signal[a]).then(a.cmp(&b)));
                kids.truncate(2 * self.context_siblings + 1);
                kids.into_iter().map(|s| Ix::new(s as u32)).collect()
            }
            Level::Document => Vec::new(),
        };
        Some(ParentContext { parent, parent_span: levels.spans.get(parent), promoted: promoted[parent.get()], window })
    }
}
