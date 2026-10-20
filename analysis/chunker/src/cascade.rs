//! A static-to-contextual retrieval cascade over `Levels`
//! units: cheap signals admit a wide candidate pool, the
//! expensive contextual space reranks only that pool, and
//! structured filters then drop or boost what remains.
//!
//! Each stage keeps its own space. A query vector carries
//! the `ModelRef` of the space it was encoded in, and
//! building a cascade refuses a query whose model is not
//! the model of the rows it would be scored against, so a
//! static query can never be scored against contextual
//! rows by accident. The two spaces must also be distinct:
//! reranking a shortlist in the space that produced it is
//! not a cascade.
//!
//! Output is deterministic for a fixed config: scores
//! descend, ties break by unit index. Every hit names the
//! stage or stages that admitted it, the signal scores it
//! received, and its exact supporting spans (the unit's
//! own fragments, discontinuous units included).

use std::collections::BTreeSet;

use turbo_analysis::{Ix, Layer, ModelRef, Span, Spans};
use turbo_analysis_layers::{Level, Levels, Vectors};

use crate::terms::{Bm25, Collection, Query, TermStats, Vocabulary};

/// A query encoded in one named vector space.
#[derive(Copy, Clone, Debug)]
pub struct QueryVector<'a> {
    pub model: &'a ModelRef,
    pub vector: &'a [f32],
}

/// One stage's vector space: the unit rows and the query
/// that belongs with them.
#[derive(Copy, Clone)]
pub struct VectorSpace<'a> {
    pub rows: &'a Vectors<Levels>,
    pub query: QueryVector<'a>,
}

/// The lexical stage's inputs: per-unit term statistics, the
/// vocabulary they count against, and the collection the
/// BM25 idf comes from.
pub struct LexicalSpace<'a> {
    pub stats: &'a TermStats<Levels>,
    pub vocabulary: &'a Vocabulary,
    pub collection: &'a dyn Collection,
}

/// Which stage admitted a candidate into the pool.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Static,
    Lexical,
}

/// What a structured filter does to the units it names.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FilterMode {
    /// Units outside the filter are dropped.
    Require,
    /// Units inside the filter have their final score
    /// multiplied by this finite, positive factor.
    Boost(f64),
}

/// A stage-3 structured filter over level units, built from
/// whatever layer carries the structure (entities, glossary
/// terms, geo regions, categories): the units it names are
/// the ones the structure touches.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitFilter {
    pub name: String,
    pub mode: FilterMode,
    pub units: BTreeSet<Ix<Levels>>,
}

impl UnitFilter {
    /// The units whose fragments overlap any fragment of
    /// `spans`, which must live in the same view as `levels`.
    pub fn overlapping<L: Layer>(
        name: impl Into<String>,
        mode: FilterMode,
        levels: &Levels,
        spans: &Spans<L>,
    ) -> Result<UnitFilter, String> {
        if spans.view() != levels.view() {
            return Err("filter spans are in a different view than the level units".into());
        }
        let mut units = BTreeSet::new();
        for unit in levels.of(Level::Chunk).chain(levels.of(Level::Section)).chain(levels.of(Level::Document)) {
            let hit = levels.spans.fragments(unit).any(|f| {
                (0..spans.len() as u32).any(|j| spans.fragments(Ix::new(j)).any(|s| f.start < s.end && s.start < f.end))
            });
            if hit {
                units.insert(unit);
            }
        }
        Ok(UnitFilter { name: name.into(), mode, units })
    }
}

/// The cascade's knobs: how wide stage 1 reaches, how many
/// hits survive, the BM25 shape, and the stage-3 filters.
#[derive(Clone, Debug)]
pub struct CascadeConfig {
    /// Per-signal stage-1 shortlist size; the pool is the
    /// union of the static and lexical shortlists.
    pub candidates: usize,
    pub top_k: usize,
    pub bm25: Bm25,
    pub filters: Vec<UnitFilter>,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        CascadeConfig { candidates: 64, top_k: 10, bm25: Bm25::LUCENE, filters: Vec::new() }
    }
}

/// One ranked unit of a cascade search.
#[derive(Clone, Debug, PartialEq)]
pub struct CascadeHit {
    pub unit: Ix<Levels>,
    pub level: Level,
    /// The unit's own fragments in the levels view.
    pub spans: Vec<Span>,
    pub score: f64,
    /// The stage-1 static signal in `[0, 1]`, when a static
    /// space scored this unit.
    pub static_score: Option<f64>,
    /// The stage-1 lexical signal in `[0, 1]`, when BM25
    /// scored this unit above zero.
    pub lexical_score: Option<f64>,
    /// The stage-2 contextual signal in `[0, 1]`, when a
    /// contextual space reranked this unit.
    pub contextual_score: Option<f64>,
    /// The stage-1 signals that admitted this unit.
    pub admitted_by: Vec<Stage>,
    /// The boost filters that raised this unit's score, in
    /// config order.
    pub boosted_by: Vec<String>,
}

/// A finished search: its hits, best first, and how much
/// work each expensive stage actually did.
#[derive(Clone, Debug, PartialEq)]
pub struct CascadeResult {
    pub hits: Vec<CascadeHit>,
    /// The stage-1 pool size before stage-3 filters.
    pub pool: usize,
    /// How many units the contextual space scored: the pool,
    /// never the whole document. Zero with no contextual
    /// stage.
    pub contextual_scored: usize,
}

impl Collection for &dyn Collection {
    fn units(&self) -> u64 {
        (**self).units()
    }
    fn mean_length(&self) -> f64 {
        (**self).mean_length()
    }
    fn df(&self, term: &str) -> u64 {
        (**self).df(term)
    }
}

/// A validated cascade over one document's levels. All
/// space and shape checks happen in [`Cascade::new`], so
/// `search` cannot meet a mismatched space.
pub struct Cascade<'a> {
    levels: &'a Levels,
    config: CascadeConfig,
    static_space: Option<VectorSpace<'a>>,
    static_norm: f64,
    contextual_space: Option<VectorSpace<'a>>,
    contextual_norm: f64,
    lexical: Option<LexicalSpace<'a>>,
}

impl<'a> Cascade<'a> {
    pub fn new(
        levels: &'a Levels,
        config: CascadeConfig,
        static_space: Option<VectorSpace<'a>>,
        contextual_space: Option<VectorSpace<'a>>,
        lexical: Option<LexicalSpace<'a>>,
    ) -> Result<Cascade<'a>, String> {
        levels.check()?;
        if config.candidates == 0 || config.top_k == 0 {
            return Err("candidates and top_k must be at least 1".into());
        }
        if !config.bm25.k1.is_finite() || config.bm25.k1 < 0.0 || !(0.0..=1.0).contains(&config.bm25.b) {
            return Err("bm25 parameters out of range".into());
        }
        // Every signal score lies in [0, 1], so a hit's final
        // score stays finite exactly when every prefix product
        // of the boost factors (applied in config order to a
        // unit every filter names) stays finite.
        let mut boost_product = 1.0f64;
        for filter in &config.filters {
            if let FilterMode::Boost(m) = filter.mode {
                if !m.is_finite() || m <= 0.0 {
                    return Err(format!("filter {:?} boost must be finite and positive", filter.name));
                }
                boost_product *= m;
                if !boost_product.is_finite() {
                    return Err(format!("filter {:?} boost overflows the combined boost product", filter.name));
                }
            }
            if filter.units.last().is_some_and(|u| u.get() >= levels.len()) {
                return Err(format!("filter {:?} names a unit outside the levels", filter.name));
            }
        }
        let static_norm = match &static_space {
            Some(space) => check_space(levels, space, "static")?,
            None => 0.0,
        };
        let contextual_norm = match &contextual_space {
            Some(space) => check_space(levels, space, "contextual")?,
            None => 0.0,
        };
        if static_space.is_none() && lexical.is_none() {
            return Err("a cascade needs a static space, a lexical space, or both".into());
        }
        if let (Some(s), Some(c)) = (&static_space, &contextual_space)
            && s.rows.model == c.rows.model
        {
            return Err("the static and contextual stages must be different spaces".into());
        }
        if let Some(lex) = &lexical {
            lex.stats.check()?;
            lex.vocabulary.check()?;
            if lex.stats.len() != levels.len() {
                return Err(format!("term stats for {} units over {} level units", lex.stats.len(), levels.len()));
            }
            if lex.stats.df.len() != lex.vocabulary.len() {
                return Err("term stats df does not match the vocabulary".into());
            }
        }
        Ok(Cascade { levels, config, static_space, static_norm, contextual_space, contextual_norm, lexical })
    }

    /// Runs the cascade for one query: its terms feed the
    /// lexical stage; the vector queries were bound to
    /// their spaces at build time.
    pub fn search(&self, terms: &[&str]) -> CascadeResult {
        let n = self.levels.len();
        let units = || (0..n as u32).map(Ix::new);
        let static_scores: Vec<Option<f64>> = match &self.static_space {
            Some(space) => units().map(|i| cosine(space, self.static_norm, i)).collect(),
            None => vec![None; n],
        };
        let lexical_scores: Vec<Option<f64>> = match &self.lexical {
            Some(lex) => {
                let q = Query::new(self.config.bm25, &lex.collection, lex.vocabulary, terms);
                let raw: Vec<f64> = units().map(|i| q.score(lex.stats, i)).collect();
                let max = raw.iter().copied().fold(0.0f64, f64::max);
                raw.iter().map(|&s| (s > 0.0 && max > 0.0).then(|| (s / max).clamp(0.0, 1.0))).collect()
            }
            None => vec![None; n],
        };

        // Stage 1: the union of the per-signal shortlists.
        let mut pool: Vec<Ix<Levels>> = Vec::new();
        let mut admitted: Vec<Vec<Stage>> = vec![Vec::new(); n];
        for (scores, stage) in [(&static_scores, Stage::Static), (&lexical_scores, Stage::Lexical)] {
            let mut ranked: Vec<Ix<Levels>> = units().filter(|&i| scores[i.get()].is_some()).collect();
            ranked.sort_by(|&a, &b| {
                scores[b.get()].unwrap().total_cmp(&scores[a.get()].unwrap()).then(a.get().cmp(&b.get()))
            });
            for unit in ranked.into_iter().take(self.config.candidates) {
                if admitted[unit.get()].is_empty() {
                    pool.push(unit);
                }
                admitted[unit.get()].push(stage);
            }
        }
        let pool_size = pool.len();

        // Stage 2: the contextual space scores the pool only.
        let mut contextual_scored = 0usize;
        let mut hits: Vec<CascadeHit> = Vec::with_capacity(pool_size);
        for unit in pool {
            let i = unit.get();
            let contextual_score = match &self.contextual_space {
                Some(space) => {
                    contextual_scored += 1;
                    cosine(space, self.contextual_norm, unit)
                }
                None => None,
            };
            let stage1 = match (static_scores[i], lexical_scores[i]) {
                (Some(s), Some(l)) => (s + l) / 2.0,
                (Some(s), None) => s,
                (None, Some(l)) => l,
                (None, None) => 0.0,
            };
            hits.push(CascadeHit {
                unit,
                level: self.levels.level(unit),
                spans: self.levels.spans.fragments(unit).collect(),
                score: contextual_score.unwrap_or(stage1),
                static_score: static_scores[i],
                lexical_score: lexical_scores[i],
                contextual_score,
                admitted_by: std::mem::take(&mut admitted[i]),
                boosted_by: Vec::new(),
            });
        }

        // Stage 3: structured filters drop or boost.
        for filter in &self.config.filters {
            match filter.mode {
                FilterMode::Require => hits.retain(|h| filter.units.contains(&h.unit)),
                FilterMode::Boost(m) => {
                    for hit in hits.iter_mut().filter(|h| filter.units.contains(&h.unit)) {
                        hit.score *= m;
                        hit.boosted_by.push(filter.name.clone());
                    }
                }
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.unit.get().cmp(&b.unit.get())));
        hits.truncate(self.config.top_k);
        CascadeResult { hits, pool: pool_size, contextual_scored }
    }
}

/// Validates one space against the levels and returns the
/// query norm. The refusal that matters most: a query
/// encoded in another space than the rows'.
fn check_space(levels: &Levels, space: &VectorSpace<'_>, stage: &str) -> Result<f64, String> {
    space.rows.check()?;
    if space.rows.len() != levels.len() {
        return Err(format!("{stage} vectors for {} units over {} level units", space.rows.len(), levels.len()));
    }
    if space.query.model != &space.rows.model {
        return Err(format!(
            "{stage} space mismatch: query encoded by {:?} over rows from {:?}",
            space.query.model, space.rows.model
        ));
    }
    if space.query.vector.len() != space.rows.dim as usize {
        return Err(format!(
            "a {stage} query vector of {} dimensions for rows of {}",
            space.query.vector.len(),
            space.rows.dim
        ));
    }
    if space.query.vector.iter().any(|v| !v.is_finite()) {
        return Err(format!("{stage} query vector must be finite"));
    }
    let norm = space.query.vector.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    if norm <= 0.0 {
        return Err(format!("{stage} query vector has no direction"));
    }
    Ok(norm)
}

/// Cosine of one row against the space's query, mapped to
/// `[0, 1]`; `None` for a row the layer marks empty or a
/// zero-norm row.
fn cosine(space: &VectorSpace<'_>, query_norm: f64, unit: Ix<Levels>) -> Option<f64> {
    if space.rows.is_empty(unit) {
        return None;
    }
    let row = space.rows.row(unit);
    if row.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let norm = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    if norm <= 0.0 {
        return None;
    }
    let dot: f64 = row.iter().zip(space.query.vector).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    Some(((dot / (norm * query_norm)).clamp(-1.0, 1.0) + 1.0) / 2.0)
}

/// Recall of the relevant units among the first `k` of a
/// ranking: the curve point a candidate-count sweep records
/// per count. Zero relevant units recall as zero.
pub fn recall_at_k(ranking: &[Ix<Levels>], relevant: &BTreeSet<Ix<Levels>>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let found: BTreeSet<Ix<Levels>> = ranking.iter().take(k).filter(|u| relevant.contains(u)).copied().collect();
    found.len() as f64 / relevant.len() as f64
}
