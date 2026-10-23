//! `ChunkFacets` and `Pragmatics` filter constructors for the
//! cascade (#857), split out of `cascade.rs` for the line cap.

use std::collections::BTreeSet;

use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::{Chunks, IncivilityLabel, Level, Levels, Polarity, Pragmatics, Unit};

use super::{FilterMode, UnitFilter};
use crate::sentiment::ChunkFacets;

impl UnitFilter {
    /// The chunk units whose [`ChunkFacets`] row satisfies
    /// `keep`. Only `Level::Chunk` units are named, so a
    /// `Require` filter built this way also drops section and
    /// document units. Refuses facets that fail their own check
    /// or that hold fewer rows than `levels` has chunks (a
    /// mismatched document), never panics on either.
    pub fn chunks_matching(
        name: impl Into<String>,
        mode: FilterMode,
        levels: &Levels,
        facets: &ChunkFacets,
        keep: impl Fn(Ix<Chunks>, &ChunkFacets) -> bool,
    ) -> Result<UnitFilter, String> {
        levels.check()?;
        facets.check()?;
        let mut units = BTreeSet::new();
        for unit in levels.of(Level::Chunk) {
            let Unit::Chunk(chunk) = levels.unit[unit.get()] else { continue };
            if chunk.get() >= facets.len() {
                return Err(format!(
                    "chunk facets hold {} rows but chunk {} is named by the levels",
                    facets.len(),
                    chunk.get()
                ));
            }
            if keep(chunk, facets) {
                units.insert(unit);
            }
        }
        Ok(UnitFilter { name: name.into(), mode, units })
    }

    /// A `Require` filter keeping only chunk units whose every
    /// incivility dimension score is at most `max` (in
    /// `0.0..=1.0`): retrieval that must not surface uncivil
    /// chunks at any rank.
    pub fn incivility_at_most(
        name: impl Into<String>,
        levels: &Levels,
        facets: &ChunkFacets,
        max: f64,
    ) -> Result<UnitFilter, String> {
        if !max.is_finite() || !(0.0..=1.0).contains(&max) {
            return Err(format!("incivility threshold {max} is not a score in 0.0..=1.0"));
        }
        Self::chunks_matching(name, FilterMode::Require, levels, facets, |c, f| {
            f.incivility_scores(c).iter().into_iter().all(|(_, s)| s <= max)
        })
    }

    /// Chunk units carrying aspect `label` (with `polarity`
    /// when given), for aspect-targeted require or boost.
    pub fn aspect_is(
        name: impl Into<String>,
        mode: FilterMode,
        levels: &Levels,
        facets: &ChunkFacets,
        label: &str,
        polarity: Option<Polarity>,
    ) -> Result<UnitFilter, String> {
        Self::chunks_matching(name, mode, levels, facets, |c, f| {
            f.aspects_of(c).any(|(_, l, p)| l == label && polarity.is_none_or(|want| p == want))
        })
    }

    /// A `Require` filter keeping only chunk units whose
    /// `label` incivility dimension is at most `max`.
    pub fn incivility_label_at_most(
        name: impl Into<String>,
        levels: &Levels,
        facets: &ChunkFacets,
        label: IncivilityLabel,
        max: f64,
    ) -> Result<UnitFilter, String> {
        if !max.is_finite() || !(0.0..=1.0).contains(&max) {
            return Err(format!("incivility threshold {max} is not a score in 0.0..=1.0"));
        }
        Self::chunks_matching(name, FilterMode::Require, levels, facets, |c, f| {
            f.incivility_scores(c).get(label) <= max
        })
    }

    /// Chunk units whose sentences' sarcasm scores are all at
    /// most `max` (and, when `drop_evidence`, none carries
    /// incongruity evidence), from the sentence-aligned
    /// `Pragmatics` layer. Use `FilterMode::Require` to drop
    /// sarcastic chunks or `Boost(f)` with `0 < f < 1` to
    /// discount the kept ones. Refuses mismatched layers
    /// (sentence ranges past the pragmatics rows) without
    /// panicking.
    pub fn sarcasm_at_most(
        name: impl Into<String>,
        mode: FilterMode,
        levels: &Levels,
        chunks: &Chunks,
        pragmatics: &Pragmatics,
        max: f64,
        drop_evidence: bool,
    ) -> Result<UnitFilter, String> {
        if !max.is_finite() || !(0.0..=1.0).contains(&max) {
            return Err(format!("sarcasm threshold {max} is not a score in 0.0..=1.0"));
        }
        levels.check()?;
        chunks.check()?;
        pragmatics.check()?;
        let mut units = BTreeSet::new();
        for unit in levels.of(Level::Chunk) {
            let Unit::Chunk(chunk) = levels.unit[unit.get()] else { continue };
            let range = chunks.sentences.get(chunk.get()).ok_or_else(|| {
                format!("chunks hold {} rows but chunk {} is named by the levels", chunks.len(), chunk.get())
            })?;
            if range.end.get() > pragmatics.len() {
                return Err(format!(
                    "chunk {} names sentences past the {} pragmatics rows",
                    chunk.get(),
                    pragmatics.len()
                ));
            }
            let keep = (range.start.get()..range.end.get())
                .all(|s| pragmatics.score[s] <= max && (!drop_evidence || pragmatics.evidence[s].is_none()));
            if keep {
                units.insert(unit);
            }
        }
        Ok(UnitFilter { name: name.into(), mode, units })
    }
}
