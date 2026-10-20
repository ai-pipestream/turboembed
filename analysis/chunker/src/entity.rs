//! Entity continuity boundary costs (`EntityContinuity`,
//! `EntityBoundaries`) and per-chunk entity headers (`ChunkEntities`,
//! `ChunkEntityHeaders`).
//!
//! A run is a stretch of sentences that keep mentioning one entity: two
//! mentions of it at most `max_gap` sentences apart are one run, and
//! every gap between them is inside the run. A cut inside a run parts
//! the sentences about one entity, so each entity whose run spans a gap
//! adds `per_entity` to that gap's cost, up to `cap`. The costs are
//! written to the `entity` instance of `turbo:boundaries`, which
//! `DistilledBoundaryScorer` sums into the default one.

mod build;
mod carry;
mod header;
mod keys;

pub use build::chunk_entities;
pub use header::{ChunkEntities, ChunkEntityHeaders, HeaderPolicy};
pub use keys::{Chains, EntityKeys, KeyMention, Keying, normalize_surface};

use std::num::NonZeroU32;

use turbo_analysis::{Annotator, Bound, Error, Groups, Refs, View};
use turbo_analysis_layers::{Boundaries, Cut, Entities, Sentences};

use crate::coref::Mentions;
use crate::weigh::ENTITY;

/// The settings of [`entity_boundaries`] and [`EntityBoundaries`], in
/// `Boundaries::UNIT`s.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EntityContinuity {
    /// What each entity whose run spans a gap adds to it. Default 500,
    /// half a sentence boundary: one entity running across a paragraph
    /// break (400) still leaves it cheaper than a plain sentence
    /// boundary, two do not.
    pub per_entity: u32,
    /// The most sentences between two mentions of one entity that still
    /// make one run: 1 counts only adjacent sentences, 2 (the default)
    /// bridges one sentence that does not name it.
    pub max_gap: NonZeroU32,
    /// The most a gap costs, however many runs span it. Default 1500.
    pub cap: u32,
    /// How entities are keyed.
    pub keying: Keying,
}

impl Default for EntityContinuity {
    fn default() -> EntityContinuity {
        EntityContinuity {
            per_entity: Boundaries::UNIT as u32 / 2,
            max_gap: NonZeroU32::new(2).expect("2 > 0"),
            cap: Boundaries::UNIT as u32 * 3 / 2,
            keying: Keying::default(),
        }
    }
}

/// The entity continuity cost after each of `n` sentences: the reason is
/// always `Cut::Sentence`, the signal only argues against cuts.
pub fn entity_boundaries(n: usize, keys: &EntityKeys, cfg: EntityContinuity) -> Boundaries {
    let mut runs = vec![0u64; n];
    let gap = cfg.max_gap.get();
    for occ in &keys.occurrences {
        for w in occ.windows(2) {
            let (s, t) = (w[0].0, w[1].0);
            // Occurrences built by `EntityKeys` ascend within `n`; any
            // other pair is skipped.
            if s < t && t as usize <= n && t - s <= gap {
                // Consecutive mentions make disjoint gap ranges, so an
                // entity counts once at each gap.
                for g in &mut runs[s as usize..t as usize] {
                    *g += 1;
                }
            }
        }
    }
    let cost = runs
        .iter()
        .map(|&r| r.saturating_mul(u64::from(cfg.per_entity)).min(u64::from(cfg.cap)).min(i32::MAX as u64) as i32)
        .collect();
    Boundaries { cost, reason: vec![Cut::Sentence; n] }
}

/// Writes the entity continuity costs to the `entity` instance of
/// `turbo:boundaries` (bind it with [`EntityBoundaries::bound`]). Without
/// entities every gap costs nothing.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct EntityBoundaries {
    pub continuity: EntityContinuity,
}

impl EntityBoundaries {
    /// The step, writing [`ENTITY`].
    pub fn bound(self) -> Bound<EntityBoundaries> {
        Bound::new(self).write(0, ENTITY)
    }
}

/// Refuses inputs in different views.
pub(crate) fn same_view(
    id: &'static str,
    sentences: &Sentences,
    view: &View,
    entities: Option<&Entities>,
    mentions: Option<&Mentions>,
) -> Result<(), Error> {
    let v = sentences.spans.view();
    if view.id() != v || entities.is_some_and(|e| e.spans.view() != v) || mentions.is_some_and(|m| m.spans.view() != v)
    {
        return Err(Error::Annotator { annotator: id, detail: "its inputs are in different views".into() });
    }
    Ok(())
}

pub(crate) fn fnv(values: &[u64]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
}

impl Annotator for EntityBoundaries {
    const ID: &'static str = "turbo:entity-boundaries";
    type Reads =
        (Sentences, View, Option<Entities>, Option<Mentions>, Option<Groups<Mentions>>, Option<Groups<Entities>>);
    type Writes = (Boundaries,);

    fn annotate(
        &self,
        (sentences, view, entities, mentions, mention_groups, entity_groups): Refs<'_, Self>,
    ) -> Result<(Boundaries,), Error> {
        same_view(Self::ID, sentences, view, entities, mentions)?;
        let chains = Chains { mentions, mention_groups, entity_groups };
        let keys = EntityKeys::build(sentences, entities, view.text(), self.continuity.keying, chains);
        Ok((entity_boundaries(sentences.spans.len(), &keys, self.continuity),))
    }

    fn config_hash(&self) -> u64 {
        let c = self.continuity;
        fnv(&[u64::from(c.per_entity), u64::from(c.max_gap.get()), u64::from(c.cap), c.keying.bits()])
    }
}

#[cfg(test)]
mod tests;
