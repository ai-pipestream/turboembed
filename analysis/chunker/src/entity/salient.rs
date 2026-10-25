//! Salience in chunk headers: when the document's consolidated entities
//! and their salience are present, a chunk's own entries can be ordered
//! by document salience, and the document's most salient entities can be
//! appended to every header that lacks them, after the carried entries.

use turbo_analysis_layers::{DocumentEntities, EntitySalience};

use super::header::ChunkEntities;
use super::keys::EntityKeys;

/// How [`super::ChunkEntityHeaders`] uses `EntitySalience`. Both are off
/// by default, which leaves headers as they were.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SalienceHeaders {
    /// Appends up to this many of the document's most salient entities to
    /// each chunk's header, most salient first, with `mentions` zero,
    /// skipping one whose key is already an entry.
    pub top_document_entities: u32,
    /// Orders a chunk's own entries by the document salience rank of
    /// their keys (keys of no ranked entity last, in their old order).
    pub order_by_salience: bool,
}

impl SalienceHeaders {
    pub(super) fn is_off(self) -> bool {
        self.top_document_entities == 0 && !self.order_by_salience
    }

    pub(super) fn bits(self) -> u64 {
        u64::from(self.top_document_entities) << 1 | u64::from(self.order_by_salience)
    }
}

/// The best salience rank of each key (`u32::MAX` for none), and the
/// keys of the document's entities in rank order, one key each: the
/// representative mention's, else the first mention's that has one.
fn rank_keys(keys: &EntityKeys, entities: &DocumentEntities, salience: &EntitySalience) -> (Vec<u32>, Vec<u32>) {
    let mut key_rank = vec![u32::MAX; keys.first.len()];
    let mut ranked: Vec<(u32, u32)> = Vec::new();
    for s in salience.as_slice() {
        let Some(e) = entities.entities.get(s.entity_ix as usize) else { continue };
        let key_of = |m: &turbo_analysis_layers::ConsolidatedMention| {
            m.entity.and_then(|row| keys.key.get(row.get()).copied().flatten())
        };
        for k in e.mentions.iter().filter_map(key_of) {
            let slot = &mut key_rank[k as usize];
            *slot = (*slot).min(s.rank);
        }
        let main = e.mentions.iter().filter(|m| m.is_representative).find_map(key_of);
        if let Some(k) = main.or_else(|| e.mentions.iter().find_map(key_of)) {
            ranked.push((s.rank, k));
        }
    }
    ranked.sort_unstable();
    let mut order = Vec::new();
    for (_, k) in ranked {
        if !order.contains(&k) {
            order.push(k);
        }
    }
    (key_rank, order)
}

/// Rewrites `h` per `cfg`.
pub(super) fn apply(
    h: &mut ChunkEntities,
    keys: &EntityKeys,
    entities: &DocumentEntities,
    salience: &EntitySalience,
    cfg: SalienceHeaders,
) {
    let (key_rank, order) = rank_keys(keys, entities, salience);
    let top = &order[..order.len().min(cfg.top_document_entities as usize)];
    let (mut row, mut key, mut mentions) = (vec![0u32], Vec::new(), Vec::new());
    for w in h.row.windows(2) {
        let r = w[0] as usize..w[1] as usize;
        let named = h.mentions[r.clone()].iter().take_while(|&&m| m > 0).count();
        let mut own: Vec<(u32, u32)> = h.key[r.start..r.start + named]
            .iter()
            .copied()
            .zip(h.mentions[r.start..r.start + named].iter().copied())
            .collect();
        if cfg.order_by_salience {
            own.sort_by_key(|&(k, _)| key_rank.get(k as usize).copied().unwrap_or(u32::MAX));
        }
        let start = key.len();
        for (k, m) in own {
            key.push(k);
            mentions.push(m);
        }
        key.extend_from_slice(&h.key[r.start + named..r.end]);
        mentions.extend_from_slice(&h.mentions[r.start + named..r.end]);
        for &k in top {
            if !key[start..].contains(&k) {
                key.push(k);
                mentions.push(0);
            }
        }
        row.push(key.len() as u32);
    }
    h.row = row;
    h.key = key;
    h.mentions = mentions;
}
