//! The hidden-layer contributions of every slot whose rows are few enough to
//! compute once per model: each POS slot over every POS row, each label slot
//! over every label row, and each word slot over the word specials
//! (`*UNK*`, `*NULL*`, `*ROOT*`). They are computed when the model is built,
//! so a model read from bytes uses them without a mutable call. Each value is
//! the same fold `FeedforwardDependencyModel::contribution` computes, so a
//! cached and an uncached score are equal to the bit.

use super::FeedforwardDependencyModel;

/// No cached row for a vocabulary id.
const NONE: u32 = u32::MAX;
/// The top two bits of a rank entry hold the slot kind (0 word, 1 POS,
/// 2 label); the rest the row's rank within that kind.
const KIND_SHIFT: u32 = 30;
/// The most f64 values the table holds; a larger model scores without it.
pub(super) const MAX_VALUES: usize = 4_000_000;

#[derive(Clone, Debug, Default)]
pub(super) struct SlotTable {
    /// Per vocabulary row: its kind and rank, or [`NONE`].
    rank: Vec<u32>,
    /// Cached rows per kind.
    rows: [usize; 3],
    /// Word slots per kind block: the layout's positions.
    positions: usize,
    hidden: usize,
    values: Vec<f64>,
}

impl SlotTable {
    pub(super) fn build(model: &FeedforwardDependencyModel) -> SlotTable {
        let v = &model.vocabulary;
        let n = model.layout.positions();
        let hidden = model.weights.hidden_bias.len();
        let specials: Vec<usize> =
            ["*UNK*", "*NULL*", "*ROOT*"].iter().filter_map(|s| v.words.get(*s).copied()).collect();
        let kinds: [Vec<usize>; 3] =
            [specials, v.pos.values().copied().collect(), v.labels.values().copied().collect()];
        let rows = [kinds[0].len(), kinds[1].len(), kinds[2].len()];
        let slots = [n, n, n - 6];
        let total: usize = (0..3).map(|k| slots[k] * rows[k]).sum::<usize>() * hidden;
        if total > MAX_VALUES || v.len() >= (1 << KIND_SHIFT) {
            return SlotTable::default();
        }
        let mut rank = vec![NONE; v.len()];
        for (k, ids) in kinds.iter().enumerate() {
            for (r, &id) in ids.iter().enumerate() {
                rank[id] = ((k as u32) << KIND_SHIFT) | r as u32;
            }
        }
        let mut values = Vec::with_capacity(total);
        let mut first = 0;
        for (k, ids) in kinds.iter().enumerate() {
            for slot in first..first + slots[k] {
                for &id in ids {
                    values.extend((0..hidden).map(|j| model.contribution(slot, id, j)));
                }
            }
            first += slots[k];
        }
        SlotTable { rank, rows, positions: n, hidden, values }
    }

    /// The cached contribution of `id` in `slot`, when the table holds it:
    /// `id` must be a row of the slot's own kind.
    #[inline]
    pub(super) fn get(&self, slot: usize, id: usize) -> Option<&[f64]> {
        let entry = *self.rank.get(id)?;
        if entry == NONE {
            return None;
        }
        let (kind, r) = ((entry >> KIND_SHIFT) as usize, (entry & ((1 << KIND_SHIFT) - 1)) as usize);
        let n = self.positions;
        let (slot_kind, within) = if slot < n {
            (0, slot)
        } else if slot < 2 * n {
            (1, slot - n)
        } else {
            (2, slot - 2 * n)
        };
        if kind != slot_kind {
            return None;
        }
        let base = match kind {
            0 => 0,
            1 => n * self.rows[0],
            _ => n * self.rows[0] + n * self.rows[1],
        };
        let at = (base + within * self.rows[kind] + r) * self.hidden;
        self.values.get(at..at + self.hidden)
    }

    /// The values held, for tests and size reports.
    pub(super) fn len(&self) -> usize {
        self.values.len()
    }
}
