use std::cmp::Reverse;
use std::collections::BTreeSet;

use super::ScoredSpan;

/// Event-updated summaries shared by exact and bucketed heatmaps.
pub(super) struct ActiveScores<'a> {
    items: &'a [ScoredSpan],
    active: BTreeSet<usize>,
    strongest: BTreeSet<(u32, Reverse<usize>)>,
    density: f64,
}

impl<'a> ActiveScores<'a> {
    pub(super) fn new(items: &'a [ScoredSpan]) -> Self {
        Self { items, active: BTreeSet::new(), strongest: BTreeSet::new(), density: 0.0 }
    }

    pub(super) fn insert(&mut self, index: usize) {
        if self.active.insert(index) {
            let score = self.items[index].score;
            // Finite non-negative f32 magnitudes sort by their bits. Reverse
            // the index to preserve the earliest-item tie rule.
            self.strongest.insert((score.abs().to_bits(), Reverse(index)));
            self.density += f64::from(score);
        }
    }

    pub(super) fn remove(&mut self, index: usize) {
        if self.active.remove(&index) {
            let score = self.items[index].score;
            self.strongest.remove(&(score.abs().to_bits(), Reverse(index)));
            self.density -= f64::from(score);
            if self.active.is_empty() {
                self.density = 0.0;
            }
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    pub(super) fn summary(&self, density: bool) -> (f32, u32, Option<u32>) {
        #[cfg(test)]
        super::AGGREGATE_QUERIES.with(|count| count.set(count.get() + 1));
        if density {
            (self.density as f32, self.active.len() as u32, self.active.first().map(|&i| self.items[i].item))
        } else if let Some((_, Reverse(index))) = self.strongest.last() {
            (self.items[*index].score, self.active.len() as u32, Some(self.items[*index].item))
        } else {
            (0.0, 0, None)
        }
    }
}
