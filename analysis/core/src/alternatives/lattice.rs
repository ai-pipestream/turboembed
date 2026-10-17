//! Path and coverage queries over [`Alternatives<L>`].

use crate::span::{Ix, IxRange, Span};

use super::{Alternatives, SplitMode};

impl<L> Alternatives<L> {
    /// The path from position `0` to `num_positions` through the alternatives
    /// whose [`ModeSet`](super::ModeSet) contains `mode`: when the mode holds
    /// several such paths, the one with the highest sum of `score` (every path,
    /// when there are no scores). Two paths with the same sum are told apart at
    /// the first position where they differ, by mode granularity
    /// (`SplitMode::A` prefers the shorter `position_length`, `SplitMode::C`
    /// the longer, `Best` and `B` the earlier row), then by the earlier row.
    /// `None` when `mode` has no complete path.
    pub fn path_for_mode(&self, mode: SplitMode) -> Option<Vec<Ix<Self>>> {
        self.best_path_through(
            |r| self.mode[r].contains(mode),
            |r| self.score.as_ref().map_or(0.0, |s| s[r]),
            Some(mode),
        )
    }

    /// The single best path from `0` to `num_positions`: the path of
    /// `SplitMode::Best` ([`path_for_mode`](Self::path_for_mode)) when it
    /// has one, else the path over all alternatives with the highest sum of
    /// `score`, or with the fewest steps when there are no scores, ties
    /// going to the earlier row at the first position that differs. Empty
    /// when the lattice is empty.
    pub fn best_path(&self) -> Vec<Ix<Self>> {
        if let Some(path) = self.path_for_mode(SplitMode::Best) {
            return path;
        }
        let weight = |r: usize| self.score.as_ref().map_or(-1.0, |s| s[r]);
        self.best_path_through(|_| true, weight, None).unwrap_or_default()
    }

    /// The path from `0` to `num_positions` over the alternatives `allowed`
    /// maximizing the sum of `weight`: one pass over the rows from the last
    /// position back, so each position's best continuation is known before
    /// the alternatives leaving it are weighed. `None` when no path exists.
    fn best_path_through(
        &self,
        allowed: impl Fn(usize) -> bool,
        weight: impl Fn(usize) -> f64,
        mode: Option<SplitMode>,
    ) -> Option<Vec<Ix<Self>>> {
        let p_len = self.num_positions as usize;
        if p_len == 0 {
            return Some(Vec::new());
        }
        // The best sum from each position to the end, and the row that
        // starts it; `None` while no allowed path reaches the end.
        let mut best: Vec<Option<(f64, usize)>> = vec![None; p_len + 1];
        best[p_len] = Some((0.0, usize::MAX));
        for r in (0..self.spans.len()).rev() {
            if !allowed(r) {
                continue;
            }
            let u = self.position[r] as usize;
            let len = self.position_length[r] as usize;
            if len == 0 || u >= p_len {
                continue;
            }
            let v = u.saturating_add(len);
            let Some(Some((rest, _))) = best.get(v) else { continue };
            let total = weight(r) + rest;
            let better = match best[u] {
                None => true,
                Some((cur, cur_row)) => match total.total_cmp(&cur) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Less => false,
                    std::cmp::Ordering::Equal => self.prefer(r, cur_row, mode),
                },
            };
            if better {
                best[u] = Some((total, r));
            }
        }
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < p_len {
            let (_, r) = best[pos]?;
            out.push(Ix::new(r as u32));
            pos += self.position_length[r] as usize;
        }
        Some(out)
    }

    /// Whether row `cand` is taken over row `current` when both start the
    /// same position with the same sum: by granularity under `mode`, then
    /// the earlier row.
    fn prefer(&self, cand: usize, current: usize, mode: Option<SplitMode>) -> bool {
        let (len_c, len_cur) = (self.position_length[cand], self.position_length[current]);
        match mode {
            Some(SplitMode::A) if len_c != len_cur => len_c < len_cur,
            Some(SplitMode::C) if len_c != len_cur => len_c > len_cur,
            _ => cand < current,
        }
    }

    /// Every path from position `0` to `num_positions`, each in row order
    /// and the paths in the order of their first differing row. The count
    /// grows exponentially with the lattice's length (a lattice with a
    /// one-position and a two-position alternative at every position has
    /// a Fibonacci number of paths), so this suits one sentence's lattice;
    /// an indexer walks [`at_position`](Self::at_position) instead.
    pub fn all_paths(&self) -> Vec<Vec<Ix<Self>>> {
        let mut out = Vec::new();
        if self.num_positions == 0 {
            out.push(Vec::new());
            return out;
        }
        // A depth-first walk with its own stack: the path so far, and for
        // each step the rows still to try at the position it left from.
        let mut current: Vec<Ix<Self>> = Vec::new();
        let mut pending: Vec<std::ops::Range<u32>> = vec![rows(self.at_position(0))];
        while let Some(range) = pending.last_mut() {
            let Some(r) = range.next() else {
                pending.pop();
                current.pop();
                continue;
            };
            let pos = self.position[r as usize];
            let len = self.position_length[r as usize];
            let Some(next) = pos.checked_add(len) else { continue };
            if len == 0 || next > self.num_positions {
                continue;
            }
            current.push(Ix::new(r));
            if next == self.num_positions {
                out.push(current.clone());
                current.pop();
            } else {
                pending.push(rows(self.at_position(next)));
            }
        }
        out
    }

    /// The spans of `path`, in path order.
    pub fn path_spans(&self, path: &[Ix<Self>]) -> Vec<Span> {
        path.iter().map(|&ix| self.spans.get(ix)).collect()
    }

    /// Returns the maximal contiguous covered byte intervals of `path`.
    ///
    /// Abutting spans along the path are merged, while any multi-position
    /// alternative bridging across an inter-token gap (`end_at[p] <
    /// start_at[p]`) is decomposed into its constituent position spans before
    /// coalescing. For any valid lattice (`check().is_ok()`), every complete
    /// path from `0` to `num_positions` returns the exact same covered byte
    /// intervals as [`best_path`](Self::best_path).
    pub fn covered_bytes(&self, path: &[Ix<Self>]) -> Vec<Span> {
        let p_len = self.num_positions as usize;
        let (starts, ends) = self.spans.columns();
        let mut start_at = vec![0u32; p_len + 1];
        let mut end_at = vec![0u32; p_len + 1];
        for i in 0..self.spans.len() {
            let u = self.position[i] as usize;
            let len = self.position_length[i] as usize;
            if len == 0 || u >= p_len {
                continue;
            }
            let v = u.saturating_add(len).min(p_len);
            start_at[u] = starts[i];
            end_at[v] = ends[i];
        }
        let mut out: Vec<Span> = Vec::new();
        let mut push_coalesced = |s: Span| match out.last_mut() {
            Some(prev) if prev.end == s.start => prev.end = s.end,
            _ => out.push(s),
        };
        for &ix in path {
            let r = ix.get();
            if r >= self.spans.len() {
                continue;
            }
            let u = self.position[r] as usize;
            let len = self.position_length[r] as usize;
            if len == 0 || u >= p_len {
                continue;
            }
            let v = u.saturating_add(len).min(p_len);
            for p in u..v {
                let (s, e) = (start_at[p], end_at[p + 1]);
                if s <= e {
                    push_coalesced(Span::new(s, e));
                }
            }
        }
        out
    }
}

/// The row numbers of a range of alternatives.
fn rows<L>(r: IxRange<L>) -> std::ops::Range<u32> {
    r.start.get() as u32..r.end.get() as u32
}

#[cfg(test)]
mod tests {
    use crate::alternatives::{Alternatives, ModeSet, NBest};
    use crate::layer::TagSet;
    use crate::span::{Ix, Span};
    use crate::text::ViewId;

    #[test]
    fn zero_length_or_out_of_bounds_rows_do_not_hang_or_panic() {
        let mut alt = Alternatives::<()>::new(ViewId::SOURCE);
        alt.push(Span::new(0, 1), 0, 0, ModeSet::BEST);
        alt.push(Span::new(0, 3), 0, 1, ModeSet::BEST);
        alt.num_positions = 1;
        assert_eq!(alt.best_path(), vec![Ix::new(1)]);
        assert_eq!(alt.all_paths(), vec![vec![Ix::new(1)]]);
        assert_eq!(alt.covered_bytes(&[Ix::new(0), Ix::new(1)]), vec![Span::new(0, 3)]);

        alt.push(Span::new(3, 5), 5, 1, ModeSet::BEST);
        alt.num_positions = 1;
        assert_eq!(alt.best_path(), vec![Ix::new(1)]);
        assert_eq!(alt.covered_bytes(&[Ix::new(1), Ix::new(2)]), vec![Span::new(0, 3)]);
    }

    #[test]
    #[should_panic(expected = "rank 0 of 0")]
    fn nbest_tags_of_rejects_rank_out_of_bounds_when_items_is_zero() {
        let nbest = NBest::<()>::new(TagSet::new("test", &["A"]), 0);
        let _ = nbest.tags_of(0);
    }
}
