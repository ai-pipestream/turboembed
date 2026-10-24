//! The best `k` rows of a scan, kept in a binary min-heap whose root
//! is the worst of them, so offering a row costs a comparison against
//! the root (and a sift only when the row enters) instead of shifting
//! a sorted buffer.

/// Whether `a` ranks strictly better than `b`: the higher score by
/// `f64::total_cmp` (so a NaN from a NaN model ranks the same as a
/// full sort ranks it), ties to the earlier row.
fn better(a: (u32, f64), b: (u32, f64)) -> bool {
    match a.1.total_cmp(&b.1) {
        core::cmp::Ordering::Greater => true,
        core::cmp::Ordering::Equal => a.0 < b.0,
        core::cmp::Ordering::Less => false,
    }
}

/// The best `k` `(row, score)` pairs offered so far.
///
/// Rows are expected in ascending order, as posting scans offer them,
/// so a row that only ties the k-th score ranks after it and stays
/// out. The ranking itself is total, so any offer order still keeps
/// the same best `k` a full sort would take.
#[derive(Debug, Clone, PartialEq)]
pub struct TopK {
    k: usize,
    /// A min-heap by rank: `heap[0]` is the worst kept row.
    heap: Vec<(u32, f64)>,
}

impl TopK {
    /// An empty keeper of the best `k` rows. `k == 0` keeps nothing.
    pub fn new(k: usize) -> TopK {
        TopK { k, heap: Vec::with_capacity(k.min(1024)) }
    }

    /// How many rows this keeps.
    pub fn capacity(&self) -> usize {
        self.k
    }

    /// How many rows are held.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether no row is held.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// The k-th score once `k` rows are held: the bound a pruning
    /// scan compares its remaining upper bounds against.
    pub fn threshold(&self) -> Option<f64> {
        (self.k > 0 && self.heap.len() == self.k).then(|| self.heap[0].1)
    }

    /// Keeps `row` at `score` if it ranks among the best `k` offered
    /// so far.
    pub fn offer(&mut self, row: u32, score: f64) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push((row, score));
            self.sift_up(self.heap.len() - 1);
        } else if better((row, score), self.heap[0]) {
            self.heap[0] = (row, score);
            self.sift_down(0);
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if !better(self.heap[parent], self.heap[i]) {
                break;
            }
            self.heap.swap(i, parent);
            i = parent;
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut worst = i;
            if l < self.heap.len() && better(self.heap[worst], self.heap[l]) {
                worst = l;
            }
            if r < self.heap.len() && better(self.heap[worst], self.heap[r]) {
                worst = r;
            }
            if worst == i {
                return;
            }
            self.heap.swap(i, worst);
            i = worst;
        }
    }

    /// The kept rows, best first: score descending, then row ascending.
    pub fn into_sorted(mut self) -> Vec<(u32, f64)> {
        self.heap.sort_unstable_by(|&a, &b| {
            if better(a, b) {
                core::cmp::Ordering::Less
            } else if better(b, a) {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        });
        self.heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naive full sort the heap must agree with: every offer,
    /// sorted best first, truncated to `k`.
    fn reference(offers: &[(u32, f64)], k: usize) -> Vec<(u32, f64)> {
        let mut all = offers.to_vec();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    #[test]
    fn heap_agrees_with_the_full_sort() {
        let mut state = 11u64;
        let mut next = move || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            state >> 40
        };
        for k in [0, 1, 3, 8, 64, 200] {
            // Scores from a small set, so ties are common, plus the
            // total_cmp corners posting scans can produce.
            let pool = [0.0, -0.0, 1.5, -2.0, 3.25, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
            let offers: Vec<(u32, f64)> = (0..150u32).map(|r| (r, pool[(next() % 8) as usize])).collect();
            let mut top = TopK::new(k);
            for &(row, score) in &offers {
                top.offer(row, score);
                let held = reference(&offers[..=row as usize], k);
                assert_eq!(top.len(), held.len(), "k={k} after row {row}");
                let want_threshold = held.last().map(|&(_, s)| s).filter(|_| held.len() == k && k > 0);
                assert_eq!(
                    top.threshold().map(f64::to_bits),
                    want_threshold.map(f64::to_bits),
                    "k={k} after row {row}"
                );
            }
            let got = top.into_sorted();
            let want = reference(&offers, k);
            assert_eq!(got.len(), want.len(), "k={k}");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.0, w.0, "k={k}");
                assert_eq!(g.1.total_cmp(&w.1), core::cmp::Ordering::Equal, "k={k}");
            }
        }
    }

    #[test]
    fn ties_keep_the_earlier_row_and_zero_k_keeps_nothing() {
        let mut top = TopK::new(2);
        top.offer(5, 1.0);
        top.offer(9, 1.0);
        top.offer(12, 1.0);
        assert_eq!(top.clone().into_sorted(), vec![(5, 1.0), (9, 1.0)]);
        top.offer(2, 1.0);
        assert_eq!(top.into_sorted(), vec![(2, 1.0), (5, 1.0)]);

        let mut none = TopK::new(0);
        none.offer(0, f64::INFINITY);
        assert!(none.is_empty());
        assert_eq!(none.threshold(), None);
        assert_eq!(none.into_sorted(), Vec::new());
    }
}
