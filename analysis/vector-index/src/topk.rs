//! The `k` best rows of a scan, kept in a heap whose root is the worst of
//! them.

use crate::Hit;

/// Whether `a` ranks below `b`: a lower score, or an equal one and a later
/// row. Equal means `==`, so `0.0` and `-0.0` tie and rank by row.
fn below(a: Hit, b: Hit) -> bool {
    a.score < b.score || (a.score == b.score && a.row > b.row)
}

pub(crate) struct TopK {
    k: usize,
    heap: Vec<Hit>,
}

impl TopK {
    pub(crate) fn new(k: usize) -> TopK {
        TopK { k, heap: Vec::with_capacity(k) }
    }

    /// Keeps `row` if it ranks among the best `k` offered so far.
    #[inline]
    pub(crate) fn offer(&mut self, row: u32, score: f32) {
        let hit = Hit { row, score };
        if self.heap.len() < self.k {
            self.heap.push(hit);
            let mut i = self.heap.len() - 1;
            while i > 0 {
                let parent = (i - 1) / 2;
                if !below(self.heap[i], self.heap[parent]) {
                    break;
                }
                self.heap.swap(i, parent);
                i = parent;
            }
        } else if self.k > 0 && below(self.heap[0], hit) {
            self.heap[0] = hit;
            self.sift_down();
        }
    }

    fn sift_down(&mut self) {
        let n = self.heap.len();
        let mut i = 0;
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut worst = i;
            if l < n && below(self.heap[l], self.heap[worst]) {
                worst = l;
            }
            if r < n && below(self.heap[r], self.heap[worst]) {
                worst = r;
            }
            if worst == i {
                return;
            }
            self.heap.swap(i, worst);
            i = worst;
        }
    }

    /// The kept rows, best first.
    pub(crate) fn into_sorted(self) -> Vec<Hit> {
        let mut out = self.heap;
        out.sort_unstable_by(|a, b| {
            if below(*b, *a) {
                std::cmp::Ordering::Less
            } else if below(*a, *b) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        out
    }
}
