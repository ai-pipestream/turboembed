//! The keys a chunk carries from earlier in its section. Mentions are
//! fed in text order; a key is live at the index of its latest mention
//! fed, so the live keys by index, latest first, are the keys by their
//! last mention. A min tree over those indexes yields the live keys of
//! any index range in key order, and finds the latest live index before
//! a point, so a lookup visits only the keys and sentence runs it takes
//! or skips, each in `O(log mentions)`, never the history behind them.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::build::op_carry;
use super::keys::KeyMention;

const NIL: u32 = u32::MAX;

pub(super) struct Carry {
    /// Leaves from `size`: the key live at that mention index, or NIL.
    tree: Vec<u32>,
    size: usize,
    /// Per key, the index of its latest mention fed, or NIL.
    last: Vec<u32>,
    /// Per mention, the first index of its run of one sentence.
    run: Vec<u32>,
    heap: BinaryHeap<Reverse<(u32, u32)>>,
}

impl Carry {
    pub(super) fn new(ms: &[KeyMention], keys: usize) -> Carry {
        let size = ms.len().next_power_of_two().max(1);
        let mut run = Vec::with_capacity(ms.len());
        for (i, m) in ms.iter().enumerate() {
            let same = i > 0 && ms[i - 1].sentence == m.sentence;
            run.push(if same { run[i - 1] } else { i as u32 });
        }
        Carry { tree: vec![NIL; 2 * size], size, last: vec![NIL; keys], run, heap: BinaryHeap::new() }
    }

    fn set(&mut self, i: usize, v: u32) {
        let mut x = self.size + i;
        self.tree[x] = v;
        while x > 1 {
            x /= 2;
            self.tree[x] = self.tree[2 * x].min(self.tree[2 * x + 1]);
        }
    }

    /// Feeds mention `i`, of key `k`: it is now the key's latest.
    pub(super) fn feed(&mut self, i: usize, k: u32) {
        let was = self.last[k as usize];
        if was != NIL {
            self.set(was as usize, NIL);
        }
        self.set(i, k);
        self.last[k as usize] = i as u32;
    }

    /// The latest live index in `lo..hi`.
    fn latest(&self, lo: usize, hi: usize) -> Option<usize> {
        let mut stack = [(0usize, 0usize, 0usize); 64];
        let (mut top, mut found) = (1, None);
        stack[0] = (1, 0, self.size);
        while top > 0 {
            top -= 1;
            let (x, a, b) = stack[top];
            if b <= lo || hi <= a || self.tree[x] == NIL {
                continue;
            }
            if b - a == 1 {
                found = Some(a);
                break;
            }
            let mid = (a + b) / 2;
            // Right child on top: it is searched first.
            stack[top] = (2 * x, a, mid);
            stack[top + 1] = (2 * x + 1, mid, b);
            top += 2;
        }
        found
    }

    /// Emits up to `need` live keys of `lo..hi` not in the chunk, in key
    /// order, and returns how many.
    fn smallest(&mut self, lo: usize, hi: usize, in_chunk: &[bool], need: u32, emit: &mut impl FnMut(u32)) -> u32 {
        let (mut a, mut b) = (lo + self.size, hi + self.size);
        while a < b {
            if a & 1 == 1 {
                self.push(a);
                a += 1;
            }
            if b & 1 == 1 {
                b -= 1;
                self.push(b);
            }
            a /= 2;
            b /= 2;
        }
        let mut got = 0;
        while got < need {
            let Some(Reverse((k, x))) = self.heap.pop() else { break };
            let x = x as usize;
            if x < self.size {
                self.push(2 * x);
                self.push(2 * x + 1);
                continue;
            }
            op_carry();
            if !in_chunk[k as usize] {
                emit(k);
                got += 1;
            }
        }
        self.heap.clear();
        got
    }

    fn push(&mut self, x: usize) {
        if self.tree[x] != NIL {
            self.heap.push(Reverse((self.tree[x], x as u32)));
        }
    }

    /// Emits the keys carried into a chunk: those whose latest mention
    /// fed is at index `lo` or later, latest sentence run first and then
    /// by key, skipping the chunk's own, at most `max`.
    pub(super) fn carried(&mut self, lo: usize, fed: usize, in_chunk: &[bool], max: u32, mut emit: impl FnMut(u32)) {
        let (mut taken, mut hi) = (0, fed);
        while taken < max {
            op_carry();
            let Some(j) = self.latest(lo, hi) else { break };
            let from = (self.run[j] as usize).max(lo);
            taken += self.smallest(from, j + 1, in_chunk, max - taken, &mut emit);
            hi = from;
        }
    }
}
