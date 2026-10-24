use super::{Node, OutcomeLabel, ROOT, compare_doubles};
use std::cmp::Ordering;

/// One reusable outcome buffer per active beam slot. Matching each surviving
/// node to a slot that held its parent updates that slot in $O(1)$ (`push`),
/// and reassigned slots walk only to the lowest common ancestor of the new
/// node and the dead branch.
pub(super) struct PathSlots<'m, L: ?Sized> {
    slots: Vec<(u32, Vec<&'m L>)>,
    assigned: Vec<usize>,
    used: Vec<bool>,
    seq_len: usize,
}

impl<'m, L: ?Sized> PathSlots<'m, L> {
    pub(super) fn new(seq_len: usize) -> Self {
        Self { slots: Vec::new(), assigned: Vec::new(), used: Vec::new(), seq_len }
    }

    pub(super) fn prepare(&mut self, nodes: &[Node], prev: &[u32], outcomes: &[&'m L]) {
        let width = prev.len();
        self.used.fill(false);
        self.assigned.clear();
        self.assigned.resize(width, usize::MAX);

        for (beam_idx, &id) in prev.iter().enumerate() {
            let parent = if id == ROOT { ROOT } else { nodes[id as usize].parent };
            if let Some(s) = (0..self.slots.len()).find(|&s| !self.used[s] && self.slots[s].0 == parent) {
                self.used[s] = true;
                self.assigned[beam_idx] = s;
            }
        }
        for (beam_idx, &id) in prev.iter().enumerate() {
            if self.assigned[beam_idx] == usize::MAX {
                if let Some(s) = (0..self.slots.len()).find(|&s| !self.used[s]) {
                    self.used[s] = true;
                    self.assigned[beam_idx] = s;
                } else {
                    let s = self.slots.len();
                    self.slots.push((ROOT, Vec::with_capacity(self.seq_len)));
                    self.used.push(true);
                    self.assigned[beam_idx] = s;
                }
            }
            let s = self.assigned[beam_idx];
            let (last_id, prior) = &mut self.slots[s];
            update_path(nodes, id, last_id, outcomes, prior);
        }
    }

    #[inline]
    pub(super) fn prior(&self, beam_idx: usize) -> &[&'m L] {
        &self.slots[self.assigned[beam_idx]].1
    }
}

/// Updates `out` (which currently holds the outcomes of the partial
/// sequence ending at `*last_id`) to hold the outcomes of the partial
/// sequence ending at `id`, walking only back to their lowest common
/// ancestor.
fn update_path<'m, L: ?Sized>(nodes: &[Node], id: u32, last_id: &mut u32, outcomes: &[&'m L], out: &mut Vec<&'m L>) {
    if id == ROOT {
        out.clear();
        *last_id = ROOT;
        return;
    }
    let len = nodes[id as usize].len as usize;
    let mut cur = id;
    let mut prev = *last_id;
    let mut prev_len = if prev == ROOT { 0 } else { nodes[prev as usize].len as usize };
    if out.len() < len {
        let dummy = outcomes[nodes[id as usize].outcome as usize];
        out.resize(len, dummy);
    } else if out.len() > len {
        out.truncate(len);
        while prev_len > len {
            prev = nodes[prev as usize].parent;
            prev_len -= 1;
        }
    }
    let mut d = len;
    while d > prev_len {
        d -= 1;
        let n = nodes[cur as usize];
        out[d] = outcomes[n.outcome as usize];
        cur = n.parent;
    }
    while cur != prev {
        d -= 1;
        let n = nodes[cur as usize];
        out[d] = outcomes[n.outcome as usize];
        cur = n.parent;
        prev = nodes[prev as usize].parent;
    }
    *last_id = id;
}

/// `SearchNode.compareTo` for two nodes of the same step (`x.len == y.len`):
/// higher scores first, then by parent prefix rank and current outcome in
/// `String.compareTo` order, in $O(1)$.
#[inline]
pub(super) fn compare_step_nodes<L: OutcomeLabel + ?Sized>(
    nodes: &[Node],
    outcomes: &[&L],
    a: u32,
    b: u32,
) -> Ordering {
    let (x, y) = (nodes[a as usize], nodes[b as usize]);
    compare_doubles(y.score, x.score)
        .then_with(|| x.parent_rank.cmp(&y.parent_rank))
        .then_with(|| L::compare(outcomes[x.outcome as usize], outcomes[y.outcome as usize]))
}

/// Computes the lexicographical outcome-path rank of each node in
/// `prev[..size.min(prev.len())]`.
pub(super) fn assign_beam_ranks<L: OutcomeLabel + ?Sized>(
    nodes: &[Node],
    outcomes: &[&L],
    prev: &[u32],
    size: usize,
    rank_order: &mut Vec<usize>,
    prev_ranks: &mut Vec<u32>,
) {
    let width = size.min(prev.len());
    rank_order.clear();
    rank_order.extend(0..width);
    rank_order.sort_by(|&ia, &ib| {
        let (x, y) = (nodes[prev[ia] as usize], nodes[prev[ib] as usize]);
        x.parent_rank
            .cmp(&y.parent_rank)
            .then_with(|| L::compare(outcomes[x.outcome as usize], outcomes[y.outcome as usize]))
    });
    prev_ranks.resize(width, 0);
    let mut r = 0u32;
    for (pos, &idx) in rank_order.iter().enumerate() {
        if pos > 0 {
            let (x, y) = (nodes[prev[rank_order[pos - 1]] as usize], nodes[prev[idx] as usize]);
            if x.parent_rank != y.parent_rank
                || L::compare(outcomes[x.outcome as usize], outcomes[y.outcome as usize]) != Ordering::Equal
            {
                r += 1;
            }
        }
        prev_ranks[idx] = r;
    }
}
