//! Building `ChunkEntities` in one pass over the chunks in text order.
//! A chunk's own entries come from the mentions whose bytes overlap its
//! fragments, so two clause chunks of one sentence each name only their
//! own entities. Its carried entries come from a recency list of keys,
//! fed with every mention that starts before the chunk: a lookup visits
//! only the keys it takes or skips, never the history behind them, so
//! the whole build is linear in chunks and mentions.

use turbo_analysis::{Ix, Span};
use turbo_analysis_layers::Chunks;

use super::header::{ChunkEntities, HeaderPolicy};
use super::keys::EntityKeys;

const NIL: u32 = u32::MAX;

#[cfg(test)]
thread_local! {
    static OPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Counts one step of the build, so tests can prove it linear.
#[inline]
fn op() {
    #[cfg(test)]
    OPS.with(|c| c.set(c.get() + 1));
}

/// The steps counted on this thread so far.
#[cfg(test)]
pub(crate) fn ops() -> u64 {
    OPS.with(|c| c.get())
}

/// Keys by their last mention, latest first: a doubly linked list over
/// key ids, each mention moving its key to the front.
struct Recent {
    next: Vec<u32>,
    prev: Vec<u32>,
    /// The (sentence, start byte) of each key's last mention fed.
    last: Vec<(u32, u32)>,
    linked: Vec<bool>,
    head: u32,
}

impl Recent {
    fn new(n: usize) -> Recent {
        Recent { next: vec![NIL; n], prev: vec![NIL; n], last: vec![(0, 0); n], linked: vec![false; n], head: NIL }
    }

    fn touch(&mut self, k: u32, sentence: u32, at: u32) {
        let i = k as usize;
        if self.linked[i] {
            let (p, n) = (self.prev[i], self.next[i]);
            if p == NIL {
                self.head = n
            } else {
                self.next[p as usize] = n
            }
            if n != NIL {
                self.prev[n as usize] = p;
            }
        }
        self.prev[i] = NIL;
        self.next[i] = self.head;
        if self.head != NIL {
            self.prev[self.head as usize] = k;
        }
        self.head = k;
        self.linked[i] = true;
        self.last[i] = (sentence, at);
    }
}

fn overlaps(a: &[Span], b: &[Span]) -> bool {
    a.iter().any(|x| b.iter().any(|y| x.start < y.end && y.start < x.end))
}

/// Builds the header of every row of `chunks` from `keys`.
pub fn chunk_entities(chunks: &Chunks, keys: &EntityKeys, policy: HeaderPolicy) -> ChunkEntities {
    let ms = &keys.mentions;
    let nk = keys.len();
    // reach[j]: the furthest end of mentions 0..=j, so the mentions that
    // start before a chunk and may reach into it are a suffix.
    let mut reach = Vec::with_capacity(ms.len());
    let mut furthest = 0;
    for m in ms {
        furthest = furthest.max(m.span.end);
        reach.push(furthest);
    }
    let n = chunks.spans.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&c| chunks.spans.get(Ix::new(c)).start);
    let mut recent = Recent::new(nk);
    let (mut fed, mut count, mut first_sentence) = (0, vec![0u32; nk], vec![u32::MAX; nk]);
    let mut in_chunk = vec![false; nk];
    let (mut here, mut group, mut frags) = (Vec::new(), Vec::new(), Vec::new());
    // (chunk, key, mentions), in processing order.
    let mut entries: Vec<(u32, u32, u32)> = Vec::new();
    for c in order {
        op();
        let cx = Ix::<Chunks>::new(c);
        let env = chunks.spans.get(cx);
        frags.clear();
        frags.extend(chunks.spans.fragments(cx));
        while fed < ms.len() && ms[fed].span.start < env.start {
            op();
            recent.touch(ms[fed].key, ms[fed].sentence, ms[fed].span.start);
            fed += 1;
        }
        let hi = ms.partition_point(|m| m.span.start < env.end);
        let lo = reach.partition_point(|&r| r <= env.start).min(hi);
        for m in &ms[lo..hi] {
            op();
            if !overlaps(keys.fragments_of(m), &frags) {
                continue;
            }
            let k = m.key as usize;
            if count[k] == 0 {
                here.push(m.key);
            }
            count[k] += 1;
            first_sentence[k] = first_sentence[k].min(m.sentence);
        }
        here.sort_unstable_by_key(|&k| (first_sentence[k as usize], k));
        for &k in &here {
            entries.push((c, k, count[k as usize]));
            in_chunk[k as usize] = true;
        }
        let scope = match chunks.parent[c as usize] {
            Some(p) => Some(chunks.spans.get(p).start),
            None if chunks.level[c as usize] == 0 => Some(0),
            None => None,
        };
        if let Some(lo) = scope.filter(|_| policy.max_carried > 0) {
            // Walk the keys latest first; a sentence's keys go in key
            // order, so a group is taken whole before it is sorted.
            let (mut taken, mut at, mut sentence) = (0u32, recent.head, None);
            let mut flush = |group: &mut Vec<u32>, taken: &mut u32| {
                group.sort_unstable();
                for &k in group.iter().take((policy.max_carried - *taken) as usize) {
                    entries.push((c, k, 0));
                    *taken += 1;
                }
                group.clear();
            };
            while at != NIL && taken < policy.max_carried {
                op();
                let k = at as usize;
                at = recent.next[k];
                let (s, pos) = recent.last[k];
                if pos < lo {
                    break;
                }
                if in_chunk[k] {
                    continue;
                }
                if sentence != Some(s) {
                    flush(&mut group, &mut taken);
                    sentence = Some(s);
                }
                group.push(k as u32);
            }
            if taken < policy.max_carried {
                flush(&mut group, &mut taken);
            }
            group.clear();
        }
        for &k in &here {
            let k = k as usize;
            (count[k], first_sentence[k], in_chunk[k]) = (0, u32::MAX, false);
        }
        here.clear();
    }
    // Stable: each chunk's entries keep their order.
    entries.sort_by_key(|e| e.0);
    let mut out = ChunkEntities {
        first: keys.first.clone(),
        row: Vec::with_capacity(n + 1),
        key: entries.iter().map(|e| e.1).collect(),
        mentions: entries.iter().map(|e| e.2).collect(),
    };
    out.row.push(0);
    let mut e = 0;
    for c in 0..n as u32 {
        while e < entries.len() && entries[e].0 == c {
            e += 1;
        }
        out.row.push(e as u32);
    }
    out
}

#[cfg(test)]
#[path = "build_tests.rs"]
mod tests;
