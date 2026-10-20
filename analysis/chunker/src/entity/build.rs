//! Building `ChunkEntities` in one pass over the chunks in text order.
//! A chunk's own entries come from the mention fragments that overlap
//! its fragments, so two clause chunks of one sentence each name only
//! their own entities. Its carried entries come from the keys of the
//! mentions that start before it (see `carry`). Mention fragments are
//! scanned by start with a running maximum of their ends, so a chunk
//! fragment visits the fragments that may reach it, not every mention
//! whose envelope does.

use turbo_analysis::{Ix, Span};
use turbo_analysis_layers::Chunks;

use super::carry::Carry;
use super::header::{ChunkEntities, HeaderPolicy};
use super::keys::EntityKeys;

#[cfg(test)]
thread_local! {
    static OPS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// Counts one step of the build, so tests can prove it linear.
#[inline]
fn op() {
    #[cfg(test)]
    OPS.with(|c| c.set((c.get().0 + 1, c.get().1)));
}

/// Counts one step of a carried lookup: a sentence run or a key visited.
#[inline]
pub(super) fn op_carry() {
    #[cfg(test)]
    OPS.with(|c| c.set((c.get().0 + 1, c.get().1 + 1)));
}

/// The steps counted on this thread so far.
#[cfg(test)]
pub(crate) fn ops() -> u64 {
    OPS.with(|c| c.get().0)
}

/// The carried lookup steps counted on this thread so far.
#[cfg(test)]
pub(crate) fn carry_ops() -> u64 {
    OPS.with(|c| c.get().1)
}

/// Whether `chunks` and `keys` are consistent: every column one entry
/// per chunk, parents in range, mentions by start with their keys and
/// fragments in range.
fn consistent(chunks: &Chunks, keys: &EntityKeys) -> bool {
    let n = chunks.spans.len();
    let (nk, nf) = (keys.len(), keys.fragments.len());
    chunks.parent.len() == n
        && chunks.level.len() == n
        && chunks.parent.iter().all(|p| p.is_none_or(|p| p.get() < n))
        && keys.mentions.windows(2).all(|w| w[0].span.start <= w[1].span.start)
        && keys
            .mentions
            .iter()
            .all(|m| (m.key as usize) < nk && m.fragments.0 <= m.fragments.1 && m.fragments.1 as usize <= nf)
}

/// Builds the header of every row of `chunks` from `keys`. Inconsistent
/// inputs (see `consistent`) give every row an empty header.
pub fn chunk_entities(chunks: &Chunks, keys: &EntityKeys, policy: HeaderPolicy) -> ChunkEntities {
    let n = chunks.spans.len();
    if !consistent(chunks, keys) {
        return ChunkEntities { first: keys.first.clone(), row: vec![0; n + 1], key: Vec::new(), mentions: Vec::new() };
    }
    let ms = &keys.mentions;
    let nk = keys.len();
    // (start, end, mention) of every mention fragment, and reach[j]: the
    // furthest end of fragments 0..=j, so the fragments that start before
    // a chunk fragment and may reach into it are a suffix.
    let mut table: Vec<(u32, u32, u32)> = Vec::new();
    for (i, m) in ms.iter().enumerate() {
        table.extend(keys.fragments_of(m).iter().map(|f| (f.start, f.end, i as u32)));
    }
    table.sort_unstable();
    let mut reach = Vec::with_capacity(table.len());
    let mut furthest = 0;
    for f in &table {
        furthest = furthest.max(f.1);
        reach.push(furthest);
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&c| chunks.spans.get(Ix::new(c)).start);
    let mut carry = Carry::new(ms, nk);
    let (mut fed, mut count, mut first_sentence) = (0, vec![0u32; nk], vec![u32::MAX; nk]);
    let (mut in_chunk, mut seen) = (vec![false; nk], vec![u32::MAX; ms.len()]);
    let (mut here, mut frags) = (Vec::new(), Vec::<Span>::new());
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
            carry.feed(fed, ms[fed].key);
            fed += 1;
        }
        for cf in &frags {
            let hi = table.partition_point(|f| f.0 < cf.end);
            let lo = reach.partition_point(|&r| r <= cf.start).min(hi);
            for &(_, end, i) in &table[lo..hi] {
                op();
                if end <= cf.start || seen[i as usize] == c {
                    continue;
                }
                seen[i as usize] = c;
                let m = &ms[i as usize];
                let k = m.key as usize;
                if count[k] == 0 {
                    here.push(m.key);
                }
                count[k] += 1;
                first_sentence[k] = first_sentence[k].min(m.sentence);
            }
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
            // The keys whose latest mention starts in scope.
            let from = ms.partition_point(|m| m.span.start < lo);
            carry.carried(from, fed, &in_chunk, policy.max_carried, |k| entries.push((c, k, 0)));
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

#[cfg(test)]
#[path = "build_cost_tests.rs"]
mod cost_tests;

#[cfg(test)]
#[path = "build_oracle.rs"]
mod oracle;
