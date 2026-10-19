//! `BPETokenizerTrainer.learnMerges` step for step, with the iteration
//! order of its three `HashMap`s, so that of pairs with the same count
//! the one the JVM merges is merged here.
//!
//! The JVM counts the words into `wordFreqs` (`merge`), puts their symbol
//! lists into `vocab` (`put`), and before each merge counts every pair
//! of `vocab`, in its order, into a new `pairCounts` (`merge`), and
//! merges the first pair of the highest count in that map's order. It
//! then merges the pair in each word of `vocab`, in order, into a new map
//! (`merge`), clears `vocab` and puts that map back into it (`putAll`).
//! Each map's order is its keys' bins and, within a bin, the order they
//! were put or, in a bin the JDK made a red-black tree, the tree's order
//! ([`turbo_jvm::hash_map`]), so the order of `vocab` and of
//! `pairCounts` is worked out here from the hash codes of the words, the
//! symbol lists and the pairs, without building the maps. `merge`
//! resizes a full table even for a key already in it, so whether such a
//! call follows the last new key counts too. Each merge
//! costs a pass over the distinct words, as it does on the JVM.

use turbo_jvm::hash_map::{HashMapOrder, Insert, concat_hash, list_hash, pair_hash, spread, string_hash};

use crate::Merge;
use crate::pieces::{FxMap, Pair, Pieces, SymbolId};

use super::symbols;

/// The merges, and whether they are certainly the JVM's: false when the
/// corpus holds a character outside the Basic Multilingual Plane, which
/// the JVM splits into surrogates, or when a tree bin of one of the maps
/// held keys with equal hash codes, which the JVM orders by identity hash
/// code (pairs and symbol lists are not `Comparable`, and the words are
/// handled the same way since only their hashes are seen here). `again` says
/// whether a word occurs again after the last new word was first seen.
pub(super) fn learn_merges(words: &[(&str, u64)], again: bool, num_merges: usize) -> (Vec<Merge>, bool) {
    let mut exact = words.iter().all(|(w, _)| w.chars().all(|c| (c as u32) < 0x1_0000));

    // `wordFreqs`: each word merged in the order first seen.
    let mut spreads: Vec<u32> = words.iter().map(|(w, _)| spread(string_hash(w))).collect();
    let mut order: Vec<u32> = Vec::new();
    let mut tmp_map = HashMapOrder::new();
    exact &= tmp_map.order(&spreads, Insert::Merge { again }, &mut order);

    let mut table = Symbols::default();
    let mut lists: Vec<Vec<SymbolId>> = words.iter().map(|(w, _)| table.intern_all(&symbols(w))).collect();
    let freqs: Vec<u64> = words.iter().map(|&(_, n)| n).collect();
    for (s, l) in spreads.iter_mut().zip(&lists) {
        *s = spread(table.list_hash(l));
    }

    // `vocab`: put in `wordFreqs`' order. Its table is kept from now on,
    // and already holds every word, so `putAll` into it does not pre-size
    // it and is a run of `put`s.
    let mut vocab = HashMapOrder::new();
    let mut scratch = Vec::new();
    exact &= reorder(&mut vocab, &mut order, &spreads, Insert::Put, &mut scratch);

    let mut merges = Vec::new();
    let mut slots: FxMap<Pair, u32> = FxMap::default();
    let mut pairs: Vec<(Pair, u64)> = Vec::new();
    let mut pair_spreads: Vec<u32> = Vec::new();
    for _ in 0..num_merges {
        // `pairCounts`: every pair of every word, in `vocab`'s order.
        slots.clear();
        pairs.clear();
        let mut again = false;
        for &w in &order {
            let freq = freqs[w as usize];
            for s in lists[w as usize].windows(2) {
                let pair = Pair { left: s[0], right: s[1] };
                let next = pairs.len() as u32;
                let slot = *slots.entry(pair).or_insert(next);
                again = slot != next;
                if again {
                    pairs[slot as usize].1 += freq;
                } else {
                    pairs.push((pair, freq));
                }
            }
        }
        if pairs.is_empty() {
            break;
        }
        pair_spreads.clear();
        pair_spreads.extend(pairs.iter().map(|&(p, _)| spread(table.pair_hash(p))));
        tmp_map.reset();
        exact &= tmp_map.order(&pair_spreads, Insert::Merge { again }, &mut scratch);
        // The first pair of the highest count in the map's order.
        let mut best: Option<(Pair, u64)> = None;
        for &k in &scratch {
            let (pair, count) = pairs[k as usize];
            if best.is_none_or(|(_, c)| count > c) {
                best = Some((pair, count));
            }
        }
        let (pair, count) = best.expect("a pair is counted");
        merges.push(Merge {
            left: table.pieces.text(pair.left).to_owned(),
            right: table.pieces.text(pair.right).to_owned(),
            count,
        });

        let merged = table.join(pair);
        for (w, list) in lists.iter_mut().enumerate() {
            if merge_in(list, pair, merged) {
                spreads[w] = spread(table.list_hash(list));
            }
        }
        // The merged words into a new map (`merge`), in `vocab`'s order,
        // then that map's order put back into `vocab` (`putAll`).
        // Each word is a new key there: no two words merge into one list.
        let merge = Insert::Merge { again: false };
        tmp_map.reset();
        exact &= reorder(&mut tmp_map, &mut order, &spreads, merge, &mut scratch);
        exact &= reorder(&mut vocab, &mut order, &spreads, Insert::Put, &mut scratch);
    }
    (merges, exact)
}

/// Puts the words of `order`, in that order, into `map` and leaves in
/// `order` the order the map iterates them.
fn reorder(
    map: &mut HashMapOrder,
    order: &mut Vec<u32>,
    spreads: &[u32],
    insert: Insert,
    scratch: &mut Vec<u32>,
) -> bool {
    let exact = map.order_keys(spreads, order, insert, scratch);
    std::mem::swap(order, scratch);
    exact
}

/// `applyMerge` in place: `pair` merged into `merged` left to right.
/// Whether the list changed.
fn merge_in(list: &mut Vec<SymbolId>, pair: Pair, merged: SymbolId) -> bool {
    let n = list.len();
    let mut read = 0;
    let mut write = 0;
    while read < n {
        if read + 1 < n && list[read] == pair.left && list[read + 1] == pair.right {
            list[write] = merged;
            read += 2;
        } else {
            list[write] = list[read];
            read += 1;
        }
        write += 1;
    }
    let changed = write < n;
    list.truncate(write);
    changed
}

/// The symbols, each with its `String.hashCode` and length in UTF-16
/// units.
#[derive(Default)]
struct Symbols {
    pieces: Pieces,
    hashes: Vec<i32>,
    units: Vec<usize>,
}

impl Symbols {
    fn intern(&mut self, text: &str, hash: impl FnOnce() -> i32, units: usize) -> SymbolId {
        let id = self.pieces.intern(text);
        if id.index() == self.hashes.len() {
            self.hashes.push(hash());
            self.units.push(units);
        }
        id
    }

    fn intern_all(&mut self, texts: &[String]) -> Vec<SymbolId> {
        texts.iter().map(|t| self.intern(t, || string_hash(t), t.encode_utf16().count())).collect()
    }

    /// The symbol `pair` makes.
    fn join(&mut self, pair: Pair) -> SymbolId {
        let (l, r) = (pair.left.index(), pair.right.index());
        let text = format!("{}{}", self.pieces.text(pair.left), self.pieces.text(pair.right));
        let (hl, hr, ur) = (self.hashes[l], self.hashes[r], self.units[r]);
        let units = self.units[l] + ur;
        self.intern(&text, || concat_hash(hl, hr, ur), units)
    }

    /// `List.hashCode` of a word's symbols.
    fn list_hash(&self, list: &[SymbolId]) -> i32 {
        list_hash(list.iter().map(|s| self.hashes[s.index()]))
    }

    /// The hash code of the record `SymbolPair(left, right)`.
    fn pair_hash(&self, pair: Pair) -> i32 {
        pair_hash(self.hashes[pair.left.index()], self.hashes[pair.right.index()])
    }
}
