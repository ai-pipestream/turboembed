//! The merge loop every trainer shares: the words of a corpus as pieces,
//! the count of every pair of adjacent pieces weighted by how often each
//! word occurs, and a queue that gives the pair to merge next.
//!
//! Counts are kept up to date as pairs are merged, never counted again
//! from the words: a merge visits only the words that hold the pair, and
//! in each only the pieces on either side of each place it is merged.
//! The queue is a binary heap of (count, key, pair) entries that may be
//! out of date. An entry is pushed when a pair is first seen and every
//! time its count goes up; when a count goes down the old entry stays,
//! and is found out when it reaches the top: it is pushed again with the
//! count the pair has now. So every pair has an entry at its count or
//! above it, and the first entry at the top whose count is its pair's
//! count is the pair to merge.
//!
//! Of pairs with the same count, the one with the smaller [`Order`] key
//! merges first, and of equal keys the smaller pair of ids. A pair the
//! order does not allow is counted but never offered.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::pieces::{FxMap, Pair, Pieces, SymbolId};

/// How a trainer orders pairs with the same count, and which pairs it
/// may merge at all.
pub trait Order {
    type Key: Ord + Clone;

    /// The key of `pair`: of two pairs with the same count, the one with
    /// the smaller key is merged first. It must depend on nothing but the
    /// two pieces, since it is worked out once per pair.
    fn key(&self, pieces: &Pieces, pair: Pair) -> Self::Key;

    /// Whether `pair` may be merged. Worked out once per pair.
    fn allows(&self, _pieces: &Pieces, _pair: Pair) -> bool {
        true
    }
}

/// A pair to merge and how often it occurs in the corpus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub pair: Pair,
    pub count: u64,
}

/// What is known of one pair.
struct PairStat<K> {
    count: u64,
    /// The words the pair has been seen in. A word can be here twice or
    /// no longer hold the pair; a merge checks each word it visits.
    words: Vec<u32>,
    /// None when the order does not allow the pair, or the trainer has
    /// set it aside (`discard`).
    key: Option<K>,
}

/// One queue entry. The heap is a max-heap, so `Ord` puts the entry to
/// merge first as the greatest: the higher count, then the smaller key,
/// then the smaller pair.
struct Entry<K> {
    count: u64,
    key: K,
    pair: Pair,
}

impl<K: Ord> Ord for Entry<K> {
    fn cmp(&self, o: &Self) -> Ordering {
        self.count.cmp(&o.count).then_with(|| o.key.cmp(&self.key)).then_with(|| o.pair.cmp(&self.pair))
    }
}

impl<K: Ord> PartialOrd for Entry<K> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl<K: Ord> PartialEq for Entry<K> {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl<K: Ord> Eq for Entry<K> {}

/// The words of a corpus being merged.
pub struct Merger<O: Order> {
    order: O,
    pieces: Pieces,
    words: Vec<Vec<SymbolId>>,
    counts: Vec<u64>,
    pairs: FxMap<Pair, PairStat<O::Key>>,
    heap: BinaryHeap<Entry<O::Key>>,
    /// For each word, the number of the last merge that visited it, so a
    /// merge visits a word once however often the pair's list names it.
    visited: Vec<u32>,
    merges: u32,
    /// The pairs whose count went up in the merge being made.
    raised: Vec<Pair>,
}

impl<O: Order> Merger<O> {
    /// The words, each its pieces and how often it occurs, with their
    /// pairs counted. `pieces` holds every piece the words use.
    pub fn new(order: O, pieces: Pieces, words: Vec<(Vec<SymbolId>, u64)>) -> Self {
        let n = words.len();
        let mut m = Merger {
            order,
            pieces,
            words: Vec::with_capacity(n),
            counts: Vec::with_capacity(n),
            pairs: FxMap::default(),
            heap: BinaryHeap::new(),
            visited: vec![0; n],
            merges: 0,
            raised: Vec::new(),
        };
        for (w, (symbols, count)) in words.into_iter().enumerate() {
            for p in symbols.windows(2) {
                m.add(Pair { left: p[0], right: p[1] }, w as u32, count);
            }
            m.words.push(symbols);
            m.counts.push(count);
        }
        m.raised.clear();
        let mut entries = Vec::with_capacity(m.pairs.len());
        for (&pair, stat) in &m.pairs {
            if let Some(key) = &stat.key {
                entries.push(Entry { count: stat.count, key: key.clone(), pair });
            }
        }
        m.heap = BinaryHeap::from(entries);
        m
    }

    pub fn pieces(&self) -> &Pieces {
        &self.pieces
    }

    /// Adds a piece that no word holds yet, such as a special token, and
    /// gives its id.
    pub fn intern(&mut self, text: &str) -> SymbolId {
        self.pieces.intern(text)
    }

    /// The words as they are now, each with how often it occurs.
    pub fn words(&self) -> impl Iterator<Item = (&[SymbolId], u64)> {
        self.words.iter().map(|w| &w[..]).zip(self.counts.iter().copied())
    }

    /// How often `pair` occurs in the words now.
    pub fn count(&self, pair: Pair) -> u64 {
        self.pairs.get(&pair).map_or(0, |s| s.count)
    }

    /// Every pair that occurs in the words now, with its count, in no
    /// particular order.
    pub fn counts(&self) -> impl Iterator<Item = (Pair, u64)> + '_ {
        self.pairs.iter().filter(|(_, s)| s.count > 0).map(|(&p, s)| (p, s.count))
    }

    /// The pairs that occurred and occur no longer, that the order allows
    /// and that were neither merged nor set aside, smallest key first,
    /// then smallest pair. A pair a merge made in a word and took away
    /// again further on in the same word (`ab a`, while `a b` is merged
    /// in `x a b a b`) is one of them, as it is to SentencePiece.
    pub fn spent(&self) -> Vec<Pair> {
        let mut v: Vec<(&O::Key, Pair)> = self
            .pairs
            .iter()
            .filter(|(_, s)| s.count == 0)
            .filter_map(|(&p, s)| s.key.as_ref().map(|k| (k, p)))
            .collect();
        v.sort_unstable();
        v.into_iter().map(|(_, p)| p).collect()
    }

    /// Every piece made, by id.
    pub fn into_pieces(self) -> Pieces {
        self.pieces
    }

    /// The pair to merge next, with its count, or None when no pair is
    /// left that the order allows. It stays the next until it is merged
    /// or discarded.
    pub fn best(&mut self) -> Option<Candidate> {
        while let Some(top) = self.heap.peek() {
            let pair = top.pair;
            match self.pairs.get(&pair) {
                Some(stat) if stat.key.is_some() && stat.count == top.count => {
                    return Some(Candidate { pair, count: stat.count });
                }
                Some(stat) if stat.key.is_some() && stat.count > 0 => {
                    let count = stat.count;
                    let mut e = self.heap.pop().expect("peeked");
                    e.count = count;
                    self.heap.push(e);
                }
                _ => {
                    self.heap.pop();
                }
            }
        }
        None
    }

    /// Sets `pair` aside: it is never offered again, though its pieces
    /// stay side by side in the words.
    pub fn discard(&mut self, pair: Pair) {
        if let Some(stat) = self.pairs.get_mut(&pair) {
            stat.key = None;
        }
    }

    /// Merges `pair` in every word that holds it, left to right in each
    /// word, and gives the id of the merged piece: the two pieces' texts
    /// joined, added if it is new.
    pub fn merge(&mut self, pair: Pair) -> SymbolId {
        let text = format!("{}{}", self.pieces.text(pair.left), self.pieces.text(pair.right));
        let merged = self.pieces.intern(&text);
        let Some(stat) = self.pairs.remove(&pair) else { return merged };
        self.merges += 1;
        for &w in &stat.words {
            if std::mem::replace(&mut self.visited[w as usize], self.merges) != self.merges {
                self.merge_word(w, pair, merged);
            }
        }
        self.raised.sort_unstable();
        self.raised.dedup();
        for &p in &self.raised {
            if let Some(stat) = self.pairs.get(&p)
                && let Some(key) = &stat.key
                && stat.count > 0
            {
                self.heap.push(Entry { count: stat.count, key: key.clone(), pair: p });
            }
        }
        self.raised.clear();
        merged
    }

    /// Merges `pair` into `merged` in word `w`, moving the counts of the
    /// pairs on either side of each place it is merged. One pass: the
    /// word is written back over itself, so a long word that holds the
    /// pair many times costs its length, not its length squared.
    fn merge_word(&mut self, w: u32, pair: Pair, merged: SymbolId) {
        let count = self.counts[w as usize];
        let mut symbols = std::mem::take(&mut self.words[w as usize]);
        let n = symbols.len();
        let mut read = 0;
        let mut write = 0;
        while read < n {
            if read + 1 < n && symbols[read] == pair.left && symbols[read + 1] == pair.right {
                if write > 0 {
                    let before = symbols[write - 1];
                    self.remove(Pair { left: before, right: pair.left }, pair, count);
                    self.add(Pair { left: before, right: merged }, w, count);
                }
                if let Some(&after) = symbols.get(read + 2) {
                    self.remove(Pair { left: pair.right, right: after }, pair, count);
                    self.add(Pair { left: merged, right: after }, w, count);
                }
                symbols[write] = merged;
                read += 2;
            } else {
                symbols[write] = symbols[read];
                read += 1;
            }
            write += 1;
        }
        symbols.truncate(write);
        self.words[w as usize] = symbols;
    }

    /// One more occurrence of `pair` in word `w`, which occurs `count`
    /// times.
    fn add(&mut self, pair: Pair, w: u32, count: u64) {
        let stat = match self.pairs.entry(pair) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let key = self.order.allows(&self.pieces, pair).then(|| self.order.key(&self.pieces, pair));
                e.insert(PairStat { count: 0, words: Vec::new(), key })
            }
        };
        stat.count += count;
        if stat.words.last() != Some(&w) {
            stat.words.push(w);
        }
        self.raised.push(pair);
    }

    /// One occurrence fewer of `pair`, unless it is `merging`, the pair
    /// being merged, whose counts are gone with it.
    fn remove(&mut self, pair: Pair, merging: Pair, count: u64) {
        if pair == merging {
            return;
        }
        let stat = self.pairs.get_mut(&pair).expect("a pair in a word is counted");
        stat.count = stat.count.checked_sub(count).expect("a count covers every occurrence");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ties go to the smaller pair of ids.
    struct ById;

    impl Order for ById {
        type Key = ();
        fn key(&self, _: &Pieces, _: Pair) {}
    }

    fn merger(words: &[(&str, u64)]) -> Merger<ById> {
        let mut pieces = Pieces::default();
        let mut chars: Vec<char> = words.iter().flat_map(|(w, _)| w.chars()).collect();
        chars.sort_unstable();
        chars.dedup();
        for c in chars {
            pieces.intern(&c.to_string());
        }
        let words =
            words.iter().map(|&(w, n)| (w.chars().map(|c| pieces.get(&c.to_string()).unwrap()).collect(), n)).collect();
        Merger::new(ById, pieces, words)
    }

    /// Every pair's count, counted again from the words.
    fn recount(m: &Merger<ById>) -> Vec<(Pair, u64)> {
        let mut c: FxMap<Pair, u64> = FxMap::default();
        for (w, n) in m.words() {
            for p in w.windows(2) {
                *c.entry(Pair { left: p[0], right: p[1] }).or_default() += n;
            }
        }
        let mut v: Vec<_> = c.into_iter().collect();
        v.sort_unstable();
        v
    }

    fn counts(m: &Merger<ById>) -> Vec<(Pair, u64)> {
        let mut v: Vec<_> = m.pairs.iter().filter(|(_, s)| s.count > 0).map(|(&p, s)| (p, s.count)).collect();
        v.sort_unstable();
        v
    }

    fn text(m: &Merger<ById>, w: usize) -> Vec<&str> {
        m.words[w].iter().map(|&s| m.pieces.text(s)).collect()
    }

    #[test]
    fn overlapping_pairs_merge_left_to_right_and_keep_their_counts() {
        let mut m = merger(&[("aaaa", 1), ("aaa", 2), ("abab", 1)]);
        let best = m.best().unwrap();
        // aa: 3 in aaaa, 2 in each of the two aaa.
        assert_eq!((m.pieces.text(best.pair.left), m.pieces.text(best.pair.right), best.count), ("a", "a", 7));
        m.merge(best.pair);
        assert_eq!(text(&m, 0), ["aa", "aa"]);
        assert_eq!(text(&m, 1), ["aa", "a"]);
        assert_eq!(text(&m, 2), ["a", "b", "a", "b"]);
        assert_eq!(counts(&m), recount(&m));
        let best = m.best().unwrap();
        m.merge(best.pair);
        assert_eq!(counts(&m), recount(&m));
    }

    #[test]
    fn a_long_run_of_one_letter_merges_in_one_pass() {
        let run = "a".repeat(100_001);
        let mut m = merger(&[(run.as_str(), 2), ("ba", 1)]);
        for _ in 0..4 {
            let best = m.best().unwrap();
            m.merge(best.pair);
            assert_eq!(counts(&m), recount(&m));
        }
        // 100001 letters: 6250 pieces of 16 and one a.
        assert_eq!(m.words[0].len(), 6251);
    }

    #[test]
    fn a_discarded_pair_is_not_offered_again() {
        let mut m = merger(&[("ab", 5), ("cd", 3)]);
        let best = m.best().unwrap();
        assert_eq!(best.count, 5);
        m.discard(best.pair);
        assert_eq!(m.best().unwrap().count, 3);
        assert_eq!(text(&m, 0), ["a", "b"]);
    }

    #[test]
    fn equal_counts_go_to_the_smaller_ids() {
        let mut m = merger(&[("ba", 2), ("ab", 2)]);
        let best = m.best().unwrap();
        assert_eq!((m.pieces.text(best.pair.left), m.pieces.text(best.pair.right)), ("a", "b"));
    }
}
