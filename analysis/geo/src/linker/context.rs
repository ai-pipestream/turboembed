//! A document's mention context, built once per [`super::GeoEntityLinker::link_mentions`]
//! call: every candidate's Geohash with its decoded centre, and small ids for
//! mention keys and candidate hash lists, so scoring one candidate compares
//! integers and reads cached coordinates instead of copying and decoding, and
//! mentions that score alike share one Geohash scoring.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::ops::Range;

use super::geohash::{GeoHashBinningScorer, decode_geohash};

pub(super) struct MentionContext {
    /// Every mention's candidate Geohashes, mention after mention.
    rows: Vec<String>,
    /// Mention `m`'s rows are `starts[m]..starts[m + 1]`.
    starts: Vec<usize>,
    /// The centre of each row's cell; each distinct hash is decoded once.
    coords: Vec<Option<(f64, f64)>>,
    /// Equal for mentions whose folded surface keys are equal.
    key_ids: Vec<usize>,
    /// Equal for mentions whose candidate hash lists are equal.
    list_ids: Vec<usize>,
}

/// Numbers `items` by first appearance: equal items get equal ids.
fn intern<K: Hash + Eq>(items: impl Iterator<Item = K>) -> Vec<usize> {
    let mut ids = HashMap::new();
    items
        .map(|k| {
            let next = ids.len();
            *ids.entry(k).or_insert(next)
        })
        .collect()
}

impl MentionContext {
    /// `rows` holds each mention's candidate hashes in turn, `starts` where
    /// each mention's begin plus the end, and `keys` each mention's folded key.
    pub(super) fn new(rows: Vec<String>, starts: Vec<usize>, keys: &[String]) -> MentionContext {
        debug_assert_eq!(starts.len(), keys.len() + 1);
        let mut decoded: HashMap<&str, Option<(f64, f64)>> = HashMap::new();
        let coords = rows.iter().map(|h| *decoded.entry(h.as_str()).or_insert_with(|| decode_geohash(h))).collect();
        let list_ids = intern(starts.windows(2).map(|w| &rows[w[0]..w[1]]));
        let key_ids = intern(keys.iter().map(String::as_str));
        MentionContext { rows, starts, coords, key_ids, list_ids }
    }

    fn range(&self, m: usize) -> Range<usize> {
        self.starts[m]..self.starts[m + 1]
    }

    /// Whether mentions `a` and `b` have the same folded key.
    pub(super) fn same_key(&self, a: usize, b: usize) -> bool {
        self.key_ids[a] == self.key_ids[b]
    }

    /// The row of mention `m`'s candidate `c`.
    pub(super) fn row(&self, m: usize, c: usize) -> usize {
        self.starts[m] + c
    }

    /// The Geohash score of every row. Mentions with the same key and the same
    /// hash list skip the same mentions and so score alike: the first of them
    /// is scored and the rest copy it.
    pub(super) fn geohash_scores(&self, scorer: &GeoHashBinningScorer) -> Vec<f64> {
        let mut scored: HashMap<(usize, usize), usize> = HashMap::new();
        let mut out = vec![0.0f64; self.rows.len()];
        for m in 0..self.key_ids.len() {
            let range = self.range(m);
            match scored.entry((self.key_ids[m], self.list_ids[m])) {
                Entry::Occupied(first) => out.copy_within(self.range(*first.get()), range.start),
                Entry::Vacant(slot) => {
                    slot.insert(m);
                    for (c, score) in out[range].iter_mut().enumerate() {
                        *score = self.geohash_score(scorer, m, c);
                    }
                }
            }
        }
        out
    }

    /// The Geohash score of mention `m`'s candidate `c`: the support of every
    /// other mention with candidates, a different key and (when it has more
    /// than one candidate) a different hash list, in mention order.
    fn geohash_score(&self, scorer: &GeoHashBinningScorer, m: usize, c: usize) -> f64 {
        let row = self.row(m, c);
        let (hash, coord) = (self.rows[row].as_str(), self.coords[row]);
        let mut other_mentions = 0usize;
        let mut support_sum = 0.0f64;
        for j in 0..self.key_ids.len() {
            let range = self.range(j);
            if j == m || range.is_empty() || self.same_key(j, m) {
                continue;
            }
            if range.len() > 1 && self.list_ids[j] == self.list_ids[m] {
                continue;
            }
            other_mentions += 1;
            let others = range.map(|r| (self.rows[r].as_str(), self.coords[r]));
            support_sum += scorer.best_support(hash, coord, others);
        }
        scorer.mean_support(other_mentions, support_sum)
    }
}
