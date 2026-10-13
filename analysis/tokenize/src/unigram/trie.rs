//! The word trie of a unigram lexicon: one node per prefix of a listed
//! word, one edge per character, and on the node that ends a word, that
//! word's score.
//!
//! The layout is for the segmenter's walk, which steps from a node to its
//! child on the next character of the text until there is none:
//!
//! - The root's edges on characters of the Basic Multilingual Plane are a
//!   table indexed by the character, so the first step is one load.
//! - Every other edge is a slot of an open addressing hash table keyed by
//!   (parent, character), probed linearly.
//! - A step yields a [`Step`]: the child's id, whether a word ends there,
//!   and a 32-bit filter of the characters the child's own edges are on
//!   (one bit per character, chosen by a hash). The next step probes only
//!   when the next character's bit is set, so most walks end without a
//!   probe, and a filter of zero means the child has no edges.
//! - The scores are an array by node id, read only for a node that ends a
//!   word.
//!
//! OpenNLP's `WordTrie` has an edge per UTF-16 code unit, each node's
//! edges sorted and found by binary search. The words found are the same:
//! a lexicon word is whole characters and the text is walked from a
//! character boundary.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// The index of a node; the root is 0.
pub(crate) type NodeId = u32;

/// Bits of a key holding the character: all of Unicode.
const CHAR_BITS: u32 = 21;

/// The bit of a child id that says a word ends at the child.
const WORD: u32 = 1 << 31;

/// The most nodes a trie may have, so that an id fits beside [`WORD`].
const MAX_NODES: usize = 1 << 31;

/// Characters with a root edge in the direct table.
const ROOT_TABLE: usize = 0x1_0000;

const MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

/// One step of a walk: the node reached and what the walk needs to know
/// about it without another load.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Step {
    /// The node id, with [`WORD`] set when a word ends at the node.
    child: u32,
    /// The filter of the characters the node's edges are on; zero when it
    /// has none.
    filter: u32,
}

impl Step {
    /// Whether a word ends at the node.
    #[inline(always)]
    pub(crate) fn is_word(self) -> bool {
        self.child & WORD != 0
    }

    /// The node's id.
    #[inline(always)]
    fn node(self) -> NodeId {
        self.child & !WORD
    }

    /// False when the node has no edge on `c`; true when it may have one.
    #[inline(always)]
    fn may_continue(self, c: char) -> bool {
        self.filter & filter_bit(c) != 0
    }
}

/// The filter bit of `c`: one of 32, by a multiplicative hash.
#[inline(always)]
fn filter_bit(c: char) -> u32 {
    1 << ((c as u32).wrapping_mul(0x9E37_79B1) >> 27)
}

/// The key of the edge from `parent` on `c`: never zero, so zero marks a
/// free slot. `parent < 2^31` and `c < 2^21`, so it fits in 53 bits.
#[inline(always)]
fn key(parent: NodeId, c: char) -> u64 {
    ((u64::from(parent) << CHAR_BITS) | u64::from(c as u32)) + 1
}

/// One slot of the edge table: the key (0 when free) and the step to the
/// child.
#[derive(Copy, Clone, Debug)]
struct Slot {
    key: u64,
    step: Step,
}

const FREE: Slot = Slot { key: 0, step: Step { child: 0, filter: 0 } };

/// A multiplicative hash of a `u64` key, for the map that numbers the
/// nodes while the trie is built.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(b)).wrapping_mul(MULTIPLIER);
        }
    }

    fn write_u64(&mut self, k: u64) {
        // The table indexes by the low bits and filters by the high ones,
        // so the product's high half is folded into both.
        let h = k.wrapping_mul(MULTIPLIER);
        self.0 = h ^ (h >> 32);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WordTrie {
    /// The root's step on each character of the Basic Multilingual Plane;
    /// a child of 0 (the root) means no edge.
    root: Box<[Step]>,
    /// The other edges, the root's on characters above the Basic
    /// Multilingual Plane included.
    slots: Box<[Slot]>,
    /// `64 - log2(slots.len())`.
    shift: u32,
    /// The score of each node, NaN where no word ends.
    scores: Box<[f64]>,
    words: usize,
}

/// A trie being built: the words' paths and each word's count. A word
/// listed twice is one node whose counts are summed.
pub(crate) struct TrieBuilder {
    /// Every edge as (parent, char, child); node ids by first appearance.
    edges: Vec<(NodeId, char, NodeId)>,
    /// The summed count of the word ending at each node, 0 where none does.
    counts: Vec<i64>,
    /// The root's child on each character of the Basic Multilingual Plane,
    /// 0 for none.
    root: Box<[NodeId]>,
    /// The other edges, by key.
    index: HashMap<u64, NodeId, BuildHasherDefault<KeyHasher>>,
    words: usize,
}

impl TrieBuilder {
    /// A builder with room for about `words` words.
    pub(crate) fn with_capacity(words: usize) -> TrieBuilder {
        // A Chinese lexicon has about one and a half nodes per word.
        let guess = words + words / 2;
        let mut counts = Vec::with_capacity(guess + 1);
        counts.push(0);
        TrieBuilder {
            edges: Vec::with_capacity(guess),
            counts,
            root: vec![0; ROOT_TABLE].into_boxed_slice(),
            index: HashMap::with_capacity_and_hasher(guess, BuildHasherDefault::default()),
            words: 0,
        }
    }

    /// The node of `word`, made with its path if it is new. The empty
    /// word is the root, which the segmenter never scores.
    ///
    /// # Panics
    ///
    /// When the trie would have more than 2^31 nodes, which ten million
    /// words (a lexicon's default limit) reach only if they average over
    /// 200 characters.
    pub(crate) fn node(&mut self, word: &str) -> NodeId {
        let mut node: NodeId = 0;
        for c in word.chars() {
            let existing = if node == 0 && (c as usize) < ROOT_TABLE {
                Some(self.root[c as usize]).filter(|&n| n != 0)
            } else {
                self.index.get(&key(node, c)).copied()
            };
            node = match existing {
                Some(child) => child,
                None => {
                    let child = self.counts.len();
                    assert!(child < MAX_NODES, "a unigram lexicon with more than 2^31 prefixes");
                    let child = child as NodeId;
                    self.counts.push(0);
                    self.edges.push((node, c, child));
                    if node == 0 && (c as usize) < ROOT_TABLE {
                        self.root[c as usize] = child;
                    } else {
                        self.index.insert(key(node, c), child);
                    }
                    child
                }
            };
        }
        node
    }

    /// The summed count of the word at `node`; 0 when no word ends there.
    pub(crate) fn count(&self, node: NodeId) -> i64 {
        self.counts[node as usize]
    }

    /// Adds `count`, which is positive, to the word at `node`. The caller
    /// keeps the total of all counts, which bounds every word's sum, from
    /// overflowing.
    pub(crate) fn add(&mut self, node: NodeId, count: i64) {
        debug_assert!(count > 0);
        let c = &mut self.counts[node as usize];
        if *c == 0 {
            self.words += 1;
        }
        *c += count;
    }

    /// The number of distinct words.
    pub(crate) fn words(&self) -> usize {
        self.words
    }

    /// The trie, each word scored by `score` of its count.
    pub(crate) fn finish(self, mut score: impl FnMut(i64) -> f64) -> WordTrie {
        let TrieBuilder { edges, counts, index, words, .. } = self;
        drop(index);
        let scores: Box<[f64]> = counts.iter().map(|&c| if c == 0 { f64::NAN } else { score(c) }).collect();
        let mut filters = vec![0u32; scores.len()];
        for &(parent, c, _) in &edges {
            filters[parent as usize] |= filter_bit(c);
        }
        let step = |child: NodeId| Step {
            child: child | if counts[child as usize] == 0 { 0 } else { WORD },
            filter: filters[child as usize],
        };
        // At most half full, so a probe ends after one or two slots.
        let len = (edges.len() * 2).next_power_of_two().max(2);
        let shift = 64 - len.trailing_zeros();
        let mut slots = vec![FREE; len].into_boxed_slice();
        let mut root = vec![Step { child: 0, filter: 0 }; ROOT_TABLE].into_boxed_slice();
        for &(parent, c, child) in &edges {
            if parent == 0 && (c as usize) < ROOT_TABLE {
                root[c as usize] = step(child);
                continue;
            }
            let k = key(parent, c);
            let mut at = (k.wrapping_mul(MULTIPLIER) >> shift) as usize;
            while slots[at].key != 0 {
                at = (at + 1) & (len - 1);
            }
            slots[at] = Slot { key: k, step: step(child) };
        }
        WordTrie { root, slots, shift, scores, words }
    }
}

impl WordTrie {
    /// The root's step on `c`.
    #[inline(always)]
    pub(crate) fn first(&self, c: char) -> Option<Step> {
        if (c as usize) < ROOT_TABLE {
            let s = self.root[c as usize];
            return (s.child != 0).then_some(s);
        }
        self.probe(0, c)
    }

    /// The step from `from` on `c`.
    #[inline(always)]
    pub(crate) fn next(&self, from: Step, c: char) -> Option<Step> {
        if !from.may_continue(c) {
            return None;
        }
        self.probe(from.node(), c)
    }

    #[inline(always)]
    fn probe(&self, parent: NodeId, c: char) -> Option<Step> {
        let k = key(parent, c);
        let mask = self.slots.len() - 1;
        let mut at = (k.wrapping_mul(MULTIPLIER) >> self.shift) as usize;
        loop {
            let slot = self.slots[at];
            if slot.key == k {
                return Some(slot.step);
            }
            if slot.key == 0 {
                return None;
            }
            at = (at + 1) & mask;
        }
    }

    /// The score of the word that ends at the step's node; NaN when none
    /// does.
    #[inline(always)]
    pub(crate) fn score(&self, step: Step) -> f64 {
        self.scores[step.node() as usize]
    }

    /// The score of `word`, if it is listed.
    pub(crate) fn get(&self, word: &str) -> Option<f64> {
        let mut chars = word.chars();
        let mut step = self.first(chars.next()?)?;
        for c in chars {
            step = self.next(step, c)?;
        }
        step.is_word().then(|| self.score(step))
    }

    /// The number of words.
    pub(crate) fn words(&self) -> usize {
        self.words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trie of `words` with their counts, each word scored minus its
    /// count.
    fn build<'a>(words: impl IntoIterator<Item = (&'a str, i64)>) -> WordTrie {
        let mut b = TrieBuilder::with_capacity(4);
        for (w, c) in words {
            let n = b.node(w);
            b.add(n, c);
        }
        b.finish(|c| -(c as f64))
    }

    #[test]
    fn finds_exactly_the_words() {
        let words = [("ab", 1), ("a", 2), ("abc", 3), ("b", 4), ("\u{20bb7}x", 5), ("\u{6211}", 6), ("\u{20bb7}", 7)];
        let trie = build(words);
        assert_eq!(trie.words(), 7);
        for (w, c) in words {
            assert_eq!(trie.get(w), Some(-(c as f64)), "{w}");
        }
        for w in ["", "abcd", "c", "x", "ba", "\u{6211}\u{6211}", "\u{20bb7}y"] {
            assert_eq!(trie.get(w), None, "{w}");
        }
    }

    #[test]
    fn a_prefix_that_is_no_word_has_no_score() {
        let trie = build([("abc", 1)]);
        let a = trie.first('a').unwrap();
        assert!(!a.is_word());
        assert!(trie.score(a).is_nan());
        let b = trie.next(a, 'b').unwrap();
        assert!(!b.is_word() && b.may_continue('c'));
        let c = trie.next(b, 'c').unwrap();
        assert!(c.is_word());
        assert_eq!(trie.score(c), -1.0);
        assert_eq!(c.filter, 0, "a leaf has no edges");
        assert_eq!(trie.next(c, 'd'), None);
        assert_eq!(trie.first('b'), None);
    }

    #[test]
    fn the_filter_has_the_bit_of_every_edge() {
        let words: Vec<String> =
            (0x4e00u32..0x4e00 + 500).map(|c| format!("x{}", char::from_u32(c).unwrap())).collect();
        let trie = build(words.iter().map(|w| (w.as_str(), 1)));
        let x = trie.first('x').unwrap();
        for w in &words {
            let c = w.chars().nth(1).unwrap();
            assert!(x.may_continue(c));
            assert!(trie.next(x, c).unwrap().is_word());
        }
        assert_eq!(x.filter, u32::MAX, "500 characters set every bit");
        assert_eq!(trie.words(), 500);
        // A character whose bit is set but which has no edge is probed and
        // not found.
        let missing = (0x5000u32..).filter_map(char::from_u32).find(|&c| x.may_continue(c)).unwrap();
        assert_eq!(trie.next(x, missing), None);
    }
}
