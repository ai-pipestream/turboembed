//! SentencePiece's BPE (bpe_model.cc): the normalized text split into
//! characters, a user-defined piece kept whole, then the adjacent pair
//! whose join is the piece of highest score merged, again and again. A
//! merged piece of type UNUSED is split back into the two it was made
//! of.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasherDefault;

use crate::tokenizer::Fx;

use super::normalizer::{Normalizer, one_char_len};
use super::proto::PieceType;
use super::{Segment, SentencePiece};

/// The library stops splitting an UNUSED piece this deep
/// (`kMaxBpeResegmentDepth`).
const MAX_RESEGMENT_DEPTH: u32 = 100;

#[derive(Clone, Copy)]
pub(crate) struct Symbol {
    prev: i32,
    next: i32,
    /// A user-defined piece, never merged.
    freeze: bool,
    start: u32,
    /// 0 once merged into the symbol before it.
    len: u32,
}

/// How each UNUSED piece BPE made was last made: the length of its left
/// part, by the piece's id. The library keys its `rev_merge` by the
/// piece's text, which is the same: one text is one piece.
pub(crate) type RevMerge = HashMap<u32, u32, BuildHasherDefault<Fx>>;

/// A pair of adjacent symbols that joins into a piece.
#[derive(Clone, Copy)]
pub(crate) struct Pair {
    /// The piece's score, as an integer that orders as the library's
    /// comparator orders scores.
    key: i32,
    left: u32,
    right: u32,
    /// The join's length when the pair was made: a pair whose symbols
    /// have changed since is passed over.
    size: u32,
}

impl Pair {
    /// The library compares the scores' bit patterns: two negative
    /// scores in reverse of their integer order, a negative one under one
    /// that is not, two others in integer order. Flipping every bit but
    /// the sign of a negative pattern gives that order as plain integers.
    fn key(score: f32) -> i32 {
        let bits = score.to_bits() as i32;
        bits ^ ((bits >> 31) & 0x7fff_ffff)
    }
}

impl Ord for Pair {
    /// The highest score first, then the leftmost pair.
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key).then(other.left.cmp(&self.left))
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Pair {}

impl SentencePiece {
    /// The id of the piece `piece` merges into, when it may be merged:
    /// a piece of type NORMAL, USER_DEFINED or UNUSED.
    #[inline]
    fn mergeable(&self, piece: &[u8]) -> Option<u32> {
        let id = self.trie.get(piece)?;
        matches!(self.kinds[id as usize], PieceType::Normal | PieceType::UserDefined | PieceType::Unused).then_some(id)
    }

    /// The pieces BPE makes of `normalized`, in order.
    ///
    /// When no piece of the model holds a space that is not part of the
    /// run of spaces it starts with (`split_words`, worked out on load),
    /// no merge can cross from a character other than a space into the
    /// space after it, so the text is merged word by word there: the
    /// merges inside one word do not depend on the others, and the result
    /// is the library's, with a heap the size of a word rather than of the
    /// whole text.
    pub(crate) fn bpe(
        &self,
        normalized: &[u8],
        symbols: &mut Vec<Symbol>,
        agenda: &mut BinaryHeap<Pair>,
        rev_merge: &mut RevMerge,
        segments: &mut Vec<Segment>,
    ) {
        segments.clear();
        symbols.clear();
        rev_merge.clear();
        let mut at = 0usize;
        while at < normalized.len() {
            let rest = &normalized[at..];
            if self.split_words
                && !symbols.is_empty()
                && rest.starts_with(self.space)
                && !normalized[..at].ends_with(self.space)
            {
                self.merge(normalized, symbols, agenda, rev_merge, segments);
                symbols.clear();
            }
            let user = self.user.as_ref().and_then(|u| Normalizer::user_match(u, rest));
            let len = user.unwrap_or_else(|| one_char_len(rest[0]).min(rest.len()));
            let index = symbols.len() as i32;
            symbols.push(Symbol {
                prev: index - 1,
                next: index + 1,
                freeze: user.is_some(),
                start: at as u32,
                len: len as u32,
            });
            at += len;
        }
        self.merge(normalized, symbols, agenda, rev_merge, segments);
    }

    /// BPE over `symbols`, a run of the normalized text's characters, as
    /// the library's `Encode` runs it over the whole text; the pieces go
    /// to `segments`.
    fn merge(
        &self,
        normalized: &[u8],
        symbols: &mut [Symbol],
        agenda: &mut BinaryHeap<Pair>,
        rev_merge: &mut RevMerge,
        segments: &mut Vec<Segment>,
    ) {
        let Some(last) = symbols.last_mut() else { return };
        last.next = -1;
        agenda.clear();
        let mut add = |symbols: &[Symbol], agenda: &mut BinaryHeap<Pair>, left: i32, right: i32| {
            if left < 0 || right < 0 {
                return;
            }
            let (l, r) = (symbols[left as usize], symbols[right as usize]);
            if l.freeze || r.freeze {
                return;
            }
            let joined = &normalized[l.start as usize..(r.start + r.len) as usize];
            let Some(id) = self.mergeable(joined) else { return };
            agenda.push(Pair {
                key: Pair::key(self.scores[id as usize]),
                left: left as u32,
                right: right as u32,
                size: joined.len() as u32,
            });
            if self.kinds[id as usize] == PieceType::Unused {
                rev_merge.insert(id, l.len);
            }
        };
        for right in 1..symbols.len() {
            add(symbols, agenda, right as i32 - 1, right as i32);
        }

        while let Some(top) = agenda.pop() {
            let (l, r) = (symbols[top.left as usize], symbols[top.right as usize]);
            if l.len == 0 || r.len == 0 || l.len + r.len != top.size {
                continue;
            }
            let left = &mut symbols[top.left as usize];
            left.len += r.len;
            left.next = r.next;
            let prev = left.prev;
            let next = left.next;
            if r.next >= 0 {
                symbols[r.next as usize].prev = top.left as i32;
            }
            symbols[top.right as usize].len = 0;
            add(symbols, agenda, prev, top.left as i32);
            add(symbols, agenda, top.left as i32, next);
        }

        let mut index = 0i32;
        while index >= 0 {
            let s = symbols[index as usize];
            self.resegment(normalized, s.start as usize, s.len as usize, rev_merge, 0, segments);
            index = s.next;
        }
    }

    /// The piece at normalized[start..start + len], split back into the
    /// pieces it was merged from while it is an UNUSED piece.
    fn resegment(
        &self,
        normalized: &[u8],
        start: usize,
        len: usize,
        rev_merge: &RevMerge,
        depth: u32,
        segments: &mut Vec<Segment>,
    ) {
        let piece = &normalized[start..start + len];
        let id = self.trie.get(piece).unwrap_or(self.unk_id);
        let split = if depth > MAX_RESEGMENT_DEPTH || self.kinds[id as usize] != PieceType::Unused {
            None
        } else {
            rev_merge.get(&id).copied()
        };
        match split {
            None => segments.push(Segment { start: start as u32, end: (start + len) as u32, id }),
            Some(left) => {
                let left = left as usize;
                self.resegment(normalized, start, left, rev_merge, depth + 1, segments);
                self.resegment(normalized, start + left, len - left, rev_merge, depth + 1, segments);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_order_as_the_library_orders_scores() {
        let pair = |score: f32, left: u32| Pair { key: Pair::key(score), left, right: left + 1, size: 2 };
        let mut heap = BinaryHeap::new();
        for (score, left) in [(-3.0, 0), (-1.0, 1), (2.0, 2), (-1.0, 0), (0.0, 3), (-0.0, 4), (f32::MIN, 5)] {
            heap.push(pair(score, left));
        }
        let order: Vec<u32> = std::iter::from_fn(|| heap.pop().map(|p| p.left)).collect();
        // 2.0, then 0.0 above -0.0 (the library compares bit patterns),
        // then -1.0 at left 0 before left 1, -3.0, the lowest float.
        assert_eq!(order, [2, 3, 4, 0, 1, 0, 5]);
    }
}
