//! SentencePiece's unigram search (unigram_model.cc, `EncodeOptimized`):
//! for each byte of the normalized text that ends a character, the best
//! path that ends there, in f32 as the library adds scores.

use super::normalizer::one_char_len;
use super::proto::PieceType;
use super::{NONE, Segment, SentencePiece};

/// The best path found to one byte of the normalized text.
#[derive(Clone, Copy)]
pub(crate) struct Best {
    score: f32,
    /// Where its last piece starts, or NONE: no path yet.
    start: u32,
    id: u32,
}

/// Past this magnitude the scores are rebased on the current position,
/// as the library does (`kScoreResetThreshold`).
const SCORE_RESET: f32 = 100_000.0;

/// The library's score for a user-defined piece of `len` bytes, worked
/// out in f64 and stored as f32 (`GetUserDefinedScore`).
#[inline]
fn user_defined_score(len: usize) -> f32 {
    (0.1 * (len as f64 - 1.0)) as f32
}

impl SentencePiece {
    /// The pieces of the best path through `normalized`, in order.
    pub(crate) fn unigram(&self, normalized: &[u8], best: &mut Vec<Best>, segments: &mut Vec<Segment>) {
        segments.clear();
        let size = normalized.len();
        if size == 0 {
            return;
        }
        best.clear();
        best.resize(size + 1, Best { score: 0.0, start: NONE, id: 0 });
        let mut max_frontier = 0usize;
        let mut start = 0usize;
        while start < size {
            let mut base = best[start].score;
            // Scores are finite (checked on load), so this is the
            // library's `base < -SCORE_RESET || base > SCORE_RESET`.
            if !(-SCORE_RESET..=SCORE_RESET).contains(&base) {
                for (i, b) in best[start..=max_frontier].iter_mut().enumerate() {
                    if i == 0 || b.start != NONE {
                        b.score -= base;
                    }
                }
                base = 0.0;
            }
            let mblen = one_char_len(normalized[start]).min(size - start);
            let mut has_single = false;
            self.trie.prefixes(&normalized[start..], |len, id| {
                let kind = self.kinds[id as usize];
                if kind == PieceType::Unused {
                    return;
                }
                let end = start + len;
                max_frontier = max_frontier.max(end);
                let score =
                    if kind == PieceType::UserDefined { user_defined_score(len) } else { self.scores[id as usize] };
                let candidate = score + base;
                let target = &mut best[end];
                if target.start == NONE || candidate > target.score {
                    *target = Best { score: candidate, start: start as u32, id };
                }
                if len == mblen {
                    has_single = true;
                }
            });
            if !has_single {
                let end = start + mblen;
                max_frontier = max_frontier.max(end);
                let candidate = self.unk_score + base;
                let target = &mut best[end];
                if target.start == NONE || candidate > target.score {
                    *target = Best { score: candidate, start: start as u32, id: self.unk_id };
                }
            }
            start += mblen;
        }
        let mut end = size;
        while end > 0 {
            let b = best[end];
            segments.push(Segment { start: b.start, end: end as u32, id: b.id });
            end = b.start as usize;
        }
        segments.reverse();
    }

    /// The library's check of a self-test sample for a unigram model: the
    /// two piece sequences, written with a space between pieces, have
    /// the same total score within 1e-7 (`VerifyOutputsEquivalent`).
    pub(crate) fn unigram_equivalent(&self, expected: &[u8], actual: &[u8]) -> bool {
        let total = |s: &[u8]| {
            let mut total = 0f32;
            for p in s.split(|&b| b == b' ') {
                let id = self.piece_to_id(p);
                total += if id == self.unk_id {
                    self.unk_score
                } else if self.kinds[id as usize] == PieceType::UserDefined {
                    user_defined_score(p.len())
                } else {
                    self.scores[id as usize]
                };
            }
            total
        };
        (total(expected) - total(actual)).abs() <= 1e-7
    }
}
