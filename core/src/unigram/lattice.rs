use super::*;
impl Unigram {
    /// One word's pieces of highest total score: the best path through
    /// the lattice of every piece at every position, as upstream's
    /// optimized encode walks it, with the unknown token for a character
    /// no piece starts at, and runs of unknown characters fused into one.
    /// `best` is room for the lattice, kept between words.
    /// `span(start, end)` is the span of the word's bytes start..end.
    pub(super) fn segment<S: Sink>(
        &self,
        word: &str,
        span: impl Fn(usize, usize) -> Span,
        best: &mut Vec<Best>,
        out: &mut S,
    ) {
        let n = word.len();
        // At most n pieces can contribute. Preserve the ordinary arithmetic
        // unless summing finite input scores could overflow and destroy ordering.
        let rescale = self.max_score > f64::MAX / (n.max(1) as f64);
        let score = |v: f64| if rescale { v / self.max_score } else { v };
        best.clear();
        best.resize(n + 1, Best { score: 0.0, start: 0, id: 0, set: false });
        let mut start = 0;
        while start < n {
            let base = best[start].score;
            let mblen = word[start..].chars().next().map_or(1, char::len_utf8);
            let mut has_single = false;
            let mut update = |end: usize, score: f64, id: u32| {
                let b = &mut best[end];
                if !b.set || score > b.score {
                    *b = Best { score, start, id, set: true };
                }
            };
            // Every piece the word continues with here, shortest first:
            // one walk down the trie, which ends where no piece goes on.
            self.trie.prefixes(&word.as_bytes()[start..], |len, id| {
                update(start + len, base + score(self.scores[id as usize]), id);
                if len == mblen {
                    has_single = true;
                }
            });
            if !has_single {
                update(start + mblen, base + score(self.unk_score), self.unk_id);
            }
            start += mblen;
        }
        let first = out.len();
        let mut end = n;
        let mut unk_run = false;
        while end > 0 {
            let b = best[end];
            if b.id == self.unk_id {
                if unk_run {
                    out.widen_last(|| span(b.start, end));
                } else {
                    out.push(self.unk_id as i32, || span(b.start, end));
                }
                unk_run = true;
            } else {
                out.push(b.id as i32, || span(b.start, end));
                unk_run = false;
            }
            end = b.start;
        }
        out.reverse_from(first);
    }
}

/// The best way found to a byte of a word.
#[derive(Clone, Copy)]
pub(super) struct Best {
    score: f64,
    start: usize,
    id: u32,
    set: bool,
}
