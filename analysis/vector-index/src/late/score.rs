use super::{LateHit, TokenIndex, TokenMatch};
use crate::rows::{self, Block};
use crate::topk::TopK;
use crate::{IndexError, Metric, Space, Which, kernel};

impl<M> TokenIndex<M> {
    /// The MaxSim score of `query`, token vectors of `space`, `space.dim`
    /// values a token one after another, against `candidate`. Panics if
    /// `candidate` is not below [`TokenIndex::len`].
    pub fn score(&self, space: &Space, query: &[f32], candidate: u32) -> Result<f32, IndexError> {
        let qs = self.prepare(space, query)?;
        Ok(self.maxsim(&qs, candidate, &mut Vec::new(), None))
    }

    /// [`TokenIndex::score`], and for each query token the candidate token
    /// that won it: the first of the candidate's tokens with the highest
    /// dot product.
    pub fn explain(&self, space: &Space, query: &[f32], candidate: u32) -> Result<(f32, Vec<TokenMatch>), IndexError> {
        let qs = self.prepare(space, query)?;
        let mut matches = Vec::with_capacity(qs.len() / self.bpr);
        let score = self.maxsim(&qs, candidate, &mut Vec::new(), Some(&mut matches));
        Ok((score, matches))
    }

    /// The `k` of `candidates` that score highest against `query`, best
    /// first; candidates that tie are in insertion order. A shortlist from
    /// a single-vector search is the usual `candidates`; one listed twice
    /// is answered twice. Panics if a candidate is not below
    /// [`TokenIndex::len`].
    pub fn rerank(
        &self,
        space: &Space,
        query: &[f32],
        candidates: &[u32],
        k: usize,
    ) -> Result<Vec<LateHit>, IndexError> {
        let qs = self.prepare(space, query)?;
        let mut best = TopK::new(k.min(candidates.len()));
        let mut buf = Vec::new();
        for &c in candidates {
            best.offer(c, self.maxsim(&qs, c, &mut buf, None));
        }
        Ok(best.into_sorted().into_iter().map(|h| LateHit { candidate: h.row, score: h.score }).collect())
    }

    /// [`TokenIndex::rerank`] over every candidate.
    pub fn top_k(&self, space: &Space, query: &[f32], k: usize) -> Result<Vec<LateHit>, IndexError> {
        let all: Vec<u32> = (0..self.len() as u32).collect();
        self.rerank(space, query, &all, k)
    }

    /// Checks that `space` is the index's and writes the query tokens as
    /// the kernels read them.
    fn prepare(&self, space: &Space, query: &[f32]) -> Result<Vec<Block>, IndexError> {
        rows::same_space(&self.space, space)?;
        let dim = self.space.dim as usize;
        if !query.len().is_multiple_of(dim) {
            return Err(IndexError::Dimension { index: self.space.dim, other: query.len() });
        }
        if query.is_empty() {
            return Err(IndexError::NoTokens(Which::Query(0)));
        }
        let mut out = vec![Block::default(); query.len() / dim * self.bpr];
        for (t, (v, o)) in query.chunks_exact(dim).zip(out.chunks_exact_mut(self.bpr)).enumerate() {
            rows::prepare(v, &self.space, Metric::Dot, Which::Query(t), o)?;
        }
        Ok(out)
    }

    /// Scores every query token against every token of `candidate` into
    /// `buf`, a query token a line, and adds up each line's maximum.
    fn maxsim(&self, qs: &[Block], candidate: u32, buf: &mut Vec<f32>, matches: Option<&mut Vec<TokenMatch>>) -> f32 {
        let tokens = self.tokens(candidate);
        let n = tokens.len();
        let nq = qs.len() / self.bpr;
        buf.resize(nq * n, 0.0);
        let rows = &self.rows[tokens.start * self.bpr..tokens.end * self.bpr];
        kernel::scores(self.path, qs, rows, self.bpr, buf);
        let mut total = 0.0f32;
        let Some(m) = matches else {
            // The maximum is exact whatever order it is taken in, so it is
            // taken 16 lanes at a time.
            for line in buf.chunks_exact(n) {
                total += line_max(line);
            }
            return total;
        };
        for line in buf.chunks_exact(n) {
            let (mut at, mut best) = (0, line[0]);
            for (j, &s) in line.iter().enumerate().skip(1) {
                if s > best {
                    (at, best) = (j, s);
                }
            }
            total += best;
            m.push(TokenMatch { token: at as u32, source: self.sources[tokens.start + at], score: best });
        }
        total
    }
}

/// The largest value of `line`, which is not empty.
#[inline]
fn line_max(line: &[f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; 16];
    let (chunks, rest) = line.as_chunks::<16>();
    for c in chunks {
        for (l, &x) in lanes.iter_mut().zip(c) {
            *l = if x > *l { x } else { *l };
        }
    }
    let mut best = rest.iter().fold(f32::NEG_INFINITY, |b, &x| if x > b { x } else { b });
    for &l in &lanes {
        best = if l > best { l } else { best };
    }
    best
}
