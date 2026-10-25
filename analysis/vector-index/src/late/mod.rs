//! Late interaction: candidates of several token vectors each, scored
//! against a query of several token vectors by MaxSim. For each query
//! token, the highest dot product over the candidate's tokens; the score
//! is the sum of those maxima over the query tokens, in query order.
//!
//! The dot products are the crate's (`f32`, one fixed order on every
//! path), computed a candidate at a time as a query-by-candidate block of
//! scores with the register-tiled kernel that answers batch queries. The
//! maximum of a query token is the first candidate token that reaches it,
//! and the sum adds the maxima one after another, so a score is the same
//! bits on every path.

mod build;
mod score;

pub use build::{Prune, TokenIndexBuilder};

use turbo_analysis::Span;

use crate::Space;
use crate::kernel::Path;
use crate::rows::{self, Block};

/// Where a candidate token came from: its row in the layer or slice it
/// was added from, and its bytes in the text when it was added from a
/// layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    pub piece: u32,
    pub span: Option<Span>,
}

/// The candidate token that won one query token, and its dot product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenMatch {
    /// The token of the candidate, from 0, among the tokens it kept.
    pub token: u32,
    pub source: Source,
    pub score: f32,
}

/// One result of a rerank: a candidate, in insertion order from 0, and its
/// MaxSim score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LateHit {
    pub candidate: u32,
    pub score: f32,
}

/// Candidates of token vectors from one [`Space`], each with the caller's
/// metadata. Immutable: queries take `&self`, so one index serves many
/// threads.
pub struct TokenIndex<M> {
    space: Space,
    bpr: usize,
    rows: Vec<Block>,
    /// Candidate `c` holds tokens `first[c]..first[c + 1]`.
    first: Vec<u32>,
    sources: Vec<Source>,
    meta: Vec<M>,
    /// Per candidate: the pruning rule (an index into `rules`) and the
    /// tokens it dropped.
    pruned: Vec<Option<(u16, u32)>>,
    rules: Vec<String>,
    path: Path,
}

impl<M> TokenIndex<M> {
    pub fn space(&self) -> &Space {
        &self.space
    }

    /// The number of candidates.
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    pub fn meta(&self, candidate: u32) -> &M {
        &self.meta[candidate as usize]
    }

    /// The tokens `candidate` kept: its rows `tokens(c)` of the index.
    pub fn tokens(&self, candidate: u32) -> std::ops::Range<usize> {
        let c = candidate as usize;
        self.first[c] as usize..self.first[c + 1] as usize
    }

    /// Where each token `candidate` kept came from, in order.
    pub fn sources(&self, candidate: u32) -> &[Source] {
        &self.sources[self.tokens(candidate)]
    }

    /// Token `token` of `candidate` as stored, or `None` when the candidate
    /// has no such token.
    pub fn token(&self, candidate: u32, token: u32) -> Option<&[f32]> {
        let tokens = self.tokens(candidate);
        if token as usize >= tokens.len() {
            return None;
        }
        let at = (tokens.start + token as usize) * self.bpr;
        Some(&rows::flat(&self.rows[at..at + self.bpr])[..self.space.dim as usize])
    }

    /// The pruning rule `candidate` was added under, and how many of its
    /// tokens the rule dropped; `None` when it was added unpruned.
    pub fn pruning(&self, candidate: u32) -> Option<(&str, u32)> {
        self.pruned[candidate as usize].map(|(r, n)| (self.rules[r as usize].as_str(), n))
    }

    /// The tokens all candidates kept, together.
    pub fn total_tokens(&self) -> usize {
        self.sources.len()
    }

    /// Serializes the [`Space`], candidate boundaries, token sources,
    /// pruning rules, and token vectors into a deterministic binary
    /// snapshot (`b"TTOKIX\x01"`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let dim = self.space.dim as usize;
        let mut out = Vec::with_capacity(64 + self.sources.len() * (dim * 4 + 13));
        out.extend_from_slice(b"TTOKIX\x01");
        rows::write_space(&self.space, &mut out);
        out.extend_from_slice(&(self.len() as u64).to_le_bytes());
        for &f in &self.first {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&(self.rules.len() as u32).to_le_bytes());
        for rule in &self.rules {
            rows::write_str(rule, &mut out);
        }
        for p in &self.pruned {
            match p {
                None => out.push(0),
                Some((r, dropped)) => {
                    out.push(1);
                    out.extend_from_slice(&r.to_le_bytes());
                    out.extend_from_slice(&dropped.to_le_bytes());
                }
            }
        }
        for s in &self.sources {
            out.extend_from_slice(&s.piece.to_le_bytes());
            match s.span {
                None => out.push(0),
                Some(span) => {
                    out.push(1);
                    out.extend_from_slice(&span.start.to_le_bytes());
                    out.extend_from_slice(&span.end.to_le_bytes());
                }
            }
        }
        for tok in 0..self.sources.len() {
            let at = tok * self.bpr;
            for &v in &rows::flat(&self.rows[at..at + self.bpr])[..dim] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Restores a [`TokenIndex`] from [`TokenIndex::to_bytes`] and caller
    /// candidate metadata `meta`.
    pub fn from_bytes(bytes: &[u8], meta: Vec<M>) -> Result<TokenIndex<M>, crate::IndexError> {
        use crate::{IndexError, Which, kernel};
        let mut r = rows::ByteReader(bytes);
        if r.take(7).map_err(IndexError::Layer)? != b"TTOKIX\x01" {
            return Err(IndexError::Layer("not a TokenIndex snapshot of version 1".into()));
        }
        let space = rows::read_space(&mut r).map_err(IndexError::Layer)?;
        let n_cand = usize::try_from(r.u64().map_err(IndexError::Layer)?).map_err(|_| IndexError::Full)?;
        if n_cand > u32::MAX as usize {
            return Err(IndexError::Full);
        }
        if meta.len() != n_cand {
            return Err(IndexError::Layer(format!("meta length {} does not match candidates {n_cand}", meta.len())));
        }
        let mut first = Vec::with_capacity(n_cand + 1);
        for _ in 0..=n_cand {
            let f = r.u32().map_err(IndexError::Layer)?;
            if let Some(&prev) = first.last()
                && f <= prev
            {
                return Err(IndexError::Layer("TokenIndex snapshot candidate offsets not strictly ascending".into()));
            }
            first.push(f);
        }
        if first[0] != 0 {
            return Err(IndexError::Layer("TokenIndex snapshot first offset must be 0".into()));
        }
        let total_tok = *first.last().unwrap() as usize;
        let n_rules = r.u32().map_err(IndexError::Layer)? as usize;
        if n_rules > u16::MAX as usize {
            return Err(IndexError::Full);
        }
        let mut rules = Vec::with_capacity(n_rules);
        for _ in 0..n_rules {
            rules.push(r.str().map_err(IndexError::Layer)?);
        }
        let mut pruned = Vec::with_capacity(n_cand);
        for _ in 0..n_cand {
            match r.u8().map_err(IndexError::Layer)? {
                0 => pruned.push(None),
                1 => {
                    let rule_ix = r.u16().map_err(IndexError::Layer)?;
                    if (rule_ix as usize) >= n_rules {
                        return Err(IndexError::Layer("TokenIndex pruning rule index out of bounds".into()));
                    }
                    let dropped = r.u32().map_err(IndexError::Layer)?;
                    pruned.push(Some((rule_ix, dropped)));
                }
                b => return Err(IndexError::Layer(format!("invalid pruning tag {b}"))),
            }
        }
        let mut sources = Vec::with_capacity(total_tok);
        for _ in 0..total_tok {
            let piece = r.u32().map_err(IndexError::Layer)?;
            let span = match r.u8().map_err(IndexError::Layer)? {
                0 => None,
                1 => {
                    let start = r.u32().map_err(IndexError::Layer)?;
                    let end = r.u32().map_err(IndexError::Layer)?;
                    if start > end {
                        return Err(IndexError::Layer("TokenIndex token span start > end".into()));
                    }
                    Some(Span::new(start, end))
                }
                b => return Err(IndexError::Layer(format!("invalid span tag {b}"))),
            };
            sources.push(Source { piece, span });
        }
        let dim = space.dim as usize;
        if r.0.len() != total_tok.saturating_mul(dim).saturating_mul(4) {
            return Err(IndexError::Layer("TokenIndex payload does not match total_tokens * dim * 4".into()));
        }
        let bpr = rows::blocks(space.dim);
        let mut row_blocks = vec![Block::default(); total_tok * bpr];
        for (tok_i, block_slice) in row_blocks.chunks_exact_mut(bpr).enumerate() {
            let flat = rows::flat_mut(block_slice);
            for (j, slot) in flat.iter_mut().enumerate().take(dim) {
                let v = r.f32().map_err(IndexError::Layer)?;
                if !v.is_finite() {
                    return Err(IndexError::NonFinite(Which::Row(tok_i), j));
                }
                *slot = v;
            }
        }
        Ok(TokenIndex { space, bpr, rows: row_blocks, first, sources, meta, pruned, rules, path: kernel::detect() })
    }
}
