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
}
