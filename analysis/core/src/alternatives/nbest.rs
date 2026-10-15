//! Top-k outcome sequences for aligned layers (`NBest<L>`).

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::layer::{Anchor, Layer, LayerName, Tag, TagSet};
use crate::span::Ix;

/// Top-k outcome sequences for an aligned layer `L` (for example `PosTags`,
/// `ChunkTags`, or `Tokens`), stored flat in row-major `[candidates, items]`
/// order: `turbo:nbest<L>`.
///
/// The layer is aligned to `L` (`Anchor::aligned::<L>()`), and its `len()` is
/// `items` (the length of `L`), so `Document::add` verifies that every
/// candidate sequence has one tag per item of `L`. Candidate sequences are
/// stored best-first in non-increasing order of `score`.
pub struct NBest<L> {
    pub tagset: Arc<TagSet>,
    /// The number of items in `L` (the length of each candidate sequence).
    pub items: usize,
    /// The sequence score (for example sum of log-probabilities) of each
    /// candidate sequence, best first.
    pub score: Vec<f64>,
    /// Flat `[candidates * items]` tag table: candidate `rank` has tags
    /// `&tags[rank * items..(rank + 1) * items]`.
    pub tags: Vec<Tag>,
    /// Optional flat `[candidates * items]` per-item probability table in
    /// `0.0..=1.0`.
    pub prob: Option<Vec<f64>>,
    _of: PhantomData<fn() -> L>,
}

impl<L> Clone for NBest<L> {
    fn clone(&self) -> Self {
        NBest {
            tagset: self.tagset.clone(),
            items: self.items,
            score: self.score.clone(),
            tags: self.tags.clone(),
            prob: self.prob.clone(),
            _of: PhantomData,
        }
    }
}

impl<L> fmt::Debug for NBest<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NBest")
            .field("tagset", &self.tagset.name)
            .field("items", &self.items)
            .field("score", &self.score)
            .field("tags", &self.tags)
            .field("prob", &self.prob)
            .finish()
    }
}

impl<L> NBest<L> {
    /// Creates an empty `NBest<L>` for a sequence of `items` items.
    pub fn new(tagset: Arc<TagSet>, items: usize) -> NBest<L> {
        NBest { tagset, items, score: Vec::new(), tags: Vec::new(), prob: None, _of: PhantomData }
    }

    /// Appends one candidate sequence of length `self.items` with its sequence
    /// `score` and optional per-item probabilities.
    pub fn push(&mut self, score: f64, tags: &[Tag], probs: Option<&[f64]>) -> Result<usize, String> {
        if tags.len() != self.items {
            return Err(format!("expected {} tags for candidate sequence, got {}", self.items, tags.len()));
        }
        if let Some(p) = probs
            && p.len() != self.items
        {
            return Err(format!("expected {} probabilities for candidate sequence, got {}", self.items, p.len()));
        }
        let rank = self.score.len();
        // Nothing is appended until every refusal is past, so a refused
        // candidate leaves the table as it was.
        match (&self.prob, probs) {
            (Some(_), None) => {
                return Err("cannot mix candidates with and without per-item probabilities".into());
            }
            (None, Some(_)) if rank > 0 => {
                return Err("cannot mix candidates without and with per-item probabilities".into());
            }
            _ => {}
        }
        self.score.push(score);
        self.tags.extend_from_slice(tags);
        match (&mut self.prob, probs) {
            (Some(table), Some(p)) => table.extend_from_slice(p),
            (None, Some(p)) => self.prob = Some(p.to_vec()),
            _ => {}
        }
        Ok(rank)
    }

    /// The number of candidate sequences.
    pub fn candidates(&self) -> usize {
        self.score.len()
    }

    /// The score of candidate sequence `rank`.
    pub fn score_of(&self, rank: usize) -> f64 {
        self.score[rank]
    }

    /// The tag sequence of candidate `rank`, one tag per item of `L`.
    pub fn tags_of(&self, rank: usize) -> &[Tag] {
        &self.tags[rank * self.items..(rank + 1) * self.items]
    }

    /// The per-item probabilities of candidate `rank`, when present.
    pub fn probs_of(&self, rank: usize) -> Option<&[f64]> {
        self.prob.as_ref().map(|p| &p[rank * self.items..(rank + 1) * self.items])
    }

    /// The tag and optional probability assigned to `item` in candidate `rank`.
    pub fn at(&self, rank: usize, item: Ix<L>) -> (Tag, Option<f64>) {
        assert!(item.get() < self.items, "item {} of {}", item.get(), self.items);
        let idx = rank * self.items + item.get();
        (self.tags[idx], self.prob.as_ref().map(|p| p[idx]))
    }
}

impl<L: Layer> Layer for NBest<L> {
    const ID: &'static str = "turbo:nbest";
    const OF: Option<&'static LayerName> = Some(L::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }

    fn len(&self) -> usize {
        self.items
    }

    fn check(&self) -> Result<(), String> {
        let k = self.score.len();
        let expected = k.checked_mul(self.items).ok_or_else(|| "candidates * items overflows usize".to_string())?;
        if self.tags.len() != expected {
            return Err(format!("{} tags for {} candidates of {} items", self.tags.len(), k, self.items));
        }
        if self.score.iter().any(|s| s.is_nan()) {
            return Err("no NaN sequence score".into());
        }
        if !self.score.windows(2).all(|w| w[0] >= w[1]) {
            return Err("candidate sequences in non-increasing score order (best first)".into());
        }
        if self.tags.iter().any(|t| t.0 as usize >= self.tagset.len()) {
            return Err(format!("a tag outside {}", self.tagset.name));
        }
        if let Some(prob) = &self.prob {
            if prob.len() != expected {
                return Err("a probability for every candidate tag".into());
            }
            if prob.iter().any(|v| !(0.0..=1.0).contains(v)) {
                return Err("a probability between 0 and 1".into());
            }
        }
        Ok(())
    }
}
