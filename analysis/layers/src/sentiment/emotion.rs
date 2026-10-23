//! `turbo:emotion`: the 28 GoEmotions scores of each sentence or chunk,
//! with the evidence token spans of its labels.

use turbo_analysis::{Anchor, Ix, IxRange, Key, KeyId, Layer, Span};

use crate::{Chunks, Sentences, Tokens};

use super::check_token_range;
use super::emotion_labels::{EMOTION_DIMENSIONS, EmotionDimension, EmotionScores};

/// The unit an [`EmotionLayer`] item scores.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EmotionTarget {
    Sentence(Ix<Sentences>),
    Chunk(Ix<Chunks>),
}

impl EmotionTarget {
    pub fn raw_index(self) -> u32 {
        match self {
            EmotionTarget::Sentence(i) => i.get() as u32,
            EmotionTarget::Chunk(i) => i.get() as u32,
        }
    }
}

/// One item handed to [`EmotionLayer::try_push`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EmotionItem<'a> {
    pub target: EmotionTarget,
    pub scores: EmotionScores,
    pub evidence: &'a [(EmotionDimension, IxRange<Tokens>)],
}

/// Multi-label emotions (`turbo:emotion`) of sentences or chunks: per item
/// its target, its 28 scores (row-major in `scores`, `28` per item, in
/// [`EmotionDimension::ALL`] order) and its evidence spans in CSR form,
/// sorted by `(label, start, end)` within each item.
#[derive(Clone, Debug)]
pub struct EmotionLayer {
    pub target: Vec<EmotionTarget>,
    pub scores: Vec<f64>,
    pub evidence_label: Vec<EmotionDimension>,
    pub evidence_tokens: Vec<IxRange<Tokens>>,
    /// End offset into the evidence columns for each item.
    pub evidence_end: Vec<u32>,
    pub tokens_key: Key<Tokens>,
    pub sentences_key: Key<Sentences>,
    pub chunks_key: Key<Chunks>,
}

impl Default for EmotionLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionLayer {
    /// An empty layer indexing the default instances.
    pub fn new() -> EmotionLayer {
        Self::on(Key::DEFAULT, Key::DEFAULT, Key::DEFAULT)
    }

    /// An empty layer indexing the given named instances.
    pub fn on(tokens_key: Key<Tokens>, sentences_key: Key<Sentences>, chunks_key: Key<Chunks>) -> EmotionLayer {
        EmotionLayer {
            target: Vec::new(),
            scores: Vec::new(),
            evidence_label: Vec::new(),
            evidence_tokens: Vec::new(),
            evidence_end: Vec::new(),
            tokens_key,
            sentences_key,
            chunks_key,
        }
    }

    /// Checks `item` against the target layer and `tokens`, then appends
    /// it: targets in strictly increasing order, scores in `0..=1`,
    /// evidence spans non-empty and inside the tokens.
    pub fn try_push(
        &mut self,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        item: EmotionItem<'_>,
    ) -> Result<Ix<EmotionLayer>, String> {
        let i = self.target.len();
        if self.target.last().is_some_and(|&prev| item.target <= prev) {
            return Err(format!("emotion item {i} target is not in strictly increasing order"));
        }
        let (n, what) = match item.target {
            EmotionTarget::Sentence(_) => (sentences.map(|s| s.spans.len()), "sentence"),
            EmotionTarget::Chunk(_) => (chunks.map(|c| c.spans.len()), "chunk"),
        };
        let Some(n) = n else {
            return Err(format!("emotion item {i} targets a {what}, but no such layer was given"));
        };
        if item.target.raw_index() as usize >= n {
            return Err(format!("emotion item {i} {what} {} is out of bounds for {n}", item.target.raw_index()));
        }
        if !item.scores.is_valid() {
            return Err(format!("emotion item {i} has a score outside 0..=1 or NaN"));
        }
        for &(_, r) in item.evidence {
            check_token_range("emotion evidence", i, r, tokens.len())?;
        }
        Ok(self.push(item))
    }

    /// Appends `item` unchecked, its evidence sorted and deduplicated.
    pub fn push(&mut self, item: EmotionItem<'_>) -> Ix<EmotionLayer> {
        let ix = Ix::new(self.target.len() as u32);
        self.target.push(item.target);
        self.scores.extend_from_slice(&item.scores.0);
        let mut sorted = item.evidence.to_vec();
        sorted.sort_unstable_by_key(|&(l, r)| (l, r.start, r.end));
        sorted.dedup();
        for (l, r) in sorted {
            self.evidence_label.push(l);
            self.evidence_tokens.push(r);
        }
        self.evidence_end.push(self.evidence_label.len() as u32);
        ix
    }

    /// Item `i`'s 28 scores.
    pub fn scores_of(&self, i: Ix<EmotionLayer>) -> EmotionScores {
        let k = i.get() * EMOTION_DIMENSIONS;
        EmotionScores(self.scores[k..k + EMOTION_DIMENSIONS].try_into().expect("28 scores"))
    }

    fn evidence_range(&self, i: Ix<EmotionLayer>) -> std::ops::Range<usize> {
        let k = i.get();
        let start = if k == 0 { 0 } else { self.evidence_end[k - 1] as usize };
        start..self.evidence_end[k] as usize
    }

    /// Item `i`'s evidence, in `(label, start, end)` order.
    pub fn evidence_of(
        &self,
        i: Ix<EmotionLayer>,
    ) -> impl ExactSizeIterator<Item = (EmotionDimension, IxRange<Tokens>)> + '_ {
        let r = self.evidence_range(i);
        self.evidence_label[r.clone()].iter().copied().zip(self.evidence_tokens[r].iter().copied())
    }

    /// Item `i`'s evidence spans for `label`, by binary search.
    pub fn evidence_for_label(&self, i: Ix<EmotionLayer>, label: EmotionDimension) -> &[IxRange<Tokens>] {
        let r = self.evidence_range(i);
        let labels = &self.evidence_label[r.clone()];
        let a = labels.partition_point(|&l| l < label);
        let b = a + labels[a..].partition_point(|&l| l == label);
        &self.evidence_tokens[r][a..b]
    }

    /// The byte span of item `i`'s target.
    pub fn target_span(
        &self,
        i: Ix<EmotionLayer>,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
    ) -> Option<Span> {
        match self.target[i.get()] {
            EmotionTarget::Sentence(s) => sentences.map(|l| l.spans.get(s)),
            EmotionTarget::Chunk(c) => chunks.map(|l| l.spans.get(c)),
        }
    }
}

impl Layer for EmotionLayer {
    const ID: &'static str = "turbo:emotion";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.target.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let max_tok = self.evidence_tokens.iter().map(|r| r.end.get()).max().unwrap_or(0);
        let (mut sent, mut chunk) = (0usize, 0usize);
        for &t in &self.target {
            match t {
                EmotionTarget::Sentence(s) => sent = sent.max(s.get() + 1),
                EmotionTarget::Chunk(c) => chunk = chunk.max(c.get() + 1),
            }
        }
        let mut out = Vec::new();
        for (id, max, named) in [
            (self.tokens_key.id(), max_tok, !self.tokens_key.name.is_empty()),
            (self.sentences_key.id(), sent, !self.sentences_key.name.is_empty()),
            (self.chunks_key.id(), chunk, !self.chunks_key.name.is_empty()),
        ] {
            if max > 0 || named {
                out.push((id, max));
            }
        }
        out
    }

    fn check(&self) -> Result<(), String> {
        let n = self.target.len();
        if self.scores.len() != n * EMOTION_DIMENSIONS || self.evidence_end.len() != n {
            return Err("28 scores and an evidence end for every emotion item".into());
        }
        if self.evidence_label.len() != self.evidence_tokens.len() {
            return Err("an evidence token range for every evidence label".into());
        }
        if !self.target.windows(2).all(|w| w[0] < w[1]) {
            return Err("emotion targets must be in strictly increasing order".into());
        }
        if self.scores.iter().any(|s| !(s.is_finite() && (0.0..=1.0).contains(s))) {
            return Err("an emotion score outside 0..=1 or NaN".into());
        }
        let mut prev_end = 0usize;
        for i in 0..n {
            let end = self.evidence_end[i] as usize;
            if end < prev_end || end > self.evidence_label.len() {
                return Err(format!("emotion item {i} evidence_end {end} is out of bounds or decreases"));
            }
            for k in prev_end..end {
                let r = self.evidence_tokens[k];
                if r.is_empty() {
                    return Err(format!("emotion item {i} has an empty evidence token range"));
                }
                if k > prev_end {
                    let before = (
                        self.evidence_label[k - 1],
                        self.evidence_tokens[k - 1].start,
                        self.evidence_tokens[k - 1].end,
                    );
                    if (self.evidence_label[k], r.start, r.end) <= before {
                        return Err(format!("emotion item {i} evidence is not in strictly increasing order"));
                    }
                }
            }
            prev_end = end;
        }
        if prev_end != self.evidence_label.len() {
            return Err("evidence_end does not reach the end of the evidence".into());
        }
        Ok(())
    }
}
