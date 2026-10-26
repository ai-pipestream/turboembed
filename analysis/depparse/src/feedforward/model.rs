use super::slots::SlotTable;
use super::{FeatureLayout, FeedforwardContext, normalize::normalize};
use crate::{Result, Transition, error};
use std::collections::{BTreeMap, HashMap, HashSet};

/// All three vocabularies share one dense embedding row space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocabulary {
    pub words: BTreeMap<String, usize>,
    pub pos: BTreeMap<String, usize>,
    pub labels: BTreeMap<String, usize>,
}
impl Vocabulary {
    pub fn len(&self) -> usize {
        self.words.len() + self.pos.len() + self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(super) fn check(&self) -> Result<()> {
        let n = self.len();
        if n > 2_000_000 {
            return Err(error("vocabulary limit exceeded"));
        }
        let mut seen = HashSet::new();
        for (map, root) in [(&self.words, true), (&self.pos, true), (&self.labels, false)] {
            for special in ["*UNK*", "*NULL*"] {
                if !map.contains_key(special) {
                    return Err(error("missing vocabulary special symbol"));
                }
            }
            if root && !map.contains_key("*ROOT*") {
                return Err(error("missing root symbol"));
            }
            for (key, &id) in map {
                if key.is_empty() || id >= n || !seen.insert(id) {
                    return Err(error("invalid or duplicate vocabulary row"));
                }
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Cube,
    Relu,
}
impl Activation {
    pub(super) fn apply(self, x: f64) -> f64 {
        match self {
            Self::Cube => x * x * x,
            Self::Relu => x.max(0.0),
        }
    }
}
/// Row-major parameter arrays. Embeddings and weights are f32; sums use f64.
#[derive(Clone, Debug, PartialEq)]
pub struct NeuralWeights {
    pub embedding_size: usize,
    pub embeddings: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub hidden_weights: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub output_bias: Vec<f32>,
}
#[derive(Clone)]
pub struct FeedforwardDependencyModel {
    pub(super) vocabulary: Vocabulary,
    pub(super) transitions: Vec<Transition>,
    pub(super) layout: FeatureLayout,
    pub(super) activation: Activation,
    pub(super) weights: NeuralWeights,
    pub(super) default_beam_size: usize,
    pub(super) label_encoding: crate::LabelEncoding,
    /// Frequent word rows in word slots, from [`Self::precompute_words`].
    pub(super) cache: HashMap<(usize, usize), Vec<f64>>,
    /// Every POS, label and word-special row in its slots, built with the model.
    pub(super) fixed: SlotTable,
}
impl FeedforwardDependencyModel {
    pub fn new(
        vocabulary: Vocabulary,
        transitions: Vec<Transition>,
        layout: FeatureLayout,
        activation: Activation,
        weights: NeuralWeights,
    ) -> Result<Self> {
        let mut model = Self {
            vocabulary,
            transitions,
            layout,
            activation,
            weights,
            default_beam_size: 1,
            label_encoding: crate::LabelEncoding::Plain,
            cache: HashMap::new(),
            fixed: SlotTable::default(),
        };
        model.check()?;
        model.fixed = SlotTable::build(&model);
        Ok(model)
    }
    pub fn label_encoding(&self) -> crate::LabelEncoding {
        self.label_encoding
    }
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }
    pub fn layout(&self) -> FeatureLayout {
        self.layout
    }
    pub fn activation(&self) -> Activation {
        self.activation
    }
    pub fn weights(&self) -> &NeuralWeights {
        &self.weights
    }
    /// Suggested decoding width recorded by the trainer or imported model.
    pub fn default_beam_size(&self) -> usize {
        self.default_beam_size
    }
    pub(super) fn check(&self) -> Result<()> {
        self.vocabulary.check()?;
        let w = &self.weights;
        let (e, h, o) = (w.embedding_size, w.hidden_bias.len(), self.transitions.len());
        if e == 0 || e > 4096 || h == 0 || h > 65536 || o == 0 || o > 100_000 {
            return Err(error("invalid neural dimensions"));
        }
        let count = self.vocabulary.len() as u128 * e as u128
            + h as u128 * self.layout.feature_count() as u128 * e as u128
            + h as u128
            + o as u128 * h as u128
            + o as u128;
        if count > 100_000_000 {
            return Err(error("neural parameter count exceeds limit"));
        }
        let input = self.layout.feature_count() * e;
        if w.embeddings.len() != self.vocabulary.len() * e
            || w.hidden_weights.len() != h * input
            || w.output_weights.len() != o * h
            || w.output_bias.len() != o
        {
            return Err(error("neural matrix dimensions disagree"));
        }
        let arrays = [&w.embeddings, &w.hidden_weights, &w.hidden_bias, &w.output_weights, &w.output_bias];
        if arrays.iter().map(|a| a.len()).sum::<usize>() > 100_000_000
            || arrays.iter().any(|a| a.iter().any(|v| !v.is_finite()))
        {
            return Err(error("non-finite or oversized neural parameters"));
        }
        let mut seen = HashSet::new();
        for t in &self.transitions {
            Transition::decode(&t.encode())?;
            if !seen.insert(t.encode()) {
                return Err(error("duplicate transition"));
            }
        }
        if !self.transitions.contains(&Transition::Shift)
            || !self.transitions.iter().any(|t| matches!(t, Transition::RightArc(_)))
        {
            return Err(error("SHIFT and RIGHT_ARC required"));
        }
        Ok(())
    }
    pub fn feature_ids(&self, context: &FeedforwardContext) -> Result<Vec<usize>> {
        let n = self.layout.positions();
        if context.words.len() != n || context.pos.len() != n || context.labels.len() != n - 6 {
            return Err(error("context layout mismatch"));
        }
        let lookup =
            |map: &BTreeMap<String, usize>, s: Option<&str>| *map.get(s.unwrap_or("*NULL*")).unwrap_or(&map["*UNK*"]);
        let mut ids = Vec::with_capacity(self.layout.feature_count());
        for s in &context.words {
            ids.push(lookup(&self.vocabulary.words, s.as_deref().map(normalize).as_deref()));
        }
        for s in &context.pos {
            ids.push(lookup(&self.vocabulary.pos, s.as_deref()));
        }
        for s in &context.labels {
            ids.push(lookup(&self.vocabulary.labels, s.as_deref()));
        }
        Ok(ids)
    }
    pub fn score(&self, ids: &[usize]) -> Result<Vec<f64>> {
        if ids.len() != self.layout.feature_count() || ids.iter().any(|&id| id >= self.vocabulary.len()) {
            return Err(error("invalid feature IDs"));
        }
        let w = &self.weights;
        let mut hidden: Vec<_> = w.hidden_bias.iter().map(|&v| f64::from(v)).collect();
        for (slot, &id) in ids.iter().enumerate() {
            if let Some(contribution) =
                self.fixed.get(slot, id).or_else(|| self.cache.get(&(slot, id)).map(Vec::as_slice))
            {
                for (h, c) in hidden.iter_mut().zip(contribution) {
                    *h += c;
                }
            } else {
                for (j, h) in hidden.iter_mut().enumerate() {
                    *h += self.contribution(slot, id, j);
                }
            }
        }
        for h in &mut hidden {
            *h = self.activation.apply(*h);
        }
        let scores: Vec<_> = w
            .output_weights
            .chunks(hidden.len())
            .zip(&w.output_bias)
            .map(|(row, &b)| row.iter().zip(&hidden).fold(f64::from(b), |s, (&v, h)| s + f64::from(v) * h))
            .collect();
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(error("non-finite neural score"));
        }
        Ok(scores)
    }
    /// The f64 values of the slot contributions computed when the model was
    /// built (POS, label and word-special rows).
    pub fn cached_slot_values(&self) -> usize {
        self.fixed.len()
    }
    pub(super) fn contribution(&self, slot: usize, id: usize, j: usize) -> f64 {
        let w = &self.weights;
        let e = w.embedding_size;
        let start = (j * self.layout.feature_count() + slot) * e;
        w.hidden_weights[start..start + e]
            .iter()
            .zip(&w.embeddings[id * e..(id + 1) * e])
            .fold(0.0, |sum, (&a, &b)| sum + f64::from(a) * f64::from(b))
    }
    /// Precomputes frequent word rows, bounded by a caller-supplied f64 value
    /// budget. Immutable weights keep cached and uncached scoring identical.
    pub fn precompute_words(&mut self, frequencies: &[(String, usize)], max_values: usize) -> usize {
        let mut words: Vec<_> = frequencies
            .iter()
            .filter_map(|(s, n)| self.vocabulary.words.get(&normalize(s)).map(|&id| (id, *n)))
            .collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        self.cache.clear();
        let h = self.weights.hidden_bias.len();
        let limit = max_values.min(8_000_000) / h;
        for (id, _) in words {
            for slot in 0..self.layout.positions() {
                if self.cache.len() >= limit {
                    return self.cache.len();
                }
                if !self.cache.contains_key(&(slot, id)) {
                    let values = (0..h).map(|j| self.contribution(slot, id, j)).collect();
                    self.cache.insert((slot, id), values);
                }
            }
        }
        self.cache.len()
    }
}
