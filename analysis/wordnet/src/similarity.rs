//! Taxonomy-based semantic similarity over a [`LexicalKnowledgeBase`] (`SynsetSimilarity`).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::WordNetError;
use crate::lexicon::LexicalKnowledgeBase;
use crate::relation::WordNetRelation;

/// Taxonomy-based semantic similarity over a [`LexicalKnowledgeBase`] (`opennlp.wordnet.SynsetSimilarity`).
///
/// Computes path similarity, Wu-Palmer similarity, and Leacock-Chodorow similarity by walking
/// [`WordNetRelation::Hypernym`] and [`WordNetRelation::InstanceHypernym`] edges upward.
#[derive(Clone, Debug)]
pub struct SynsetSimilarity<L: LexicalKnowledgeBase> {
    knowledge_base: L,
}

impl<L: LexicalKnowledgeBase> SynsetSimilarity<L> {
    /// Creates a new [`SynsetSimilarity`] over `knowledge_base`.
    pub fn new(knowledge_base: L) -> Self {
        Self { knowledge_base }
    }

    /// Returns a reference to the underlying [`LexicalKnowledgeBase`].
    pub fn knowledge_base(&self) -> &L {
        &self.knowledge_base
    }

    /// Computes path similarity: `1.0 / (1.0 + d)` over the shortest hypernym-graph distance `d`,
    /// or `0.0` when the synsets share no ancestor.
    pub fn path(&self, synset_id: &str, other_synset_id: &str) -> f64 {
        match self.shortest_distance(synset_id, other_synset_id) {
            None => 0.0,
            Some(d) => 1.0 / (1.0 + d as f64),
        }
    }

    /// Computes Wu-Palmer similarity: `2 * depth(lcs) / (depth(a) + depth(b))`,
    /// with depths counted in nodes from the taxonomy root (the root itself at depth `1`),
    /// or `0.0` when the synsets share no ancestor.
    pub fn wu_palmer(&self, synset_id: &str, other_synset_id: &str) -> f64 {
        let up = self.depths_above(synset_id);
        let other_up = self.depths_above(other_synset_id);
        if up.is_empty() || other_up.is_empty() {
            return 0.0;
        }
        let depth_a = (self.depth_from_root(synset_id).unwrap_or(0) + 1) as f64;
        let depth_b = (self.depth_from_root(other_synset_id).unwrap_or(0) + 1) as f64;
        let mut best = 0.0_f64;
        for common_id in up.keys() {
            if other_up.contains_key(common_id) {
                let lcs_depth = (self.depth_from_root(common_id).unwrap_or(0) + 1) as f64;
                let score = (2.0 * lcs_depth) / (depth_a + depth_b);
                if score > best {
                    best = score;
                }
            }
        }
        best
    }

    /// Computes Leacock-Chodorow similarity `-ln((d + 1) / (2 * taxonomy_depth))` over the
    /// shortest hypernym-graph distance `d`, or `0.0` when the synsets share no ancestor.
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] if `taxonomy_depth == 0`.
    pub fn leacock_chodorow(
        &self,
        synset_id: &str,
        other_synset_id: &str,
        taxonomy_depth: usize,
    ) -> Result<f64, WordNetError> {
        if taxonomy_depth == 0 {
            return Err(WordNetError::invalid(format!("taxonomyDepth must be positive: {taxonomy_depth}")));
        }
        let Some(distance) = self.shortest_distance(synset_id, other_synset_id) else {
            return Ok(0.0);
        };
        Ok(-(((distance as f64) + 1.0) / (2.0 * (taxonomy_depth as f64))).ln())
    }

    /// Finds the shortest edge distance between two synsets through a common ancestor,
    /// or `None` when no common ancestor exists.
    pub fn shortest_distance(&self, synset_id: &str, other_synset_id: &str) -> Option<usize> {
        let up = self.depths_above(synset_id);
        let other_up = self.depths_above(other_synset_id);
        let mut best: Option<usize> = None;
        for (common, &dist_a) in &up {
            if let Some(&dist_b) = other_up.get(common) {
                let total = dist_a + dist_b;
                best = Some(best.map_or(total, |b| b.min(total)));
            }
        }
        best
    }

    fn depths_above(&self, synset_id: &str) -> HashMap<String, usize> {
        let mut depths = HashMap::new();
        if self.knowledge_base.synset(synset_id).is_none() {
            return depths;
        }
        let mut queue = VecDeque::new();
        depths.insert(synset_id.to_string(), 0);
        queue.push_back(synset_id.to_string());
        while let Some(current) = queue.pop_front() {
            let depth = depths[&current];
            for parent in self.hypernyms(&current) {
                if self.knowledge_base.synset(&parent).is_none() {
                    continue;
                }
                if !depths.contains_key(&parent) || depths[&parent] > depth + 1 {
                    depths.insert(parent.clone(), depth + 1);
                    queue.push_back(parent);
                }
            }
        }
        depths
    }

    fn depth_from_root(&self, synset_id: &str) -> Option<usize> {
        let mut path = HashSet::new();
        self.depth_from_root_rec(synset_id, &mut path)
    }

    fn depth_from_root_rec(&self, synset_id: &str, path: &mut HashSet<String>) -> Option<usize> {
        if self.knowledge_base.synset(synset_id).is_none() || !path.insert(synset_id.to_string()) {
            return None;
        }
        let mut deepest = 0usize;
        for parent in self.hypernyms(synset_id) {
            if let Some(parent_depth) = self.depth_from_root_rec(&parent, path) {
                deepest = deepest.max(parent_depth + 1);
            }
        }
        path.remove(synset_id);
        Some(deepest)
    }

    fn hypernyms(&self, synset_id: &str) -> Vec<String> {
        let direct = self.knowledge_base.related(synset_id, WordNetRelation::Hypernym);
        let instance = self.knowledge_base.related(synset_id, WordNetRelation::InstanceHypernym);
        let mut all = Vec::with_capacity(direct.len() + instance.len());
        all.extend(direct.iter().cloned());
        all.extend(instance.iter().cloned());
        all
    }
}
