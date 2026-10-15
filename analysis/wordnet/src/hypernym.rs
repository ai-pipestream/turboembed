//! Coarse semantic typing via upward hypernym-chain lookup (`HypernymTyper`).

use std::collections::{HashMap, VecDeque};

use crate::error::WordNetError;
use crate::lexicon::LexicalKnowledgeBase;
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;

const UPWARD_RELATIONS: [WordNetRelation; 2] = [
    WordNetRelation::Hypernym,
    WordNetRelation::InstanceHypernym,
];

/// Types a noun using the nearest caller-defined anchor in its hypernym graph (`opennlp.wordnet.HypernymTyper`).
#[derive(Clone, Debug)]
pub struct HypernymTyper<L: LexicalKnowledgeBase> {
    knowledge_base: L,
    label_by_synset_id: HashMap<String, String>,
}

impl<L: LexicalKnowledgeBase> HypernymTyper<L> {
    /// Initializes the [`HypernymTyper`] from an iterable of `(anchor_noun_lemma, label)` pairs.
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] if `anchors` is empty, if any anchor lemma or label
    /// is blank, if an anchor lemma is unknown as a noun in `knowledge_base`, or if two anchors
    /// assign different labels to the same synset.
    pub fn new<I, K, V>(knowledge_base: L, anchors: I) -> Result<Self, WordNetError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut labels: HashMap<String, String> = HashMap::new();
        let mut count = 0usize;
        for (anchor_key, anchor_val) in anchors {
            count += 1;
            let key = anchor_key.as_ref();
            let val = anchor_val.as_ref();
            if key.trim().is_empty() || val.trim().is_empty() {
                return Err(WordNetError::invalid(
                    "anchors must not contain blank entries",
                ));
            }
            let senses = knowledge_base.lookup(key, WordNetPos::Noun);
            if senses.is_empty() {
                return Err(WordNetError::invalid(format!(
                    "anchor lemma is unknown to the knowledge base: {key}"
                )));
            }
            for sense in senses {
                if let Some(prev) = labels.get(sense.id()) {
                    if prev != val {
                        return Err(WordNetError::invalid(format!(
                            "anchor lemmas assign conflicting labels to synset {}: {prev} and {val}",
                            sense.id()
                        )));
                    }
                } else {
                    labels.insert(sense.id().to_string(), val.to_string());
                }
            }
        }
        if count == 0 {
            return Err(WordNetError::invalid("anchors must not be null or empty"));
        }
        Ok(Self {
            knowledge_base,
            label_by_synset_id: labels,
        })
    }

    /// Types a noun lemma using the nearest anchor in its hypernym graph across all noun senses.
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] if `lemma` is blank.
    pub fn type_lemma(&self, lemma: &str) -> Result<Option<&str>, WordNetError> {
        if lemma.trim().is_empty() {
            return Err(WordNetError::invalid("lemma must not be null or blank"));
        }
        let mut best_label: Option<&str> = None;
        let mut best_distance = usize::MAX;
        for sense in self.knowledge_base.lookup(lemma, WordNetPos::Noun) {
            if let Some((label, dist)) = self.nearest_anchor(sense.id())
                && dist < best_distance
            {
                best_distance = dist;
                best_label = Some(label);
            }
        }
        Ok(best_label)
    }

    /// Types a synset by its identifier using the nearest anchor in its hypernym graph.
    pub fn type_synset(&self, synset_id: &str) -> Option<&str> {
        self.nearest_anchor(synset_id).map(|(label, _)| label)
    }

    fn nearest_anchor(&self, synset_id: &str) -> Option<(&str, usize)> {
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut depths: HashMap<String, usize> = HashMap::new();
        queue.push_back(synset_id.to_string());
        depths.insert(synset_id.to_string(), 0);
        while let Some(current) = queue.pop_front() {
            let current_depth = depths[&current];
            if let Some(label) = self.label_by_synset_id.get(&current) {
                return Some((label.as_str(), current_depth));
            }
            let parent_depth = current_depth + 1;
            for relation in UPWARD_RELATIONS {
                for parent in self.knowledge_base.related(&current, relation) {
                    if !depths.contains_key(parent) {
                        depths.insert(parent.clone(), parent_depth);
                        queue.push_back(parent.clone());
                    }
                }
            }
        }
        None
    }
}
