//! Coreference scores (`CorefScorer`): MUC, B-cubed, CEAF-m, CEAF-e,
//! mention detection and the CoNLL average, following Pradhan et al.
//! (ACL 2014), "Scoring Coreference Partitions of Predicted Mentions: A
//! Reference Implementation".
//!
//! A partition is a list of entities, each a set of mentions; mentions
//! are matched by equality, so predicted mentions missing from the key
//! count against recall and extra ones against precision. Singleton
//! entities are scored when present. Documents accumulate numerators and
//! denominators, so the scores are micro-averaged over the corpus.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use turbo_analysis::{Groups, Ix};

use crate::layer::Mentions;

mod assign;
mod scores;

pub use assign::maximum_assignment;
pub use scores::{CorefScores, Score};

/// Running sums over the documents added so far.
#[derive(Clone, Debug, Default)]
pub struct CorefScorer {
    muc_recall_numerator: f64,
    muc_recall_denominator: f64,
    muc_precision_numerator: f64,
    muc_precision_denominator: f64,
    b_cubed_recall_numerator: f64,
    b_cubed_precision_numerator: f64,
    ceaf_m_numerator: f64,
    ceaf_e_numerator: f64,
    key_entities: u64,
    response_entities: u64,
    key_mentions: u64,
    response_mentions: u64,
    matched_mentions: u64,
}

fn index<'m, M: Eq + Hash + Debug>(partition: &'m [HashSet<M>], name: &str) -> Result<HashMap<&'m M, usize>, String> {
    let mut entity_of = HashMap::new();
    for (e, mentions) in partition.iter().enumerate() {
        if mentions.is_empty() {
            return Err(format!("{name} must not contain an empty entity"));
        }
        for m in mentions {
            if entity_of.insert(m, e).is_some() {
                return Err(format!("{name} places mention {m:?} in two entities"));
            }
        }
    }
    Ok(entity_of)
}

fn muc_numerator<M: Eq + Hash>(entities: &[HashSet<M>], other: &HashMap<&M, usize>) -> f64 {
    let mut numerator = 0.0;
    for entity in entities {
        let mut parts = HashSet::new();
        let mut unmatched = 0usize;
        for m in entity {
            match other.get(m) {
                Some(&o) => {
                    parts.insert(o);
                }
                None => unmatched += 1,
            }
        }
        numerator += entity.len() as f64 - parts.len() as f64 - unmatched as f64;
    }
    numerator
}

fn muc_denominator<M>(entities: &[HashSet<M>]) -> f64 {
    entities.iter().map(|e| e.len() as f64 - 1.0).sum()
}

fn best_alignment<M>(overlap: &[Vec<f64>], keys: &[HashSet<M>], responses: &[HashSet<M>], entity_based: bool) -> f64 {
    let size = keys.len().max(responses.len());
    if size == 0 {
        return 0.0;
    }
    let mut similarity = vec![vec![0.0; size]; size];
    for i in 0..keys.len() {
        for j in 0..responses.len() {
            similarity[i][j] = if entity_based {
                2.0 * overlap[i][j] / (keys[i].len() + responses[j].len()) as f64
            } else {
                overlap[i][j]
            };
        }
    }
    maximum_assignment(&similarity)
}

impl CorefScorer {
    pub fn new() -> CorefScorer {
        CorefScorer::default()
    }

    /// Adds one document's key and response partitions; `Err` when a
    /// partition holds an empty entity or a mention in two entities.
    pub fn add<M: Eq + Hash + Debug>(&mut self, key: &[HashSet<M>], response: &[HashSet<M>]) -> Result<(), String> {
        let key_of = index(key, "key")?;
        let response_of = index(response, "response")?;
        self.muc_recall_numerator += muc_numerator(key, &response_of);
        self.muc_recall_denominator += muc_denominator(key);
        self.muc_precision_numerator += muc_numerator(response, &key_of);
        self.muc_precision_denominator += muc_denominator(response);
        let overlap: Vec<Vec<f64>> = key
            .iter()
            .map(|k| response.iter().map(|r| k.iter().filter(|m| r.contains(m)).count() as f64).collect())
            .collect();
        for (i, row) in overlap.iter().enumerate() {
            for &shared in row {
                if shared > 0.0 {
                    self.b_cubed_recall_numerator += shared * shared / key[i].len() as f64;
                }
            }
        }
        for row in &overlap {
            for (j, &shared) in row.iter().enumerate() {
                if shared > 0.0 {
                    self.b_cubed_precision_numerator += shared * shared / response[j].len() as f64;
                }
            }
        }
        self.ceaf_m_numerator += best_alignment(&overlap, key, response, false);
        self.ceaf_e_numerator += best_alignment(&overlap, key, response, true);
        self.key_entities += key.len() as u64;
        self.response_entities += response.len() as u64;
        self.key_mentions += key_of.len() as u64;
        self.response_mentions += response_of.len() as u64;
        self.matched_mentions += response_of.keys().filter(|m| key_of.contains_key(*m)).count() as u64;
        Ok(())
    }

    /// The scores over every document added so far; all zero before the
    /// first.
    pub fn scores(&self) -> CorefScores {
        let muc = Score::of(
            self.muc_precision_numerator,
            self.muc_precision_denominator,
            self.muc_recall_numerator,
            self.muc_recall_denominator,
        );
        let (km, rm) = (self.key_mentions as f64, self.response_mentions as f64);
        let b_cubed = Score::of(self.b_cubed_precision_numerator, rm, self.b_cubed_recall_numerator, km);
        let ceaf_m = Score::of(self.ceaf_m_numerator, rm, self.ceaf_m_numerator, km);
        let ceaf_e = Score::of(
            self.ceaf_e_numerator,
            self.response_entities as f64,
            self.ceaf_e_numerator,
            self.key_entities as f64,
        );
        let matched = self.matched_mentions as f64;
        let mentions = Score::of(matched, rm, matched, km);
        CorefScores::derived(muc, b_cubed, ceaf_m, ceaf_e, mentions)
    }
}

/// The chains of a `Groups<Mentions>` layer as a partition of mention
/// envelopes `(start, end)` in the mentions' view, for scoring two layers
/// made over one text.
pub fn partition(mentions: &Mentions, chains: &Groups<Mentions>) -> Vec<HashSet<(u32, u32)>> {
    chains
        .iter()
        .map(|(_, members)| {
            members
                .iter()
                .map(|&m: &Ix<Mentions>| {
                    let s = mentions.spans.get(m);
                    (s.start, s.end)
                })
                .collect()
        })
        .collect()
}
