//! Resolution with a model: the precise sieves, then for each anaphor
//! still heading its own cluster, the best scoring candidate.

use super::{Resolver, Sieve};
use crate::features::Features;
use crate::model::CorefModel;

impl Resolver<'_> {
    /// Resolves with a model. A ranker links the best option when it
    /// beats starting a new chain; a pair classifier links the best
    /// candidate whose link probability reaches `threshold`.
    pub fn resolve_with_model(&mut self, model: &CorefModel, threshold: f64) -> Result<(), String> {
        self.resolve_precise();
        let ranking = model.is_ranking();
        for j in 0..self.mentions.len() {
            if self.clusters.find(j) != j || !self.rankable(j) {
                continue;
            }
            let mut best = None;
            let mut best_score = if ranking {
                let option = self.new_chain_features(j)?;
                score(model, &option)?
            } else {
                f64::NEG_INFINITY
            };
            for i in self.ranker_candidates(j) {
                let option = self.features(i, j)?;
                let s = score(model, &option)?;
                if s > best_score {
                    best_score = s;
                    best = Some(i);
                }
            }
            if let Some(i) = best
                && (ranking || best_score >= threshold)
            {
                let sieve = if ranking { Sieve::Ranker } else { Sieve::Classifier };
                self.link(i, j, sieve, best_score);
            }
        }
        Ok(())
    }
}

fn score(model: &CorefModel, option: &Features) -> Result<f64, String> {
    let (names, values) = option.expand()?;
    model.link_probability(&names, values.as_deref())
}
