//! [`GazetteerProvider`]: candidates from a [`turbo_geo::Gazetteer`].

use turbo_geo::{FuzzyStringScorer, Gazetteer};
use turbo_vector_index::Space;

use super::{KbProvider, RawCandidate};
use crate::similarity::{extract_context_terms, fold_alias};

/// [`KbProvider`] adapter over any [`turbo_geo::Gazetteer`].
#[derive(Clone, Debug)]
pub struct GazetteerProvider<G> {
    provider_id: String,
    gazetteer: G,
    fuzzy_scorer: FuzzyStringScorer,
}

impl<G: Gazetteer> GazetteerProvider<G> {
    pub fn new(provider_id: impl Into<String>, gazetteer: G) -> Self {
        Self { provider_id: provider_id.into(), gazetteer, fuzzy_scorer: FuzzyStringScorer::default() }
    }

    pub fn gazetteer(&self) -> &G {
        &self.gazetteer
    }

    fn entry_to_candidate(&self, entry: &turbo_geo::Entry<'_>, alias_prior: f64, sim: f64) -> RawCandidate {
        let kb_id = format!("{}:{}", entry.source(), entry.record_id());
        let pop = entry.population() as f64;
        let popularity_prior = if pop > 0.0 { ((pop + 1.0).log10() / 8.0).clamp(0.1, 1.0) } else { 0.25 };
        let mut entity_types = vec!["GPE".to_owned(), "LOC".to_owned()];
        if let Some(fc) = entry.feature_class() {
            entity_types.push(fc.to_owned());
            if fc == "POI" {
                entity_types.push("FAC".to_owned());
            }
        }
        let mut terms = extract_context_terms(entry.name());
        if let Some(cc) = entry.country_code() {
            terms.push(cc.to_ascii_lowercase());
        }
        for c in entry.containment() {
            terms.extend(extract_context_terms(c));
        }
        for alt in entry.alternate_names() {
            terms.extend(extract_context_terms(alt));
        }
        terms.sort_unstable();
        terms.dedup();

        let mut related_ids = Vec::new();
        if let Some(cc) = entry.country_code() {
            related_ids.push(format!("country:{cc}"));
        }
        for c in entry.containment() {
            related_ids.push(format!("region:{}", fold_alias(c)));
        }

        RawCandidate {
            kb_id,
            provider_id: self.provider_id.clone(),
            canonical_name: entry.name().to_owned(),
            matched_alias: entry.name().to_owned(),
            alias_prior: alias_prior.clamp(0.0, 1.0),
            popularity_prior,
            match_similarity: sim.clamp(0.0, 1.0),
            entity_types,
            context_terms: terms,
            related_ids,
            vector_score: None,
        }
    }
}

impl<G: Gazetteer + Send + Sync> KbProvider for GazetteerProvider<G> {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn generate_candidates(
        &self,
        mention_text: &str,
        _query_vector: Option<(&Space, &[f32])>,
        max_candidates: usize,
        fuzzy_threshold: f64,
    ) -> Vec<RawCandidate> {
        if max_candidates == 0 || mention_text.trim().is_empty() {
            return Vec::new();
        }
        let entries = self.gazetteer.lookup(mention_text);
        if entries.is_empty() {
            return Vec::new();
        }
        let total_weight: f64 = entries
            .iter()
            .enumerate()
            .map(|(rank, e)| {
                let pop_w = ((e.population() as f64) + 10.0).log10();
                pop_w / ((rank + 1) as f64)
            })
            .sum();

        let threshold = fuzzy_threshold.clamp(0.0, 1.0);
        let mut out = Vec::new();
        for (rank, entry) in entries.iter().enumerate() {
            let sim = self.fuzzy_scorer.score_entry(mention_text, entry).max(if threshold <= 1.0 { 1.0 } else { 0.0 });
            let pop_w = ((entry.population() as f64) + 10.0).log10() / ((rank + 1) as f64);
            let prior = if total_weight > 0.0 { pop_w / total_weight } else { 1.0 / (entries.len() as f64) };
            out.push(self.entry_to_candidate(entry, prior, sim));
            if out.len() >= max_candidates {
                break;
            }
        }
        out
    }

    fn lookup_by_id(&self, kb_id: &str) -> Option<RawCandidate> {
        let (source, record_id) = kb_id.split_once(':')?;
        let entry = self.gazetteer.by_id(source, record_id)?;
        Some(self.entry_to_candidate(&entry, 1.0, 1.0))
    }
}
