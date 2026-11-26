//! [`WordNetProvider`]: candidates from a [`turbo_wordnet::LexicalKnowledgeBase`].

use turbo_vector_index::Space;
use turbo_wordnet::{HypernymTyper, LexicalKnowledgeBase, Synset, WordNetPos, WordNetRelation};

use super::{KbProvider, RawCandidate};
use crate::similarity::extract_context_terms;

/// [`KbProvider`] adapter over any [`turbo_wordnet::LexicalKnowledgeBase`].
#[derive(Clone, Debug)]
pub struct WordNetProvider<L: LexicalKnowledgeBase> {
    provider_id: String,
    lexicon: L,
    typer: Option<HypernymTyper<L>>,
}

impl<L: LexicalKnowledgeBase + Clone> WordNetProvider<L> {
    pub fn new(provider_id: impl Into<String>, lexicon: L) -> Self {
        Self { provider_id: provider_id.into(), lexicon, typer: None }
    }

    pub fn with_hypernym_typer(mut self, typer: HypernymTyper<L>) -> Self {
        self.typer = Some(typer);
        self
    }

    fn synset_to_candidate(&self, synset: &Synset, matched_alias: &str, rank: usize, total: usize) -> RawCandidate {
        let rank_weight = 1.0 / ((rank + 1) as f64);
        let norm: f64 = (0..total.max(1)).map(|i| 1.0 / ((i + 1) as f64)).sum();
        let alias_prior = (rank_weight / norm.max(1e-9)).clamp(0.0, 1.0);
        let popularity_prior = (1.0 / (1.0 + 0.35 * (rank as f64))).clamp(0.1, 1.0);

        let mut entity_types = vec![synset.pos().file_suffix().to_ascii_uppercase(), "CONCEPT".to_owned()];
        if let Some(typer) = &self.typer
            && let Some(label) = typer.type_synset(synset.id())
        {
            entity_types.push(label.to_owned());
        }

        let mut context_terms = extract_context_terms(synset.gloss());
        for lemma in synset.lemmas() {
            context_terms.extend(extract_context_terms(lemma));
        }
        context_terms.sort_unstable();
        context_terms.dedup();

        let mut related_ids = Vec::new();
        for rel in [
            WordNetRelation::Hypernym,
            WordNetRelation::InstanceHypernym,
            WordNetRelation::MemberHolonym,
            WordNetRelation::PartHolonym,
            WordNetRelation::DomainTopic,
        ] {
            for target in synset.related(rel) {
                related_ids.push(target.clone());
            }
        }

        let canonical_name = synset.lemmas().first().cloned().unwrap_or_else(|| synset.id().to_owned());
        RawCandidate {
            kb_id: synset.id().to_owned(),
            provider_id: self.provider_id.clone(),
            canonical_name,
            matched_alias: matched_alias.to_owned(),
            alias_prior,
            popularity_prior,
            match_similarity: 1.0,
            entity_types,
            context_terms,
            related_ids,
            vector_score: None,
        }
    }
}

impl<L: LexicalKnowledgeBase + Clone + Send + Sync> KbProvider for WordNetProvider<L> {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn generate_candidates(
        &self,
        mention_text: &str,
        _query_vector: Option<(&Space, &[f32])>,
        max_candidates: usize,
        _fuzzy_threshold: f64,
    ) -> Vec<RawCandidate> {
        if max_candidates == 0 || mention_text.trim().is_empty() {
            return Vec::new();
        }
        let mut synsets = Vec::new();
        for pos in [WordNetPos::Noun, WordNetPos::Verb, WordNetPos::Adjective, WordNetPos::Adverb] {
            for s in self.lexicon.lookup(mention_text, pos) {
                if !synsets.iter().any(|existing: &&Synset| existing.id() == s.id()) {
                    synsets.push(s);
                }
            }
        }
        let total = synsets.len();
        synsets
            .into_iter()
            .take(max_candidates)
            .enumerate()
            .map(|(rank, s)| self.synset_to_candidate(s, mention_text, rank, total))
            .collect()
    }

    fn lookup_by_id(&self, kb_id: &str) -> Option<RawCandidate> {
        let synset = self.lexicon.synset(kb_id)?;
        let name = synset.lemmas().first().map_or(kb_id, String::as_str);
        Some(self.synset_to_candidate(synset, name, 0, 1))
    }
}
