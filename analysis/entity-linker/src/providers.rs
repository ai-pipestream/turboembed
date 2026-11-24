//! Pluggable [`KbProvider`] backends ([`CatalogProvider`], [`GazetteerProvider`],
//! [`WordNetProvider`], and [`GlossaryProvider`]) for candidate generation.

mod gazetteer;
mod wordnet;

use std::collections::HashMap;
use std::sync::Arc;

use turbo_glossary_layers::Glossary;
use turbo_matcher::FuzzyMatchPolicy;
use turbo_vector_index::Space;

use crate::kb::KnowledgeBase;
use crate::similarity::{alias_similarity, fold_alias};

pub use gazetteer::GazetteerProvider;
pub use wordnet::WordNetProvider;

/// An unscored candidate entity produced by a [`KbProvider`].
#[derive(Clone, Debug, PartialEq)]
pub struct RawCandidate {
    /// Canonical identifier within the provider's knowledge base.
    pub kb_id: String,
    /// Identifier of the [`KbProvider`] that generated this candidate.
    pub provider_id: String,
    /// Canonical display name of the entity.
    pub canonical_name: String,
    /// Surface alias that matched the mention.
    pub matched_alias: String,
    /// Conditional alias prior $P(e \mid m)$ in `[0.0, 1.0]`.
    pub alias_prior: f64,
    /// Global popularity or prominence prior in `[0.0, 1.0]`.
    pub popularity_prior: f64,
    /// String similarity between the mention text and `matched_alias` in `[0.0, 1.0]`.
    pub match_similarity: f64,
    /// Semantic / NER entity types associated with this KB entity.
    pub entity_types: Vec<String>,
    /// Pre-folded content terms from the entity's description, name, and types.
    pub context_terms: Vec<String>,
    /// Related KB entity IDs for document-level coherence scoring.
    pub related_ids: Vec<String>,
    /// Optional dense vector similarity score mapped to `[0.0, 1.0]`.
    pub vector_score: Option<f64>,
}

/// Trait implemented by knowledge-base backends that supply entity candidates.
pub trait KbProvider: Send + Sync {
    /// Stable identifier of this provider (for example `"wikidata"`, `"geonames"`, `"wordnet"`).
    fn provider_id(&self) -> &str;

    /// Generates up to `max_candidates` candidate entities for `mention_text` and optional
    /// query embedding `query_vector`.
    fn generate_candidates(
        &self,
        mention_text: &str,
        query_vector: Option<(&Space, &[f32])>,
        max_candidates: usize,
        fuzzy_threshold: f64,
    ) -> Vec<RawCandidate>;

    /// Like [`generate_candidates`](KbProvider::generate_candidates) with a fuzzy match policy;
    /// the default delegates when disabled and otherwise broadens retrieval, policy-filters, caps.
    fn generate_candidates_with_policy(
        &self,
        mention_text: &str,
        query_vector: Option<(&Space, &[f32])>,
        max_candidates: usize,
        fuzzy_threshold: f64,
        fuzzy_policy: FuzzyMatchPolicy,
    ) -> Vec<RawCandidate> {
        crate::policy::generate_with_policy(
            self,
            mention_text,
            query_vector,
            max_candidates,
            fuzzy_threshold,
            fuzzy_policy,
        )
    }

    /// Looks up a single entity by its canonical `kb_id`.
    fn lookup_by_id(&self, kb_id: &str) -> Option<RawCandidate>;

    /// Returns the inverse document frequency weight of a context term, if known.
    fn term_idf(&self, _term: &str) -> f64 {
        1.0
    }

    /// Returns a 64-bit content fingerprint of this provider for provenance hashing.
    fn fingerprint(&self) -> u64 {
        turbo_analysis::fnv1a(self.provider_id().as_bytes())
    }
}

/// [`KbProvider`] backed by an in-memory [`KnowledgeBase`] catalog.
#[derive(Clone, Debug)]
pub struct CatalogProvider {
    kb: Arc<KnowledgeBase>,
}

impl CatalogProvider {
    pub fn new(kb: KnowledgeBase) -> Self {
        Self { kb: Arc::new(kb) }
    }

    pub fn from_arc(kb: Arc<KnowledgeBase>) -> Self {
        Self { kb }
    }

    pub fn knowledge_base(&self) -> &KnowledgeBase {
        &self.kb
    }

    /// Deterministic fuzzy-index work for `mention_text` at `fuzzy_threshold`.
    ///
    /// Folds the mention the same way [`KbProvider::generate_candidates`] does,
    /// then reports [`crate::FuzzyLookupWork`] from the catalog's fuzzy index.
    pub fn fuzzy_lookup_work(&self, mention_text: &str, fuzzy_threshold: f64) -> crate::FuzzyLookupWork {
        let query_folded = fold_alias(mention_text);
        let threshold = fuzzy_threshold.clamp(0.0, 1.0);
        self.kb.fuzzy_lookup_work(&query_folded, threshold)
    }

    fn candidate_from_idx(
        &self,
        idx: u32,
        matched_alias: String,
        alias_prior: f64,
        match_similarity: f64,
        vector_score: Option<f64>,
    ) -> Option<RawCandidate> {
        let ent = self.kb.entity(idx)?;
        Some(RawCandidate {
            kb_id: ent.id.clone(),
            provider_id: self.kb.provider_id().to_owned(),
            canonical_name: ent.name.clone(),
            matched_alias,
            alias_prior: alias_prior.clamp(0.0, 1.0),
            popularity_prior: ent.popularity_prior.clamp(0.0, 1.0),
            match_similarity: match_similarity.clamp(0.0, 1.0),
            entity_types: ent.entity_types.clone(),
            context_terms: self.kb.context_terms(idx).to_vec(),
            related_ids: ent.related_ids.clone(),
            vector_score,
        })
    }
}

impl KbProvider for CatalogProvider {
    fn provider_id(&self) -> &str {
        self.kb.provider_id()
    }

    fn generate_candidates(
        &self,
        mention_text: &str,
        query_vector: Option<(&Space, &[f32])>,
        max_candidates: usize,
        fuzzy_threshold: f64,
    ) -> Vec<RawCandidate> {
        if max_candidates == 0 {
            return Vec::new();
        }
        let query_folded = fold_alias(mention_text);
        let mut by_entity: HashMap<u32, RawCandidate> = HashMap::new();

        // 1. Exact folded alias lookup
        if !query_folded.is_empty()
            && let Some(postings) = self.kb.exact_postings(&query_folded)
        {
            for p in postings.iter().take(max_candidates) {
                if let Some(c) =
                    self.candidate_from_idx(p.entity_idx, p.matched_alias.clone(), p.alias_prior, 1.0, None)
                {
                    by_entity.insert(p.entity_idx, c);
                }
            }
        }

        // 2. Fuzzy alias lookup when threshold < 1.0 and we have room for more candidates
        let threshold = fuzzy_threshold.clamp(0.0, 1.0);
        if !query_folded.is_empty() && threshold < 1.0 && by_entity.len() < max_candidates {
            let needed = max_candidates - by_entity.len();
            let fuzzy_hits =
                self.kb.fuzzy_match_entities(&query_folded, threshold, needed, |id| by_entity.contains_key(&id));
            for (ent_idx, alias, prior, sim) in fuzzy_hits {
                if let Some(c) = self.candidate_from_idx(ent_idx, alias, prior, sim, None) {
                    by_entity.insert(ent_idx, c);
                }
            }
        }

        // 3. Optional dense vector nearest neighbors from turbo-vector-index
        if let (Some((space, vec_slice)), Some(vindex)) = (query_vector, self.kb.vector_index())
            && let Ok(hits) = vindex.top_k(space, vec_slice, max_candidates)
        {
            for hit in hits {
                let ent_idx = *vindex.meta(hit.row);
                let norm_vec_score = ((hit.score as f64) + 1.0).mul_add(0.5, 0.0).clamp(0.0, 1.0);
                if let Some(existing) = by_entity.get_mut(&ent_idx) {
                    existing.vector_score =
                        Some(existing.vector_score.map_or(norm_vec_score, |prev| prev.max(norm_vec_score)));
                } else if by_entity.len() < max_candidates
                    && let Some(ent) = self.kb.entity(ent_idx)
                    && let Some(c) = self.candidate_from_idx(
                        ent_idx,
                        ent.name.clone(),
                        norm_vec_score * 0.5,
                        alias_similarity(&query_folded, &fold_alias(&ent.name)),
                        Some(norm_vec_score),
                    )
                {
                    by_entity.insert(ent_idx, c);
                }
            }
        }

        let mut out: Vec<RawCandidate> = by_entity.into_values().collect();
        out.sort_by(|a, b| {
            b.match_similarity
                .total_cmp(&a.match_similarity)
                .then_with(|| b.alias_prior.total_cmp(&a.alias_prior))
                .then_with(|| b.popularity_prior.total_cmp(&a.popularity_prior))
                .then_with(|| a.kb_id.cmp(&b.kb_id))
        });
        out.truncate(max_candidates);
        out
    }

    fn lookup_by_id(&self, kb_id: &str) -> Option<RawCandidate> {
        let (idx, ent) = self.kb.entity_by_id(kb_id)?;
        self.candidate_from_idx(idx, ent.name.clone(), 1.0, 1.0, None)
    }

    fn term_idf(&self, term: &str) -> f64 {
        self.kb.term_idf(term)
    }

    fn fingerprint(&self) -> u64 {
        self.kb.fingerprint()
    }
}

/// [`KbProvider`] adapter over a [`turbo_glossary_layers::Glossary`].
#[derive(Clone, Debug)]
pub struct GlossaryProvider {
    kb: CatalogProvider,
}

impl GlossaryProvider {
    /// Converts a [`Glossary`] into a [`KbProvider`] with entity type `default_type`.
    pub fn from_glossary(provider_id: impl Into<String>, glossary: &Glossary, default_type: &str) -> Self {
        let mut by_id: HashMap<String, crate::kb::KbEntity> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for (_, entry) in glossary.iter() {
            let ent = by_id.entry(entry.id.to_owned()).or_insert_with(|| {
                order.push(entry.id.to_owned());
                crate::kb::KbEntity::new(entry.id, entry.term).with_types([default_type]).with_prior(0.6)
            });
            ent.aliases.push((entry.term.to_owned(), 1.0));
        }
        let entities: Vec<_> = order.into_iter().filter_map(|id| by_id.remove(&id)).collect();
        let kb =
            KnowledgeBase::from_entities(provider_id, entities).expect("glossary entries have non-blank IDs and terms");
        Self { kb: CatalogProvider::new(kb) }
    }
}

impl KbProvider for GlossaryProvider {
    fn provider_id(&self) -> &str {
        self.kb.provider_id()
    }

    fn generate_candidates(
        &self,
        mention_text: &str,
        query_vector: Option<(&Space, &[f32])>,
        max_candidates: usize,
        fuzzy_threshold: f64,
    ) -> Vec<RawCandidate> {
        self.kb.generate_candidates(mention_text, query_vector, max_candidates, fuzzy_threshold)
    }

    fn lookup_by_id(&self, kb_id: &str) -> Option<RawCandidate> {
        self.kb.lookup_by_id(kb_id)
    }

    fn fingerprint(&self) -> u64 {
        self.kb.fingerprint()
    }
}
