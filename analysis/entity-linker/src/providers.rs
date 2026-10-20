//! Pluggable [`KbProvider`] backends ([`CatalogProvider`], [`GazetteerProvider`],
//! [`WordNetProvider`], and [`GlossaryProvider`]) for candidate generation.

use std::collections::HashMap;
use std::sync::Arc;

use turbo_geo::{FuzzyStringScorer, Gazetteer};
use turbo_glossary_layers::Glossary;
use turbo_vector_index::Space;
use turbo_wordnet::{HypernymTyper, LexicalKnowledgeBase, Synset, WordNetPos, WordNetRelation};

use crate::kb::KnowledgeBase;
use crate::similarity::{alias_similarity, extract_context_terms, fold_alias};

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

    /// Looks up a single entity by its canonical `kb_id`.
    fn lookup_by_id(&self, kb_id: &str) -> Option<RawCandidate>;

    /// Returns the inverse document frequency weight of a context term, if known.
    fn term_idf(&self, _term: &str) -> f64 {
        1.0
    }

    /// Returns a 64-bit content fingerprint of this provider for provenance hashing.
    fn fingerprint(&self) -> u64 {
        self.provider_id()
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x1000_0000_01b3))
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
            let mut fuzzy_hits: Vec<(u32, String, f64, f64)> = Vec::new();
            let keys = self.kb.folded_keys();
            for k_idx in self.kb.fuzzy_candidate_keys(&query_folded, threshold) {
                let key = &keys[k_idx as usize];
                if key != &query_folded
                    && let sim = alias_similarity(&query_folded, key)
                    && sim >= threshold
                    && let Some(postings) = self.kb.exact_postings(key)
                {
                    for p in postings {
                        fuzzy_hits.push((p.entity_idx, p.matched_alias.clone(), p.alias_prior * sim, sim));
                    }
                }
            }
            fuzzy_hits
                .sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| b.2.total_cmp(&a.2)).then_with(|| a.0.cmp(&b.0)));
            for (ent_idx, alias, prior, sim) in fuzzy_hits {
                if by_entity.len() < max_candidates || by_entity.contains_key(&ent_idx) {
                    by_entity.entry(ent_idx).or_insert_with(|| {
                        self.candidate_from_idx(ent_idx, alias, prior, sim, None).expect("valid entity index")
                    });
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
