//! Knowledge base entity records, CSV/TSV catalog loading, binary snapshot
//! serialization, normalized alias indexing, and optional vector index attachment.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use turbo_vector_index::VectorIndex;

use crate::similarity::{extract_context_terms, fold_alias};

const SNAPSHOT_MAGIC: &[u8; 8] = b"TBKB\x01\x00\x00\x00";

/// One entity record in a [`KnowledgeBase`].
#[derive(Clone, Debug, PartialEq)]
pub struct KbEntity {
    /// Canonical knowledge-base identifier (for example `"Q312"`, `"US-CA"`, `"00001740-n"`).
    pub id: String,
    /// Canonical display name (for example `"Apple Inc."`).
    pub name: String,
    /// Known surface aliases paired with positive link weights (unnormalized $P(e \mid m)$ counts or weights).
    pub aliases: Vec<(String, f64)>,
    /// Semantic or NER entity types (for example `["ORG", "COMPANY"]`, `["PERSON"]`, `["GPE", "LOC"]`).
    pub entity_types: Vec<String>,
    /// Short textual description or gloss used for local context overlap scoring.
    pub description: String,
    /// Global popularity or prominence prior in `[0.0, 1.0]`.
    pub popularity_prior: f64,
    /// Related KB entity IDs used for document-level coherence scoring.
    pub related_ids: Vec<String>,
}

impl KbEntity {
    /// Constructs a new [`KbEntity`] with default popularity `0.5` and empty optional fields.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            aliases: Vec::new(),
            entity_types: Vec::new(),
            description: String::new(),
            popularity_prior: 0.5,
            related_ids: Vec::new(),
        }
    }

    /// Builder helper to add an alias with weight.
    pub fn with_alias(mut self, alias: impl Into<String>, weight: f64) -> Self {
        self.aliases.push((alias.into(), weight));
        self
    }

    /// Builder helper to set entity types.
    pub fn with_types<I, S>(mut self, types: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.entity_types = types.into_iter().map(Into::into).collect();
        self
    }

    /// Builder helper to set description text.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Builder helper to set popularity prior in `[0.0, 1.0]`.
    pub fn with_prior(mut self, prior: f64) -> Self {
        self.popularity_prior = prior;
        self
    }

    /// Builder helper to set related KB entity IDs.
    pub fn with_related<I, S>(mut self, related: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.related_ids = related.into_iter().map(Into::into).collect();
        self
    }
}

/// Errors returned when building, parsing, or decoding a [`KnowledgeBase`].
#[derive(Clone, Debug, PartialEq)]
pub enum CatalogError {
    BlankProviderId,
    BlankEntityId { row: usize },
    DuplicateEntityId { id: String },
    BlankEntityName { id: String },
    InvalidPrior { id: String, value: f64 },
    InvalidAliasWeight { id: String, alias: String, weight: f64 },
    ParseError { line: usize, detail: String },
    SnapshotError(String),
    VectorIndexMismatch { detail: String },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatalogError::BlankProviderId => f.write_str("provider_id must not be blank"),
            CatalogError::BlankEntityId { row } => write!(f, "entity at row {row} has a blank id"),
            CatalogError::DuplicateEntityId { id } => write!(f, "duplicate entity id '{id}'"),
            CatalogError::BlankEntityName { id } => write!(f, "entity '{id}' has a blank canonical name"),
            CatalogError::InvalidPrior { id, value } => {
                write!(f, "entity '{id}' has popularity_prior {value} outside [0.0, 1.0]")
            }
            CatalogError::InvalidAliasWeight { id, alias, weight } => {
                write!(f, "entity '{id}' alias '{alias}' has invalid weight {weight} (must be finite and > 0)")
            }
            CatalogError::ParseError { line, detail } => write!(f, "catalog parse error at line {line}: {detail}"),
            CatalogError::SnapshotError(detail) => write!(f, "snapshot decode error: {detail}"),
            CatalogError::VectorIndexMismatch { detail } => write!(f, "vector index mismatch: {detail}"),
        }
    }
}

impl std::error::Error for CatalogError {}

/// An indexed posting entry for a folded alias.
#[derive(Clone, Debug)]
pub(crate) struct AliasPosting {
    pub entity_idx: u32,
    pub matched_alias: String,
    pub alias_prior: f64,
}

/// In-memory knowledge base catalog with normalized alias posting lists,
/// precomputed context terms and IDF weights, and optional [`VectorIndex`].
#[derive(Clone)]
pub struct KnowledgeBase {
    provider_id: String,
    entities: Vec<KbEntity>,
    id_to_idx: HashMap<String, u32>,
    alias_map: HashMap<String, Vec<AliasPosting>>,
    folded_keys: Vec<String>,
    fuzzy_index: crate::fuzzy_index::FuzzyAliasIndex,
    entity_context_terms: Vec<Vec<String>>,
    term_idf: HashMap<String, f64>,
    vector_index: Option<Arc<VectorIndex<u32>>>,
}

impl fmt::Debug for KnowledgeBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KnowledgeBase")
            .field("provider_id", &self.provider_id)
            .field("entities", &self.entities.len())
            .field("folded_keys", &self.folded_keys.len())
            .field("has_vector_index", &self.vector_index.is_some())
            .finish()
    }
}

impl KnowledgeBase {
    /// Builds an indexed [`KnowledgeBase`] from a provider identifier and entity records.
    pub fn from_entities(provider_id: impl Into<String>, entities: Vec<KbEntity>) -> Result<Self, CatalogError> {
        let provider_id = provider_id.into();
        if provider_id.trim().is_empty() {
            return Err(CatalogError::BlankProviderId);
        }

        let mut id_to_idx = HashMap::with_capacity(entities.len());
        let mut raw_alias_map: HashMap<String, Vec<(u32, String, f64)>> = HashMap::new();
        let mut entity_context_terms = Vec::with_capacity(entities.len());
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for (row, ent) in entities.iter().enumerate() {
            if ent.id.trim().is_empty() {
                return Err(CatalogError::BlankEntityId { row });
            }
            if ent.name.trim().is_empty() {
                return Err(CatalogError::BlankEntityName { id: ent.id.clone() });
            }
            if !ent.popularity_prior.is_finite() || !(0.0..=1.0).contains(&ent.popularity_prior) {
                return Err(CatalogError::InvalidPrior { id: ent.id.clone(), value: ent.popularity_prior });
            }
            let idx = row as u32;
            if id_to_idx.insert(ent.id.clone(), idx).is_some() {
                return Err(CatalogError::DuplicateEntityId { id: ent.id.clone() });
            }

            // Collect folded aliases for this entity, keeping max weight per folded key
            let mut per_entity_aliases: HashMap<String, (String, f64)> = HashMap::new();
            let canonical_folded = fold_alias(&ent.name);
            if !canonical_folded.is_empty() {
                per_entity_aliases.insert(canonical_folded, (ent.name.clone(), 1.0));
            }
            for (alias, weight) in &ent.aliases {
                if !weight.is_finite() || *weight <= 0.0 {
                    return Err(CatalogError::InvalidAliasWeight {
                        id: ent.id.clone(),
                        alias: alias.clone(),
                        weight: *weight,
                    });
                }
                let f = fold_alias(alias);
                if f.is_empty() {
                    continue;
                }
                match per_entity_aliases.get_mut(&f) {
                    Some((best_surf, best_w)) => {
                        if *weight > *best_w {
                            *best_surf = alias.clone();
                            *best_w = *weight;
                        }
                    }
                    None => {
                        per_entity_aliases.insert(f, (alias.clone(), *weight));
                    }
                }
            }
            for (folded, (surf, w)) in per_entity_aliases {
                raw_alias_map.entry(folded).or_default().push((idx, surf, w));
            }

            // Extract unique context terms from description + canonical name + types
            let mut terms = extract_context_terms(&ent.description);
            terms.extend(extract_context_terms(&ent.name));
            for t in &ent.entity_types {
                terms.extend(extract_context_terms(t));
            }
            terms.sort_unstable();
            terms.dedup();
            for term in &terms {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
            entity_context_terms.push(terms);
        }

        // Normalize alias priors P(e | m) per folded alias key and sort deterministically
        let mut alias_map = HashMap::with_capacity(raw_alias_map.len());
        let mut folded_keys = Vec::with_capacity(raw_alias_map.len());
        for (folded, list) in raw_alias_map {
            let max_w = list.iter().map(|(_, _, w)| *w).fold(0.0_f64, f64::max);
            let total_scaled: f64 = if max_w > 0.0 { list.iter().map(|(_, _, w)| *w / max_w).sum() } else { 0.0 };
            let mut postings: Vec<AliasPosting> = list
                .into_iter()
                .map(|(entity_idx, matched_alias, w)| AliasPosting {
                    entity_idx,
                    matched_alias,
                    alias_prior: if total_scaled > 0.0 { ((w / max_w) / total_scaled).clamp(0.0, 1.0) } else { 1.0 },
                })
                .collect();
            postings.sort_by(|a, b| {
                b.alias_prior
                    .total_cmp(&a.alias_prior)
                    .then_with(|| {
                        entities[b.entity_idx as usize]
                            .popularity_prior
                            .total_cmp(&entities[a.entity_idx as usize].popularity_prior)
                    })
                    .then_with(|| a.entity_idx.cmp(&b.entity_idx))
            });
            folded_keys.push(folded.clone());
            alias_map.insert(folded, postings);
        }
        folded_keys.sort_unstable();
        let fuzzy_index = crate::fuzzy_index::FuzzyAliasIndex::build(&folded_keys);

        let n_docs = entities.len().max(1) as f64;
        let term_idf = doc_freq
            .into_iter()
            .map(|(term, df)| {
                let idf = ((n_docs - (df as f64) + 0.5) / ((df as f64) + 0.5) + 1.0).ln().max(0.1);
                (term, idf)
            })
            .collect();

        Ok(Self {
            provider_id,
            entities,
            id_to_idx,
            alias_map,
            folded_keys,
            fuzzy_index,
            entity_context_terms,
            term_idf,
            vector_index: None,
        })
    }

    /// Attaches a [`VectorIndex<u32>`] whose row metadata values are entity indices in this knowledge base.
    pub fn with_vector_index(mut self, index: Arc<VectorIndex<u32>>) -> Result<Self, CatalogError> {
        for row in 0..index.len() as u32 {
            let ent_idx = *index.meta(row);
            if (ent_idx as usize) >= self.entities.len() {
                return Err(CatalogError::VectorIndexMismatch {
                    detail: format!(
                        "row {row} maps to entity index {ent_idx} outside {} entities",
                        self.entities.len()
                    ),
                });
            }
        }
        self.vector_index = Some(index);
        Ok(self)
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn entities(&self) -> &[KbEntity] {
        &self.entities
    }

    pub fn entity(&self, idx: u32) -> Option<&KbEntity> {
        self.entities.get(idx as usize)
    }

    pub fn entity_by_id(&self, id: &str) -> Option<(u32, &KbEntity)> {
        let &idx = self.id_to_idx.get(id)?;
        Some((idx, &self.entities[idx as usize]))
    }

    pub fn context_terms(&self, idx: u32) -> &[String] {
        self.entity_context_terms.get(idx as usize).map_or(&[], Vec::as_slice)
    }

    pub fn term_idf(&self, term: &str) -> f64 {
        self.term_idf.get(term).copied().unwrap_or(1.0)
    }

    pub(crate) fn exact_postings(&self, folded_key: &str) -> Option<&[AliasPosting]> {
        self.alias_map.get(folded_key).map(Vec::as_slice)
    }

    pub(crate) fn fuzzy_match_entities(
        &self,
        query_folded: &str,
        threshold: f64,
        needed: usize,
        should_skip_entity: impl Fn(u32) -> bool,
    ) -> Vec<(u32, String, f64, f64)> {
        if needed == 0 || query_folded.is_empty() {
            return Vec::new();
        }
        let mut best_fuzzy: HashMap<u32, (f64, f64, String)> = HashMap::new();
        let mut sims_buf: Vec<f64> = Vec::new();
        self.fuzzy_index.for_each_candidate(query_folded, threshold, |k_idx, running_cutoff| {
            let key = &self.folded_keys[k_idx as usize];
            if key == query_folded {
                return;
            }
            let sim = crate::similarity::alias_similarity(query_folded, key);
            if sim < *running_cutoff || sim < threshold {
                return;
            }
            let Some(postings) = self.exact_postings(key) else {
                return;
            };
            let mut updated = false;
            for p in postings {
                if should_skip_entity(p.entity_idx) {
                    continue;
                }
                let scaled_prior = p.alias_prior * sim;
                match best_fuzzy.get_mut(&p.entity_idx) {
                    Some(entry) => {
                        if sim
                            .total_cmp(&entry.0)
                            .then_with(|| scaled_prior.total_cmp(&entry.1))
                            .then_with(|| entry.2.cmp(&p.matched_alias))
                            .is_gt()
                        {
                            *entry = (sim, scaled_prior, p.matched_alias.clone());
                            updated = true;
                        }
                    }
                    None => {
                        best_fuzzy.insert(p.entity_idx, (sim, scaled_prior, p.matched_alias.clone()));
                        updated = true;
                    }
                }
            }
            if updated && best_fuzzy.len() >= needed {
                sims_buf.clear();
                sims_buf.extend(best_fuzzy.values().map(|&(s, _, _)| s));
                let (_, &mut kth_sim, _) = sims_buf.select_nth_unstable_by(needed - 1, |a, b| b.total_cmp(a));
                let new_cutoff = (kth_sim - 1e-9).max(threshold);
                if new_cutoff > *running_cutoff {
                    *running_cutoff = new_cutoff;
                }
            }
        });
        let mut hits: Vec<(u32, String, f64, f64)> = best_fuzzy
            .into_iter()
            .map(|(ent_idx, (sim, scaled_prior, alias))| (ent_idx, alias, scaled_prior, sim))
            .collect();
        hits.sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| b.2.total_cmp(&a.2)).then_with(|| a.0.cmp(&b.0)));
        hits.truncate(needed);
        hits
    }

    pub(crate) fn vector_index(&self) -> Option<&Arc<VectorIndex<u32>>> {
        self.vector_index.as_ref()
    }

    /// Serializes the catalog into a deterministic binary snapshot.
    pub fn encode_snapshot(&self) -> Vec<u8> {
        crate::snapshot::encode_kb(self)
    }

    /// Restores a [`KnowledgeBase`] from a binary snapshot produced by [`Self::encode_snapshot`].
    pub fn decode_snapshot(bytes: &[u8]) -> Result<Self, CatalogError> {
        crate::snapshot::decode_kb(bytes, SNAPSHOT_MAGIC)
    }

    /// Parses a tab-separated catalog (`id\tname\taliases\ttypes\tdescription\tprior[\trelated_ids]`).
    pub fn from_tsv(provider_id: impl Into<String>, text: &str) -> Result<Self, CatalogError> {
        crate::snapshot::parse_delimited(provider_id.into(), text, '\t')
    }

    /// Parses a comma-separated catalog (`id,name,aliases,types,description,prior[,related_ids]`).
    pub fn from_csv(provider_id: impl Into<String>, text: &str) -> Result<Self, CatalogError> {
        crate::snapshot::parse_delimited(provider_id.into(), text, ',')
    }

    /// Computes a deterministic 64-bit FNV-1a content fingerprint over the catalog.
    pub fn fingerprint(&self) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mix = |mut h: u64, bytes: &[u8]| -> u64 {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(PRIME);
            }
            h ^= 0xff;
            h.wrapping_mul(PRIME)
        };
        let mut h = mix(OFFSET, self.provider_id.as_bytes());
        h = mix(h, &(self.entities.len() as u64).to_le_bytes());
        for ent in &self.entities {
            h = mix(h, ent.id.as_bytes());
            h = mix(h, ent.name.as_bytes());
            h = mix(h, &ent.popularity_prior.to_bits().to_le_bytes());
            for (alias, w) in &ent.aliases {
                h = mix(h, alias.as_bytes());
                h = mix(h, &w.to_bits().to_le_bytes());
            }
            for t in &ent.entity_types {
                h = mix(h, t.as_bytes());
            }
            h = mix(h, ent.description.as_bytes());
            for rel in &ent.related_ids {
                h = mix(h, rel.as_bytes());
            }
        }
        if let Some(vi) = &self.vector_index {
            h = mix(h, &(vi.len() as u64).to_le_bytes());
            h = mix(h, &(vi.space().dim as u64).to_le_bytes());
        }
        h
    }
}
