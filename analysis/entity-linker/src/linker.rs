//! [`EntityLinker`] annotator and standalone disambiguation engine.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Error, Groups, Ix, Key, Layer, Provenance, Refs, View};
use turbo_analysis_layers::{Entities, Sentences, Tokens, Vectors};
use turbo_coref::Mentions;
use turbo_vector_index::{Normalization, Space};

use crate::layer::{LinkedEntities, NilReason};
use crate::providers::{KbProvider, RawCandidate};
use crate::scorer::{
    LinkerConfig, context_overlap_score, propagate_coref_candidates, score_all_mentions, type_compatibility,
};
use crate::similarity::{extract_context_terms, fold_alias};

/// Generic entity linker and [`Annotator`] that disambiguates [`Entities`] against
/// one or more [`KbProvider`]s and writes [`LinkedEntities`].
#[derive(Clone)]
pub struct EntityLinker {
    providers: Vec<Arc<dyn KbProvider>>,
    config: LinkerConfig,
    vector_normalization: Normalization,
}

impl std::fmt::Debug for EntityLinker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let provider_ids: Vec<&str> = self.providers.iter().map(|p| p.provider_id()).collect();
        f.debug_struct("EntityLinker")
            .field("providers", &provider_ids)
            .field("config", &self.config)
            .field("vector_normalization", &self.vector_normalization)
            .finish()
    }
}

impl Default for EntityLinker {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityLinker {
    /// Creates an [`EntityLinker`] with [`LinkerConfig::default()`] and no providers yet.
    pub fn new() -> Self {
        Self { providers: Vec::new(), config: LinkerConfig::default(), vector_normalization: Normalization::Unit }
    }

    /// Sets the [`LinkerConfig`].
    pub fn with_config(mut self, config: LinkerConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the [`Normalization`] assumed when querying a provider's vector index with `Vectors<Entities>`.
    pub fn with_vector_normalization(mut self, normalization: Normalization) -> Self {
        self.vector_normalization = normalization;
        self
    }

    /// Registers a [`KbProvider`] (such as [`crate::CatalogProvider`], [`crate::GazetteerProvider`],
    /// [`crate::WordNetProvider`], or [`crate::GlossaryProvider`]).
    pub fn add_provider<P: KbProvider + 'static>(mut self, provider: P) -> Self {
        self.providers.push(Arc::new(provider));
        self
    }

    /// Registers an already-wrapped `Arc<dyn KbProvider>`.
    pub fn add_shared_provider(mut self, provider: Arc<dyn KbProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    pub fn config(&self) -> &LinkerConfig {
        &self.config
    }

    pub fn providers(&self) -> &[Arc<dyn KbProvider>] {
        &self.providers
    }

    /// Links the default [`Entities`] layer on `doc` and returns the resulting [`LinkedEntities`].
    pub fn link(&self, doc: &Document) -> Result<LinkedEntities, Error> {
        let view = doc.require(&Key::<View>::DEFAULT)?;
        let entities = doc.require(&Key::<Entities>::DEFAULT)?;
        let sentences = doc.get_key(&Key::<Sentences>::DEFAULT);
        let tokens = doc.get_key(&Key::<Tokens>::DEFAULT);
        let mentions = doc.get_key(&Key::<Mentions>::DEFAULT);
        let mention_groups = doc.get_key(&Key::<Groups<Mentions>>::DEFAULT);
        let entity_groups = doc.get_key(&Key::<Groups<Entities>>::DEFAULT);
        let entity_vectors = doc.get_key(&Key::<Vectors<Entities>>::DEFAULT);
        let (linked,) = self.annotate((
            view,
            entities,
            sentences,
            tokens,
            mentions,
            mention_groups,
            entity_groups,
            entity_vectors,
        ))?;
        Ok(linked)
    }

    /// Links the default [`Entities`] layer on `doc` and adds [`LinkedEntities`] to `doc`.
    pub fn annotate_document(&self, doc: &mut Document) -> Result<(), Error> {
        let linked = self.link(doc)?;
        let mut reads = vec![Key::<View>::DEFAULT.id(), Key::<Entities>::DEFAULT.id()];
        for opt_id in [
            Key::<Sentences>::DEFAULT.id(),
            Key::<Tokens>::DEFAULT.id(),
            Key::<Mentions>::DEFAULT.id(),
            Key::<Groups<Mentions>>::DEFAULT.id(),
            Key::<Groups<Entities>>::DEFAULT.id(),
            Key::<Vectors<Entities>>::DEFAULT.id(),
        ] {
            if doc.present(opt_id) {
                reads.push(opt_id);
            }
        }
        doc.add(
            &Key::<LinkedEntities>::DEFAULT,
            linked,
            Provenance {
                annotator: Self::ID,
                version: Self::VERSION,
                config: self.config_hash(),
                model: None,
                execution: None,
                reads,
                on: Vec::new(),
            },
        )
    }
}

impl Annotator for EntityLinker {
    const ID: &'static str = "turbo:entity_linker";
    const VERSION: &'static str = "1";

    type Reads = (
        View,
        Entities,
        Option<Sentences>,
        Option<Tokens>,
        Option<Mentions>,
        Option<Groups<Mentions>>,
        Option<Groups<Entities>>,
        Option<Vectors<Entities>>,
    );
    type Writes = (LinkedEntities,);

    fn config_hash(&self) -> u64 {
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
        let w = &self.config.weights;
        let mut h = OFFSET;
        for bits in [
            w.prior_weight.to_bits(),
            w.type_weight.to_bits(),
            w.context_weight.to_bits(),
            w.coherence_weight.to_bits(),
            self.config.nil_threshold.to_bits(),
            self.config.min_type_compatibility.to_bits(),
            self.config.fuzzy_threshold.to_bits(),
            self.config.top_k as u64,
            self.config.max_candidates_per_provider as u64,
            u64::from(self.config.coref_propagation),
            self.config.context_window_tokens as u64,
            match self.vector_normalization {
                Normalization::Unit => 1,
                Normalization::None => 0,
            },
            self.providers.len() as u64,
        ] {
            h = mix(h, &bits.to_le_bytes());
        }
        for p in &self.providers {
            h = mix(h, p.provider_id().as_bytes());
            h = mix(h, &p.fingerprint().to_le_bytes());
        }
        h
    }

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.config.validate().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        let (view, entities, sentences, tokens, mentions, mention_groups, entity_groups, entity_vectors) = input;
        if entities.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("Entities view {:?} does not match input View {:?}", entities.spans.view(), view.id()),
            });
        }

        let n = entities.len();
        if n == 0 {
            return Ok((LinkedEntities::default(),));
        }

        let text = view.text();
        let provider_refs: Vec<&dyn KbProvider> = self.providers.iter().map(AsRef::as_ref).collect();

        // Extract surface text and NER kind for each entity
        let mut mention_texts = Vec::with_capacity(n);
        let mut ner_kinds = Vec::with_capacity(n);
        for i in 0..n {
            let ix = Ix::<Entities>::new(i as u32);
            let span = entities.spans.get(ix);
            let (s, e) = (span.start as usize, span.end as usize);
            if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("entity {i} span {s}..{e} is not valid UTF-8 in view of length {}", text.len()),
                });
            }
            mention_texts.push(text[s..e].to_owned());
            ner_kinds.push(entities.kinds.label(entities.kind[i]).to_owned());
        }

        // Precompute local and document context term sets per entity
        let (local_contexts, doc_contexts) = build_context_sets(
            text,
            entities,
            sentences.filter(|s| s.spans.view() == view.id()),
            tokens.filter(|t| t.spans.view() == view.id()),
            self.config.context_window_tokens,
        );

        let vector_space = entity_vectors.map(|v| Space::of(v, self.vector_normalization));

        // Step 1: Candidate generation across all providers
        let mut raw_per_mention: Vec<Vec<RawCandidate>> = Vec::with_capacity(n);
        let mut type_scores: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let ix = Ix::<Entities>::new(i as u32);
            let qvec = match (entity_vectors, &vector_space) {
                (Some(vecs), Some(space)) if i < vecs.len() && !vecs.is_empty(ix) => Some((space, vecs.row(ix))),
                _ => None,
            };

            let mut cands = Vec::new();
            for p in &self.providers {
                let mut pc = p.generate_candidates(
                    &mention_texts[i],
                    qvec,
                    self.config.max_candidates_per_provider,
                    self.config.fuzzy_threshold,
                );
                cands.append(&mut pc);
            }
            let ts: Vec<f64> = cands.iter().map(|c| type_compatibility(&ner_kinds[i], &c.entity_types)).collect();
            raw_per_mention.push(cands);
            type_scores.push(ts);
        }

        // Step 2: Optional coreference cluster candidate propagation
        let clusters = extract_entity_clusters(n, entities, mentions, mention_groups, entity_groups);
        let mut coref_boost: Vec<HashMap<(String, String), f64>> = vec![HashMap::new(); n];
        if self.config.coref_propagation && !clusters.is_empty() {
            propagate_coref_candidates(
                &clusters,
                &mention_texts,
                &mut raw_per_mention,
                &mut type_scores,
                &mut coref_boost,
            );
        }

        // Step 3: Compute context overlap scores for all candidates
        let mut context_scores: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let xs = raw_per_mention[i]
                .iter()
                .map(|c| context_overlap_score(c, &local_contexts[i], &doc_contexts[i], &provider_refs))
                .collect();
            context_scores.push(xs);
        }

        // Step 4: Score all candidates with coherence and apply NIL threshold
        let mut all_scored =
            score_all_mentions(&self.config, &raw_per_mention, &type_scores, &context_scores, &coref_boost);

        // Build coref cluster lookup for NIL clustering
        let mut entity_to_coref_group: Vec<Option<usize>> = vec![None; n];
        for (g_idx, members) in clusters.iter().enumerate() {
            for &m in members {
                if m < n {
                    entity_to_coref_group[m] = Some(g_idx);
                }
            }
        }
        let mut nil_key_to_cluster: HashMap<String, u32> = HashMap::new();
        let mut next_nil_cluster = 0u32;

        let mut out = LinkedEntities::with_capacity(n);
        for i in 0..n {
            let had_candidates = !raw_per_mention[i].is_empty();
            let had_type_compatible = type_scores[i].iter().any(|&ts| ts >= self.config.min_type_compatibility);
            let mut cands = std::mem::take(&mut all_scored[i]);

            let resolved = cands
                .iter()
                .find(|c| c.type_score >= self.config.min_type_compatibility && c.score >= self.config.nil_threshold)
                .cloned();
            let top_score = cands
                .iter()
                .find(|c| c.type_score >= self.config.min_type_compatibility)
                .or_else(|| cands.first())
                .map_or(0.0, |c| c.score);

            if self.config.top_k > 0 && cands.len() > self.config.top_k {
                cands.truncate(self.config.top_k);
            } else if self.config.top_k == 0 {
                cands.clear();
            }

            match resolved {
                Some(best) => {
                    out.kb_id.push(Some(best.kb_id));
                    out.provider_id.push(best.provider_id);
                    out.canonical_name.push(best.canonical_name);
                    out.matched_alias.push(best.matched_alias);
                    out.score.push(best.score);
                    out.nil_reason.push(None);
                    out.nil_cluster.push(None);
                }
                None => {
                    let reason = if !had_candidates {
                        NilReason::NoCandidates
                    } else if !had_type_compatible {
                        NilReason::TypeIncompatible
                    } else {
                        NilReason::BelowThreshold
                    };
                    let nil_key = if let Some(g) = entity_to_coref_group[i] {
                        format!("coref:{g}")
                    } else {
                        let f = fold_alias(&mention_texts[i]);
                        if f.is_empty() { format!("span:{i}") } else { format!("surf:{f}") }
                    };
                    let cluster_id = *nil_key_to_cluster.entry(nil_key).or_insert_with(|| {
                        let id = next_nil_cluster;
                        next_nil_cluster += 1;
                        id
                    });
                    out.kb_id.push(None);
                    out.provider_id.push(String::new());
                    out.canonical_name.push(String::new());
                    out.matched_alias.push(mention_texts[i].clone());
                    out.score.push(top_score);
                    out.nil_reason.push(Some(reason));
                    out.nil_cluster.push(Some(cluster_id));
                }
            }
            out.candidates.push(cands);
        }

        Ok((out,))
    }
}

fn build_context_sets(
    text: &str,
    entities: &Entities,
    sentences: Option<&Sentences>,
    tokens: Option<&Tokens>,
    window_tokens: usize,
) -> (Vec<HashSet<String>>, Vec<HashSet<String>>) {
    let n = entities.len();
    let mut local_sets = Vec::with_capacity(n);
    let mut doc_sets = Vec::with_capacity(n);
    let all_doc_terms = extract_context_terms(text);

    for i in 0..n {
        let ix = Ix::<Entities>::new(i as u32);
        let ent_span = entities.spans.get(ix);
        let ent_terms: HashSet<String> =
            extract_context_terms(&text[ent_span.start as usize..ent_span.end as usize]).into_iter().collect();

        // Local context: surrounding sentence if available, else token window, else character window
        let local_slice = if let Some(sents) = sentences
            && let Some((_, s_span)) =
                sents.spans.iter().find(|(_, s)| s.start <= ent_span.start && ent_span.end <= s.end)
        {
            &text[s_span.start as usize..s_span.end as usize]
        } else if let Some(toks) = tokens
            && toks.len() > 0
        {
            let range = entities.tokens[i];
            let start_tok = range.start.get().saturating_sub(window_tokens).min(toks.len() - 1);
            let end_tok = range.end.get().saturating_add(window_tokens).clamp(1, toks.len());
            let start_b = toks.spans.get(Ix::new(start_tok as u32)).start as usize;
            let end_b = toks.spans.get(Ix::new((end_tok - 1) as u32)).end as usize;
            let lo = start_b.min(text.len());
            let hi = end_b.clamp(lo, text.len());
            &text[lo..hi]
        } else {
            text
        };

        let local_set: HashSet<String> =
            extract_context_terms(local_slice).into_iter().filter(|t| !ent_terms.contains(t)).collect();
        let doc_set: HashSet<String> = all_doc_terms.iter().filter(|t| !ent_terms.contains(*t)).cloned().collect();
        local_sets.push(local_set);
        doc_sets.push(doc_set);
    }
    (local_sets, doc_sets)
}

fn extract_entity_clusters(
    n_entities: usize,
    entities: &Entities,
    mentions: Option<&Mentions>,
    mention_groups: Option<&Groups<Mentions>>,
    entity_groups: Option<&Groups<Entities>>,
) -> Vec<Vec<usize>> {
    let mut clusters = Vec::new();
    if let Some(eg) = entity_groups {
        for (_, members) in eg.iter() {
            let group: Vec<usize> = members.iter().map(|ix| ix.get()).filter(|&i| i < n_entities).collect();
            if group.len() >= 2 {
                clusters.push(group);
            }
        }
    }
    if let (Some(ments), Some(mg)) = (mentions, mention_groups) {
        for (_, members) in mg.iter() {
            let mut group = Vec::new();
            for &m_ix in members {
                if m_ix.get() >= ments.len() {
                    continue;
                }
                if let Some(e_ix) = ments.entity_of(m_ix) {
                    if e_ix.get() < n_entities && !group.contains(&e_ix.get()) {
                        group.push(e_ix.get());
                    }
                } else if ments.spans.view() == entities.spans.view() {
                    let m_span = ments.spans.get(m_ix);
                    if let Some((e_ix, _)) = entities.spans.iter().find(|(_, es)| *es == m_span)
                        && !group.contains(&e_ix.get())
                    {
                        group.push(e_ix.get());
                    }
                }
            }
            if group.len() >= 2 {
                clusters.push(group);
            }
        }
    }
    clusters
}
