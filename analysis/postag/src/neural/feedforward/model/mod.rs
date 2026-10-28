//! Feedforward neural POS model parameters, scoring cache, and binary
//! `ONLP-FFPT-1` / `ONLP-FFPT-2` serialization (`FeedforwardPOSModel`).

use std::collections::HashMap;
use std::sync::atomic::AtomicIsize;
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock, RwLock};

use turbo_analysis::ModelRef;

use super::context::{ABSENT, FeedforwardPosContext, PRETRAINED_SLOTS, SLOTS, SentenceFeatures, UNKNOWN};

mod io;
mod score;

const MAGIC_V1: &str = "ONLP-FFPT-1";
const MAGIC_V2: &str = "ONLP-FFPT-2";
pub(super) const CACHE_ENTRY_LIMIT: isize = 32768;
const INITIAL_CAPACITY: usize = 4096;
const MAX_LENGTH: i32 = 1 << 30;

#[derive(Debug)]
pub(super) struct ContributionCache {
    by_slot: Vec<Vec<OnceLock<Vec<f64>>>>,
    remaining: AtomicIsize,
}

impl ContributionCache {
    fn new(row_count: usize) -> Self {
        let mut by_slot = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let mut slot_vec = Vec::with_capacity(row_count);
            slot_vec.resize_with(row_count, OnceLock::new);
            by_slot.push(slot_vec);
        }
        Self { by_slot, remaining: AtomicIsize::new(CACHE_ENTRY_LIMIT) }
    }

    #[cfg(test)]
    pub(super) fn set_budget(&self, budget: isize) {
        self.remaining.store(budget, Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(super) fn budget(&self) -> isize {
        self.remaining.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(super) fn cached_entries(&self) -> usize {
        self.by_slot.iter().flat_map(|slot| slot.iter()).filter(|cell| cell.get().is_some()).count()
    }
}

/// Immutable parameters and vocabularies of the feedforward neural POS tagger (`ONLP-FFPT-1` / `ONLP-FFPT-2`).
#[derive(Clone, Debug)]
pub struct FeedforwardPosModel {
    word_entries: Vec<(String, usize)>,
    word_ids: HashMap<String, usize>,
    suffix_entries: Vec<(String, usize)>,
    suffix_ids: HashMap<String, usize>,
    shape_entries: Vec<(String, usize)>,
    shape_ids: HashMap<String, usize>,
    tag_feature_entries: Vec<(String, usize)>,
    tag_feature_ids: HashMap<String, usize>,
    tags: Vec<String>,
    embedding_size: usize,
    embeddings: Vec<Vec<f32>>,
    hidden_weights: Vec<Vec<f32>>,
    hidden_bias: Vec<f32>,
    output_weights: Vec<Vec<f32>>,
    output_bias: Vec<f32>,
    pretrained_size: usize,
    pretrained_entries: Vec<(String, usize)>,
    pretrained_ids: HashMap<String, usize>,
    pretrained_vectors: Vec<Vec<f32>>,
    pub(super) cache: Arc<RwLock<Option<Arc<ContributionCache>>>>,
}

impl FeedforwardPosModel {
    /// Creates a `FeedforwardPosModel` without pretrained word vectors (`ONLP-FFPT-1`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        word_entries: Vec<(String, usize)>,
        suffix_entries: Vec<(String, usize)>,
        shape_entries: Vec<(String, usize)>,
        tag_feature_entries: Vec<(String, usize)>,
        tags: Vec<String>,
        embedding_size: usize,
        embeddings: Vec<Vec<f32>>,
        hidden_weights: Vec<Vec<f32>>,
        hidden_bias: Vec<f32>,
        output_weights: Vec<Vec<f32>>,
        output_bias: Vec<f32>,
    ) -> Self {
        Self::with_pretrained(
            word_entries,
            suffix_entries,
            shape_entries,
            tag_feature_entries,
            tags,
            embedding_size,
            embeddings,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            0,
            Vec::new(),
            Vec::new(),
        )
    }

    /// Creates a `FeedforwardPosModel` with optional pretrained word vectors (`ONLP-FFPT-2` when `pretrained_size > 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn with_pretrained(
        word_entries: Vec<(String, usize)>,
        suffix_entries: Vec<(String, usize)>,
        shape_entries: Vec<(String, usize)>,
        tag_feature_entries: Vec<(String, usize)>,
        tags: Vec<String>,
        embedding_size: usize,
        embeddings: Vec<Vec<f32>>,
        hidden_weights: Vec<Vec<f32>>,
        hidden_bias: Vec<f32>,
        output_weights: Vec<Vec<f32>>,
        output_bias: Vec<f32>,
        pretrained_size: usize,
        pretrained_entries: Vec<(String, usize)>,
        pretrained_vectors: Vec<Vec<f32>>,
    ) -> Self {
        let word_ids = entries_to_map(&word_entries);
        let suffix_ids = entries_to_map(&suffix_entries);
        let shape_ids = entries_to_map(&shape_entries);
        let tag_feature_ids = entries_to_map(&tag_feature_entries);
        let pretrained_ids = entries_to_map(&pretrained_entries);
        Self {
            word_entries,
            word_ids,
            suffix_entries,
            suffix_ids,
            shape_entries,
            shape_ids,
            tag_feature_entries,
            tag_feature_ids,
            tags,
            embedding_size,
            embeddings,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            pretrained_size,
            pretrained_entries,
            pretrained_ids,
            pretrained_vectors,
            cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Maps the 12 context feature strings to their embedding row indices.
    pub fn feature_ids(&self, context: &[impl AsRef<str>; SLOTS]) -> [usize; SLOTS] {
        let mut ids = [0usize; SLOTS];
        // Words are matched case-insensitively, as Java's `normalize` does.
        for s in 0..5 {
            ids[s] = self.lookup(&self.word_ids, &FeedforwardPosContext::lowercase(context[s].as_ref()));
        }
        ids[5] = self.lookup(&self.suffix_ids, context[5].as_ref());
        ids[6] = self.lookup(&self.suffix_ids, context[6].as_ref());
        for s in 7..10 {
            ids[s] = self.lookup(&self.shape_ids, context[s].as_ref());
        }
        ids[10] = self.lookup(&self.tag_feature_ids, context[10].as_ref());
        ids[11] = self.lookup(&self.tag_feature_ids, context[11].as_ref());
        ids
    }

    /// Maps the 12 context features at `index` of a pre-lowercased sentence to
    /// their embedding row indices without allocating per-slot `String`s.
    pub(super) fn feature_ids_at(
        &self,
        sentence: &SentenceFeatures<'_>,
        index: usize,
        prev_tag: Option<&str>,
        prev_prev_tag: Option<&str>,
    ) -> [usize; SLOTS] {
        let idx = index as isize;
        [
            self.lookup(&self.word_ids, sentence.word_key(idx - 2)),
            self.lookup(&self.word_ids, sentence.word_key(idx - 1)),
            self.lookup(&self.word_ids, sentence.word_key(idx)),
            self.lookup(&self.word_ids, sentence.word_key(idx + 1)),
            self.lookup(&self.word_ids, sentence.word_key(idx + 2)),
            self.lookup(&self.suffix_ids, sentence.suffix_key(index, 2)),
            self.lookup(&self.suffix_ids, sentence.suffix_key(index, 3)),
            self.lookup(&self.shape_ids, sentence.shape_key(idx - 1)),
            self.lookup(&self.shape_ids, sentence.shape_key(idx)),
            self.lookup(&self.shape_ids, sentence.shape_key(idx + 1)),
            self.lookup(&self.tag_feature_ids, prev_tag.unwrap_or(ABSENT)),
            self.lookup(&self.tag_feature_ids, prev_prev_tag.unwrap_or(ABSENT)),
        ]
    }

    fn lookup(&self, table: &HashMap<String, usize>, symbol: &str) -> usize {
        table.get(symbol).copied().unwrap_or_else(|| table[UNKNOWN])
    }

    /// Resolves the 5 pretrained word-vector row indices (`-1` when absent) around `tokens[index]`.
    pub fn pretrained_rows(&self, tokens: &[impl AsRef<str>], index: usize) -> Option<[isize; PRETRAINED_SLOTS]> {
        if self.pretrained_size == 0 {
            return None;
        }
        let mut rows = [-1isize; PRETRAINED_SLOTS];
        let idx = index as isize;
        for w in 0..PRETRAINED_SLOTS {
            let pos = idx + (w as isize) - 2;
            if pos >= 0 && (pos as usize) < tokens.len() {
                let norm = FeedforwardPosContext::lowercase(tokens[pos as usize].as_ref());
                if let Some(&r) = self.pretrained_ids.get(&norm) {
                    rows[w] = r as isize;
                }
            }
        }
        Some(rows)
    }

    /// Resolves the 5 pretrained word-vector row indices (`-1` when absent)
    /// around `sentence[index]` without allocating per-window `String`s.
    pub(super) fn pretrained_rows_at(
        &self,
        sentence: &SentenceFeatures<'_>,
        index: usize,
    ) -> Option<[isize; PRETRAINED_SLOTS]> {
        if self.pretrained_size == 0 {
            return None;
        }
        let mut rows = [-1isize; PRETRAINED_SLOTS];
        let idx = index as isize;
        let len = sentence.len() as isize;
        for w in 0..PRETRAINED_SLOTS {
            let pos = idx + (w as isize) - 2;
            if pos >= 0
                && pos < len
                && let Some(&r) = self.pretrained_ids.get(sentence.word_key(pos))
            {
                rows[w] = r as isize;
            }
        }
        Some(rows)
    }

    /// Enables the thread-safe contribution cache (bounded at 32,768 `(slot, row)` entries).
    pub fn enable_scoring_cache(&self) {
        if let Ok(mut guard) = self.cache.write()
            && guard.is_none()
        {
            *guard = Some(Arc::new(ContributionCache::new(self.embeddings.len())));
        }
    }

    /// Returns the provenance [`ModelRef`] for this model (including the SHA-256 of its serialized bytes).
    pub fn model_ref(&self) -> ModelRef {
        let mut bytes = Vec::new();
        self.serialize(&mut bytes).expect("serializing an in-memory FeedforwardPosModel to Vec<u8> cannot fail");
        let revision = if self.pretrained_size > 0 { MAGIC_V2 } else { MAGIC_V1 };
        ModelRef {
            id: "opennlp:FeedforwardPOSModel".into(),
            revision: revision.into(),
            sha256: turbo_maxent::sha256(&bytes).to_hex(),
        }
    }

    /// Tag outcome inventory in index order.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Width of each discrete embedding row.
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    /// Learned discrete embedding table.
    pub fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// Hidden-layer weight matrix.
    pub fn hidden_weights(&self) -> &[Vec<f32>] {
        &self.hidden_weights
    }

    /// Hidden-layer bias vector.
    pub fn hidden_bias(&self) -> &[f32] {
        &self.hidden_bias
    }

    /// Output-layer weight matrix.
    pub fn output_weights(&self) -> &[Vec<f32>] {
        &self.output_weights
    }

    /// Output-layer bias vector.
    pub fn output_bias(&self) -> &[f32] {
        &self.output_bias
    }

    /// Mutable output-layer bias vector (for test fault injection).
    pub fn output_bias_mut(&mut self) -> &mut [f32] {
        &mut self.output_bias
    }

    /// Width of each pretrained word vector (`0` when unused).
    pub fn pretrained_size(&self) -> usize {
        self.pretrained_size
    }

    /// Whether the model includes a pretrained word-vector table.
    pub fn uses_pretrained_vectors(&self) -> bool {
        self.pretrained_size > 0
    }

    /// Frozen pretrained word-vector rows.
    pub fn pretrained_vectors(&self) -> &[Vec<f32>] {
        &self.pretrained_vectors
    }
}

fn entries_to_map(entries: &[(String, usize)]) -> HashMap<String, usize> {
    let mut map = HashMap::with_capacity(entries.len());
    for (k, v) in entries {
        map.insert(k.clone(), *v);
    }
    map
}
