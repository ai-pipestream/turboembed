//! In-memory search index registry, collection statistics, aliases, and
//! vocabulary drift watchers (`#1221`).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use tokio::sync::broadcast;
use turbo::TURBO_TASK_EMBED;
use turbo_analysis::{ModelRef, Span};
use turbo_chunker::terms::{Bm25PostingIndex, Bm25PostingIndexBuilder};
use turbo_query::QueryAnalysisProfile;
use turbo_vector_index::{Metric, Normalization, QuantizedIndex, Space, TokenIndex, TurboQuant, VectorIndex};

use crate::api::{
    Failure, Result, TURBO_E_BUNDLE_INTEGRITY, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT,
    TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED, field,
};
use crate::proto::search::{
    ChunkInput, CollectionEvent, CollectionInfo, DistanceMetric, IndexSpec, IndexSummary, ModelIdentity,
    NormalizationMode, SearchProviderDescriptor, SpaceSpec, SparseEntryProto, TermFrequencyEntry, TokenVectorInput,
    VectorStorageKind, VocabularyDriftReport,
};
use crate::state::{Loaded, State};

pub const ALREADY_EXISTS_FIELD: u32 = u32::MAX;

pub fn shared_query_profile() -> &'static QueryAnalysisProfile {
    static PROFILE: OnceLock<QueryAnalysisProfile> = OnceLock::new();
    PROFILE.get_or_init(QueryAnalysisProfile::new)
}

#[derive(Clone, Debug)]
pub struct StoredChunk {
    pub chunk_id: String,
    pub doc_id: String,
    pub level_kind: String,
    pub span: Span,
    pub parent_span: Option<Span>,
    pub text: String,
    pub parent_text: String,
    pub terms: Vec<(String, Span)>,
    pub dense_vector: Option<Vec<f32>>,
    pub sparse_terms: Vec<SparseEntryProto>,
    pub token_vectors: Vec<TokenVectorInput>,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct RetainedDocument {
    pub doc_id: String,
    pub text: String,
    pub graph: Vec<u8>,
    pub raw_chunks: Vec<ChunkInput>,
    pub chunks: Vec<ChunkInput>,
    pub attributes: BTreeMap<String, String>,
}

pub enum CompiledDense {
    Empty,
    Flat(VectorIndex<u32>),
    Quantized(Box<QuantizedIndex<u32>>),
}

pub struct HybridIndex {
    pub spec: IndexSpec,
    pub space: Space,
    pub metric: Metric,
    pub storage: VectorStorageKind,
    pub quant: Option<TurboQuant>,
    pub sealed: bool,
    pub docs: Vec<RetainedDocument>,
    pub chunks: Vec<StoredChunk>,
    pub bm25: Bm25PostingIndex,
    pub dense: CompiledDense,
    pub token_index: Option<TokenIndex<u32>>,
    pub token_candidates: Vec<u32>,
    pub seen_terms: BTreeSet<String>,
    pub term_df: BTreeMap<String, u64>,
}

#[derive(Clone)]
pub struct CollectionState {
    pub name: String,
    pub baseline_terms: BTreeSet<String>,
    pub known_terms: BTreeSet<String>,
    pub term_df: BTreeMap<String, u64>,
    pub total_units: u64,
    pub total_length: u64,
    pub oov_alert_threshold: f64,
}

impl CollectionState {
    pub fn to_info(&self) -> CollectionInfo {
        let mean_length = if self.total_units == 0 { 0.0 } else { self.total_length as f64 / self.total_units as f64 };
        CollectionInfo {
            collection_name: self.name.clone(),
            total_units: self.total_units,
            mean_length,
            vocabulary_size: self.known_terms.len() as u64,
            oov_alert_threshold: self.oov_alert_threshold,
            top_terms: top_df_entries(&self.term_df, 16),
        }
    }
}

pub fn top_df_entries(df: &BTreeMap<String, u64>, limit: usize) -> Vec<TermFrequencyEntry> {
    let mut items: Vec<(&String, &u64)> = df.iter().collect();
    items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    items.into_iter().take(limit).map(|(t, &c)| TermFrequencyEntry { term: t.clone(), document_frequency: c }).collect()
}

pub struct SearchRegistry {
    pub indexes: RwLock<BTreeMap<String, Arc<RwLock<HybridIndex>>>>,
    pub aliases: RwLock<BTreeMap<String, String>>,
    pub collections: RwLock<BTreeMap<String, CollectionState>>,
    pub events: broadcast::Sender<CollectionEvent>,
    pub snapshot_dir: RwLock<PathBuf>,
}

impl Default for SearchRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchRegistry {
    pub fn new() -> Self {
        let (events, _) = broadcast::channel(256);
        let default_snap = std::env::temp_dir().join(format!("turboembed-snapshots-{}", std::process::id()));
        Self {
            indexes: RwLock::new(BTreeMap::new()),
            aliases: RwLock::new(BTreeMap::new()),
            collections: RwLock::new(BTreeMap::new()),
            events,
            snapshot_dir: RwLock::new(default_snap),
        }
    }

    pub fn set_snapshot_dir(&self, dir: PathBuf) {
        *self.snapshot_dir.write().unwrap_or_else(|e| e.into_inner()) = dir;
    }

    pub fn snapshot_dir(&self) -> PathBuf {
        self.snapshot_dir.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn resolve_index(&self, name_or_alias: &str) -> Result<(String, Arc<RwLock<HybridIndex>>)> {
        if name_or_alias.is_empty() {
            return Err(bad_arg("index_name must not be empty"));
        }
        let target = {
            let aliases = self.aliases.read().unwrap_or_else(|e| e.into_inner());
            aliases.get(name_or_alias).cloned().unwrap_or_else(|| name_or_alias.to_owned())
        };
        let indexes = self.indexes.read().unwrap_or_else(|e| e.into_inner());
        let ix = indexes
            .get(&target)
            .cloned()
            .ok_or_else(|| not_found(format!("no search index or alias `{name_or_alias}` exists")))?;
        Ok((target, ix))
    }

    pub(crate) fn create_index(&self, state: &State, spec: IndexSpec) -> Result<IndexSummary> {
        if spec.index_name.is_empty() {
            return Err(bad_arg("spec.index_name must not be empty"));
        }
        let space = parse_space(spec.space.as_ref())?;
        if !spec.embed_bundle.is_empty() {
            verify_embed_bundle_matches_space(state, &spec.embed_bundle, &space)?;
        }
        let metric = parse_metric(spec.metric)?;
        let storage = VectorStorageKind::try_from(spec.storage)
            .map_err(|_| bad_arg(format!("unknown storage kind {}", spec.storage)))?;
        let quant = match storage {
            VectorStorageKind::FlatF32 => None,
            VectorStorageKind::TurboQuant => {
                let bits = if spec.quant_bits == 0 { 4 } else { spec.quant_bits as u8 };
                let tq = TurboQuant::new(bits, spec.quant_calibrate)
                    .map_err(|e| bad_arg(format!("invalid TurboQuant configuration: {e}")))?;
                if !space.dim.is_multiple_of(8) {
                    return Err(bad_arg(format!(
                        "TurboQuant requires space.dim ({}) to be a multiple of 8",
                        space.dim
                    )));
                }
                Some(tq)
            }
        };
        let mut indexes = self.indexes.write().unwrap_or_else(|e| e.into_inner());
        if indexes.contains_key(&spec.index_name) {
            return Err(already_exists(format!("search index `{}` already exists", spec.index_name)));
        }
        let hybrid = HybridIndex {
            spec: spec.clone(),
            space,
            metric,
            storage,
            quant,
            sealed: false,
            docs: Vec::new(),
            chunks: Vec::new(),
            bm25: Bm25PostingIndexBuilder::new().build(),
            dense: CompiledDense::Empty,
            token_index: None,
            token_candidates: Vec::new(),
            seen_terms: BTreeSet::new(),
            term_df: BTreeMap::new(),
        };
        let summary = hybrid.summary();
        indexes.insert(spec.index_name.clone(), Arc::new(RwLock::new(hybrid)));
        Ok(summary)
    }

    pub fn record_collection_update(
        &self,
        collection_name: &str,
        source_index: &str,
        chunk_terms: &[Vec<String>],
    ) -> Option<VocabularyDriftReport> {
        let mut cols = self.collections.write().unwrap_or_else(|e| e.into_inner());
        let state = cols.entry(collection_name.to_owned()).or_insert_with(|| CollectionState {
            name: collection_name.to_owned(),
            baseline_terms: BTreeSet::new(),
            known_terms: BTreeSet::new(),
            term_df: BTreeMap::new(),
            total_units: 0,
            total_length: 0,
            oov_alert_threshold: 0.0,
        });

        let mut total_occurrences = 0u64;
        let mut oov_occurrences = 0u64;
        let mut new_terms_set = BTreeSet::new();
        let baseline_non_empty = !state.baseline_terms.is_empty();

        for unit in chunk_terms {
            if !unit.is_empty() {
                state.total_units += 1;
                state.total_length += unit.len() as u64;
            }
            let mut unit_unique = BTreeSet::new();
            for t in unit {
                total_occurrences += 1;
                let in_base =
                    if baseline_non_empty { state.baseline_terms.contains(t) } else { state.known_terms.contains(t) };
                if !in_base {
                    oov_occurrences += 1;
                    new_terms_set.insert(t.clone());
                }
                state.known_terms.insert(t.clone());
                unit_unique.insert(t.clone());
            }
            for t in unit_unique {
                *state.term_df.entry(t).or_insert(0) += 1;
            }
        }
        let oov_ratio = if total_occurrences == 0 { 0.0 } else { oov_occurrences as f64 / total_occurrences as f64 };
        let vocabulary_coverage = (1.0 - oov_ratio).clamp(0.0, 1.0);
        let report = VocabularyDriftReport {
            collection_name: collection_name.to_owned(),
            vocabulary_coverage,
            oov_ratio,
            new_terms: new_terms_set.into_iter().collect(),
            top_terms: top_df_entries(&state.term_df, 16),
            total_units: state.total_units,
        };
        if oov_ratio >= state.oov_alert_threshold {
            let _ = self.events.send(CollectionEvent {
                collection_name: collection_name.to_owned(),
                source_index: source_index.to_owned(),
                report: Some(report.clone()),
            });
        }
        Some(report)
    }
}

impl HybridIndex {
    pub fn summary(&self) -> IndexSummary {
        let dense_row_count = match &self.dense {
            CompiledDense::Empty => 0,
            CompiledDense::Flat(ix) => ix.len() as u32,
            CompiledDense::Quantized(ix) => ix.len() as u32,
        };
        let sparse_row_count = self.chunks.iter().filter(|c| !c.sparse_terms.is_empty()).count() as u32;
        let token_candidate_count = self.token_index.as_ref().map_or(0, |t| t.len() as u32);
        IndexSummary {
            index_name: self.spec.index_name.clone(),
            space: Some(space_to_proto(&self.space)),
            metric: match self.metric {
                Metric::Cosine => DistanceMetric::Cosine as i32,
                Metric::Dot => DistanceMetric::Dot as i32,
            },
            storage: self.storage as i32,
            quant_bits: self.quant.map_or(0, |q| u32::from(q.bits)),
            sealed: self.sealed,
            document_count: self.docs.len() as u32,
            chunk_count: self.chunks.len() as u32,
            dense_row_count,
            sparse_row_count,
            token_candidate_count,
            collection_name: self.spec.collection_name.clone(),
        }
    }
}

pub(crate) fn verify_embed_bundle_matches_space(
    state: &State,
    bundle_name: &str,
    space: &Space,
) -> Result<Arc<Loaded>> {
    let (_, loaded) = state.ready(bundle_name, "")?;
    if loaded.task != TURBO_TASK_EMBED {
        return Err(bad_arg(format!("bundle `{bundle_name}` is not a TASK_EMBED model")));
    }
    let mi = &loaded.model_info;
    let id = field(&mi.model_id);
    let rev = field(&mi.revision);
    let sha = field(&mi.manifest_sha256);
    if id != space.model.id || rev != space.model.revision || sha != space.model.sha256 {
        return Err(bad_arg(format!(
            "embed_bundle `{bundle_name}` model ({id}@{rev}#{sha}) does not match SpaceSpec ({}@{}#{})",
            space.model.id, space.model.revision, space.model.sha256
        )));
    }
    let extra = &mi.output_dims[..(mi.output_dims_count as usize).min(mi.output_dims.len())];
    if mi.dim != space.dim && !extra.contains(&space.dim) {
        return Err(bad_arg(format!(
            "embed_bundle `{bundle_name}` does not support SpaceSpec.dim {} (model dim {})",
            space.dim, mi.dim
        )));
    }
    Ok(loaded)
}

pub fn parse_space(spec: Option<&SpaceSpec>) -> Result<Space> {
    let Some(s) = spec else {
        return Err(bad_arg("SpaceSpec is required"));
    };
    if s.dim == 0 {
        return Err(bad_arg("SpaceSpec.dim must be > 0"));
    }
    let m = s.model.as_ref().ok_or_else(|| bad_arg("SpaceSpec.model is required"))?;
    if m.id.is_empty() {
        return Err(bad_arg("SpaceSpec.model.id must not be empty"));
    }
    if m.revision.is_empty() {
        return Err(bad_arg("SpaceSpec.model.revision must not be empty"));
    }
    if m.sha256.len() != 64 || !m.sha256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(bad_arg("SpaceSpec.model.sha256 must be a 64-character lowercase hex digest"));
    }
    let norm = match NormalizationMode::try_from(s.normalization) {
        Ok(NormalizationMode::None) => Normalization::None,
        Ok(NormalizationMode::Unit) => Normalization::Unit,
        Err(_) => return Err(bad_arg(format!("unknown normalization mode {}", s.normalization))),
    };
    Ok(Space::new(ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: m.sha256.clone() }, s.dim, norm))
}

pub fn space_to_proto(space: &Space) -> SpaceSpec {
    SpaceSpec {
        model: Some(ModelIdentity {
            id: space.model.id.clone(),
            revision: space.model.revision.clone(),
            sha256: space.model.sha256.clone(),
        }),
        dim: space.dim,
        normalization: match space.normalization {
            Normalization::None => NormalizationMode::None as i32,
            Normalization::Unit => NormalizationMode::Unit as i32,
        },
    }
}

pub fn parse_metric(raw: i32) -> Result<Metric> {
    match DistanceMetric::try_from(raw) {
        Ok(DistanceMetric::Cosine) => Ok(Metric::Cosine),
        Ok(DistanceMetric::Dot) => Ok(Metric::Dot),
        Err(_) => Err(bad_arg(format!("unknown distance metric {raw}"))),
    }
}

pub fn providers_list() -> Vec<SearchProviderDescriptor> {
    vec![
        SearchProviderDescriptor {
            id: "turbo-chunker:bm25-posting".into(),
            kind: "lexical".into(),
            description: "BM25 inverted posting index with MaxScore pruning and positional phrase slop".into(),
        },
        SearchProviderDescriptor {
            id: "turbo-vector-index:flat-f32".into(),
            kind: "dense".into(),
            description: "Exact deterministic SIMD f32 dot/cosine vector index with Space verification".into(),
        },
        SearchProviderDescriptor {
            id: "turbo-vector-index:turboquant".into(),
            kind: "quantized_dense".into(),
            description: "TurboQuant 2-to-4-bit quantized vector index with calibration".into(),
        },
        SearchProviderDescriptor {
            id: "turbo-analysis-layers:sparse-terms".into(),
            kind: "sparse".into(),
            description: "Learned SPLADE sparse term inverted weight scorer".into(),
        },
        SearchProviderDescriptor {
            id: "turbo-vector-index:token-maxsim".into(),
            kind: "late_interaction".into(),
            description: "ColBERT-style late-interaction MaxSim scorer with winning subword token attribution".into(),
        },
        SearchProviderDescriptor {
            id: "turbo-query:profile".into(),
            kind: "query_analyzer".into(),
            description: "Short-query BM25 analyzer and structured temporal/money/quantity/place/entity extractor"
                .into(),
        },
    ]
}

pub fn bad_arg(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg.into())
}

pub fn bad_state(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_STATE, 0, msg.into())
}

pub fn not_found(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, msg.into())
}

pub fn already_exists(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_STATE, ALREADY_EXISTS_FIELD, msg.into())
}

pub fn corrupt_snapshot(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_BUNDLE_INTEGRITY, 0, msg.into())
}

pub fn io_err(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INTERNAL, 0, msg.into())
}

pub fn not_admin(what: &str) -> Failure {
    Failure::new(TURBO_E_UNSUPPORTED, 0, format!("{what} is admin-only: start the server with --admin to serve it"))
}

pub type ScoreMap = HashMap<u32, (f64, Vec<crate::proto::search::MatchedSpan>)>;
