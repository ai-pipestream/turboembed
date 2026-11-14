//! Unified pure-Rust thick client for `turboembed` (`#2338`).
//!
//! Combines zero-RPC in-process lexical lattices (`#2340`), one-pass document
//! NLP projection (`#2345`), one-pass query analysis (`#2339`), memory-mapped
//! stem-to-surface completion and contextual spellcheck (`#2341`), optional
//! in-process `libturbo` neural sessions (`feature = "local-core"`), and
//! runtime-scoped pooled `turbo-kserve` gRPC offload (`feature = "grpc"`)
//! behind cryptographic [`AnalysisFingerprint`] and [`BundleScoringFingerprint`]
//! handshakes.

pub mod error;
pub mod fingerprint;
pub mod lexical;
pub mod nlp;

#[cfg(feature = "local-core")]
pub mod local;

#[cfg(feature = "grpc")]
pub mod coalesce;
#[cfg(feature = "grpc")]
pub mod remote;

use std::collections::HashMap;
#[cfg(feature = "local-core")]
use std::path::PathBuf;
use std::sync::Arc;

pub use error::{ClientError, Result};
pub use fingerprint::{
    AnalysisFingerprint, BundleScoringFingerprint, BundleScoringSpec, FieldAnalysisConfig, IndexGenerationHandshake,
    NormalizerKind, StemmerKind, TermSourceKind, TokenizerKind,
};
pub use lexical::{
    AnalyzedField, AnalyzedSentence, AnalyzedTokenArc, DualAnalyzedField, LexicalAnalyzer, ProjectedSpanSet,
};
pub use nlp::{
    DocumentNlpProjection, DocumentSketch, EntityPolarityHit, FieldQueryPlan, FieldQuerySpec, FieldTokenizerKind,
    GeoHit, MappedSurfaceLexicon, MoneyHit, OnePassDocumentNlp, OnePassNlpBuilder, OnePassQueryAnalyzer,
    OnePassQueryBuilder, OnePassQueryPlan, OrdinalQueryTerm, QualitySummary, QuantityHit, QueryComparisonOp,
    QuerySpellResult, QuerySpellchecker, QuerySynonymMatch, SentimentSummary, SpellEdit, SpellcheckConfig, StemSurface,
    StructuredFilterPayload, StructuredFilterPlan, SurfaceCompletion, SurfaceLexiconBuilder, TemporalHit,
    UtcDateSummary, VocabularyCorpusIdf,
};

#[cfg(feature = "local-core")]
pub use local::{
    EmbedPromptRole, ImpactBits, ImpactQuantizerConfig, LocalEmbedEngine, LocalEncoderOptions, LocalHighlightSnippet,
    LocalPrecision, LocalRerankEngine, LocalRerankOptions, LocalRerankSessionOptions, MultiResOptions, MultiResVectors,
    QuantizedImpacts, RerankCalibrator, RerankScore, SparseOptions, VectorTier, VoronoiCentroids, VoronoiOptions,
    calibrate_voronoi_centroids,
};

#[cfg(feature = "grpc")]
pub use coalesce::{CoalesceConfig, EmbedCoalescer};
#[cfg(feature = "tls")]
pub use remote::RemoteTlsConfig;
#[cfg(feature = "grpc")]
pub use remote::{RemoteChannelPool, RemoteClient, RemoteEndpointConfig, RemotePriority, proto};

/// Builder assembling an immutable, `Send + Sync` [`TurboClient`].
#[derive(Default)]
pub struct TurboClientBuilder {
    default_field: FieldAnalysisConfig,
    named_fields: HashMap<String, FieldAnalysisConfig>,
    handshake: Option<IndexGenerationHandshake>,
    nlp: Option<OnePassDocumentNlp>,
    query: Option<OnePassQueryAnalyzer>,
    #[cfg(feature = "local-core")]
    local_embed: Option<(PathBuf, LocalEncoderOptions, Option<SparseOptions>)>,
    #[cfg(feature = "local-core")]
    local_rerank: Option<(PathBuf, LocalRerankSessionOptions, RerankCalibrator)>,
    #[cfg(feature = "grpc")]
    remote_cfg: Option<(RemoteEndpointConfig, Option<(String, String)>)>,
    #[cfg(feature = "grpc")]
    coalesce_cfg: Option<(String, Option<String>, CoalesceConfig)>,
}

impl TurboClientBuilder {
    /// Creates a new builder with `FieldAnalysisConfig::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the default lexical field configuration verified against `handshake.analysis`.
    pub fn with_default_field(mut self, config: FieldAnalysisConfig) -> Self {
        self.default_field = config;
        self
    }

    /// Registers an additional per-field lexical configuration.
    pub fn with_field(mut self, field_name: impl Into<String>, config: FieldAnalysisConfig) -> Self {
        self.named_fields.insert(field_name.into(), config);
        self
    }

    /// Pins an [`IndexGenerationHandshake`] verified during `build` / `connect`.
    pub fn with_handshake(mut self, handshake: IndexGenerationHandshake) -> Self {
        self.handshake = Some(handshake);
        self
    }

    /// Overrides the one-pass document NLP extractor.
    pub fn with_document_nlp(mut self, nlp: OnePassDocumentNlp) -> Self {
        self.nlp = Some(nlp);
        self
    }

    /// Overrides the one-pass query analyzer.
    pub fn with_query_analyzer(mut self, query: OnePassQueryAnalyzer) -> Self {
        self.query = Some(query);
        self
    }

    /// Configures an in-process `libturbo` embedding (and optional SPLADE) bundle.
    #[cfg(feature = "local-core")]
    pub fn with_local_embed_bundle(
        mut self,
        bundle_dir: impl Into<PathBuf>,
        encoder_opts: LocalEncoderOptions,
        sparse_opts: Option<SparseOptions>,
    ) -> Self {
        self.local_embed = Some((bundle_dir.into(), encoder_opts, sparse_opts));
        self
    }

    /// Configures an in-process `libturbo` cross-encoder reranker bundle.
    #[cfg(feature = "local-core")]
    pub fn with_local_rerank_bundle(
        mut self,
        bundle_dir: impl Into<PathBuf>,
        session_opts: LocalRerankSessionOptions,
        calibrator: RerankCalibrator,
    ) -> Self {
        self.local_rerank = Some((bundle_dir.into(), session_opts, calibrator));
        self
    }

    /// Configures pooled gRPC offload to `turbo-kserve`, optionally naming a
    /// `(model_name, model_version)` verified via `ModelMetadata` during [`Self::connect`].
    #[cfg(feature = "grpc")]
    pub fn with_remote_endpoint(
        mut self,
        config: RemoteEndpointConfig,
        handshake_model: Option<(String, String)>,
    ) -> Self {
        self.remote_cfg = Some((config, handshake_model));
        self
    }

    /// Enables runtime-scoped query embedding coalescing for `model_name`.
    #[cfg(feature = "grpc")]
    pub fn with_embed_coalescer(
        mut self,
        model_name: impl Into<String>,
        role: Option<String>,
        config: CoalesceConfig,
    ) -> Self {
        self.coalesce_cfg = Some((model_name.into(), role, config));
        self
    }

    /// Builds a synchronous [`TurboClient`], verifying lexical and local-bundle
    /// handshakes immediately.
    pub fn build(self) -> Result<TurboClient> {
        let default_lexical = LexicalAnalyzer::new(self.default_field);
        if let Some(hs) = &self.handshake {
            hs.verify_analysis(&default_lexical.fingerprint())?;
        }
        let named_lexical = self.named_fields.into_iter().map(|(k, cfg)| (k, LexicalAnalyzer::new(cfg))).collect();
        let nlp = match self.nlp {
            Some(n) => n,
            None => OnePassDocumentNlp::builder().build()?,
        };
        let query = match self.query {
            Some(q) => q,
            None => OnePassQueryBuilder::new().build()?,
        };
        #[cfg(feature = "local-core")]
        let local_embed = match self.local_embed {
            Some((dir, eo, so)) => Some(LocalEmbedEngine::open(&dir, &eo, so.as_ref(), self.handshake.as_ref())?),
            None => None,
        };
        #[cfg(feature = "local-core")]
        let local_rerank = match self.local_rerank {
            Some((dir, ro, cal)) => Some(LocalRerankEngine::open(&dir, &ro, cal, self.handshake.as_ref())?),
            None => None,
        };
        #[cfg(feature = "grpc")]
        let remote = self.remote_cfg.map(|(cfg, _)| RemoteClient::new(cfg));
        #[cfg(feature = "grpc")]
        let coalescer = match (&remote, self.coalesce_cfg) {
            (Some(r), Some((m, role, c))) => Some(EmbedCoalescer::new(r.clone(), m, role, c)),
            _ => None,
        };

        Ok(TurboClient {
            inner: Arc::new(TurboClientInner {
                default_lexical,
                named_lexical,
                nlp,
                query,
                #[cfg(feature = "local-core")]
                local_embed,
                #[cfg(feature = "local-core")]
                local_rerank,
                #[cfg(feature = "grpc")]
                remote,
                #[cfg(feature = "grpc")]
                coalescer,
            }),
        })
    }

    /// Builds [`TurboClient`] and, when a remote `handshake_model` and
    /// `IndexGenerationHandshake` are configured, verifies the remote model's
    /// [`BundleScoringFingerprint`] over gRPC `ModelMetadata`.
    #[cfg(feature = "grpc")]
    pub async fn connect(self) -> Result<TurboClient> {
        let hs_check = self.handshake.clone().zip(self.remote_cfg.as_ref().and_then(|(_, m)| m.clone()));
        let client = self.build()?;
        if let (Some((hs, (model_name, model_version))), Some(remote)) = (hs_check, client.remote()) {
            remote.verify_model_handshake(&model_name, &model_version, &hs).await?;
        }
        Ok(client)
    }
}

struct TurboClientInner {
    default_lexical: LexicalAnalyzer,
    named_lexical: HashMap<String, LexicalAnalyzer>,
    nlp: OnePassDocumentNlp,
    query: OnePassQueryAnalyzer,
    #[cfg(feature = "local-core")]
    local_embed: Option<LocalEmbedEngine>,
    #[cfg(feature = "local-core")]
    local_rerank: Option<LocalRerankEngine>,
    #[cfg(feature = "grpc")]
    remote: Option<RemoteClient>,
    #[cfg(feature = "grpc")]
    coalescer: Option<EmbedCoalescer>,
}

/// Thread-safe, runtime-independent unified thick client (`Clone + Send + Sync`).
#[derive(Clone)]
pub struct TurboClient {
    inner: Arc<TurboClientInner>,
}

impl TurboClient {
    /// Creates a [`TurboClientBuilder`].
    pub fn builder() -> TurboClientBuilder {
        TurboClientBuilder::new()
    }

    /// Default in-process lexical analyzer.
    pub fn lexical(&self) -> &LexicalAnalyzer {
        &self.inner.default_lexical
    }

    /// Field-specific lexical analyzer, falling back to `lexical()`.
    pub fn field_lexical(&self, field: &str) -> &LexicalAnalyzer {
        self.inner.named_lexical.get(field).unwrap_or(&self.inner.default_lexical)
    }

    /// In-process one-pass document NLP extractor.
    pub fn document_nlp(&self) -> &OnePassDocumentNlp {
        &self.inner.nlp
    }

    /// In-process one-pass query analyzer.
    pub fn query_analyzer(&self) -> &OnePassQueryAnalyzer {
        &self.inner.query
    }

    /// Optional in-process `libturbo` embedding / SPLADE session.
    #[cfg(feature = "local-core")]
    pub fn local_embed(&self) -> Option<&LocalEmbedEngine> {
        self.inner.local_embed.as_ref()
    }

    /// Optional in-process `libturbo` cross-encoder rerank session.
    #[cfg(feature = "local-core")]
    pub fn local_rerank(&self) -> Option<&LocalRerankEngine> {
        self.inner.local_rerank.as_ref()
    }

    /// Optional pooled remote `turbo-kserve` gRPC client.
    #[cfg(feature = "grpc")]
    pub fn remote(&self) -> Option<&RemoteClient> {
        self.inner.remote.as_ref()
    }

    /// Optional runtime-scoped query embedding micro-batch coalescer.
    #[cfg(feature = "grpc")]
    pub fn coalescer(&self) -> Option<&EmbedCoalescer> {
        self.inner.coalescer.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_network_embedded_client_analyzes_lexical_nlp_and_query_in_process() {
        let client = TurboClient::builder().build().expect("build embedded TurboClient");
        let lex = client.lexical().analyze("Contract value $120.50 in Paris on 2026-04-10.").unwrap();
        assert!(!lex.arcs.is_empty());
        let nlp = client.document_nlp().analyze("Contract value $120.50 in Paris on 2026-04-10.").unwrap();
        assert!(nlp.money.iter().any(|m| m.currency == "USD"));
        let q = client.query_analyzer().analyze("contracts under $200 in Paris").unwrap();
        assert!(!q.filters.is_empty());
    }
}
