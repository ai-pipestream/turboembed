//! In-process `libturbo` (`core/`) + `turbo-analysis-embed` engine (`#2338`).
//!
//! Provides zero-RPC multi-resolution dense + `TurboQuant` + sentence range
//! embedding (`#2344`), calibrated SPLADE `u8`/`u16` impact quantization
//! (`#2343`), calibrated cross-encoder reranking with source-projected semantic
//! highlights (`#2342`), and cryptographic [`BundleScoringFingerprint`]
//! verification against [`IndexGenerationHandshake`].

use std::ffi::c_char;
use std::path::Path;
use std::sync::Mutex;

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::rerank::{RerankOptions, RerankSessionOptions, Reranker};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::{
    TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_POOLING_CLS, TURBO_POOLING_LAST, TURBO_POOLING_MEAN,
    TURBO_TASK_EMBED, TURBO_TASK_EXTRACTIVE_QA, TURBO_TASK_RERANK, turbo_model_info,
};
use turbo_analysis::{Annotator, Document, Key, ModelRef, OffsetTable, Span, Spans, View, ViewId};
use turbo_analysis_embed::{
    CalibratedRerankScore, Encoder, EncoderOptions, MultiResInput, Rows, SourcePassage, SparseAnnotator,
    SubwordAnnotator, VectorAnnotator, VectorOptions, build_tier_from_unnormalized, normalize_matryoshka_dims,
    score_spans_pooled_cosine, score_spans_subword_tokenizer,
};
use turbo_analysis_layers::Sentences;

use crate::error::{ClientError, Result};
use crate::fingerprint::{BundleScoringFingerprint, BundleScoringSpec, IndexGenerationHandshake};
use crate::lexical::ProjectedSpanSet;

pub use turbo::manifest::PromptRole as EmbedPromptRole;
pub use turbo::rerank::{RerankOptions as LocalRerankOptions, RerankSessionOptions as LocalRerankSessionOptions};
pub use turbo_analysis_embed::impacts::ImpactBits;
pub use turbo_analysis_embed::{
    CalibratedRerankScore as RerankScore, EncoderOptions as LocalEncoderOptions, ImpactQuantizerConfig,
    MultiResOptions, MultiResVectors, Precision as LocalPrecision, QuantizedImpacts, RerankCalibrator, SparseOptions,
    VectorTier, VoronoiCentroids, VoronoiOptions, train_voronoi_centroids as calibrate_voronoi_centroids,
};

/// One source-attributed semantic highlight snippet in UTF-8, UTF-16, and code points.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalHighlightSnippet {
    pub spans: ProjectedSpanSet,
    pub score: f64,
    pub surface: String,
}

/// Calibrated reranking result and attributed semantic highlights for one passage.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalRerankHit {
    pub candidate_index: usize,
    pub score: CalibratedRerankScore,
    pub snippets: Vec<LocalHighlightSnippet>,
}

fn c_field(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| u8::from_ne_bytes(c.to_ne_bytes())).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn spec_from_model_info(mi: &turbo_model_info) -> BundleScoringSpec {
    let task = match mi.task {
        TURBO_TASK_EMBED => "embed",
        TURBO_TASK_RERANK => "rerank",
        TURBO_TASK_EXTRACTIVE_QA => "extractive_qa",
        _ => "unknown",
    };
    let pooling = match mi.pooling {
        TURBO_POOLING_MEAN => "mean",
        TURBO_POOLING_CLS => "cls",
        TURBO_POOLING_LAST => "last",
        _ => "model",
    };
    let normalize = match mi.normalize {
        TURBO_NORMALIZE_L2 => "l2",
        TURBO_NORMALIZE_NONE => "none",
        _ => "model",
    };
    BundleScoringSpec {
        model_id: c_field(&mi.model_id),
        revision: c_field(&mi.revision),
        manifest_sha256: c_field(&mi.manifest_sha256),
        task: task.to_owned(),
        dim: mi.dim,
        pooling: pooling.to_owned(),
        normalize: normalize.to_owned(),
    }
}

/// Thread-safe in-process embedding and SPLADE session backed by `libturbo`.
pub struct LocalEmbedEngine {
    tokenizer: Tokenizer,
    subwords: SubwordAnnotator,
    encoder: Mutex<Encoder>,
    vector_annotator: Option<VectorAnnotator>,
    sparse: Option<SparseAnnotator>,
    declared_dims: Vec<u32>,
    model_ref: ModelRef,
    scoring_spec: BundleScoringSpec,
    fingerprint: BundleScoringFingerprint,
}

impl LocalEmbedEngine {
    /// Opens an embedding bundle at `dir`, optionally loading its MLM SPLADE head
    /// when `sparse_opts` is `Some`, and verifies `handshake` before returning.
    pub fn open(
        dir: &Path,
        encoder_opts: &EncoderOptions,
        sparse_opts: Option<&SparseOptions>,
        handshake: Option<&IndexGenerationHandshake>,
    ) -> Result<Self> {
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let subwords = SubwordAnnotator::open(dir)?;
        let encoder = Encoder::open(dir, encoder_opts)?;
        let mi = encoder.model_info();
        let scoring_spec = spec_from_model_info(mi);
        let fingerprint = scoring_spec.fingerprint();
        if let Some(hs) = handshake {
            hs.verify_bundle(&fingerprint)?;
        }
        let declared_dims: Vec<u32> =
            mi.output_dims.iter().copied().take((mi.output_dims_count as usize).min(mi.output_dims.len())).collect();
        let model_ref = ModelRef {
            id: scoring_spec.model_id.clone(),
            revision: scoring_spec.revision.clone(),
            sha256: scoring_spec.manifest_sha256.clone(),
        };
        let vector_annotator = if mi.pooling == TURBO_POOLING_MEAN {
            VectorAnnotator::open(
                dir,
                &VectorOptions {
                    encoder: encoder_opts.clone(),
                    prompt: PromptRole::Document,
                    stride: 0,
                    output_dim: 0,
                },
            )
            .ok()
        } else {
            None
        };
        let sparse = match sparse_opts {
            Some(so) => Some(SparseAnnotator::open(dir, so)?),
            None => None,
        };
        Ok(Self {
            tokenizer,
            subwords,
            encoder: Mutex::new(encoder),
            vector_annotator,
            sparse,
            declared_dims,
            model_ref,
            scoring_spec,
            fingerprint,
        })
    }

    /// Canonical scoring descriptor of the loaded bundle.
    pub fn scoring_spec(&self) -> &BundleScoringSpec {
        &self.scoring_spec
    }

    /// Deterministic cryptographic scoring fingerprint of the loaded bundle.
    pub fn fingerprint(&self) -> BundleScoringFingerprint {
        self.fingerprint
    }

    /// Encodes a batch of texts (queries or passages) in one forward pass per
    /// batch window, emitting the full vector tier, requested Matryoshka
    /// prefixes, optional `TurboQuant` codes, `inv_norm`, and SHA-256 digests.
    pub fn embed_texts_multires(
        &self,
        texts: &[&str],
        role: PromptRole,
        opts: &MultiResOptions,
    ) -> Result<MultiResVectors> {
        let mut enc = self.encoder.lock().map_err(|_| ClientError::Internal("embed session poisoned".into()))?;
        let info = enc.info().clone();
        let prefix_dims =
            normalize_matryoshka_dims("turbo-client:embed", &opts.matryoshka_dims, info.dim, &self.declared_dims)?;
        let encode_cfg =
            Encode { add_special_tokens: true, truncation: Truncation::Right, max_tokens: info.max_seq, prompt: role };
        let max_batch = (info.max_batch as usize).max(1);
        let dim = info.dim as usize;
        let mut hidden_all = Vec::with_capacity(texts.len() * dim);
        let mut empty_rows = Vec::new();
        let mut chunk_out = Vec::new();

        for (base, batch_slice) in texts.chunks(max_batch).enumerate() {
            let offset = base * max_batch;
            let mut rows = Rows::new(info.max_seq, self.tokenizer.fill_id());
            for (i, &text) in batch_slice.iter().enumerate() {
                if text.trim().is_empty() {
                    empty_rows.push((offset + i) as u32);
                }
                let ids = self.tokenizer.encode(text, encode_cfg)?;
                rows.push(&[ids.as_slice()]);
            }
            enc.run_rows_pooled(&rows, info.pooling, false, &mut chunk_out)?;
            hidden_all.extend_from_slice(&chunk_out);
        }

        for &r in &empty_rows {
            let start = r as usize * dim;
            if let Some(slice) = hidden_all.get_mut(start..start + dim) {
                slice.fill(0.0);
            }
        }

        let full = build_tier_from_unnormalized(
            "turbo-client:embed",
            &self.model_ref,
            &hidden_all,
            dim,
            info.dim,
            info.l2,
            &empty_rows,
            opts.emit_row_sha256,
            opts.turboquant,
        )?;
        let mut prefixes = Vec::with_capacity(prefix_dims.len());
        for &d in &prefix_dims {
            prefixes.push(build_tier_from_unnormalized(
                "turbo-client:embed",
                &self.model_ref,
                &hidden_all,
                dim,
                d,
                info.l2,
                &empty_rows,
                opts.emit_row_sha256,
                None,
            )?);
        }
        Ok(MultiResVectors { model: self.model_ref.clone(), l2_normalized: info.l2, full, prefixes, sentences: None })
    }

    /// Embeds a single document and its `sentence_spans` (`ViewId::SOURCE` UTF-8
    /// offsets) in one encoder pass via `turbo_embed_write_ranges` when the
    /// bundle uses mean pooling, falling back to batch encoding on CLS models.
    pub fn embed_document_multires(
        &self,
        source: &str,
        sentence_spans: &[Span],
        opts: &MultiResOptions,
    ) -> Result<MultiResVectors> {
        if let Some(va) = &self.vector_annotator {
            let doc = Document::new(source);
            let view = doc.require(&Key::<View>::DEFAULT)?;
            let (sw,) = self.subwords.annotate((view, None))?;
            let mut sents = Spans::<Sentences>::new(ViewId::SOURCE);
            for &sp in sentence_spans {
                sents.push(sp);
            }
            let input = MultiResInput {
                subwords: &sw,
                sentences: (!sentence_spans.is_empty()).then_some(&sents),
                chunks: None,
                doc_span: Some(Span::new(0, source.len() as u32)),
            };
            return Ok(va
                .embed_multires("turbo-client:embed", &[input], opts)
                .into_iter()
                .next()
                .expect("single doc")?);
        }
        let mut out = self.embed_texts_multires(&[source], PromptRole::Document, opts)?;
        if !sentence_spans.is_empty() {
            let sent_texts: Vec<&str> =
                sentence_spans.iter().map(|sp| source.get(sp.range()).unwrap_or_default()).collect();
            let sent_res = self.embed_texts_multires(
                &sent_texts,
                PromptRole::Document,
                &MultiResOptions { emit_row_sha256: opts.emit_row_sha256, ..Default::default() },
            )?;
            out.sentences = Some(sent_res.full);
        }
        Ok(out)
    }

    /// Encodes `text` with the loaded MLM SPLADE head and quantizes its active
    /// weights into sorted `(vocab_id, impact)` pairs with L1 / dot certificates.
    pub fn encode_sparse_impacts(
        &self,
        text: &str,
        is_query: bool,
        quant: &ImpactQuantizerConfig,
    ) -> Result<QuantizedImpacts> {
        let sa = self.sparse.as_ref().ok_or_else(|| {
            ClientError::FailedPrecondition("bundle was opened without SparseOptions / MLM SPLADE head".into())
        })?;
        let doc = Document::new(text);
        let view = doc.require(&Key::<View>::DEFAULT)?;
        let (sw,) = self.subwords.annotate((view, None))?;
        let (terms,) = sa.annotate((view, &sw))?;
        quant.quantize_sparse_terms(&terms, 0, is_query).map_err(ClientError::InvalidArgument)
    }
}

/// Thread-safe in-process cross-encoder reranker and semantic highlighter (`#2342`).
pub struct LocalRerankEngine {
    tokenizer: Tokenizer,
    reranker: Mutex<Reranker>,
    calibrator: RerankCalibrator,
    scoring_spec: BundleScoringSpec,
    fingerprint: BundleScoringFingerprint,
}

/// Optional `(query_vector, per_candidate_sentence_vectors, dim)` MaxSim input
/// for pooled vector span attribution during local reranking.
pub type PooledHighlightInput<'a> = (&'a [f32], &'a [&'a [f32]], usize);

impl LocalRerankEngine {
    /// Opens a `TURBO_TASK_RERANK` bundle at `dir` and verifies `handshake`.
    pub fn open(
        dir: &Path,
        session_opts: &RerankSessionOptions,
        calibrator: RerankCalibrator,
        handshake: Option<&IndexGenerationHandshake>,
    ) -> Result<Self> {
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let reranker = Reranker::open_with(dir, session_opts)?;
        let id = reranker.identity();
        let scoring_spec = BundleScoringSpec {
            model_id: id.model_id.clone(),
            revision: id.revision.clone(),
            manifest_sha256: id.manifest_sha256.clone(),
            task: "rerank".to_owned(),
            dim: 0,
            pooling: "cls".to_owned(),
            normalize: "none".to_owned(),
        };
        let fingerprint = scoring_spec.fingerprint();
        if let Some(hs) = handshake {
            hs.verify_bundle(&fingerprint)?;
        }
        Ok(Self { tokenizer, reranker: Mutex::new(reranker), calibrator, scoring_spec, fingerprint })
    }

    /// Canonical scoring descriptor of the loaded reranker bundle.
    pub fn scoring_spec(&self) -> &BundleScoringSpec {
        &self.scoring_spec
    }

    /// Deterministic cryptographic scoring fingerprint of the loaded reranker bundle.
    pub fn fingerprint(&self) -> BundleScoringFingerprint {
        self.fingerprint
    }

    /// Scores `(query, passage)` candidates, maps raw logits into calibrated
    /// `[0, 1]` probabilities, and attributes non-overlapping semantic highlight
    /// spans back onto `ViewId::SOURCE` UTF-8, UTF-16, and code-point coordinates.
    pub fn rerank_and_highlight(
        &self,
        query: &str,
        candidates: &[(&str, &[Span])],
        rerank_opts: RerankOptions,
        max_snippets: usize,
        language: &str,
        pooled_vectors: Option<PooledHighlightInput<'_>>,
    ) -> Result<Vec<LocalRerankHit>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let pairs: Vec<(&str, &str)> = candidates.iter().map(|&(p, _)| (query, p)).collect();
        let raw_scores = {
            let mut rr = self.reranker.lock().map_err(|_| ClientError::Internal("rerank session poisoned".into()))?;
            rr.score(&pairs, rerank_opts)?.scores
        };

        let mut hits = Vec::with_capacity(candidates.len());
        for (idx, (&(passage_text, explicit_spans), &logit)) in candidates.iter().zip(&raw_scores).enumerate() {
            let score = self.calibrator.score(logit).map_err(ClientError::InvalidArgument)?;
            let mut snippets = Vec::new();
            if max_snippets > 0 && !passage_text.is_empty() {
                let sp = SourcePassage::new(passage_text, language, explicit_spans)?;
                let raw_snips = if let Some((q_vec, sent_mats, dim)) = pooled_vectors
                    && let Some(&s_vecs) = sent_mats.get(idx)
                {
                    score_spans_pooled_cosine(q_vec, s_vecs, dim, &sp, max_snippets)
                } else {
                    score_spans_subword_tokenizer(&self.tokenizer, query, &sp, max_snippets)
                        .map_err(ClientError::Internal)?
                };
                let table = OffsetTable::build(passage_text);
                for cand in raw_snips {
                    snippets.push(LocalHighlightSnippet {
                        spans: ProjectedSpanSet::from_source_span(passage_text, &table, cand.source_span),
                        score: cand.score,
                        surface: cand.surface,
                    });
                }
            }
            hits.push(LocalRerankHit { candidate_index: idx, score, snippets });
        }
        hits.sort_by(|a, b| {
            b.score
                .calibrated_probability
                .total_cmp(&a.score.calibrated_probability)
                .then_with(|| a.candidate_index.cmp(&b.candidate_index))
        });
        Ok(hits)
    }
}
