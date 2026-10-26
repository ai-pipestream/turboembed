//! Trained-checkpoint encoding for the four-arm SPLATE study (#913).
//!
//! Builds an [`EncodedCorpus`] from pinned trained bundles instead of
//! the synthetic seeded encoder (#711): BM25 terms as in `corpus.rs`,
//! static dense vectors from a potion bundle, learned-sparse terms from
//! the OpenSearch neural-sparse bundle's MLM head (SPLADE max-log
//! pooling), and late-interaction token vectors from the ColBERTv2
//! bundle's projection head, each token L2-normalized as ColBERT does.
//!
//! Arm 4's adapter is trained here by per-token ridge distillation:
//! for a deterministic prefix of the corpus documents, every ColBERT
//! token vector (128) is regressed onto the SPLADE teacher's per-token
//! `relu(logit)` vocabulary activations over the heaviest vocabulary
//! terms, in closed form. Both encoders share the BERT WordPiece
//! vocabulary, so token `i` of one is token `i` of the other; a
//! document whose piece ids disagree between the two tokenizers is
//! left out of the fit and counted in the provenance.

use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo::bundle::Bundle;
use turbo::manifest::PromptRole;
use turbo::mlm::MlmHead;
use turbo::sparse::{MaxLogPool, SparseOptions as SparseCut, SparseTerm};
use turbo_analysis::{Document, ModelRef, View};
use turbo_analysis_embed::vectors::ContextualTokens;
use turbo_analysis_embed::{Device, EncoderOptions, Precision, SubwordAnnotator, VectorAnnotator, VectorOptions};

use super::corpus::{EncodedCorpus, EncodedDoc, EncodedQuery, extract_query_terms, tokenize_bm25};
use crate::Result;
use crate::embedder::{Embedder, Role, TurboEmbedder};
use crate::set::LoadedSet;

/// The three bundle directories a trained run encodes with.
#[derive(Debug, Clone)]
pub struct TrainedBundles {
    pub sparse: PathBuf,
    pub colbert: PathBuf,
    pub static_bundle: PathBuf,
}

/// Knobs of the trained encoder and the adapter fit.
#[derive(Debug, Clone, Copy)]
pub struct TrainedConfig {
    /// Most subword pieces kept per document (ColBERT and SPLADE alike).
    pub max_doc_tokens: usize,
    /// Most subword pieces kept per query.
    pub max_query_tokens: usize,
    /// Documents (a prefix, in corpus order) the adapter is fit on.
    pub fit_docs: usize,
    /// Vocabulary terms the adapter covers, by total teacher weight.
    pub adapter_terms: usize,
    /// Most teacher terms kept per token as regression targets.
    pub token_targets: usize,
    /// Ridge penalty of the adapter fit.
    pub ridge_lambda: f64,
}

impl Default for TrainedConfig {
    fn default() -> Self {
        Self {
            max_doc_tokens: 256,
            max_query_tokens: 64,
            fit_docs: 400,
            adapter_terms: 4_096,
            token_targets: 64,
            ridge_lambda: 1.0,
        }
    }
}

/// What a trained encoding ran, for the study record.
#[derive(Debug, Clone)]
pub struct TrainedProvenance {
    pub sparse_model: ModelRef,
    pub colbert_model: ModelRef,
    pub static_model: ModelRef,
    pub token_dim: usize,
    pub static_dim: usize,
    pub vocab_size: usize,
    pub fit_docs_used: usize,
    pub fit_tokens: usize,
    pub fit_terms: usize,
    pub fit_misaligned_docs: usize,
    pub fit_mse: f64,
}

fn full_text(title: &Option<String>, text: &str) -> String {
    match title {
        Some(t) if !t.is_empty() => format!("{t}\n{text}"),
        _ => text.to_owned(),
    }
}

fn unit_rows(data: &mut [f32], dim: usize) {
    for row in data.chunks_exact_mut(dim) {
        let n = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        if n > 0.0 {
            for x in row.iter_mut() {
                *x = (*x as f64 / n) as f32;
            }
        }
    }
}

fn model_ref(dir: &Path) -> Result<ModelRef> {
    let b = Bundle::open(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(ModelRef {
        id: b.manifest.model.id.clone(),
        revision: b.manifest.model.revision.clone(),
        sha256: b.manifest_sha256.clone(),
    })
}

fn annotator(dir: &Path, device: &Device) -> Result<VectorAnnotator> {
    let o = VectorOptions {
        encoder: EncoderOptions { device: device.clone(), ..EncoderOptions::default() },
        prompt: PromptRole::None,
        stride: 0,
    };
    VectorAnnotator::open(dir, &o).map_err(|e| format!("{}: {e}", dir.display()))
}

/// Per-token vectors for every text, encoded in batches so the
/// encoder sessions run full rows instead of one document at a time.
fn token_batch(
    va: &VectorAnnotator,
    subs: &SubwordAnnotator,
    texts: &[String],
    what: &str,
) -> Result<Vec<ContextualTokens>> {
    let mut out = Vec::with_capacity(texts.len());
    for (ci, chunk) in texts.chunks(32).enumerate() {
        let docs: Vec<Document> = chunk.iter().map(|t| Document::new(t.clone())).collect();
        let mut pieces = Vec::with_capacity(chunk.len());
        for d in &docs {
            let view = d.get::<View>().ok_or_else(|| "missing source view".to_owned())?;
            pieces.push(subs.pieces(view).map_err(|e| e.to_string())?);
        }
        let inputs: Vec<(
            &turbo_analysis_layers::Subwords,
            Option<&turbo_analysis::Spans<turbo_analysis_layers::Sentences>>,
        )> = pieces.iter().map(|p| (p, None)).collect();
        for r in va.embed_subword_tokens_batch("turbo:subword-vectors", &inputs, 0) {
            out.push(r.map_err(|e| e.to_string())?);
        }
        eprintln!("trained: {what} tokens for {} of {} texts", out.len(), texts.len());
        let _ = ci;
    }
    Ok(out)
}

use super::adapter_fit::{FitToken, fit_adapter};

/// One document's SPLADE outcome: its terms, its adapter-fit tokens
/// (fit documents only), and whether the two tokenizers agreed.
type SpladeDoc = Result<(Vec<SparseTerm>, Vec<FitToken>, bool)>;

/// Encodes `set` with the trained bundles; see the module comment.
pub fn encode_trained(
    set: &LoadedSet,
    bundles: &TrainedBundles,
    device: Device,
    cfg: &TrainedConfig,
    doc_cut: &SparseCut,
    query_cut: &SparseCut,
) -> Result<(EncodedCorpus, TrainedProvenance)> {
    set.check()?;
    let colbert = annotator(&bundles.colbert, &device)?;
    let sparse_ann = annotator(&bundles.sparse, &device)?;
    let colbert_subs = SubwordAnnotator::open(&bundles.colbert).map_err(|e| e.to_string())?;
    let sparse_subs = SubwordAnnotator::open(&bundles.sparse).map_err(|e| e.to_string())?;
    let sparse_bundle = Bundle::open(&bundles.sparse).map_err(|e| e.to_string())?;
    let mlm = MlmHead::load(&sparse_bundle).map_err(|e| e.to_string())?;
    let vocab_size = mlm.vocab;
    let token_dim = colbert.token_dim() as usize;
    let mut statik = TurboEmbedder::open(&bundles.static_bundle, device.clone(), Precision::Model)?;
    let static_dim = statik.dim();

    // Static document vectors in batches (not timed; queries are below).
    let texts: Vec<String> = set.docs.iter().map(|d| full_text(&d.title, &d.text)).collect();
    let mut doc_static: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(64) {
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        let mut out = Vec::new();
        statik.embed(&refs, Role::Document, &mut out)?;
        doc_static.extend(out.chunks_exact(static_dim).map(<[f32]>::to_vec));
    }

    let colbert_toks = token_batch(&colbert, &colbert_subs, &texts, "colbert")?;
    let sparse_toks = token_batch(&sparse_ann, &sparse_subs, &texts, "sparse")?;

    // SPLADE terms (and adapter-fit targets) per document, in parallel
    // over contiguous slices; the MLM head is Sync and its logits are
    // computed once per document for both uses.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(16));
    let per_thread = texts.len().div_ceil(threads).max(1);
    let splade_all: Vec<SpladeDoc> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (ci, chunk) in texts.chunks(per_thread).enumerate() {
            let (mlm, cfg, doc_cut) = (&mlm, cfg, doc_cut);
            let base = ci * per_thread;
            let (cts, sts) = (&colbert_toks[base..base + chunk.len()], &sparse_toks[base..base + chunk.len()]);
            handles.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(chunk.len());
                for (li, (ct, st)) in cts.iter().zip(sts).enumerate() {
                    let di = base + li;
                    let n = ct.ids.len().min(cfg.max_doc_tokens);
                    let m = n.min(st.ids.len());
                    let states = &st.data[..m * (st.dim as usize)];
                    let one = (|| -> Result<(Vec<SparseTerm>, Vec<FitToken>, bool)> {
                        let logits = mlm.logits(states).map_err(|e| e.to_string())?;
                        let mut pool = MaxLogPool::new(mlm.vocab);
                        for row in logits.chunks_exact(mlm.vocab) {
                            pool.observe(row).map_err(|e| e.to_string())?;
                        }
                        let terms = pool.terms(doc_cut).map_err(|e| e.to_string())?;
                        let aligned = ct.ids[..n] == st.ids[..m] && st.ids.len() >= n;
                        let mut fit_toks = Vec::new();
                        if di < cfg.fit_docs && aligned {
                            let mut token_vecs = ct.data[..n * token_dim].to_vec();
                            unit_rows(&mut token_vecs, token_dim);
                            fit_from_logits(&logits, mlm.vocab, &token_vecs, token_dim, cfg, &mut fit_toks);
                        }
                        Ok((terms, fit_toks, aligned))
                    })();
                    out.push(one);
                }
                out
            }));
        }
        let mut all = Vec::with_capacity(texts.len());
        for h in handles {
            match h.join() {
                Ok(part) => all.extend(part),
                Err(_) => all.push(Err("splade worker thread panicked".into())),
            }
        }
        all
    });

    let mut fit: Vec<FitToken> = Vec::new();
    let mut misaligned = 0usize;
    let mut docs = Vec::with_capacity(set.docs.len());
    for (di, d) in set.docs.iter().enumerate() {
        let text = &texts[di];
        let (vocab, _indexed, stats) = tokenize_bm25(text)?;
        let ct = &colbert_toks[di];
        let n = ct.ids.len().min(cfg.max_doc_tokens);
        let mut token_vecs = ct.data[..n * token_dim].to_vec();
        unit_rows(&mut token_vecs, token_dim);
        let (splade, fit_toks, aligned) = match &splade_all[di] {
            Ok(v) => v,
            Err(e) => return Err(e.clone()),
        };
        if di < cfg.fit_docs {
            if *aligned {
                fit.extend(fit_toks.iter().map(|t| FitToken { x: t.x.clone(), y: t.y.clone() }));
            } else {
                misaligned += 1;
            }
        }
        docs.push(EncodedDoc {
            id: d.id.clone(),
            text: text.clone(),
            bm25_stats: stats,
            bm25_vocab: vocab,
            token_ids: ct.ids[..n].iter().map(|&i| i as u32).collect(),
            static_vec: std::mem::take(&mut doc_static[di]),
            token_vecs,
            splade_terms: splade.clone(),
        });
        if (di + 1) % 250 == 0 {
            eprintln!("trained: assembled {} of {} docs", di + 1, set.docs.len());
        }
    }

    let (adapter, fit_terms, fit_mse) = fit_adapter(&fit, vocab_size, token_dim, cfg)?;
    let fit_tokens = fit.len();

    let mut queries = Vec::with_capacity(set.queries.len());
    for q in &set.queries {
        let t0 = Instant::now();
        let bm25_terms = extract_query_terms(&q.text)?;
        let ns_bm25 = t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();
        let mut sv = Vec::new();
        statik.embed(&[q.text.as_str()], Role::Query, &mut sv)?;
        let ns_static = t1.elapsed().as_nanos() as u64;
        let t2 = Instant::now();
        let ct = colbert.embed_text_tokens(&q.text).map_err(|e| e.to_string())?;
        let ns_colbert = t2.elapsed().as_nanos() as u64;
        let n = ct.ids.len().min(cfg.max_query_tokens);
        let mut token_vecs = ct.data[..n * token_dim].to_vec();
        unit_rows(&mut token_vecs, token_dim);
        let t3 = Instant::now();
        let st = sparse_ann.embed_text_tokens(&q.text).map_err(|e| e.to_string())?;
        let m = st.ids.len().min(cfg.max_query_tokens);
        let splade = pooled_terms(&mlm, &st.data[..m * (st.dim as usize)], query_cut)?;
        let ns_splade = t3.elapsed().as_nanos() as u64;
        queries.push(EncodedQuery {
            id: q.id.clone(),
            text: q.text.clone(),
            bm25_terms,
            token_ids: ct.ids[..n].iter().map(|&i| i as u32).collect(),
            static_vec: sv,
            token_vecs,
            splade_terms: splade,
            encode_ns: [ns_bm25, ns_static, ns_colbert + ns_splade, ns_colbert],
        });
    }

    let prov = TrainedProvenance {
        sparse_model: model_ref(&bundles.sparse)?,
        colbert_model: model_ref(&bundles.colbert)?,
        static_model: statik.identity().model_ref(),
        token_dim,
        static_dim,
        vocab_size,
        fit_docs_used: cfg.fit_docs.min(set.docs.len()) - misaligned,
        fit_tokens,
        fit_terms,
        fit_misaligned_docs: misaligned,
        fit_mse,
    };
    let corpus = EncodedCorpus {
        set_name: set.name.clone(),
        dim: token_dim,
        static_dim,
        vocab_size,
        model: prov.colbert_model.clone(),
        vocab_basis: Vec::new(),
        adapter: Some(adapter),
        docs,
        queries,
    };
    Ok((corpus, prov))
}

/// SPLADE terms of token hidden states: MLM logits pooled by max of
/// `log(1 + relu(logit))`, cut as `cut` says; also returns the logits
/// to a caller that collects fit targets, through `collect_fit`.
fn pooled_terms(mlm: &MlmHead, states: &[f32], cut: &SparseCut) -> Result<Vec<SparseTerm>> {
    let logits = mlm.logits(states).map_err(|e| e.to_string())?;
    let mut pool = MaxLogPool::new(mlm.vocab);
    for row in logits.chunks_exact(mlm.vocab) {
        pool.observe(row).map_err(|e| e.to_string())?;
    }
    pool.terms(cut).map_err(|e| e.to_string())
}

fn fit_from_logits(
    logits: &[f32],
    vocab: usize,
    token_vecs: &[f32],
    dim: usize,
    cfg: &TrainedConfig,
    fit: &mut Vec<FitToken>,
) {
    for (ti, row) in logits.chunks_exact(vocab).enumerate() {
        let mut y: Vec<(u32, f32)> =
            row.iter().enumerate().filter(|&(_, &v)| v > 0.0).map(|(t, &v)| (t as u32, v)).collect();
        y.sort_by(|a, b| b.1.total_cmp(&a.1));
        y.truncate(cfg.token_targets);
        fit.push(FitToken { x: token_vecs[ti * dim..(ti + 1) * dim].to_vec(), y });
    }
}
