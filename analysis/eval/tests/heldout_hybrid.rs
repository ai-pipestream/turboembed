//! Held-out hybrid weight receipt (#955): choose the weighted-fusion
//! BM25 weight on the even-position scored queries of each cached set
//! and score that choice plus pure dense/BM25 on the odd-position queries.
//!
//! Env-gated like the other trained receipts: `TURBO_EVAL_CACHE` names
//! the set cache, `TURBO_HELDOUT955_BUNDLE` the dense bundle directory,
//! `TURBO_GOLDENS_MODELS` the sentence model directory when it is not
//! under the workspace target directory, and `TURBO_HELDOUT955_OUT`,
//! when set, the directory for the JSON record. Without the bundle
//! variable the receipt says so and passes.

use std::path::{Path, PathBuf};

use turbo_analysis_embed::{Device, Precision};
use turbo_chunker::terms::Bm25;
use turbo_chunker::{Budget, Costs};
use turbo_eval::Result;
use turbo_eval::analysis::{Analysis, Sentencer};
use turbo_eval::chunkers::{Chunker, Signals, TurboChunker};
use turbo_eval::embedder::{Embedder, Role, TurboEmbedder};
use turbo_eval::hybrid::{Fusion, HeldOutWeightChoice, Hybrid, HybridSearch};
use turbo_eval::lexical::TermRule;
use turbo_eval::set::LoadedSet;
use turbo_eval::sets;
use turbo_structure::Syntax;

const SETS: [&str; 4] = ["scifact", "nfcorpus", "arguana", "qasper"];

fn sentence_model() -> PathBuf {
    if let Ok(dir) = std::env::var("TURBO_GOLDENS_MODELS") {
        return PathBuf::from(dir).join("opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin");
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/turbo-goldens/models/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin")
}

fn embed_all(embedder: &mut dyn Embedder, texts: &[String], role: Role) -> Result<Vec<f32>> {
    let dim = embedder.dim();
    let mut out = Vec::with_capacity(texts.len() * dim);
    let mut part = Vec::new();
    for group in texts.chunks(1024) {
        let refs: Vec<&str> = group.iter().map(String::as_str).collect();
        embedder.embed(&refs, role, &mut part)?;
        out.extend_from_slice(&part);
    }
    Ok(out)
}

fn choose(
    set: &LoadedSet,
    analysis: &Analysis,
    chunker: &dyn Chunker,
    embedder: &mut dyn Embedder,
    weights: &[f64],
) -> Result<(HeldOutWeightChoice, usize)> {
    set.check()?;
    let docs = set
        .docs
        .iter()
        .map(|d| analysis.analyze(&d.text, embedder).map_err(|e| format!("document {}: {e}", d.id)))
        .collect::<Result<Vec<_>>>()?;
    let (mut doc_of, mut spans) = (Vec::new(), Vec::new());
    for (i, (d, doc)) in set.docs.iter().zip(&docs).enumerate() {
        let (chunks, pooled) = chunker.chunk_vectors(doc, embedder).map_err(|e| format!("document {}: {e}", d.id))?;
        if pooled.is_some() {
            return Err("vectors a chunker pools are retrieved by dense retrieval only".into());
        }
        for span in chunks {
            doc_of.push(i as u32);
            spans.push(span);
        }
    }
    drop(docs);
    if spans.is_empty() {
        return Err(format!("set {}: the chunker made no chunks", set.name));
    }
    let ids: Vec<String> = set.docs.iter().map(|d| d.id.clone()).collect();
    let texts: Vec<String> =
        doc_of.iter().zip(&spans).map(|(&d, s)| set.docs[d as usize].text[s.start..s.end].to_owned()).collect();
    let vectors = embed_all(embedder, &texts, Role::Document)?;
    let query_texts: Vec<String> = set.queries.iter().map(|q| q.text.clone()).collect();
    let query_vectors = embed_all(embedder, &query_texts, Role::Query)?;
    let hybrid =
        Hybrid { model: Bm25::LUCENE, terms: TermRule::Folded, fusion: Fusion::Weighted { bm25_weight: 0.35 } };
    let index = hybrid.index(&texts, &doc_of)?;
    let queries: Vec<(&str, &str)> = set.queries.iter().map(|q| (q.id.as_str(), q.text.as_str())).collect();
    let search =
        HybridSearch { bm25: &index, dim: embedder.dim(), vectors: &vectors, doc: &doc_of, span: &spans, docs: &ids };
    let chunks = spans.len();
    Ok((hybrid.choose_bm25_weight_held_out(&search, &queries, &query_vectors, &set.qrels, weights)?, chunks))
}

/// Held-out hybrid weight receipt (#955); run explicitly:
///
/// `TURBO_EVAL_CACHE=<cache> TURBO_HELDOUT955_BUNDLE=<bundle> TURBO_HELDOUT955_OUT=<dir> cargo test --locked --release -p turbo-eval --test heldout_hybrid -- --ignored --nocapture`
#[test]
#[ignore = "held-out hybrid weight receipt for docs/analysis.md (#955); needs a bundle, the sentence model and the set cache"]
fn heldout_hybrid_receipt() {
    let Ok(bundle) = std::env::var("TURBO_HELDOUT955_BUNDLE") else {
        eprintln!("TURBO_HELDOUT955_BUNDLE is not set: no dense bundle to run");
        return;
    };
    let cache = PathBuf::from(std::env::var("TURBO_EVAL_CACHE").expect("TURBO_EVAL_CACHE"));
    let analysis = Analysis {
        sentences: Sentencer::maxent(&sentence_model()).expect("sentence model"),
        structure: Some(Syntax::Plain),
    };
    let mut embedder = TurboEmbedder::open(Path::new(&bundle), Device::Cpu, Precision::Model).expect("dense bundle");
    let chunker = TurboChunker {
        budget: Budget { max: 254, target: 200, size_weight: 1000, overhead: embedder.overhead(Role::Document) },
        costs: Costs::default(),
        signals: Signals::Structure,
        cohesion: None,
    };
    let weights: Vec<f64> = (0..=10).map(|i| f64::from(i) / 10.0).collect();
    let model = embedder.identity().model_ref();
    let mut record_sets = Vec::new();

    println!(
        "| Set | Chosen bm25_weight | Selection queries | Selection nDCG@10 | Held-out queries | Held-out nDCG@10 | Held-out dense (w=0) | Held-out BM25 (w=1) |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for name in SETS {
        let set = match sets::find(name).and_then(|s| s.load(&cache)) {
            Ok(set) => set,
            Err(e) => {
                eprintln!("{name}: not loadable from the cache, skipped: {e}");
                continue;
            }
        };
        let (choice, chunks) =
            choose(&set, &analysis, &chunker, &mut embedder, &weights).expect("choose held-out weight");
        println!(
            "| {} | {:.2} | {} | {:.4} | {} | {:.4} | {:.4} | {:.4} |",
            set.name,
            choice.bm25_weight,
            choice.selection_queries,
            choice.selection_ndcg_at_10,
            choice.held_out_queries,
            choice.held_out_ndcg_at_10,
            choice.held_out_dense_ndcg_at_10,
            choice.held_out_bm25_ndcg_at_10
        );
        record_sets.push(serde_json::json!({
            "set": set.name,
            "set_sha256": set.sha256,
            "docs": set.docs.len(),
            "chunks": chunks,
            "queries": set.queries.len(),
            "chosen_bm25_weight": choice.bm25_weight,
            "selection_queries": choice.selection_queries,
            "selection_ndcg_at_10": choice.selection_ndcg_at_10,
            "held_out_queries": choice.held_out_queries,
            "held_out_ndcg_at_10": choice.held_out_ndcg_at_10,
            "held_out_dense_ndcg_at_10": choice.held_out_dense_ndcg_at_10,
            "held_out_bm25_ndcg_at_10": choice.held_out_bm25_ndcg_at_10,
            "candidates": choice.candidates.iter().map(|c| serde_json::json!({
                "bm25_weight": c.bm25_weight,
                "selection_ndcg_at_10": c.selection_ndcg_at_10,
            })).collect::<Vec<_>>(),
        }));
    }

    if let Ok(out) = std::env::var("TURBO_HELDOUT955_OUT") {
        let dir = PathBuf::from(out);
        std::fs::create_dir_all(&dir).expect("create out dir");
        let record = serde_json::json!({
            "issue": 955,
            "kind": "held-out hybrid BM25 weight choice",
            "model": {"id": model.id, "revision": model.revision, "manifest_sha256": model.sha256},
            "chunker": "turbo-chunker max 254 target 200",
            "fusion": "weighted",
            "split": "scored queries in byte order; even positions select, odd positions held out",
            "candidate_weights": weights,
            "sets": record_sets,
        });
        std::fs::write(dir.join("heldout-hybrid-955.json"), serde_json::to_string_pretty(&record).expect("json"))
            .expect("write record");
    }
}
