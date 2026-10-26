//! Trained cascade latency receipt (#879): per-query p50/p95 latency
//! including query encoding, static only vs. contextual only vs. the
//! cascade at C in {10, 25, 50, 100, 200, 400}, arms interleaved,
//! plus first-round Recall@10/nDCG@10 per arm for the default-C
//! choice. Env-gated like the other bundle receipts:
//! `TURBO_EVAL_CACHE`, `TURBO_CASCADE879_BUNDLES` (a directory holding
//! `potion-base-8m` and `all-minilm-l6-v2` bundles),
//! `TURBO_GOLDENS_MODELS` (the sentence model directory, default
//! `target/turbo-goldens/models` under the workspace root), and
//! `TURBO_CASCADE879_OUT` for the JSON record. Without the bundle
//! variable the receipt says so and passes.

use std::path::{Path, PathBuf};

use turbo_analysis_embed::{Device, Precision};
use turbo_chunker::{Budget, Costs};
use turbo_eval::analysis::{Analysis, Sentencer};
use turbo_eval::cascade_latency::{LatencyArm, measure};
use turbo_eval::chunkers::{Signals, TurboChunker};
use turbo_eval::embedder::{Embedder, Role, TurboEmbedder};
use turbo_eval::sets;
use turbo_structure::Syntax;

fn sentence_model() -> PathBuf {
    if let Ok(dir) = std::env::var("TURBO_GOLDENS_MODELS") {
        return PathBuf::from(dir).join("opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin");
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/turbo-goldens/models/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin")
}

/// Trained cascade latency receipt (#879); run explicitly:
///
/// `TURBO_EVAL_CACHE=/work/eval-cache TURBO_CASCADE879_BUNDLES=<dir> TURBO_CASCADE879_OUT=<dir> taskset -c 14 cargo test --locked --release -p turbo-eval --test cascade_latency -- --ignored --nocapture`
#[test]
#[ignore = "cascade latency receipt for docs/analysis.md (#879); needs bundles, the sentence model and the set cache"]
fn cascade_latency_receipt() {
    let Ok(bundles_root) = std::env::var("TURBO_CASCADE879_BUNDLES") else {
        eprintln!("TURBO_CASCADE879_BUNDLES is not set: no trained bundles to run");
        return;
    };
    let root = PathBuf::from(bundles_root);
    let cache = PathBuf::from(std::env::var("TURBO_EVAL_CACHE").expect("TURBO_EVAL_CACHE"));
    let rounds: usize = std::env::var("TURBO_CASCADE879_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let analysis = Analysis {
        sentences: Sentencer::maxent(&sentence_model()).expect("sentence model"),
        structure: Some(Syntax::Plain),
    };
    let mut statics =
        TurboEmbedder::open(&root.join("potion-base-8m"), Device::Cpu, Precision::Model).expect("static bundle");
    let mut contextual =
        TurboEmbedder::open(&root.join("all-minilm-l6-v2"), Device::Cpu, Precision::Model).expect("contextual bundle");
    let chunker = TurboChunker {
        budget: Budget { max: 254, target: 200, size_weight: 1000, overhead: contextual.overhead(Role::Document) },
        costs: Costs::default(),
        signals: Signals::Structure,
        cohesion: None,
    };
    let depths = [10usize, 25, 50, 100, 200, 400];
    let mut record_sets: Vec<serde_json::Value> = Vec::new();

    for name in ["scifact", "nfcorpus", "arguana", "qasper", "gutenqa", "conteb-squad"] {
        let set = match sets::find(name).and_then(|s| s.load(&cache)) {
            Ok(set) => set,
            Err(e) => {
                eprintln!("{name}: not loadable from the cache, skipped: {e}");
                continue;
            }
        };
        let report =
            measure(&set, &analysis, &chunker, &mut statics, &mut contextual, &depths, rounds).expect("measure");
        println!(
            "\n=== {} (docs={}, chunks={}, queries={}, rounds={}) ===",
            report.set, report.docs, report.chunks, report.queries, report.rounds
        );
        print!("{}", report.to_markdown_table());
        let arms: Vec<serde_json::Value> = report
            .arms
            .iter()
            .map(|a| {
                serde_json::json!({
                    "arm": a.arm.label(),
                    "candidates": match a.arm { LatencyArm::Cascade(c) => Some(c), _ => None },
                    "encode_p50_us": a.encode_p50_us, "encode_p95_us": a.encode_p95_us,
                    "search_p50_us": a.search_p50_us, "search_p95_us": a.search_p95_us,
                    "total_p50_us": a.total_p50_us, "total_p95_us": a.total_p95_us,
                    "recall_at_1": a.scores.recall.at_1, "recall_at_5": a.scores.recall.at_5,
                    "recall_at_10": a.scores.recall.at_10, "recall_at_100": a.scores.recall.at_100,
                    "ndcg_at_10": a.scores.ndcg_at_10,
                })
            })
            .collect();
        record_sets.push(serde_json::json!({
            "set": report.set, "docs": report.docs, "chunks": report.chunks,
            "queries": report.queries, "rounds": report.rounds, "arms": arms,
            "markdown": report.to_markdown_table(),
        }));
    }
    if let Ok(out) = std::env::var("TURBO_CASCADE879_OUT") {
        let dir = PathBuf::from(out);
        std::fs::create_dir_all(&dir).expect("create out dir");
        let record = serde_json::json!({
            "issue": 879, "kind": "cascade per-query latency, encoding included",
            "static": "minishlab/potion-base-8M", "contextual": "sentence-transformers/all-MiniLM-L6-v2",
            "chunker": "turbo-chunker max 254 target 200", "sets": record_sets,
        });
        std::fs::write(dir.join("cascade-latency-879.json"), serde_json::to_string_pretty(&record).expect("json"))
            .expect("write record");
    }
}
