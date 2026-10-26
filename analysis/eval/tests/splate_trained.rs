//! Trained-checkpoint receipt for the four-arm SPLATE study (#913).
//!
//! Env-gated like the synthetic receipt in `tests/splate.rs`:
//! `TURBO_EVAL_CACHE` names the BEIR cache root and
//! `TURBO_SPLATE913_BUNDLES` a directory holding `neural-sparse`,
//! `colbertv2`, `potion-base-8m` and `ms-marco-minilm` bundles (built
//! by `turbo-bundle make` from the recipes in `bundle/recipes/`).
//! `TURBO_SPLATE913_OUT`, when set, is the directory the JSON record
//! is written to. Without the bundle variable the receipt says so and
//! passes, as the other bundle-gated eval tests do.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbo::rerank::RerankOptions;
use turbo::sparse::SparseOptions;
use turbo_analysis_embed::Device;
use turbo_eval::metrics::{Qrel, Qrels};
use turbo_eval::read::beir;
use turbo_eval::set::LoadedSet;
use turbo_eval::splate::receipt::{self, PinnedBundle, ReceiptBundles, ReceiptRun};
use turbo_eval::splate::{
    BundleCrossEncoder, SplateConfig, TrainedBundles, TrainedConfig, encode_trained, run_study_with_encoded,
};

fn subset_with_distractors(set: &mut LoadedSet, max_docs: usize, max_queries: usize) -> (usize, usize) {
    let mut selected_qids: Vec<String> = set.qrels.scored_queries().map(str::to_owned).collect();
    selected_qids.truncate(max_queries);
    let keep_qset: BTreeSet<String> = selected_qids.into_iter().collect();
    let mut rel_docs = BTreeSet::new();
    for d in &set.docs {
        for qid in &keep_qset {
            if set.qrels.grade(qid, &d.id) > 0 {
                rel_docs.insert(d.id.clone());
                break;
            }
        }
    }
    let rel = rel_docs.len();
    let target_docs = max_docs.max(rel);
    let mut keep_docs = rel_docs;
    for d in &set.docs {
        if keep_docs.len() >= target_docs {
            break;
        }
        keep_docs.insert(d.id.clone());
    }
    let dist = keep_docs.len() - rel;
    set.docs.retain(|d| keep_docs.contains(&d.id));
    set.queries.retain(|q| keep_qset.contains(&q.id));
    let mut new_qrels = Vec::new();
    for q in &set.queries {
        for d in &set.docs {
            let g = set.qrels.grade(&q.id, &d.id);
            if g > 0 {
                new_qrels.push(Qrel { query: q.id.clone(), doc: d.id.clone(), grade: g });
            }
        }
    }
    set.qrels = Qrels::new(new_qrels).expect("valid truncated qrels");
    (rel, dist)
}

/// Trained four-arm receipt (#913); run explicitly:
///
/// `TURBO_EVAL_CACHE=<cache> TURBO_SPLATE913_BUNDLES=<dir> TURBO_SPLATE913_OUT=<dir> cargo test --locked --release -p turbo-eval --test splate_trained -- --ignored --nocapture`
#[test]
#[ignore = "trained SPLATE receipt for docs/splate.md (#913); needs bundles and the BEIR cache"]
fn splate_trained_receipt() {
    let Ok(bundles_root) = std::env::var("TURBO_SPLATE913_BUNDLES") else {
        eprintln!("TURBO_SPLATE913_BUNDLES is not set: no trained bundles to run");
        return;
    };
    let root = PathBuf::from(bundles_root);
    let bundles = TrainedBundles {
        sparse: root.join("neural-sparse"),
        colbert: root.join("colbertv2"),
        static_bundle: root.join("potion-base-8m"),
    };
    let device_name = std::env::var("TURBO_EVAL_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        name => Device::Backend(name.into()),
    };
    let ce_threads: u32 = std::env::var("TURBO_SPLATE913_CE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let rounds: usize = std::env::var("TURBO_SPLATE913_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let ce_dir = root.join("ms-marco-minilm");
    let ce =
        BundleCrossEncoder::open(&ce_dir, ce_threads, RerankOptions::default()).expect("open cross-encoder bundle");
    let pinned = ReceiptBundles {
        sparse: PinnedBundle::open(&bundles.sparse).expect("pin the sparse bundle"),
        colbert: PinnedBundle::open(&bundles.colbert).expect("pin the ColBERT bundle"),
        static_model: PinnedBundle::open(&bundles.static_bundle).expect("pin the static bundle"),
        cross_encoder: PinnedBundle::open(&ce_dir).expect("pin the cross-encoder bundle"),
    };
    let cache = PathBuf::from(std::env::var("TURBO_EVAL_CACHE").expect("TURBO_EVAL_CACHE"));
    let tcfg = TrainedConfig::default();
    let doc_cut = SparseOptions { top_k: 64, floor: 0.05 };
    let query_cut = SparseOptions { top_k: 24, floor: 0.05 };
    let mut record_sets: Vec<serde_json::Value> = Vec::new();

    for (name, max_docs, max_queries) in [("scifact", 1_500usize, 100usize), ("nfcorpus", 3_633, 100)] {
        let dir = cache.join(name);
        if !dir.join("corpus.jsonl").is_file() {
            eprintln!("{name}: not in the cache, skipped");
            continue;
        }
        let mut set = beir::read(name, &dir).expect("load cached BEIR set");
        let (rel, dist) = subset_with_distractors(&mut set, max_docs, max_queries);
        println!(
            "\n=== DATASET: {name} (docs={}, rel={rel}, distractors={dist}, queries={}) ===",
            set.docs.len(),
            set.queries.len()
        );
        let (encoded, prov) =
            encode_trained(&set, &bundles, device.clone(), &tcfg, &doc_cut, &query_cut).expect("encode trained");
        println!(
            "provenance: sparse={} colbert={} static={} cross_encoder={} token_dim={} static_dim={} vocab={} fit_docs={} fit_tokens={} fit_terms={} misaligned={} fit_mse={:.6}",
            pinned.sparse.pinned(),
            pinned.colbert.pinned(),
            pinned.static_model.pinned(),
            pinned.cross_encoder.pinned(),
            prov.token_dim,
            prov.static_dim,
            prov.vocab_size,
            prov.fit_docs_used,
            prov.fit_tokens,
            prov.fit_terms,
            prov.fit_misaligned_docs,
            prov.fit_mse,
        );
        let mut tables: Vec<(usize, String)> = Vec::new();
        for c in [10usize, 25, 50, 100] {
            let cfg = SplateConfig {
                dim: prov.token_dim,
                max_vocab: prov.vocab_size,
                shortlist_k: c,
                top_k: 100,
                splade_doc_cut: doc_cut,
                splade_query_cut: query_cut,
                splate_doc_cut: doc_cut,
                splate_query_cut: query_cut,
                rounds,
                ..SplateConfig::default()
            };
            let rep = run_study_with_encoded(&encoded, &set.qrels, &ce, &cfg).expect("study");
            println!("\n--- Shortlist C = {c} (trained) ---");
            print!("{}", rep.to_markdown_table());
            tables.push((c, rep.to_markdown_table()));
        }
        record_sets.push(receipt::set_record(name, set.docs.len(), set.queries.len(), &pinned, &prov, &tables));
    }
    if let Ok(out) = std::env::var("TURBO_SPLATE913_OUT") {
        let dir = PathBuf::from(out);
        std::fs::create_dir_all(&dir).expect("create out dir");
        let run = ReceiptRun { device: &device_name, rounds, encoder: &tcfg, doc_cut: &doc_cut, query_cut: &query_cut };
        let record = receipt::record(&run, record_sets);
        std::fs::write(dir.join("splate-trained-913.json"), serde_json::to_string_pretty(&record).expect("json"))
            .expect("write record");
    }
}
