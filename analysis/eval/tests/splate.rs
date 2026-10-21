//! Unit, parity, and integration tests for the four-arm SPLATE retrieval study (`#711`).

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbo::sparse::{SparseOptions, SparseTerm};
use turbo_eval::metrics::{Qrel, Qrels};
use turbo_eval::read::{beir, legalbench_rag};
use turbo_eval::set::LoadedSet;
use turbo_eval::splate::{
    CascadeArm, CrossEncoderScorer, EncodedCorpus, EncodedQuery, SparsePostingIndexBuilder, SplateAdapter,
    SplateConfig, encode_loaded_set, run_shortlist_sweep, run_study, run_study_with_encoded, synthetic_benchmark_set,
};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

#[test]
fn sparse_posting_index_max_score_matches_exhaustive_top_k() {
    let mut builder = SparsePostingIndexBuilder::new();
    for doc_id in 0..80u32 {
        let mut terms = Vec::with_capacity(6);
        for slot in 0..6u32 {
            let term = (doc_id.wrapping_mul(7).wrapping_add(slot * 5)) % 24;
            let weight = 0.15 + ((doc_id + slot) % 11) as f32 * 0.12;
            terms.push(SparseTerm { term, weight });
        }
        builder.add(&terms).expect("add doc");
    }
    let index = builder.build();
    assert_eq!(index.docs(), 80);
    assert_eq!(index.active_terms(), 24);
    assert!(index.byte_size() > 0);

    for q_idx in 0..25u32 {
        let query = vec![
            SparseTerm { term: q_idx % 24, weight: 1.25 },
            SparseTerm { term: (q_idx * 3 + 1) % 24, weight: 0.85 },
            SparseTerm { term: (q_idx * 5 + 2) % 24, weight: 0.45 },
            SparseTerm { term: (q_idx * 11 + 4) % 24, weight: 1.60 },
            SparseTerm { term: 999, weight: 5.0 }, // out-of-vocabulary term ignored cleanly
        ];
        for k in [1usize, 5, 10, 25, 80] {
            let fast = index.top_k(&query, k);
            let exact = index.exhaustive_top_k(&query, k);
            assert_eq!(fast.len(), exact.len(), "length mismatch for q={q_idx}, k={k}");
            for (a, b) in fast.iter().zip(exact.iter()) {
                assert_eq!(a.0, b.0, "doc_id mismatch for q={q_idx}, k={k}");
                assert!((a.1 - b.1).abs() <= 1e-5, "score mismatch for q={q_idx}, k={k}: {} vs {}", a.1, b.1);
            }
        }
    }
}

#[test]
fn splate_adapter_validates_inputs_and_projects_sparse_terms() {
    assert!(SplateAdapter::new(Vec::new(), Vec::new(), 0, 8).is_err());
    assert!(SplateAdapter::new(Vec::new(), Vec::new(), 4, 0).is_err());
    assert!(SplateAdapter::new(Vec::new(), Vec::new(), usize::MAX, 2).is_err());
    assert!(SplateAdapter::from_basis(&[], 0, 8, 1.0, 0.0).is_err());
    assert!(SplateAdapter::from_basis(&[], 4, 0, 1.0, 0.0).is_err());
    assert!(SplateAdapter::from_basis(&[], usize::MAX, 2, 1.0, 0.0).is_err());
    assert!(SplateAdapter::new(vec![1.0; 7], vec![0.0; 2], 2, 4).is_err());
    assert!(SplateAdapter::new(vec![1.0; 8], vec![0.0; 3], 2, 4).is_err());
    assert!(SplateAdapter::new(vec![1.0, 0.0, 0.0, 0.0, f32::NAN, 0.0, 0.0, 0.0], vec![0.0, 0.0], 2, 4).is_err());

    let weights = vec![
        1.0, 0.0, 0.0, 0.0, // vocab 0 aligns with dim 0
        0.0, 1.0, 0.0, 0.0, // vocab 1 aligns with dim 1
        0.0, 0.0, 1.0, 0.0, // vocab 2 aligns with dim 2
    ];
    let adapter = SplateAdapter::from_basis(&weights, 3, 4, 1.0, 0.0).expect("valid adapter");
    assert_eq!(adapter.vocab(), 3);
    assert_eq!(adapter.dim(), 4);
    assert_eq!(adapter.byte_size(), (3 * 4 + 3) * 4);

    let cut = SparseOptions { top_k: 2, floor: 0.1 };
    assert!(adapter.encode_tokens(&[1.0, 2.0, 3.0], &cut).is_err());

    let tokens = vec![
        2.0, 0.0, 0.0, 0.0, // activates vocab 0 strongly
        0.0, 0.5, 0.0, 0.0, // activates vocab 1 moderately
    ];
    let sparse = adapter.encode_tokens(&tokens, &cut).expect("encode");
    assert_eq!(sparse.len(), 2);
    assert_eq!(sparse[0].term, 0);
    assert_eq!(sparse[1].term, 1);
    assert!(sparse[0].weight > sparse[1].weight);
}

#[test]
fn query_and_doc_sharing_same_id_are_self_excluded_per_beir_rule() {
    use turbo_eval::set::{Doc, Query};

    let single_self = LoadedSet {
        name: "self-only".into(),
        sha256: "b".repeat(64),
        docs: vec![Doc {
            id: "1".into(),
            title: Some("hypertension study".into()),
            text: "clinical hypertension vascular arterial myocardium regulation".into(),
        }],
        queries: vec![Query { id: "1".into(), text: "hypertension vascular arterial".into() }],
        qrels: Qrels::new(vec![Qrel { query: "1".into(), doc: "1".into(), grade: 1 }]).expect("valid qrels"),
        spans: None,
        unplaced: Vec::new(),
        absent: Vec::new(),
    };
    let cfg = SplateConfig { shortlist_k: 1, top_k: 1, rounds: 1, ..SplateConfig::default() };
    let self_report = run_study(&single_self, &cfg).expect("self-only study");
    for rep in &self_report.arms {
        assert_eq!(rep.scores.recall.at_1, 0.0, "arm {:?} failed to exclude doc whose id == query.id", rep.arm);
    }

    let arguana_like = LoadedSet {
        name: "arguana-like".into(),
        sha256: "c".repeat(64),
        docs: vec![
            Doc {
                id: "1".into(),
                title: Some("hypertension study".into()),
                text: "clinical hypertension vascular arterial myocardium regulation".into(),
            },
            Doc {
                id: "2".into(),
                title: Some("hypertension counterargument".into()),
                text: "clinical hypertension vascular arterial myocardium regulation therapy".into(),
            },
        ],
        queries: vec![Query { id: "1".into(), text: "hypertension vascular arterial".into() }],
        qrels: Qrels::new(vec![Qrel { query: "1".into(), doc: "2".into(), grade: 1 }]).expect("valid qrels"),
        spans: None,
        unplaced: Vec::new(),
        absent: Vec::new(),
    };
    let report = run_study(&arguana_like, &cfg).expect("arguana-like study");
    for rep in &report.arms {
        assert!(
            (rep.scores.recall.at_1 - 1.0).abs() <= 1e-9,
            "arm {:?} failed to rank non-self relevant doc at top-1 after excluding self (Recall@1 = {})",
            rep.arm,
            rep.scores.recall.at_1
        );
        assert!((rep.scores.ndcg_at_10 - 1.0).abs() <= 1e-9, "arm {:?} ndcg_at_10 != 1.0", rep.arm);
    }

    for (shortlist_k, top_k) in [(1, usize::MAX), (usize::MAX, 1), (usize::MAX, usize::MAX)] {
        let max_cfg = SplateConfig { shortlist_k, top_k, rounds: 1, ..SplateConfig::default() };
        let max_report = run_study(&arguana_like, &max_cfg).expect("usize::MAX study");
        for rep in &max_report.arms {
            assert!(
                (rep.scores.recall.at_100 - 1.0).abs() <= 1e-9,
                "arm {:?} failed at shortlist_k={shortlist_k}, top_k={top_k} (Recall@100 = {})",
                rep.arm,
                rep.scores.recall.at_100
            );
        }
    }
}

#[test]
fn four_arm_study_runs_on_synthetic_and_sample_datasets() {
    let synth = synthetic_benchmark_set(20, 3);
    let cfg = SplateConfig { shortlist_k: 25, top_k: 50, rounds: 2, ..SplateConfig::default() };
    let report = run_study(&synth, &cfg).expect("synthetic study");
    assert_eq!(report.doc_count, 160);
    assert_eq!(report.query_count, 24);
    assert_eq!(report.arms[0].arm, CascadeArm::Bm25Only);
    assert_eq!(report.arms[1].arm, CascadeArm::StaticThenCrossEncoder);
    assert_eq!(report.arms[2].arm, CascadeArm::LearnedSparseThenMaxSim);
    assert_eq!(report.arms[3].arm, CascadeArm::SplateThenMaxSim);

    for rep in &report.arms {
        assert!(rep.scores.ndcg_at_10 > 0.35, "arm {:?} ndcg too low", rep.arm);
        assert!(rep.scores.recall.at_5 >= rep.scores.recall.at_1 - 1e-9);
        assert!(rep.scores.recall.at_10 >= rep.scores.recall.at_5 - 1e-9);
        assert!(rep.scores.recall.at_100 >= rep.scores.recall.at_10 - 1e-9);
        assert!(rep.index_bytes > 0);
        assert!(rep.encode_p50_us >= 0.0);
        assert!(rep.retrieve_p50_us >= 0.0);
        assert!(rep.rerank_p50_us >= 0.0);
        assert!(rep.p50_us >= 0.0);
        assert!(rep.p99_us >= rep.p50_us);
    }

    let table = report.to_markdown_table();
    assert!(table.contains("Arm 1: BM25 only"));
    assert!(table.contains("Arm 4: SPLATE adapter + ColBERT MaxSim"));
    assert!(table.contains("Encode p50 (us)"));

    // Verify sample datasets in tests/data also run cleanly through all 4 arms.
    let beir_set = beir::read("beir-sample", &data_dir().join("beir")).expect("beir");
    let beir_rep =
        run_study(&beir_set, &SplateConfig { shortlist_k: 4, top_k: 4, rounds: 1, ..SplateConfig::default() })
            .expect("beir study");
    assert_eq!(beir_rep.arms.len(), 4);
    assert!(beir_rep.arms[2].scores.ndcg_at_10 > 0.4);
    assert!(beir_rep.arms[3].scores.ndcg_at_10 > 0.4);

    let lb_set = legalbench_rag::read("legalbench-rag-mini", &data_dir().join("legalbench-rag")).expect("lb");
    let lb_rep = run_study(&lb_set, &SplateConfig { shortlist_k: 3, top_k: 3, rounds: 1, ..SplateConfig::default() })
        .expect("legalbench study");
    assert_eq!(lb_rep.arms.len(), 4);
}

#[test]
fn custom_cross_encoder_scorer_integrates_with_arm2() {
    struct LexicalOverlapReranker;
    impl CrossEncoderScorer for LexicalOverlapReranker {
        fn score_candidates(
            &self,
            corpus: &EncodedCorpus,
            query: &EncodedQuery,
            candidates: &[u32],
        ) -> turbo_eval::Result<Vec<f32>> {
            let q_bytes = query.text.as_bytes();
            Ok(candidates
                .iter()
                .map(|&doc_id| {
                    corpus.docs[doc_id as usize].text.as_bytes().iter().filter(|b| q_bytes.contains(b)).count() as f32
                })
                .collect())
        }

        fn byte_size(&self) -> usize {
            16
        }
    }

    let beir_set = beir::read("beir-sample", &data_dir().join("beir")).expect("beir");
    let cfg = SplateConfig { shortlist_k: 3, top_k: 4, rounds: 1, ..SplateConfig::default() };
    let encoded = encode_loaded_set(&beir_set, cfg.dim, cfg.max_vocab, &cfg.splade_doc_cut, &cfg.splade_query_cut)
        .expect("encode");
    let report = run_study_with_encoded(&encoded, &beir_set.qrels, &LexicalOverlapReranker, &cfg)
        .expect("custom reranker study");
    assert_eq!(report.arms[1].arm, CascadeArm::StaticThenCrossEncoder);
    assert_eq!(report.arms[1].scores.per_query.len(), 2);
}

/// Generates the full multi-dataset, multi-shortlist synthetic harness receipt for `docs/splate.md`.
///
/// Run in `--release` on a single pinned core:
/// `taskset -c 14 cargo test --locked --release -p turbo-eval --test splate splate_benchmark_receipt -- --ignored --nocapture`
#[test]
#[ignore = "benchmark receipt generator for docs/splate.md; run explicitly with --release --ignored"]
fn splate_benchmark_receipt() {
    let mut datasets: Vec<(LoadedSet, usize, usize)> = Vec::new();
    if let Ok(cache_root) = std::env::var("TURBO_EVAL_CACHE") {
        let root = PathBuf::from(cache_root);
        for (name, max_docs, max_queries) in [("scifact", 1_500usize, 100usize), ("nfcorpus", 3_633, 100)] {
            let dir = root.join(name);
            if dir.join("corpus.jsonl").is_file() {
                let mut set = beir::read(name, &dir).expect("load cached BEIR set");
                let (rel_docs, distractor_docs) = subset_with_distractors(&mut set, max_docs, max_queries);
                datasets.push((set, rel_docs, distractor_docs));
            }
        }
    }
    let synth = synthetic_benchmark_set(80, 4);
    let synth_rel =
        synth.docs.iter().filter(|d| synth.queries.iter().any(|q| synth.qrels.grade(&q.id, &d.id) > 0)).count();
    let synth_dist = synth.docs.len() - synth_rel;
    datasets.push((synth, synth_rel, synth_dist));

    let depths = [10usize, 25, 50, 100];
    let base_cfg =
        SplateConfig { dim: 32, max_vocab: 2_048, shortlist_k: 50, top_k: 100, rounds: 5, ..SplateConfig::default() };
    for (set, rel_docs, distractor_docs) in &datasets {
        let reports = run_shortlist_sweep(set, &depths, &base_cfg).expect("sweep");
        println!(
            "\n=== DATASET: {} (docs={}, rel_docs={}, distractors={}, queries={}) ===",
            set.name,
            set.docs.len(),
            rel_docs,
            distractor_docs,
            set.queries.len()
        );
        for rep in &reports {
            println!("\n--- Shortlist C = {} ---", rep.config.shortlist_k);
            print!("{}", rep.to_markdown_table());
            println!(
                "Index breakdown (bytes): Arm1={} | Arm2={} | Arm3={} (sparse={}, tok={}) | Arm4={} (sparse+adapter={}, tok={})",
                rep.arms[0].index_bytes,
                rep.arms[1].index_bytes,
                rep.arms[2].index_bytes,
                rep.arms[2].first_stage_bytes,
                rep.arms[2].token_index_bytes,
                rep.arms[3].index_bytes,
                rep.arms[3].first_stage_bytes,
                rep.arms[3].token_index_bytes,
            );
        }
    }
}

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
    let rel_count = rel_docs.len();
    let target_docs = max_docs.max(rel_count);
    let mut keep_docs = rel_docs;
    for d in &set.docs {
        if keep_docs.len() >= target_docs {
            break;
        }
        keep_docs.insert(d.id.clone());
    }
    let distractor_count = keep_docs.len().saturating_sub(rel_count);

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
    (rel_count, distractor_count)
}
