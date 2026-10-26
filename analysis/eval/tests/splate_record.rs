//! Check documentation against the original, unmodified trained receipt,
//! and the receipt writer against that record.
//!
//! The archived record was supplemented by hand: the writer it came from
//! named each checkpoint by the recipe revision ("1") and left out the
//! run, cross-encoder and fit fields. The writer tests rebuild the
//! archived sets from the recipes and the record's own numbers and
//! require the writer to produce every field the archive holds except
//! `run`, so a regenerated record loses nothing. The archived `run` names
//! `--release`; the writer's generic command asserts no build, and the
//! actual Cargo settings go in its `build` block.
use serde_json::Value;
use turbo::manifest::Model;
use turbo::sparse::SparseOptions;
use turbo_analysis::ModelRef;
use turbo_eval::splate::receipt::{self, PinnedBundle, ReceiptBundles, ReceiptRun};
use turbo_eval::splate::{TrainedConfig, TrainedProvenance};

const DOC: &str = include_str!("../../../docs/splate.md");
const RECORD: &str = include_str!("../records/splate-trained-913/splate-trained-913.json");

fn cells(line: &str) -> Vec<&str> {
    line.trim_matches('|').split('|').map(str::trim).collect()
}

fn arms(table: &Value) -> Vec<Vec<f64>> {
    table["markdown"]
        .as_str()
        .unwrap()
        .lines()
        .filter(|line| line.starts_with("| Arm ") && !line.starts_with("| Arm |"))
        .map(|line| cells(line)[1..].iter().map(|v| v.parse().unwrap()).collect())
        .collect()
}

#[test]
fn trained_summary_cells_match_the_original_record() {
    let record: Value = serde_json::from_str(RECORD).unwrap();
    for set in record["sets"].as_array().unwrap() {
        let name = set["set"].as_str().unwrap();
        let heading = format!("Trained run on `{name}`");
        let section = DOC.split_once(&heading).unwrap().1.split("### ").next().unwrap();
        let summaries: Vec<_> =
            section.lines().filter(|s| s.starts_with("| Arm ") && !s.starts_with("| Arm |")).collect();
        assert_eq!(summaries.len(), 4);
        for (depth, table) in set["tables"].as_array().unwrap().iter().enumerate() {
            let rows = arms(table);
            assert_eq!(rows.len(), 4);
            for (arm, row) in rows.iter().enumerate() {
                let summary = cells(summaries[arm]);
                let metrics: Vec<f64> = summary[depth + 1].split('/').map(|s| s.trim().parse().unwrap()).collect();
                // Receipt columns: C, R@1, R@5, R@10, R@100, nDCG@10, ...
                assert_eq!(metrics, [row[5], row[4]], "{name}, arm {}, depth {depth}", arm + 1);
                if depth == 3 {
                    let ms: f64 = summary[5].trim_end_matches(" ms").parse().unwrap();
                    assert!((ms - row[9] / 1000.0).abs() <= 0.005, "search latency rounding");
                }
            }
        }
    }
}

#[test]
fn decision_discloses_each_arm_two_quality_win() {
    let record: Value = serde_json::from_str(RECORD).unwrap();
    let decision = DOC.split_once("### 5.3").unwrap().1;
    for set in record["sets"].as_array().unwrap() {
        let name = set["set"].as_str().unwrap();
        for (metric, column) in [("nDCG@10", 5), ("Recall@100", 4)] {
            let winning_depths: Vec<_> = set["tables"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|table| {
                    let rows = arms(table);
                    rows[1][column] > rows[2][column]
                })
                .map(|table| format!("C={}", table["c"]))
                .collect();
            if !winning_depths.is_empty() {
                let disclosure = format!("`{name}` {metric}: Arm 2 leads at {}.", winning_depths.join(", "));
                assert!(decision.contains(&disclosure), "missing receipt evidence: {disclosure}");
            }
        }
    }
}

const SPARSE: &str = include_str!("../../../bundle/recipes/opensearch-neural-sparse-encoding-v1.json");
const COLBERT: &str = include_str!("../../../bundle/recipes/colbertv2.json");
const STATIC: &str = include_str!("../../../bundle/recipes/potion/potion-base-8M.json");
const CROSS_ENCODER: &str = include_str!("../../../bundle/recipes/ms-marco-minilm-l6-v2-rerank.json");

/// Fields the writer adds that the archived run did not capture.
const NEW_SET_FIELDS: [&str; 1] = ["bundles"];
const NEW_TOP_FIELDS: [&str; 2] = ["config", "build"];

fn model(recipe: &str) -> Model {
    let v: Value = serde_json::from_str(recipe).unwrap();
    serde_json::from_value(v["manifest"]["model"].clone()).unwrap()
}

fn pin(recipe: &str, digest: char) -> PinnedBundle {
    PinnedBundle::from_model(&model(recipe), &digest.to_string().repeat(64)).unwrap()
}

fn bundles() -> ReceiptBundles {
    ReceiptBundles {
        sparse: pin(SPARSE, 'a'),
        colbert: pin(COLBERT, 'b'),
        static_model: pin(STATIC, 'c'),
        cross_encoder: pin(CROSS_ENCODER, 'd'),
    }
}

/// What `encode_trained` returns for an archived set; the `ModelRef`s
/// carry the recipe revision, as the bundle's manifest does.
fn provenance(set: &Value) -> TrainedProvenance {
    let r = |recipe: &str| {
        let m = model(recipe);
        ModelRef { id: m.id, revision: m.revision, sha256: String::new() }
    };
    let n = |k: &str| set[k].as_u64().unwrap() as usize;
    TrainedProvenance {
        sparse_model: r(SPARSE),
        colbert_model: r(COLBERT),
        static_model: r(STATIC),
        token_dim: n("token_dim"),
        static_dim: n("static_dim"),
        vocab_size: n("vocab"),
        fit_docs_used: n("fit_docs"),
        fit_tokens: n("fit_tokens"),
        fit_terms: n("fit_terms"),
        fit_misaligned_docs: n("fit_misaligned_docs"),
        // The run's unrounded error; the writer keeps six places.
        fit_mse: set["fit_mse"].as_f64().unwrap() + 3.0e-8,
    }
}

fn tables(set: &Value) -> Vec<(usize, String)> {
    set["tables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| (t["c"].as_u64().unwrap() as usize, t["markdown"].as_str().unwrap().to_owned()))
        .collect()
}

/// Every key of `archived` is in `written` with the same value, and
/// `written` holds no key beyond those and `new`.
fn covers(archived: &Value, written: &Value, new: &[&str], at: &str) {
    let (a, w) = (archived.as_object().unwrap(), written.as_object().unwrap());
    for (k, v) in a {
        if k == "sets" || k == "run" {
            continue;
        }
        assert_eq!(w.get(k), Some(v), "{at}: {k}");
    }
    for k in w.keys() {
        assert!(a.contains_key(k) || new.contains(&k.as_str()), "{at}: unexpected field {k}");
    }
    for k in new {
        assert!(w.contains_key(*k), "{at}: {k} missing");
    }
}

#[test]
fn the_writer_reproduces_every_archived_field_but_run() {
    let archived: Value = serde_json::from_str(RECORD).unwrap();
    let pinned = bundles();
    let mut sets = Vec::new();
    for set in archived["sets"].as_array().unwrap() {
        let name = set["set"].as_str().unwrap();
        let docs = set["docs"].as_u64().unwrap() as usize;
        let queries = set["queries"].as_u64().unwrap() as usize;
        let written = receipt::set_record(name, docs, queries, &pinned, &provenance(set), &tables(set));
        covers(set, &written, &NEW_SET_FIELDS, name);
        assert_eq!(written["tables"], set["tables"], "{name}: tables");
        assert_eq!(written["bundles"]["colbert"], "b".repeat(64), "{name}: manifest digest");
        sets.push(written);
    }
    let encoder = TrainedConfig::default();
    let doc_cut = SparseOptions { top_k: 64, floor: 0.05 };
    let query_cut = SparseOptions { top_k: 24, floor: 0.05 };
    let run = ReceiptRun {
        device: "cpu",
        rounds: 3,
        cross_encoder_threads: 4,
        encoder: &encoder,
        doc_cut: &doc_cut,
        query_cut: &query_cut,
    };
    let written = receipt::record(&run, sets);
    covers(&archived, &written, &NEW_TOP_FIELDS, "record");
    let config = &written["config"];
    assert_eq!(config["max_doc_tokens"], 256);
    assert_eq!(config["max_query_tokens"], 64);
    assert_eq!(config["doc_cut"]["floor"], 0.05);
    assert_eq!(config["query_cut"]["top_k"], 24);
}

#[test]
fn the_receipt_names_the_source_commit_not_the_recipe_revision() {
    for recipe in [SPARSE, COLBERT, STATIC, CROSS_ENCODER] {
        let m = model(recipe);
        let p = PinnedBundle::from_model(&m, "").unwrap();
        assert_eq!(p.pinned(), format!("{}@{}", m.id, m.source.commit));
        assert_ne!(p.commit, m.revision, "{}", m.id);
    }
}

#[test]
fn a_model_without_a_source_commit_is_refused() {
    let mut m = model(COLBERT);
    let upper = "c1e84128e85ef755c096a95bdb06b47793b13acf".to_ascii_uppercase();
    for bad in ["1", "", "c1e84128", &"g".repeat(40), &upper] {
        m.source.commit = bad.to_owned();
        let e = PinnedBundle::from_model(&m, "").unwrap_err();
        assert!(e.contains("40-digit lowercase hex commit"), "{bad}: {e}");
    }
}

#[test]
fn debug_writer_does_not_claim_a_release_run() {
    let encoder = TrainedConfig::default();
    let cut = SparseOptions { top_k: 1, floor: 0.0 };
    let run = ReceiptRun {
        device: "cpu",
        rounds: 1,
        cross_encoder_threads: 7,
        encoder: &encoder,
        doc_cut: &cut,
        query_cut: &cut,
    };
    let written = receipt::record(&run, Vec::new());
    assert!(!written["run"].as_str().unwrap().contains("--release"));
    assert_eq!(written["build"]["profile"], env!("TURBO_EVAL_PROFILE"));
    assert_eq!(written["build"]["opt_level"], env!("TURBO_EVAL_OPT_LEVEL"));
    assert_eq!(written["build"]["debug_info"], env!("TURBO_EVAL_DEBUG"));
    assert_eq!(written["build"]["debug_assertions"], cfg!(debug_assertions));
    assert_eq!(written["config"]["cross_encoder_threads"], 7);
}

#[test]
fn opened_bundle_provenance_has_the_actual_manifest_digest() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let bundle = turbo::bundle::Bundle::open(&dir).unwrap();
    let pinned = PinnedBundle::open(&dir).unwrap();
    assert_eq!(pinned.manifest_sha256, bundle.manifest_sha256);
    assert_eq!(pinned.commit, bundle.manifest.model.source.commit);
    assert_ne!(pinned.commit, bundle.manifest.model.revision);
}
