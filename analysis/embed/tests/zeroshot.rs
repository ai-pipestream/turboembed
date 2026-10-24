//! End-to-end tests for [`BundleNliScorer`] and [`ZeroShotCategorizer`] (`#909`).
//!
//! Verifies:
//! 1. Pair encoding via `turbo_tokenizer_encode_pair` (with `PAIR_BERT` segment
//!    types `0`/`1` and `PAIR_ROBERTA` `</s></s>` separator), CLS pooling, and
//!    `CLASSIFIER_SEQUENCE` head evaluation (`entailment`, `neutral`,
//!    `contradiction` and 2-way `entailment`, `not_entailment`).
//! 2. Single-run hypothesis batching per document (`run_count() == 1` when all
//!    `(window, hypothesis)` pairs fit in `max_batch`) and batch-size invariance.
//! 3. `ZeroShotAnnotator` and `ZeroShotSpanAnnotator` pipeline integration with
//!    `ModelRef` and `Execution` provenance.
//! 4. Real-checkpoint conformance against PyTorch `AutoModelForSequenceClassification`
//!    and zero-shot categorization accuracy when `TURBO_TEST_NLI_BUNDLE` is set.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::Head;
use turbo::safetensors::{Dtype, File};
use turbo::tokenizer::Tokenizer;
use turbo_analysis::{Analyzer, Document, Key, Layer, Provenance, Span, Spans, ViewId};
use turbo_analysis_embed::{BundleNliScorer, Device, Encoder, EncoderOptions, Precision, Rows};
use turbo_analysis_layers::{Categories, Sentences};
use turbo_doccat::{NliScorer, WindowAggregation, ZeroShotConfig, ZeroShotMode, ZeroShotSpanAnnotator};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn tiny_bundle() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn safetensors_file(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for t in tensors {
        let end = offset + t.data.len();
        header.insert(
            t.name.clone(),
            json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [offset, end]
            }),
        );
        offset = end;
    }
    let mut hdr_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !hdr_bytes.len().is_multiple_of(8) {
        hdr_bytes.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hdr_bytes.len() + offset);
    out.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    for t in tensors {
        out.extend_from_slice(&t.data);
    }
    out
}

fn linspace_f32(start: f32, end: f32, count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|i| {
            let t = i as f32 / (count.saturating_sub(1).max(1)) as f32;
            (start + (end - start) * t).to_le_bytes()
        })
        .collect()
}

fn slice_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn make_nli_bundle(name: &str, labels: &[&str], edit: impl FnOnce(&mut Value)) -> TempDir {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-nli-{}-{name}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny_bundle().join(f), dir.join(f)).unwrap();
    }
    let src_bytes = fs::read(tiny_bundle().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src_bytes[..8].try_into().unwrap()) as usize;
    let raw_header: serde_json::Map<String, Value> = serde_json::from_slice(&src_bytes[8..8 + n]).unwrap();
    let data = &src_bytes[8 + n..];
    let mut tensors = Vec::new();
    for (k, v) in raw_header {
        if k == "__metadata__" {
            continue;
        }
        let shape: Vec<u64> = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        let offs = v["data_offsets"].as_array().unwrap();
        let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
        tensors.push(Tensor { name: k, dtype: "F32", shape, data: data[a..b].to_vec() });
    }
    let num_labels = labels.len();
    tensors.push(Tensor {
        name: "classifier.weight".into(),
        dtype: "F32",
        shape: vec![num_labels as u64, 32],
        data: linspace_f32(0.35, -0.25, num_labels * 32),
    });
    let biases: Vec<f32> = (0..num_labels).map(|i| (i as f32 - 1.0) * 0.07).collect();
    tensors.push(Tensor {
        name: "classifier.bias".into(),
        dtype: "F32",
        shape: vec![num_labels as u64],
        data: slice_f32(&biases),
    });
    let weights_bytes = safetensors_file(&tensors);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights_bytes).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    m["classifier"] = json!({
        "kind": "CLASSIFIER_SEQUENCE",
        "labels": labels,
    });
    let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("pooler_weight".into(), json!("pooler.dense.weight"));
    tn.insert("pooler_bias".into(), json!("pooler.dense.bias"));
    tn.insert("classifier_weight".into(), json!("classifier.weight"));
    tn.insert("classifier_bias".into(), json!("classifier.bias"));
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w_entry = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w_entry["size"] = json!(weights_bytes.len());
    w_entry["sha256"] = json!(sha256_hex(&weights_bytes));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    TempDir(dir)
}

#[test]
fn bundle_nli_scorer_encodes_pairs_and_matches_encoder_cls_plus_head() {
    let dir = make_nli_bundle("three-way", &["contradiction", "entailment", "neutral"], |_| {});
    let enc_opts =
        EncoderOptions { device: Device::Cpu, precision: Precision::Model, max_seq: 16, ..Default::default() };
    let scorer = BundleNliScorer::open(&dir.0, &enc_opts).unwrap();
    assert_eq!(scorer.outcome_labels(), &["contradiction", "entailment", "neutral"]);
    assert_eq!(scorer.pair_special_tokens(), 3);
    assert_eq!(scorer.max_pair_tokens(), 16);

    let pairs = [
        ("The quick brown fox", "This example is animals."),
        ("Café naïve", "This example is food."),
        ("A short premise", "This example is technology."),
    ];
    let got = scorer.score_pairs(&pairs).unwrap();
    assert_eq!(got.len(), pairs.len());
    assert_eq!(scorer.run_count(), 1);

    // Compare against manual Tokenizer::encode_pair + Rows::push_with_types + Encoder::run_cls + Head::logits_into.
    let bundle = Bundle::open(&dir.0).unwrap();
    let tok = Tokenizer::load(&bundle).unwrap();
    let head = Head::load(&bundle).unwrap();
    let mut eo = enc_opts.clone();
    eo.max_ranges = eo.max_batch.max(8);
    let mut encoder = Encoder::open(&dir.0, &eo).unwrap();
    let mut rows = Rows::new(encoder.info().max_seq, tok.fill_id());
    for &(premise, hypothesis) in &pairs {
        let hyp_tokens = scorer.count_tokens(hypothesis);
        let query_tokens = (16 - 3 - hyp_tokens.max(1)).max(1) as u32;
        let win = &tok.encode_pair(premise, hypothesis, query_tokens, 16, 0).unwrap()[0];
        assert_eq!(win.ids[0], 101); // [CLS]
        assert_eq!(*win.ids.last().unwrap(), 102); // [SEP]
        // First segment has type 0, second segment has type 1.
        let first_sep = win.ids.iter().position(|&id| id == 102).unwrap();
        assert!(win.types[..=first_sep].iter().all(|&t| t == 0));
        assert!(win.types[first_sep + 1..].iter().all(|&t| t == 1));
        rows.push_with_types(&win.ids, &win.types);
    }
    let mut cls = Vec::new();
    encoder.run_cls(&rows, &mut cls).unwrap();
    let mut want = vec![0.0f32; pairs.len() * 3];
    head.logits_into(&cls, &mut want).unwrap();
    for (row_idx, want_row) in want.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            assert!(
                (got[row_idx][c] - want_row[c]).abs() < 1e-6,
                "pair {row_idx} col {c}: got {}, want {}",
                got[row_idx][c],
                want_row[c]
            );
        }
    }

    // Also verify that segment type 1 vs 0 changes the CLS hidden state on BERT.
    let mut rows_zero_types = Rows::new(encoder.info().max_seq, tok.fill_id());
    let win0 = &tok.encode_pair(pairs[0].0, pairs[0].1, 8, 16, 0).unwrap()[0];
    rows_zero_types.push_with_types(&win0.ids, &vec![0; win0.ids.len()]);
    let mut cls_zero = Vec::new();
    encoder.run_cls(&rows_zero_types, &mut cls_zero).unwrap();
    let diff: f32 = cls[..32].iter().zip(&cls_zero).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-4, "segment token types must influence BERT CLS state (diff={diff})");
}

#[test]
fn zero_shot_categorizer_batches_document_hypotheses_in_one_run() {
    let dir = make_nli_bundle("batching", &["contradiction", "entailment", "neutral"], |_| {});
    let labels: Vec<String> = ["finance", "sports", "weather", "cooking"].into_iter().map(String::from).collect();
    let cfg = ZeroShotConfig {
        template: "This is {}.".into(),
        mode: ZeroShotMode::SingleLabel,
        aggregation: WindowAggregation::Max,
        overlap_words: 0,
    };

    // With max_batch = 8, all 4 hypotheses for a 1-window premise run in a single encoder batch.
    let opts_b8 = EncoderOptions { device: Device::Cpu, max_batch: 8, ..Default::default() };
    let cat_b8 = BundleNliScorer::open_categorizer(&dir.0, &opts_b8, labels.clone(), cfg.clone()).unwrap();
    let scores_b8 = cat_b8.categorize("stocks rose today").unwrap();
    assert_eq!(cat_b8.scorer().run_count(), 1);
    assert_eq!(scores_b8.len(), 4);
    assert!((scores_b8.iter().sum::<f64>() - 1.0).abs() < 1e-9);

    // With max_batch = 2, the 4 hypotheses run across 2 batches of 2 and give the exact same probabilities.
    let opts_b2 = EncoderOptions { device: Device::Cpu, max_batch: 2, ..Default::default() };
    let cat_b2 = BundleNliScorer::open_categorizer(&dir.0, &opts_b2, labels.clone(), cfg.clone()).unwrap();
    let scores_b2 = cat_b2.categorize("stocks rose today").unwrap();
    assert_eq!(cat_b2.scorer().run_count(), 2);
    for (a, b) in scores_b8.iter().zip(&scores_b2) {
        assert!((a - b).abs() < 1e-6, "{scores_b8:?} vs {scores_b2:?}");
    }

    // Multi-window long premise in MultiLabel mode with 2-way NLI outcomes.
    let dir_2way = make_nli_bundle("two-way", &["entailment", "not_entailment"], |_| {});
    let ml_cfg = ZeroShotConfig {
        template: "This is {}.".into(),
        mode: ZeroShotMode::MultiLabel,
        aggregation: WindowAggregation::LengthWeightedMean,
        overlap_words: 1,
    };
    let opts_seq16 = EncoderOptions { device: Device::Cpu, max_batch: 8, max_seq: 16, ..Default::default() };
    let cat_2way = BundleNliScorer::open_categorizer(&dir_2way.0, &opts_seq16, labels, ml_cfg).unwrap();
    let long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi";
    let windows = cat_2way.premise_windows(long_text).unwrap();
    assert!(windows.len() >= 2, "expected multiple premise windows, got {}", windows.len());
    let ml_scores = cat_2way.categorize(long_text).unwrap();
    assert_eq!(ml_scores.len(), 4);
    assert!(ml_scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
}

#[test]
fn zero_shot_annotators_write_categories_and_execution_provenance() {
    let dir = make_nli_bundle("annotator", &["contradiction", "entailment", "neutral"], |_| {});
    let labels: Vec<String> = ["finance", "sports", "tech"].into_iter().map(String::from).collect();
    let cfg = ZeroShotConfig { template: "Topic: {}.".into(), ..ZeroShotConfig::default() };
    let enc_opts = EncoderOptions { device: Device::Cpu, ..Default::default() };

    let doc_annotator = BundleNliScorer::open_annotator(&dir.0, &enc_opts, labels.clone(), cfg.clone()).unwrap();
    let span_categorizer = BundleNliScorer::open_categorizer(&dir.0, &enc_opts, labels, cfg).unwrap();
    let span_annotator = ZeroShotSpanAnnotator::<BundleNliScorer, Sentences>::new(span_categorizer);

    let text = "Revenue grew strongly. The team won the game.";
    let mut doc = Document::new(text);
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, 22));
    sent_spans.push(Span::new(23, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test:sentences")).unwrap();

    let analyzer =
        Analyzer::builder().given(Key::<Sentences>::DEFAULT).add(doc_annotator).add(span_annotator).build().unwrap();
    let mut docs = [doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }

    let doc_cats = docs[0].get::<Categories>().unwrap();
    assert_eq!(doc_cats.len(), 1);
    let doc_prov = docs[0].provenance(Key::<Categories>::DEFAULT.id()).unwrap();
    assert_eq!(doc_prov.annotator, "turbo:zero-shot");
    assert!(doc_prov.model.is_some());
    assert_eq!(doc_prov.execution.as_ref().unwrap().backend, "cpu");

    let sent_cats = docs[0].get::<Categories<Sentences>>().unwrap();
    assert_eq!(sent_cats.len(), 2);
    let sent_prov = docs[0].provenance(Key::<Categories<Sentences>>::DEFAULT.id()).unwrap();
    assert_eq!(sent_prov.annotator, "turbo:zero-shot-spans");
    assert_eq!(sent_prov.execution.as_ref().unwrap().backend, "cpu");
}

#[test]
fn bundle_nli_scorer_rejects_invalid_heads_and_tiny_max_seq() {
    let bad_labels = make_nli_bundle("bad-labels", &["NEG", "POS"], |_| {});
    let err = BundleNliScorer::open(&bad_labels.0, &EncoderOptions::default()).err().unwrap();
    assert!(err.message.contains("NLI classifier labels"), "unexpected error: {err}");

    let tiny_seq = make_nli_bundle("tiny-seq", &["contradiction", "entailment", "neutral"], |_| {});
    let opts = EncoderOptions { max_seq: 4, ..EncoderOptions::default() };
    let err = BundleNliScorer::open(&tiny_seq.0, &opts).err().unwrap();
    assert!(err.message.contains("leaves no room"), "unexpected error: {err}");
}

#[test]
fn real_nli_bundle_conformance_and_zero_shot_classification() {
    let Ok(bundle_dir) = std::env::var("TURBO_TEST_NLI_BUNDLE") else {
        eprintln!("TURBO_TEST_NLI_BUNDLE not set; skipping real NLI checkpoint test");
        return;
    };
    let dir = Path::new(&bundle_dir);
    let mut devices = vec![Device::Cpu];
    if std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| !v.is_empty() && v != "0") {
        devices.push(Device::Backend("cuda".into()));
    }
    let bundle = Bundle::open(dir).unwrap();
    let ref_bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
    let ref_st = File::parse(&bundle.manifest.reference.file, &ref_bytes).unwrap();
    let ref_logits = ref_st.get("logits", Dtype::F32, 2).unwrap().f32s();

    for device in devices {
        let enc_opts = EncoderOptions { device: device.clone(), precision: Precision::Exact, ..Default::default() };
        let scorer = BundleNliScorer::open(dir, &enc_opts).unwrap();
        assert_eq!(scorer.outcome_labels(), &["contradiction", "entailment", "neutral"]);
        assert_eq!(scorer.pair_special_tokens(), 4);
        assert_eq!(scorer.max_pair_tokens(), 512);

        // 1. Verify BundleNliScorer::score_pairs against PyTorch AutoModelForSequenceClassification
        //    reference logits stored in reference/reference.safetensors for every </s></s> pair case.
        let mut pair_cases: Vec<(usize, (&str, &str))> = Vec::new();
        for (i, case) in bundle.manifest.reference.cases.iter().enumerate() {
            if let Some((premise, hypothesis)) = case.text.split_once("</s></s>")
                && scorer.count_tokens(premise) + scorer.count_tokens(hypothesis) + 4 <= 512
            {
                pair_cases.push((i, (premise, hypothesis)));
            }
        }
        assert!(!pair_cases.is_empty(), "expected </s></s> pair cases in reference.cases");
        let pairs: Vec<(&str, &str)> = pair_cases.iter().map(|&(_, p)| p).collect();
        let got_logits = scorer.score_pairs(&pairs).unwrap();
        let mut worst_diff = 0.0f32;
        for ((case_idx, (p, h)), got_row) in pair_cases.iter().zip(&got_logits) {
            let want_row = &ref_logits[case_idx * 3..(case_idx + 1) * 3];
            for c in 0..3 {
                let diff = (got_row[c] - want_row[c]).abs();
                worst_diff = worst_diff.max(diff);
                assert!(
                    diff < 1e-4,
                    "device {device:?} case {case_idx} ({p:?} / {h:?}) col {c}: got {}, want {} (diff {diff})",
                    got_row[c],
                    want_row[c]
                );
            }
        }
        eprintln!(
            "real NLI bundle reference conformance ({device:?}): {} pair cases, worst_abs_diff = {worst_diff:.2e}",
            pairs.len()
        );

        // 2. End-to-end ZeroShotCategorizer accuracy and single-run batching on real text.
        let labels: Vec<String> = ["finance", "sports", "travel", "food"].into_iter().map(String::from).collect();
        let cfg = ZeroShotConfig { template: "This text is about {}.".into(), ..ZeroShotConfig::default() };
        let cat = BundleNliScorer::open_categorizer(dir, &enc_opts, labels, cfg).unwrap();
        let examples = [
            ("A man is eating fresh pizza at a restaurant in Naples.", "food", 0.35),
            ("Quarterly revenue rose 14% year over year to $2.4B on strong enterprise cloud demand.", "finance", 0.35),
            (
                "The championship final went into overtime after a last-second three-pointer tied the game.",
                "sports",
                0.35,
            ),
            ("Our flight to Tokyo landed early and we took the train to the hotel.", "travel", 0.35),
        ];
        let t0 = Instant::now();
        for (idx, &(text, want_label, min_prob)) in examples.iter().enumerate() {
            let before = cat.scorer().run_count();
            let (best_label, prob) = cat.best(text).unwrap();
            let after = cat.scorer().run_count();
            assert_eq!(after - before, 1, "example {idx} must batch all 4 hypotheses in 1 encoder run");
            eprintln!("  [{device:?} {want_label}] best={best_label} prob={prob:.4} text={text:?}");
            assert_eq!(best_label, want_label, "wrong zero-shot category for {text:?}");
            assert!(prob >= min_prob, "expected prob >= {min_prob} for {text:?}, got {prob}");
        }
        let elapsed = t0.elapsed();
        eprintln!(
            "real zero-shot categorization ({device:?}): {} docs (16 pairs across {} encoder runs) in {:.2?} ({:.2?}/doc)",
            examples.len(),
            cat.scorer().run_count(),
            elapsed,
            elapsed / examples.len() as u32
        );
    }
}
