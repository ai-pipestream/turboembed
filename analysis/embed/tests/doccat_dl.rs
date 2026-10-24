//! End-to-end tests for `BundleSequenceScorer` and `DocumentCategorizerDL`
//! (`#208`) on turboembed backends (`CPU` and any hardware backend enabled via
//! `TURBO_TEST_REQUIRE_*`).
//!
//! Verifies:
//! 1. Exact sequence-classification logit and softmax probability parity
//!    against PyTorch `BertForSequenceClassification` and OpenNLP's
//!    `DocumentCategorizerDL` (`pooler.dense` + `tanh` + `classifier` + `f64`
//!    softmax + `AverageClassificationScoringStrategy`) on the small BERT
//!    bundle.
//! 2. Multi-chunk sliding-window categorization (`whitespace_chunks` and
//!    `opennlp:subwords` windowing).
//! 3. Pipeline `Annotator` integration reading `opennlp:subwords`,
//!    `turbo:chunks`, `turbo:structure`, and `opennlp:sentences` and writing
//!    `turbo:categories` and `turbo:categories<L>`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::softmax as head_softmax;
use turbo::safetensors::{Dtype, File};
use turbo_analysis::{Analyzer, Annotator, Document, IxRange, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_embed::{BundleSequenceScorer, Device, EncoderOptions, Precision, SubwordAnnotator};
use turbo_analysis_layers::{Block, Categories, Chunks, Cut, Sentences, Structure, Subwords};
use turbo_doccat::{DlChunkingMode, DlInferenceOptions, DlSpanCategorizer, DocumentCategorizerDL};

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
            let t = i as f32 / (count - 1) as f32;
            (start + (end - start) * t).to_le_bytes()
        })
        .collect()
}

fn slice_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Builds a temporary bundle from `testdata/tiny-bert-bundle` with
/// `extra` tensors appended to `weights/model.safetensors` and `edit` applied
/// to `manifest.json`.
fn make_seq_bundle(name: &str, extra: Vec<Tensor>, edit: impl FnOnce(&mut Value)) -> TempDir {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-doccat-dl-{}-{name}-{id}", std::process::id()));
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
    tensors.extend(extra);
    let weights_bytes = safetensors_file(&tensors);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights_bytes).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w_entry = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w_entry["size"] = json!(weights_bytes.len());
    w_entry["sha256"] = json!(sha256_hex(&weights_bytes));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    TempDir(dir)
}

fn bert_seq_cls_bundle() -> TempDir {
    let extra = vec![
        Tensor {
            name: "classifier.weight".into(),
            dtype: "F32",
            shape: vec![2, 32],
            data: linspace_f32(0.3, -0.2, 64),
        },
        Tensor { name: "classifier.bias".into(), dtype: "F32", shape: vec![2], data: slice_f32(&[-0.05, 0.15]) },
    ];
    make_seq_bundle("seq-cls", extra, |m| {
        m["classifier"] = json!({
            "kind": "CLASSIFIER_SEQUENCE",
            "labels": ["NEG", "POS"]
        });
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("pooler_weight".into(), json!("pooler.dense.weight"));
        tn.insert("pooler_bias".into(), json!("pooler.dense.bias"));
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        tn.insert("classifier_bias".into(), json!("classifier.bias"));
    })
}

fn claimed_devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    for (env_key, backend) in [
        ("TURBO_TEST_REQUIRE_CUDA", "cuda"),
        ("TURBO_TEST_REQUIRE_METAL", "metal"),
        ("TURBO_TEST_REQUIRE_LEVELZERO", "levelzero"),
        ("TURBO_TEST_REQUIRE_NPU", "npu"),
    ] {
        if std::env::var(env_key).is_ok_and(|v| !v.is_empty() && v != "0") {
            devices.push(Device::Backend(backend.into()));
        }
    }
    devices
}

#[test]
fn bundle_sequence_scorer_matches_pytorch_and_jvm_document_categorizer_dl() {
    let dir = bert_seq_cls_bundle();
    let ref_bytes = fs::read(dir.0.join("reference/reference.safetensors")).unwrap();
    let ref_st = File::parse("reference.safetensors", &ref_bytes).unwrap();
    let ref_ids = ref_st.get("ids", Dtype::I32, 2).unwrap().i32s();
    let ref_lens = ref_st.get("lengths", Dtype::I32, 1).unwrap().i32s();
    let width = ref_st.get("ids", Dtype::I32, 2).unwrap().shape[1] as usize;

    // Cases 1 and 2 from reference.safetensors ("The quick brown fox..." and "Café naïve RÉSUMÉ").
    // Strip the leading [CLS] (101) and trailing [SEP] (102) so we pass content subwords.
    let row1_full = &ref_ids[width..width + ref_lens[1] as usize];
    let row2_full = &ref_ids[2 * width..2 * width + ref_lens[2] as usize];
    let row1_content = &row1_full[1..row1_full.len() - 1];
    let row2_content = &row2_full[1..row2_full.len() - 1];

    // Exact PyTorch BertForSequenceClassification logits from turbo-reference:latest.
    let want_logits_1 = [0.16030255f32, 0.06783607];
    let want_logits_2 = [0.15911731f32, 0.06804016];
    let mut want_probs_1 = [0.0f64; 2];
    let mut want_probs_2 = [0.0f64; 2];
    head_softmax(&want_logits_1, &mut want_probs_1).unwrap();
    head_softmax(&want_logits_2, &mut want_probs_2).unwrap();

    for device in claimed_devices() {
        let enc_opts = EncoderOptions { device: device.clone(), precision: Precision::Model, ..Default::default() };
        // Tiny-bert tokenizer is cased, so use lower_case = false to match the cased reference rows.
        let dl_opts = DlInferenceOptions::default().with_lower_case(false);
        let cat = BundleSequenceScorer::open_categorizer(&dir.0, &enc_opts, dl_opts).unwrap();
        assert_eq!(cat.number_of_categories(), 2);
        assert_eq!(cat.get_category(0), Some("NEG"));
        assert_eq!(cat.get_category(1), Some("POS"));

        // 1. Subword-slice parity against PyTorch BertForSequenceClassification + JVM softmax.
        let probs1 = cat.categorize_subword_slice(row1_content).unwrap();
        let probs2 = cat.categorize_subword_slice(row2_content).unwrap();
        for i in 0..2 {
            assert!(
                (probs1[i] - want_probs_1[i]).abs() < 1e-5,
                "device {device:?} case 1 prob[{i}]: got {}, want {}",
                probs1[i],
                want_probs_1[i]
            );
            assert!(
                (probs2[i] - want_probs_2[i]).abs() < 1e-5,
                "device {device:?} case 2 prob[{i}]: got {}, want {}",
                probs2[i],
                want_probs_2[i]
            );
        }
        assert_eq!(cat.best_category(&probs1).unwrap(), "NEG");
        assert_eq!(cat.best_category(&probs2).unwrap(), "NEG");

        // 2. Direct text categorization parity on "The quick brown fox jumps over the lazy dog."
        let text_probs1 = cat.categorize(&["The quick brown fox jumps over the lazy dog."]).unwrap();
        for i in 0..2 {
            assert!(
                (text_probs1[i] - want_probs_1[i]).abs() < 1e-5,
                "device {device:?} text case 1 prob[{i}]: got {}, want {}",
                text_probs1[i],
                want_probs_1[i]
            );
        }

        // 3. Multi-chunk whitespace splitting parity (documentSplitSize = 5, splitOverlapSize = 2).
        // "The quick brown fox jumps over the lazy dog." has 9 words ->
        // chunk 0: "The quick brown fox jumps" (words 0..5)
        // chunk 1: "fox jumps over the lazy" (words 3..8)
        // chunk 2: "the lazy dog." (words 6..9)
        let split_opts =
            DlInferenceOptions::default().with_lower_case(false).with_document_split_size(5).with_split_overlap_size(2);
        let split_cat = BundleSequenceScorer::open_categorizer(&dir.0, &enc_opts, split_opts).unwrap();
        let c0 = cat.categorize(&["The quick brown fox jumps"]).unwrap();
        let c1 = cat.categorize(&["fox jumps over the lazy"]).unwrap();
        let c2 = cat.categorize(&["the lazy dog."]).unwrap();
        let multi = split_cat.categorize(&["The quick brown fox jumps over the lazy dog."]).unwrap();
        for i in 0..2 {
            let expected_avg = (c0[i] + c1[i] + c2[i]) / 3.0;
            assert!(
                (multi[i] - expected_avg).abs() < 1e-12,
                "multi-chunk average[{i}]: got {}, want {expected_avg}",
                multi[i]
            );
        }
    }
}

#[test]
fn pipeline_annotators_categorize_document_chunks_and_structure_on_bundle() {
    let dir = bert_seq_cls_bundle();
    let subword_annotator = SubwordAnnotator::open(&dir.0).unwrap();
    let text = "The quick brown fox jumps over the lazy dog. Café naïve RÉSUMÉ";

    for mode in [DlChunkingMode::WhitespaceWords, DlChunkingMode::Subwords] {
        let dl_opts = DlInferenceOptions::default().with_lower_case(false).with_chunking_mode(mode);
        let doc_cat = Arc::new(
            BundleSequenceScorer::open_categorizer(&dir.0, &EncoderOptions::default(), dl_opts.clone()).unwrap(),
        );
        let span_scorer = BundleSequenceScorer::open(&dir.0, &EncoderOptions::default()).unwrap();
        let span_cat_inner = Arc::new(DocumentCategorizerDL::with_options(span_scorer, dl_opts).unwrap());
        let chunk_cat = DlSpanCategorizer::<_, Chunks>::new(Arc::clone(&span_cat_inner));
        let struct_cat = DlSpanCategorizer::<_, Structure>::new(Arc::clone(&span_cat_inner));
        let sent_cat = DlSpanCategorizer::<_, Sentences>::new(Arc::clone(&span_cat_inner));

        let mut doc = Document::new(text);
        let (subwords,) = subword_annotator.annotate((doc.get::<View>().unwrap(), None)).unwrap();
        doc.add(&Key::DEFAULT, subwords, Provenance::by("turbo:subwords")).unwrap();

        let first_end = "The quick brown fox jumps over the lazy dog.".len() as u32;
        let second_start = first_end + 1;
        let full_end = text.len() as u32;

        let mut sent_spans = Spans::new(ViewId::SOURCE);
        sent_spans.push(Span::new(0, first_end));
        sent_spans.push(Span::new(second_start, full_end));
        doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();

        let mut chunk_spans = Spans::new(ViewId::SOURCE);
        chunk_spans.push(Span::new(0, first_end));
        chunk_spans.push(Span::new(second_start, full_end));
        doc.add(
            &Key::DEFAULT,
            Chunks {
                spans: chunk_spans,
                level: vec![0, 0],
                parent: vec![None, None],
                sentences: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
                cut: vec![Cut::DocumentStart, Cut::Sentence],
                cost: vec![0, 1000],
                model_tokens: vec![12, 5],
                over_budget: vec![false, false],
                prompt_overhead: 0,
            },
            Provenance::by("test"),
        )
        .unwrap();

        let mut struct_spans = Spans::new(ViewId::SOURCE);
        struct_spans.push(Span::new(0, first_end));
        struct_spans.push(Span::new(second_start, full_end));
        doc.add(
            &Key::DEFAULT,
            Structure {
                spans: struct_spans,
                kind: vec![Block::Paragraph, Block::Paragraph],
                parent: vec![None, None],
                depth: vec![0, 0],
            },
            Provenance::by("test"),
        )
        .unwrap();

        let view = doc.get::<View>().unwrap();
        let sw = doc.get::<Subwords>();
        let (doc_cats,) = doc_cat.annotate((view, sw)).unwrap();
        assert_eq!(doc_cats.len(), 1);
        assert_eq!(doc_cats.best_category(0), "NEG");
        assert!((doc_cats.row_scores(0).iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let analyzer = Analyzer::builder()
            .given(Key::<Subwords>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .given(Key::<Structure>::DEFAULT)
            .add(chunk_cat)
            .add(struct_cat)
            .add(sent_cat)
            .build()
            .unwrap();
        let mut docs = [doc];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        let chunk_cats = docs[0].get::<Categories<Chunks>>().unwrap();
        assert_eq!(chunk_cats.len(), 2);
        assert_eq!(chunk_cats.best_category(0), "NEG");
        assert_eq!(chunk_cats.best_category(1), "NEG");
        assert!((chunk_cats.row_scores(0)[0] - 0.5231001887498664).abs() < 1e-5);
        assert!((chunk_cats.row_scores(1)[0] - 0.5227535869725517).abs() < 1e-5);

        let struct_cats = docs[0].get::<Categories<Structure>>().unwrap();
        assert_eq!(struct_cats.len(), 2);
        assert_eq!(struct_cats.row_scores(0), chunk_cats.row_scores(0));
        assert_eq!(struct_cats.row_scores(1), chunk_cats.row_scores(1));

        let sent_cats = docs[0].get::<Categories<Sentences>>().unwrap();
        assert_eq!(sent_cats.len(), 2);
        assert_eq!(sent_cats.row_scores(0), chunk_cats.row_scores(0));
        assert_eq!(sent_cats.row_scores(1), chunk_cats.row_scores(1));
    }
}

#[test]
fn bundle_sequence_scorer_rejects_token_classifier_bundle() {
    let extra = vec![
        Tensor {
            name: "classifier.weight".into(),
            dtype: "F32",
            shape: vec![3, 32],
            data: linspace_f32(-0.25, 0.35, 96),
        },
        Tensor { name: "classifier.bias".into(), dtype: "F32", shape: vec![3], data: slice_f32(&[0.1, -0.2, 0.05]) },
    ];
    let dir = make_seq_bundle("token-cls-reject", extra, |m| {
        m["classifier"] = json!({
            "kind": "CLASSIFIER_TOKEN",
            "labels": ["O", "B-PER", "I-PER"]
        });
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        tn.insert("classifier_bias".into(), json!("classifier.bias"));
    });
    let b = Bundle::open(&dir.0).unwrap();
    assert_eq!(b.manifest.classifier.as_ref().unwrap().labels.len(), 3);
    let err = match BundleSequenceScorer::open(&dir.0, &EncoderOptions::default()) {
        Ok(_) => panic!("expected token classifier bundle to be rejected"),
        Err(e) => e,
    };
    assert!(err.message.contains("CLASSIFIER_SEQUENCE"), "unexpected error: {err}");
}

#[path = "doccat/batch.rs"]
mod batch;
