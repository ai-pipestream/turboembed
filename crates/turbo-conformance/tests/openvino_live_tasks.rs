//! Live rerank, classification, and token-classification checks for the
//! OpenVINO provider on real ONNX bundles.
//!
//! Skipped (with a printed reason) unless `TURBO_OPENVINO_LIB` is set plus the
//! bundle for each test: `TURBO_OPENVINO_RERANK_BUNDLE`
//! (cross-encoder/ms-marco-MiniLM-L6-v2), `TURBO_OPENVINO_CLASSIFY_BUNDLE`
//! (distilbert-base-uncased-finetuned-sst-2-english), and
//! `TURBO_OPENVINO_NER_BUNDLE` (dslim/bert-base-NER). Optional
//! `TURBO_OPENVINO_ORDINAL` selects the device (default: the CPU device).
//!
//! These assert semantic properties (ranking order, label names, entity
//! spans), not exact numbers, so they hold across FP32 devices.

use std::path::PathBuf;
use std::sync::Arc;

use turbo::abi;
use turbo::{
    Aggregation, ClassifyOptions, Context, ContextDesc, DeviceKind, DeviceSelector, ModelDesc, RerankOptions,
    RuntimeDesc, SelectPolicy, SessionDesc,
};

fn context() -> Option<Arc<Context>> {
    let lib = match std::env::var("TURBO_OPENVINO_LIB") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            eprintln!("skipping: TURBO_OPENVINO_LIB is not set");
            return None;
        }
    };
    let rt = turbo::create_runtime(RuntimeDesc { provider_paths: vec![lib], ..Default::default() })
        .unwrap_or_else(|e| panic!("load the openvino provider: {e}"));
    let devices: Vec<_> = rt.devices().into_iter().filter(|d| d.info.provider_id == "openvino").collect();
    let ordinal = match std::env::var("TURBO_OPENVINO_ORDINAL") {
        Ok(v) => v.parse::<u32>().expect("TURBO_OPENVINO_ORDINAL"),
        Err(_) => devices.iter().find(|d| d.info.kind == DeviceKind::Cpu).expect("an openvino CPU device").info.ordinal,
    };
    let idx = rt
        .select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: "openvino".into(),
            ordinal,
            ..Default::default()
        })
        .expect("select the openvino device");
    eprintln!("device: {}", rt.device(idx).unwrap().info.name);
    Some(Context::create(rt, idx, &ContextDesc::default()).expect("context"))
}

fn bundle(var: &str) -> Option<PathBuf> {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => {
            eprintln!("skipping: {var} is not set");
            None
        }
    }
}

fn read_f32(r: &turbo::ResultHandle, index: u32) -> Vec<f32> {
    let out = r.output(index).unwrap();
    let mut bytes = vec![0u8; out.logical_bytes().unwrap() as usize];
    r.read(index, &mut bytes).unwrap();
    bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn read_i32(r: &turbo::ResultHandle, index: u32) -> Vec<i32> {
    let out = r.output(index).unwrap();
    let mut bytes = vec![0u8; out.logical_bytes().unwrap() as usize];
    r.read(index, &mut bytes).unwrap();
    bytes.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn openvino_rerank_orders_relevant_documents_first() {
    let Some(ctx) = context() else { return };
    let Some(dir) = bundle("TURBO_OPENVINO_RERANK_BUNDLE") else { return };
    let model = ctx.load_model(&dir, &ModelDesc::default()).expect("load reranker");
    assert_eq!(model.info().kind, turbo::ModelKind::Reranker);
    let session = model.create_session(&SessionDesc { max_batch: 4, max_seq: 128, ..Default::default() }).unwrap();
    let query = "How many people live in Berlin?";
    let docs = [
        "Berlin has a population of 3.5 million registered inhabitants in an area of 891 square kilometers.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "New York City is the most populous city in the United States.",
    ];
    session.write_pairs(query, &docs, &RerankOptions { return_sorted: true, ..Default::default() }).unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.outputs().len(), 2);
    let scores = read_f32(&r, 0);
    let sorted = read_i32(&r, 1);
    eprintln!("scores {scores:?} sorted {sorted:?}");
    assert_eq!(scores.len(), 3);
    assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)), "sigmoid scores in [0, 1]");
    assert_eq!(sorted[0], 0, "the Berlin passage ranks first");
    assert!(scores[0] > 0.5 && scores[1] < 0.5 && scores[2] < 0.5, "{scores:?}");
    drop(r);
    // top_n limits the sorted output length.
    session.write_pairs(query, &docs, &RerankOptions { top_n: 2, ..Default::default() }).unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.output(1).unwrap().shape, vec![2]);
    drop(r);
    // Options the provider cannot honor are rejected with the field index.
    let e = session.write_pairs(query, &docs, &RerankOptions { raw_scores: true, ..Default::default() }).unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
    assert_eq!(e.field(), RerankOptions::FIELD_RAW_SCORES);
}

#[test]
fn openvino_classify_sentiment_labels() {
    let Some(ctx) = context() else { return };
    let Some(dir) = bundle("TURBO_OPENVINO_CLASSIFY_BUNDLE") else { return };
    let model = ctx.load_model(&dir, &ModelDesc::default()).expect("load classifier");
    assert_eq!(model.info().labels, vec!["NEGATIVE", "POSITIVE"]);
    let session = model.create_session(&SessionDesc { max_batch: 4, max_seq: 64, ..Default::default() }).unwrap();
    let texts = ["I absolutely loved this movie, it was wonderful.", "This was a dreadful, boring waste of time."];
    session.write_text_classify(&texts, &ClassifyOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    let scores = read_f32(&r, 0);
    assert_eq!(r.output(0).unwrap().shape, vec![2, 2]);
    for row in scores.chunks(2) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "softmax row sums to 1: {row:?}");
    }
    eprintln!("sentiment scores {scores:?}");
    assert!(scores[1] > 0.9, "positive review: {:?}", &scores[..2]);
    assert!(scores[2] > 0.9, "negative review: {:?}", &scores[2..]);
}

#[test]
fn openvino_token_classify_finds_entities() {
    let Some(ctx) = context() else { return };
    let Some(dir) = bundle("TURBO_OPENVINO_NER_BUNDLE") else { return };
    let model = ctx.load_model(&dir, &ModelDesc::default()).expect("load NER");
    let labels = model.info().labels.clone();
    assert_eq!(labels[0], "O");
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();
    let text = "Ada Lovelace visited Berlin with colleagues from Microsoft.";
    session
        .write_text_classify(&[text], &ClassifyOptions { aggregation: Aggregation::Simple, ..Default::default() })
        .unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.output(0).unwrap().shape, vec![1, 64, labels.len() as u64]);
    let found: Vec<(String, String)> = r
        .spans()
        .iter()
        .map(|s| (text[s.byte_start as usize..s.byte_end as usize].to_string(), labels[s.label as usize].clone()))
        .collect();
    eprintln!("spans {found:?}");
    let has = |t: &str, ent: &str| found.iter().any(|(w, l)| w == t && l.ends_with(ent));
    assert!(has("Ada Lovelace", "PER"), "{found:?}");
    assert!(has("Berlin", "LOC"), "{found:?}");
    assert!(has("Microsoft", "ORG"), "{found:?}");
    drop(r);
    session
        .write_text_classify(&[text], &ClassifyOptions { aggregation: Aggregation::None, ..Default::default() })
        .unwrap();
    let r = session.run(&Default::default()).unwrap();
    let per_word: Vec<&str> = r.spans().iter().map(|s| &text[s.byte_start as usize..s.byte_end as usize]).collect();
    eprintln!("per-word entity spans {per_word:?}");
    assert!(per_word.contains(&"Ada") && per_word.contains(&"Lovelace"), "{per_word:?}");
}
