//! Live rerank, classification, and token-classification checks for any
//! provider on real ONNX bundles.
//!
//! Selection is described in `turbo_conformance::live`. Bundles:
//! `TURBO_LIVE_RERANK_BUNDLE` (cross-encoder/ms-marco-MiniLM-L6-v2),
//! `TURBO_LIVE_CLASSIFY_BUNDLE` (distilbert-base-uncased-finetuned-sst-2-english),
//! `TURBO_LIVE_NER_BUNDLE` (dslim/bert-base-NER).
//!
//! These assert semantic properties (ranking order, label names, entity
//! spans), not exact numbers, so they hold across FP32 devices.

use turbo::abi;
use turbo::{Aggregation, ClassifyOptions, ModelDesc, RerankOptions, SessionDesc};
use turbo_conformance::live::{bundle, live};
use turbo_conformance::{read_f32, read_i32};

#[test]
fn live_rerank_orders_relevant_documents_first() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle("TURBO_LIVE_RERANK_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load reranker");
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
    // top_n limits the sorted output length and never exceeds the row count.
    session.write_pairs(query, &docs, &RerankOptions { top_n: 2, ..Default::default() }).unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.output(1).unwrap().shape, vec![2]);
    drop(r);
    let e = session.write_pairs(query, &docs, &RerankOptions { top_n: 9, ..Default::default() }).unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "top_n larger than the batch is an argument error");
    assert_eq!(e.field(), RerankOptions::FIELD_TOP_N);
    // raw_scores is either honored (logits, so the ranking is preserved) or
    // rejected naming the field; there is no third behavior.
    match session.write_pairs(query, &docs, &RerankOptions { raw_scores: true, ..Default::default() }) {
        Ok(()) => {
            let r = session.run(&Default::default()).unwrap();
            let logits = read_f32(&r, 0);
            eprintln!("raw logits {logits:?}");
            assert!(logits[0] > logits[1] && logits[0] > logits[2]);
            assert!(logits.iter().any(|l| !(0.0..=1.0).contains(l)), "logits, not activated scores");
        }
        Err(e) => {
            assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
            assert_eq!(e.field(), RerankOptions::FIELD_RAW_SCORES);
        }
    }
}

#[test]
fn live_classify_sentiment_labels() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle("TURBO_LIVE_CLASSIFY_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load classifier");
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
fn live_token_classify_finds_entities() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle("TURBO_LIVE_NER_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load NER");
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
    for s in r.spans() {
        assert!(s.score > 0.0 && s.score <= 1.0, "span score is a probability: {s:?}");
    }
    drop(r);
    session
        .write_text_classify(&[text], &ClassifyOptions { aggregation: Aggregation::None, ..Default::default() })
        .unwrap();
    let r = session.run(&Default::default()).unwrap();
    let per_word: Vec<&str> = r.spans().iter().map(|s| &text[s.byte_start as usize..s.byte_end as usize]).collect();
    eprintln!("per-word entity spans {per_word:?}");
    assert!(per_word.contains(&"Ada") && per_word.contains(&"Lovelace"), "{per_word:?}");
}
