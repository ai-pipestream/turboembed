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
use turbo::{Aggregation, ClassifyOptions, ModelDesc, RerankOptions, SessionDesc, Truncate};
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
    drop(r);
    // raw_scores is either honored (logits, so the rows no longer sum to 1)
    // or rejected naming the field; there is no third behavior.
    match session.write_text_classify(&texts, &ClassifyOptions { raw_scores: true, ..Default::default() }) {
        Ok(()) => {
            let r = session.run(&Default::default()).unwrap();
            let logits = read_f32(&r, 0);
            eprintln!("raw logits {logits:?}");
            for row in logits.chunks(2) {
                assert!((row.iter().sum::<f32>() - 1.0).abs() > 1e-3, "logits, not activated scores: {row:?}");
            }
            assert!(logits[1] > logits[0] && logits[2] > logits[3], "the ranking survives: {logits:?}");
        }
        Err(e) => {
            assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
            assert_eq!(e.field(), ClassifyOptions::FIELD_RAW_SCORES);
        }
    }
}

#[test]
fn live_rerank_right_truncation_keeps_the_query_whole() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle("TURBO_LIVE_RERANK_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load reranker");
    // 16 columns leaves 13 for content, and the query alone is longer than
    // that, so under RIGHT truncation (query kept whole, document cut) no
    // document token fits: both pairs are the same row and score the same.
    // Under the model's own policy the budget is split, so the two
    // documents survive in part and score differently.
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 16, ..Default::default() }).unwrap();
    let query = "How many people live in the city of Berlin in Germany and its surrounding metropolitan area";
    let docs = ["Berlin has 3.5 million registered inhabitants.", "Mount Fuji is the highest mountain in Japan."];
    match session.write_pairs(query, &docs, &RerankOptions { truncate: Truncate::Right, ..Default::default() }) {
        Ok(()) => {
            let r = session.run(&Default::default()).unwrap();
            let kept = read_f32(&r, 0);
            eprintln!("query-priority scores {kept:?}");
            assert_eq!(kept[0], kept[1], "with the document truncated away both rows are the query alone");
            drop(r);
            session.write_pairs(query, &docs, &RerankOptions::default()).unwrap();
            let r = session.run(&Default::default()).unwrap();
            let split = read_f32(&r, 0);
            eprintln!("model-policy scores {split:?}");
            assert_ne!(split[0], split[1], "the model policy keeps part of each document");
        }
        Err(e) => {
            assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
            assert_eq!(e.field(), RerankOptions::FIELD_TRUNCATE);
        }
    }
    // LEFT would drop the [CLS] and the query; a provider either offers it
    // or names the field.
    if let Err(e) = session.write_pairs(query, &docs, &RerankOptions { truncate: Truncate::Left, ..Default::default() })
    {
        assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
        assert_eq!(e.field(), RerankOptions::FIELD_TRUNCATE);
    }
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

#[test]
fn live_rerank_raw_scores_preserve_the_activated_ranking() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle("TURBO_LIVE_RERANK_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load reranker");
    let session = model.create_session(&SessionDesc { max_batch: 5, max_seq: 128, ..Default::default() }).unwrap();
    let query = "What is the capital of France?";
    let docs = [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe whose capital is Paris.",
        "The capital of Japan is Tokyo, a city of 14 million people.",
        "Sourdough bread needs a starter, flour, water, and salt.",
        "Rust is a systems programming language without a garbage collector.",
    ];
    session.write_pairs(query, &docs, &Default::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    let activated = read_f32(&r, 0);
    drop(r);
    assert!(activated.iter().all(|s| (0.0..=1.0).contains(s)), "activated scores are in [0, 1]: {activated:?}");

    let raw = RerankOptions { raw_scores: true, ..Default::default() };
    match session.write_pairs(query, &docs, &raw) {
        Ok(()) => {
            let r = session.run(&Default::default()).unwrap();
            let logits = read_f32(&r, 0);
            eprintln!("activated {activated:?} logits {logits:?}");
            assert_eq!(logits.len(), activated.len(), "raw_scores does not change the result shape");
            let order = |v: &[f32]| {
                let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
                idx
            };
            assert_eq!(
                order(&logits),
                order(&activated),
                "the activation is monotonic, so logits rank the documents exactly as the scores do"
            );
            assert!(
                logits.iter().any(|l| !(0.0..=1.0).contains(l)),
                "raw_scores must return logits, not activated scores: {logits:?}"
            );
        }
        Err(e) => {
            assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "raw_scores is honored or rejected: {e}");
            assert_eq!(e.field(), RerankOptions::FIELD_RAW_SCORES, "the rejection names raw_scores: {e}");
            assert!(!live.has_cap(abi::TURBO_CAP_OPT_RAW_SCORES), "a rejected option must not advertise its bit");
        }
    }
}

/// A text long enough to be truncated by a short session, with multi-byte
/// characters so span boundaries are checked against real UTF-8.
const NER_LONG_TEXT: &str = "Ada Lovelace besuchte Zürich und München mit Kollegen von Microsoft, \
bevor sie über Köpenick nach Berlin reiste und dort Grace Hopper traf, die für die \
United States Navy arbeitete und später in São Paulo lehrte.";

#[test]
fn live_token_classify_truncation_keeps_spans_inside_the_text() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle("TURBO_LIVE_NER_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load NER");
    let labels = model.info().labels.clone();
    // max_seq far below the text's token count: every run truncates.
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 16, ..Default::default() }).unwrap();
    for aggregation in [Aggregation::Model, Aggregation::None, Aggregation::Simple] {
        let opts = ClassifyOptions { aggregation, ..Default::default() };
        if let Err(e) = session.write_text_classify(&[NER_LONG_TEXT], &opts) {
            assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{aggregation:?} is honored or rejected: {e}");
            assert_eq!(e.field(), ClassifyOptions::FIELD_AGGREGATION, "the rejection names aggregation: {e}");
            continue;
        }
        let r = session.run(&Default::default()).unwrap();
        for s in r.spans() {
            assert_eq!(s.row, 0, "a single-row run produces spans on row 0 only: {s:?}");
            assert!(s.byte_start < s.byte_end, "{aggregation:?}: empty or inverted span {s:?}");
            assert!(
                s.byte_end as usize <= NER_LONG_TEXT.len(),
                "{aggregation:?}: span {s:?} runs past the {} byte input",
                NER_LONG_TEXT.len()
            );
            assert!(
                NER_LONG_TEXT.is_char_boundary(s.byte_start as usize),
                "{aggregation:?}: byte_start {} is inside a UTF-8 sequence",
                s.byte_start
            );
            assert!(
                NER_LONG_TEXT.is_char_boundary(s.byte_end as usize),
                "{aggregation:?}: byte_end {} is inside a UTF-8 sequence",
                s.byte_end
            );
            assert!((s.label as usize) < labels.len(), "{aggregation:?}: label {} is out of range", s.label);
            assert!(s.score > 0.0 && s.score <= 1.0, "{aggregation:?}: span score is a probability: {s:?}");
        }
        // Truncation keeps the head of the text, so nothing near the tail
        // can be reported.
        let last = r.spans().iter().map(|s| s.byte_end).max().unwrap_or(0);
        assert!(
            last < NER_LONG_TEXT.len() as u64,
            "{aggregation:?}: a 16-token window cannot reach byte {last} of the text"
        );
        drop(r);
    }
    // Refusing to truncate is a capacity error, never a silent short read.
    let e = session
        .write_text_classify(
            &[NER_LONG_TEXT],
            &ClassifyOptions { truncate: turbo::Truncate::None, ..Default::default() },
        )
        .unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "truncate=NONE over the budget is a capacity error: {e}");
}

#[test]
fn live_token_classify_aggregation_modes_group_words_differently() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle("TURBO_LIVE_NER_BUNDLE") else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load NER");
    let labels = model.info().labels.clone();
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 64, ..Default::default() }).unwrap();
    let text = "Ada Lovelace visited Berlin with colleagues from Microsoft.";

    let spans = |aggregation: Aggregation| -> Option<Vec<(String, String)>> {
        let opts = ClassifyOptions { aggregation, ..Default::default() };
        match session.write_text_classify(&[text], &opts) {
            Ok(()) => {}
            Err(e) => {
                assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{aggregation:?} is honored or rejected: {e}");
                assert_eq!(e.field(), ClassifyOptions::FIELD_AGGREGATION, "the rejection names aggregation: {e}");
                assert!(
                    !live.has_cap(abi::TURBO_CAP_OPT_AGGREGATION),
                    "{aggregation:?} was rejected although the device advertises TURBO_CAP_OPT_AGGREGATION"
                );
                return None;
            }
        }
        let r = session.run(&Default::default()).unwrap();
        Some(
            r.spans()
                .iter()
                .map(|s| {
                    (text[s.byte_start as usize..s.byte_end as usize].to_string(), labels[s.label as usize].clone())
                })
                .collect(),
        )
    };

    let none = spans(Aggregation::None);
    let simple = spans(Aggregation::Simple);
    let max = spans(Aggregation::Max);
    eprintln!("none {none:?}\nsimple {simple:?}\nmax {max:?}");

    if let Some(none) = &none {
        let words: Vec<&str> = none.iter().map(|(w, _)| w.as_str()).collect();
        assert!(
            words.contains(&"Ada") && words.contains(&"Lovelace"),
            "NONE reports one span per word, not a grouped entity: {words:?}"
        );
        assert!(none.iter().all(|(_, l)| l != "O"), "NONE reports entity words only: {none:?}");
    }
    for (name, grouped) in [("SIMPLE", &simple), ("MAX", &max)] {
        let Some(grouped) = grouped else { continue };
        let has = |t: &str, ent: &str| grouped.iter().any(|(w, l)| w == t && l.ends_with(ent));
        assert!(has("Ada Lovelace", "PER"), "{name} groups the two words of the person: {grouped:?}");
        assert!(has("Berlin", "LOC"), "{name}: {grouped:?}");
        assert!(has("Microsoft", "ORG"), "{name}: {grouped:?}");
        if let Some(none) = &none {
            assert!(
                grouped.len() < none.len(),
                "{name} merges words that NONE reports separately: {} vs {}",
                grouped.len(),
                none.len()
            );
        }
    }
    if let (Some(simple), Some(max)) = (&simple, &max) {
        let boundaries = |v: &[(String, String)]| v.iter().map(|(w, _)| w.clone()).collect::<Vec<_>>();
        assert_eq!(
            boundaries(simple),
            boundaries(max),
            "SIMPLE and MAX differ in how a word's label is chosen, not in where words are grouped"
        );
    }
}
