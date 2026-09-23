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
use turbo::{Aggregation, CapStatus, ClassifyOptions, Modality, ModelDesc, RerankOptions, SessionDesc, Task, Truncate};
use turbo_conformance::live::{live, Live};
use turbo_conformance::{read_f32, read_i32};

/// The bundle directory `var` names, for a task the device under test
/// offers. A device whose capability cell does not offer the task prints
/// `not applicable` and the case returns; a device that does offer it with
/// no bundle configured is a configuration error and panics naming the
/// variable, so a live run never skips a case it could have run (the rule
/// `Target::offered` applies in `crates/turbo-conformance/src/lib.rs`).
fn bundle_for(live: &Live, task: Task, var: &str) -> Option<std::path::PathBuf> {
    let cell = live
        .ctx
        .runtime()
        .capability(live.ctx.device_index(), task, Modality::Text)
        .unwrap_or_else(|e| panic!("{task:?} x TEXT capability of `{}`: {e}", live.device.name));
    if matches!(cell.status, CapStatus::Unsupported | CapStatus::Planned) {
        println!(
            "not applicable: {} device {} (`{}`) does not offer {task:?} for Text (capability {:?})",
            live.provider, live.device.ordinal, live.device.name, cell.status
        );
        return None;
    }
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Some(std::path::PathBuf::from(v)),
        _ => panic!(
            "{} device {} (`{}`) offers {task:?} but {var} is not set; point it at a bundle of that kind",
            live.provider, live.device.ordinal, live.device.name
        ),
    }
}

#[test]
fn live_rerank_orders_relevant_documents_first() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle_for(&live, Task::Rerank, "TURBO_LIVE_RERANK_BUNDLE") else { return };
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
    // The bundle's contract says whether the head's logits are activated:
    // a sigmoid reranker scores in [0, 1] with 0.5 as the midpoint, an
    // identity one (the ms-marco cross-encoders declare Identity) hands
    // back logits with 0 as the midpoint. Either way Berlin ranks first.
    let sigmoid = model.bundle().contract().activation.as_deref() == Some("sigmoid");
    let midpoint = if sigmoid { 0.5 } else { 0.0 };
    if sigmoid {
        assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)), "sigmoid scores in [0, 1]");
    }
    assert_eq!(sorted[0], 0, "the Berlin passage ranks first");
    assert!(scores[0] > midpoint && scores[1] < midpoint && scores[2] < midpoint, "{scores:?}");
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
            if sigmoid {
                assert!(logits.iter().any(|l| !(0.0..=1.0).contains(l)), "logits, not activated scores");
            } else {
                assert_eq!(logits, scores, "an identity head's raw scores are its scores");
            }
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
    let Some(dir) = bundle_for(&live, Task::Classify, "TURBO_LIVE_CLASSIFY_BUNDLE") else { return };
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
    let Some(dir) = bundle_for(&live, Task::Rerank, "TURBO_LIVE_RERANK_BUNDLE") else { return };
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
    let Some(dir) = bundle_for(&live, Task::TokenClassify, "TURBO_LIVE_NER_BUNDLE") else { return };
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
    let Some(dir) = bundle_for(&live, Task::Rerank, "TURBO_LIVE_RERANK_BUNDLE") else { return };
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
    let sigmoid = model.bundle().contract().activation.as_deref() == Some("sigmoid");
    if sigmoid {
        assert!(activated.iter().all(|s| (0.0..=1.0).contains(s)), "activated scores are in [0, 1]: {activated:?}");
    }

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
            if sigmoid {
                assert!(
                    logits.iter().any(|l| !(0.0..=1.0).contains(l)),
                    "raw_scores must return logits, not activated scores: {logits:?}"
                );
                for (l, a) in logits.iter().zip(&activated) {
                    let s = 1.0 / (1.0 + (-l).exp());
                    assert!((s - a).abs() < 1e-4, "the score is the sigmoid of the logit: {l} -> {a}");
                }
            } else {
                assert_eq!(logits, activated, "an identity head's raw scores are its scores");
            }
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
    let Some(dir) = bundle_for(&live, Task::TokenClassify, "TURBO_LIVE_NER_BUNDLE") else { return };
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
    let Some(dir) = bundle_for(&live, Task::TokenClassify, "TURBO_LIVE_NER_BUNDLE") else { return };
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

// ---------------------------------------------------------------------------
// Precision checks against the PyTorch references
// ---------------------------------------------------------------------------
//
// The cases above assert orderings and that a softmax row sums to one, which
// every wrong-but-monotonic implementation also satisfies. The cases below
// compare the numbers a provider returns against references computed from the
// same Hugging Face checkpoints in PyTorch, float32, on the CPU
// (`scripts/gen-reference-tasks.py`), the way `live_embed.rs` compares
// embeddings against `testdata/reference_embeddings/`.
//
// The tolerances are absolute. Each is eight times the worst difference
// measured between ONNX Runtime on the `cuda` provider (RTX 4080 SUPER,
// 2026-09-22) and PyTorch float32 on the CPU, over exactly the cases
// below. Eight is headroom for another GPU, driver or execution provider
// without leaving room for a real regression: the OpenVINO CPU device on
// the same machine and day lands two to three orders of magnitude closer
// than cuda does, so nothing in between is near a gate. Every case prints
// the three largest differences it saw, so a later run shows its headroom
// without a debugger. The measurements are recorded in
// `testdata/receipts/turbo/precision-tasks-rtx4080-2026-09-22.json`.

/// Reranker activated scores: 8 x the 1.064e-5 measured on cuda.
const RERANK_SCORE_ATOL: f32 = 8.5e-5;
/// Reranker logits: 8 x the 1.190e-3 measured on cuda. Logits are two
/// orders of magnitude larger than the scores they activate to, and these
/// run to +-11, so the same relative error is a much larger absolute one.
const RERANK_LOGIT_ATOL: f32 = 9.5e-3;
/// Classifier softmax probabilities: 8 x the 4.085e-4 measured on cuda.
const CLASSIFY_PROB_ATOL: f32 = 3.3e-3;
/// Classifier logits: 8 x the 2.351e-3 measured on cuda.
const CLASSIFY_LOGIT_ATOL: f32 = 1.9e-2;
/// Token-classifier per-token softmax probabilities: 8 x the 3.310e-4
/// measured on cuda. This doubles as the width below which two labels are a
/// tie the arithmetic decides rather than a disagreement about the model.
const NER_PROB_ATOL: f32 = 2.6e-3;
/// Token-classifier aggregated span scores (a mean over word scores, so no
/// wider than the per-token probabilities): 8 x the 3.310e-4 measured on
/// cuda.
const NER_SPAN_ATOL: f32 = 2.6e-3;

/// Absolute differences seen in one comparison, worst first when reported.
/// A single worst value can hide a tokenization difference among float
/// noise, so the three largest are printed, not just the one.
#[derive(Default)]
struct Worst {
    seen: Vec<(f32, String)>,
}

impl Worst {
    fn see(&mut self, got: f32, want: f32, at: impl FnOnce() -> String) {
        self.seen.push(((got - want).abs(), at()));
    }

    fn worst(&self) -> f32 {
        self.seen.iter().map(|(d, _)| *d).fold(0.0f32, f32::max)
    }

    /// Print the three largest differences and hold the largest to `atol`.
    fn check(&self, what: &str, atol: f32) {
        let mut top = self.seen.clone();
        top.sort_by(|a, b| b.0.total_cmp(&a.0));
        top.truncate(3);
        eprintln!(
            "{what}: {} comparisons, worst |device - pytorch| = {:.3e}, tolerance {atol:.3e}",
            self.seen.len(),
            self.worst()
        );
        for (d, at) in &top {
            eprintln!("    {d:.3e}  {at}");
        }
        assert!(self.worst() <= atol, "{what}: {:.3e} at {} exceeds the tolerance {atol:.3e}", top[0].0, top[0].1);
    }
}

/// A reference file under `testdata/`, or `TURBO_TESTDATA_DIR` when the test
/// binary runs somewhere else.
fn reference(rel: &str) -> serde_json::Value {
    let root = match std::env::var("TURBO_TESTDATA_DIR") {
        Ok(v) if !v.is_empty() => std::path::PathBuf::from(v),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata"),
    };
    let path = root.join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn num(v: &serde_json::Value) -> f32 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number")) as f32
}

fn nums(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap_or_else(|| panic!("{v} is not an array")).iter().map(num).collect()
}

fn text_of(v: &serde_json::Value, key: &str) -> String {
    v[key].as_str().unwrap_or_else(|| panic!("{key} is not a string in {v}")).to_string()
}

/// Print what the reference was produced from, so a run in a log says which
/// checkpoint it was held to.
fn announce(what: &str, r: &serde_json::Value) {
    eprintln!(
        "{what} reference: {} revision {} (torch {}, transformers {}, {} on {})",
        r["model_id"].as_str().unwrap_or("?"),
        r["revision"].as_str().unwrap_or("?"),
        r["produced_by"]["torch"].as_str().unwrap_or("?"),
        r["produced_by"]["transformers"].as_str().unwrap_or("?"),
        r["date"].as_str().unwrap_or("?"),
        r["machine"].as_str().unwrap_or("?"),
    );
}

/// The BIO prefix stripped off a label, matching the providers' `entity_of`
/// and the `entity_group` transformers' pipeline reports.
fn entity_of(label: &str) -> &str {
    let b = label.as_bytes();
    if b.len() > 2 && b[1] == b'-' && matches!(b[0], b'B' | b'I' | b'L' | b'U' | b'E' | b'S') {
        &label[2..]
    } else {
        label
    }
}

/// Cases that share a query, in order, so a batch is one `write_pairs` call.
fn rerank_groups(cases: &[serde_json::Value], max_batch: usize) -> Vec<(String, Vec<usize>)> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let q = text_of(c, "query");
        match groups.last_mut() {
            Some((prev, idx)) if *prev == q && idx.len() < max_batch => idx.push(i),
            _ => groups.push((q, vec![i])),
        }
    }
    groups
}

#[test]
fn live_rerank_matches_the_pytorch_reference() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle_for(&live, Task::Rerank, "TURBO_LIVE_RERANK_BUNDLE") else { return };
    let r = reference("reference_rerank/ms_marco_minilm_l6.json");
    announce("rerank", &r);
    let cases = r["cases"].as_array().expect("cases").clone();
    let max_seq = r["tokenization"]["max_length"].as_u64().expect("max_length") as u32;

    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load reranker");
    eprintln!("bundle model_id {}", model.bundle().manifest().model_id);
    let sigmoid = model.bundle().contract().activation.as_deref() == Some("sigmoid");
    assert!(
        model.info().max_seq >= max_seq,
        "the reference truncates at {max_seq} tokens but the model offers {}",
        model.info().max_seq
    );
    let session = model.create_session(&SessionDesc { max_batch: 4, max_seq, ..Default::default() }).unwrap();
    let groups = rerank_groups(&cases, 4);

    // Activated scores: what the bundle's contract says the head produces.
    let mut worst = Worst::default();
    for (query, idx) in &groups {
        let docs: Vec<&str> = idx.iter().map(|&i| cases[i]["document"].as_str().unwrap()).collect();
        session.write_pairs(query, &docs, &RerankOptions::default()).unwrap();
        let run = session.run(&Default::default()).unwrap();
        let got = read_f32(&run, 0);
        assert_eq!(got.len(), idx.len());
        for (k, &i) in idx.iter().enumerate() {
            let want = num(if sigmoid { &cases[i]["sigmoid"] } else { &cases[i]["logit"] });
            worst.see(got[k], want, || format!("{} (got {} want {want})", text_of(&cases[i], "id"), got[k]));
        }
    }
    worst.check("rerank scores", if sigmoid { RERANK_SCORE_ATOL } else { RERANK_LOGIT_ATOL });

    // raw_scores: the logit itself, on a provider that offers it.
    match session.write_pairs(&groups[0].0, &["probe"], &RerankOptions { raw_scores: true, ..Default::default() }) {
        Ok(()) => {
            drop(session.run(&Default::default()).unwrap());
            let mut worst = Worst::default();
            for (query, idx) in &groups {
                let docs: Vec<&str> = idx.iter().map(|&i| cases[i]["document"].as_str().unwrap()).collect();
                session.write_pairs(query, &docs, &RerankOptions { raw_scores: true, ..Default::default() }).unwrap();
                let run = session.run(&Default::default()).unwrap();
                let got = read_f32(&run, 0);
                for (k, &i) in idx.iter().enumerate() {
                    let want = num(&cases[i]["logit"]);
                    worst.see(got[k], want, || format!("{} (got {} want {want})", text_of(&cases[i], "id"), got[k]));
                }
            }
            worst.check("rerank logits (raw_scores)", RERANK_LOGIT_ATOL);
        }
        Err(e) => {
            assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "raw_scores is honored or rejected: {e}");
            assert_eq!(e.field(), RerankOptions::FIELD_RAW_SCORES);
            eprintln!("rerank logits: {} does not offer raw_scores, logits not compared", live.provider);
        }
    }
}

#[test]
fn live_classify_matches_the_pytorch_reference() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle_for(&live, Task::Classify, "TURBO_LIVE_CLASSIFY_BUNDLE") else { return };
    let r = reference("reference_classify/sst2_distilbert.json");
    announce("classify", &r);
    let cases = r["cases"].as_array().expect("cases").clone();
    let max_seq = r["tokenization"]["max_length"].as_u64().expect("max_length") as u32;
    let labels: Vec<String> = r["labels"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();

    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load classifier");
    assert_eq!(model.info().labels, labels, "the bundle's labels are the reference's");
    let width = labels.len();
    let texts: Vec<&str> = cases.iter().map(|c| c["text"].as_str().unwrap()).collect();
    let session =
        model.create_session(&SessionDesc { max_batch: texts.len() as u32, max_seq, ..Default::default() }).unwrap();

    session.write_text_classify(&texts, &ClassifyOptions::default()).unwrap();
    let run = session.run(&Default::default()).unwrap();
    assert_eq!(run.output(0).unwrap().shape, vec![texts.len() as u64, width as u64]);
    let got = read_f32(&run, 0);
    let mut worst = Worst::default();
    for (i, c) in cases.iter().enumerate() {
        let want = nums(&c["probs"]);
        for j in 0..width {
            worst.see(got[i * width + j], want[j], || {
                format!("{} {} (got {} want {})", text_of(c, "id"), labels[j], got[i * width + j], want[j])
            });
        }
    }
    drop(run);
    worst.check("classify probabilities", CLASSIFY_PROB_ATOL);

    match session.write_text_classify(&texts, &ClassifyOptions { raw_scores: true, ..Default::default() }) {
        Ok(()) => {
            let run = session.run(&Default::default()).unwrap();
            let got = read_f32(&run, 0);
            let mut worst = Worst::default();
            for (i, c) in cases.iter().enumerate() {
                let want = nums(&c["logits"]);
                for j in 0..width {
                    worst.see(got[i * width + j], want[j], || {
                        format!("{} {} (got {} want {})", text_of(c, "id"), labels[j], got[i * width + j], want[j])
                    });
                }
            }
            worst.check("classify logits (raw_scores)", CLASSIFY_LOGIT_ATOL);
        }
        Err(e) => {
            assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "raw_scores is honored or rejected: {e}");
            assert_eq!(e.field(), ClassifyOptions::FIELD_RAW_SCORES);
            eprintln!("classify logits: {} does not offer raw_scores, logits not compared", live.provider);
        }
    }
}

/// The spans one aggregation strategy is expected to produce for one case:
/// `(byte_start, byte_end, entity, score)`.
fn expected_spans(
    case: &serde_json::Value,
    aggregation: Aggregation,
    model_default: Aggregation,
) -> Vec<(u64, u64, String, f32)> {
    let resolved = if aggregation == Aggregation::Model { model_default } else { aggregation };
    match resolved {
        // Transformers has no word-aligned "none": its own "none" is one
        // entry per sub-token. The providers report one span per word whose
        // first sub-token is not `O`, which is `words` in the reference.
        Aggregation::None => case["words"]
            .as_array()
            .expect("words")
            .iter()
            .filter(|w| w["first_label"].as_str() != Some("O"))
            .map(|w| {
                (
                    w["byte_start"].as_u64().unwrap(),
                    w["byte_end"].as_u64().unwrap(),
                    entity_of(w["first_label"].as_str().unwrap()).to_string(),
                    num(&w["first_score"]),
                )
            })
            .collect(),
        // `SIMPLE` and `FIRST` are both word-aligned in these providers (a
        // word's label is its first sub-token's), which is transformers'
        // `first`. Transformers' `simple` is token-aligned and can split a
        // word, so it is not the rule these providers implement; it is in
        // the reference for comparison, not as a gate.
        Aggregation::Simple | Aggregation::First => hf_spans(case, "first"),
        Aggregation::Max => hf_spans(case, "max"),
        Aggregation::Model => unreachable!("resolved above"),
    }
}

fn hf_spans(case: &serde_json::Value, strategy: &str) -> Vec<(u64, u64, String, f32)> {
    case["aggregation"][strategy]
        .as_array()
        .unwrap_or_else(|| panic!("aggregation.{strategy}"))
        .iter()
        .map(|s| {
            (s["byte_start"].as_u64().unwrap(), s["byte_end"].as_u64().unwrap(), text_of(s, "entity"), num(&s["score"]))
        })
        .collect()
}

#[test]
fn live_token_classify_matches_the_pytorch_reference() {
    let Some(live) = live() else { return };
    let Some(dir) = bundle_for(&live, Task::TokenClassify, "TURBO_LIVE_NER_BUNDLE") else { return };
    let r = reference("reference_token_classify/bert_base_ner.json");
    announce("token_classify", &r);
    let cases = r["cases"].as_array().expect("cases").clone();
    let labels: Vec<String> = r["labels"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();

    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load NER");
    assert_eq!(model.info().labels, labels, "the bundle's labels are the reference's");
    let width = labels.len();
    let model_default = model.bundle().contract().aggregation.as_deref().unwrap_or("simple");
    let model_default = match model_default {
        "none" => Aggregation::None,
        "simple" => Aggregation::Simple,
        "first" => Aggregation::First,
        "max" => Aggregation::Max,
        other => panic!("contract.aggregation `{other}`"),
    };
    eprintln!("contract.aggregation resolves TURBO_AGGREGATE_MODEL to {model_default:?}");

    let longest = cases.iter().map(|c| c["n_tokens"].as_u64().unwrap()).max().unwrap() as u32;
    let seq = longest.next_power_of_two().max(32);
    let texts: Vec<&str> = cases.iter().map(|c| c["text"].as_str().unwrap()).collect();
    let session = model
        .create_session(&SessionDesc { max_batch: texts.len() as u32, max_seq: seq, ..Default::default() })
        .unwrap();

    // Per token: the label the provider's softmax argmax picks, and the
    // whole probability row, for every column the reference covers.
    session
        .write_text_classify(&texts, &ClassifyOptions { aggregation: Aggregation::None, ..Default::default() })
        .unwrap();
    let run = session.run(&Default::default()).unwrap();
    assert_eq!(run.output(0).unwrap().shape, vec![texts.len() as u64, seq as u64, width as u64]);
    let probs = read_f32(&run, 0);
    let mut worst = Worst::default();
    let mut disagreements = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        for t in c["tokens"].as_array().unwrap() {
            let col = t["column"].as_u64().unwrap() as usize;
            let want = nums(&t["probs"]);
            let base = (i * seq as usize + col) * width;
            let row = &probs[base..base + width];
            for j in 0..width {
                worst.see(row[j], want[j], || {
                    format!("{} column {col} {} (got {} want {})", text_of(c, "id"), labels[j], row[j], want[j])
                });
            }
            let got_label = (0..width).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            let want_label = t["label_id"].as_u64().unwrap() as usize;
            if got_label != want_label {
                // Two labels closer together than the tolerance are a tie
                // the arithmetic decides, not a disagreement about the
                // model; anything wider is one.
                let gap = (want[want_label] - want[got_label]).abs();
                if gap > NER_PROB_ATOL {
                    disagreements.push(format!(
                        "{} column {col} `{}`: {} not {} (reference gap {gap:.3e})",
                        text_of(c, "id"),
                        text_of(t, "token"),
                        labels[got_label],
                        labels[want_label]
                    ));
                }
            }
        }
    }
    drop(run);
    assert!(disagreements.is_empty(), "per-token labels differ from the reference: {disagreements:#?}");
    worst.check("token_classify per-token probabilities", NER_PROB_ATOL);

    // Aggregated spans, for every strategy this provider offers.
    for aggregation in
        [Aggregation::Model, Aggregation::None, Aggregation::Simple, Aggregation::First, Aggregation::Max]
    {
        let opts = ClassifyOptions { aggregation, ..Default::default() };
        if let Err(e) = session.write_text_classify(&texts, &opts) {
            assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{aggregation:?} is honored or rejected: {e}");
            assert_eq!(e.field(), ClassifyOptions::FIELD_AGGREGATION, "the rejection names aggregation: {e}");
            assert!(!live.has_cap(abi::TURBO_CAP_OPT_AGGREGATION), "a rejected option must not advertise its bit");
            eprintln!("token_classify {aggregation:?}: rejected by {}, not compared", live.provider);
            continue;
        }
        let run = session.run(&Default::default()).unwrap();
        let mut worst = Worst::default();
        for (i, c) in cases.iter().enumerate() {
            let want = expected_spans(c, aggregation, model_default);
            let got: Vec<(u64, u64, String, f32)> = run
                .spans()
                .iter()
                .filter(|s| s.row == i as u32)
                .map(|s| (s.byte_start, s.byte_end, entity_of(&labels[s.label as usize]).to_string(), s.score))
                .collect();
            let shape =
                |v: &[(u64, u64, String, f32)]| v.iter().map(|(a, b, e, _)| (*a, *b, e.clone())).collect::<Vec<_>>();
            assert_eq!(
                shape(&got),
                shape(&want),
                "{aggregation:?} on `{}`: spans differ from the reference in offsets or label",
                text_of(c, "id")
            );
            for (g, w) in got.iter().zip(&want) {
                worst.see(g.3, w.3, || {
                    format!("{} span {}..{} {} (got {} want {})", text_of(c, "id"), g.0, g.1, g.2, g.3, w.3)
                });
            }
        }
        drop(run);
        worst.check(&format!("token_classify {aggregation:?} span scores"), NER_SPAN_ATOL);
    }
}
