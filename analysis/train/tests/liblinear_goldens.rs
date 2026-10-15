//! The liblinear add-on's trainer against the JVM: the goldens of
//! `LiblinearTrainGolden.java` in `analysis/goldens-tool`, under
//! `analysis/goldens/data/train-liblinear/<run>`. Each golden holds the
//! events given to `LiblinearTrainer`, then either the exception it threw
//! or the liblinear text model inside the bytes `serialize` wrote, their
//! SHA-256, and `eval` and `getBestOutcome` on the contexts of the first
//! events, as given and reversed.
//!
//! For each golden this test trains on the same events with the run's
//! parameters and compares:
//!
//! - a refusal: the same message;
//! - the model: the text model's header exactly, and every weight within
//!   [`WEIGHT_TOLERANCE`] of the JVM's (`docs/liblinear.md` says why the
//!   weights are not always the same to the bit);
//! - the JVM's own model, rebuilt from its text with the index's outcome
//!   labels and predicate map: the bytes written from it have the JVM's
//!   SHA-256 and length, and its scores are within [`SCORE_TOLERANCE`] of
//!   the JVM's, with the same best outcome;
//! - the scores of the model trained here, within [`TRAINED_SCORE_TOLERANCE`].

use std::path::{Path, PathBuf};

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape, data_dir};
use turbo_train::liblinear::{self, LiblinearKey, LiblinearModel, format};
use turbo_train::{Event, Key, TrainingParameters};

mod common;
use common::{config, events};

/// The command that writes the goldens.
const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

/// Weights trained here against the JVM's, absolute.
const WEIGHT_TOLERANCE: f64 = 1e-9;
/// The JVM's model scored here against the JVM's scores: the sigmoid's
/// `Math.exp` may differ in the last bit.
const SCORE_TOLERANCE: f64 = 1e-12;
/// The model trained here scored against the JVM's scores.
const TRAINED_SCORE_TOLERANCE: f64 = 1e-9;

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;

const LAYERS: &[(&str, &[ColumnSpec])] = &[
    ("turbo:events", &[col("outcome", STR), col("context", STR), col("values", STR)]),
    ("turbo:liblinear-error", &[col("class", STR), col("message", STR)]),
    ("turbo:liblinear-text", &[col("line", STR)]),
    ("turbo:liblinear-model", &[col("sha256", STR), col("length", U32)]),
    ("turbo:liblinear-scores", &[col("event", U32), col("order", STR), col("outcome", U32), col("value", F64)]),
    ("turbo:liblinear-best", &[col("event", U32), col("order", STR), col("outcome", STR)]),
];

fn shapes() -> Vec<Shape> {
    LAYERS.iter().map(|(id, columns)| Shape { id: id.to_string(), anchor: GoldenAnchor::Document, columns }).collect()
}

fn layer<'g>(g: &'g Golden, id: &str) -> Option<&'g GoldenLayer> {
    g.layers.iter().find(|l| l.id == id)
}

fn golden_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.unwrap().path();
        if p.is_dir() {
            golden_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "golden") {
            out.push(p);
        }
    }
}

/// The trainer's parameters from the run's config records, as strings, as
/// the generator puts them.
fn params(g: &Golden) -> TrainingParameters {
    let mut p = TrainingParameters::new();
    for key in [LiblinearKey::SOLVER_TYPE, LiblinearKey::C, LiblinearKey::EPS, LiblinearKey::P, LiblinearKey::BIAS] {
        p.put(key, config(g, key));
    }
    p.put(Key::CUTOFF, config(g, "Cutoff"));
    p.put(Key::DATA_INDEXER, config(g, "DataIndexer"));
    p
}

#[derive(Default)]
struct Totals {
    goldens: usize,
    refused: usize,
    models: usize,
    identical_text: usize,
    weights: usize,
    identical_weights: usize,
    max_weight_diff: f64,
    scores: usize,
    identical_scores: usize,
    max_score_diff: f64,
    trained_scores: usize,
    identical_trained_scores: usize,
    max_trained_score_diff: f64,
    best: usize,
    trained_best_differs: usize,
}

/// `eval` on a context as the generator passes it, given or reversed.
fn eval(model: &LiblinearModel, context: &[&[u8]], reversed: bool, out: &mut [f64]) {
    if reversed {
        let r: Vec<&[u8]> = context.iter().rev().copied().collect();
        model.eval_modified_utf8(&r, out);
    } else {
        model.eval_modified_utf8(context, out);
    }
}

fn check(path: &Path, shapes: &[Shape], totals: &mut Totals) {
    let what = path.strip_prefix(data_dir()).unwrap_or(path).display().to_string();
    let g = Golden::read_with(path, shapes).unwrap_or_else(|e| panic!("{what}: {e}"));
    let events: Vec<Event> = events(&g);
    let result = liblinear::train_events(&events, &params(&g));
    totals.goldens += 1;

    if let Some(error) = layer(&g, "turbo:liblinear-error") {
        let message = error.strs("message").unwrap().get(0);
        let got = match result {
            Ok(_) => panic!("{what}: the JVM refused ({message}), training here did not"),
            Err(e) => e.to_string(),
        };
        // The unsorted refusal names the event after the JVM's message.
        assert!(got.starts_with(message), "{what}: refused with {got:?}, the JVM with {message:?}");
        totals.refused += 1;
        return;
    }
    let (model, _) = result.unwrap_or_else(|e| panic!("{what}: the JVM trained, training here refused: {e}"));
    totals.models += 1;

    // The text model: the header exactly, the weights to a tolerance.
    let mut jvm_text = String::new();
    for line in layer(&g, "turbo:liblinear-text").expect("a text model").strs("line").unwrap().iter() {
        jvm_text.push_str(line);
        jvm_text.push('\n');
    }
    let text = format::save_model(model.linear());
    totals.identical_text += usize::from(text == jvm_text);
    let header = |t: &str| t.lines().take_while(|l| *l != "w").map(str::to_owned).collect::<Vec<_>>();
    assert_eq!(header(&text), header(&jvm_text), "{what}: the text model's header");
    let jvm_linear = format::load_model(&jvm_text).unwrap_or_else(|e| panic!("{what}: {e}"));
    let (ours, theirs) = (model.linear().weights(), jvm_linear.weights());
    assert_eq!(ours.len(), theirs.len(), "{what}: weights");
    for (k, (&a, &b)) in ours.iter().zip(theirs).enumerate() {
        let diff = (a - b).abs();
        assert!(diff <= WEIGHT_TOLERANCE, "{what}: weight {k}: {a:e} here, {b:e} on the JVM");
        totals.weights += 1;
        totals.identical_weights += usize::from(a.to_bits() == b.to_bits());
        totals.max_weight_diff = totals.max_weight_diff.max(diff);
    }

    // The JVM's model with this index's labels and predicate map writes the
    // JVM's bytes: the text, the labels and the map in HashMap order.
    let jvm_model = LiblinearModel::from_parts(jvm_linear, model.outcomes().to_vec(), model.predicates().to_vec())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let bytes = format::write(&jvm_model).unwrap();
    let m = layer(&g, "turbo:liblinear-model").expect("the model's digest");
    assert_eq!(bytes.len() as u32, m.u32s("length").unwrap()[0], "{what}: the model's length");
    assert_eq!(turbo_maxent::sha256(&bytes).to_hex(), m.strs("sha256").unwrap().get(0), "{what}: the model's SHA-256");
    assert_eq!(format::read(&bytes).as_ref(), Ok(&jvm_model), "{what}: the bytes read back");

    // The scores, row by row: an array per event and order.
    let s = layer(&g, "turbo:liblinear-scores").expect("scores");
    let (event, order, outcome, value) =
        (s.u32s("event").unwrap(), s.strs("order").unwrap(), s.u32s("outcome").unwrap(), s.f64s("value").unwrap());
    let b = layer(&g, "turbo:liblinear-best").expect("best outcomes");
    let (best_event, best_order, best_outcome) =
        (b.u32s("event").unwrap(), b.strs("order").unwrap(), b.strs("outcome").unwrap());
    let n = model.num_outcomes();
    assert_eq!(value.len(), best_event.len() * n, "{what}: one score per outcome per array");
    let (mut jvm_probs, mut probs) = (vec![0.0; n], vec![0.0; n]);
    for (a, &e) in best_event.iter().enumerate() {
        let e = e as usize;
        let reversed = match best_order.get(a) {
            "given" => false,
            "reversed" => true,
            other => panic!("{what}: order {other:?}"),
        };
        let context: Vec<&[u8]> = events[e].context().iter().map(|p| p.as_bytes()).collect();
        eval(&jvm_model, &context, reversed, &mut jvm_probs);
        eval(&model, &context, reversed, &mut probs);
        for i in 0..n {
            let row = a * n + i;
            assert_eq!(
                (event[row] as usize, order.get(row), outcome[row] as usize),
                (e, best_order.get(a), i),
                "{what}"
            );
            let jvm = value[row];
            let diff = (jvm_probs[i] - jvm).abs();
            assert!(diff <= SCORE_TOLERANCE, "{what}: event {e} outcome {i}: {} here, {jvm} on the JVM", jvm_probs[i]);
            totals.scores += 1;
            totals.identical_scores += usize::from(jvm_probs[i].to_bits() == jvm.to_bits());
            totals.max_score_diff = totals.max_score_diff.max(diff);
            let diff = (probs[i] - jvm).abs();
            assert!(diff <= TRAINED_SCORE_TOLERANCE, "{what}: event {e} outcome {i}: trained {} , JVM {jvm}", probs[i]);
            totals.trained_scores += 1;
            totals.identical_trained_scores += usize::from(probs[i].to_bits() == jvm.to_bits());
            totals.max_trained_score_diff = totals.max_trained_score_diff.max(diff);
        }
        let best = jvm_model.outcome(LiblinearModel::best_outcome(&jvm_probs));
        assert!(
            best.eq_str(best_outcome.get(a)),
            "{what}: event {e}: best outcome {best:?}, the JVM's {}",
            best_outcome.get(a)
        );
        totals.best += 1;
        let trained = model.outcome(LiblinearModel::best_outcome(&probs));
        totals.trained_best_differs += usize::from(!trained.eq_str(best_outcome.get(a)));
    }
}

#[test]
fn every_golden_matches_the_jvm() {
    let root = data_dir().join("train-liblinear");
    if !root.is_dir() {
        eprintln!("skipped: {} is absent; write the goldens with `{GENERATE}`", root.display());
        return;
    }
    let shapes = shapes();
    let mut paths = Vec::new();
    golden_files(&root, &mut paths);
    paths.sort();
    let mut totals = Totals::default();
    for path in &paths {
        check(path, &shapes, &mut totals);
    }
    assert!(totals.models > 0 && totals.refused > 0, "{}: the goldens cover models and refusals", root.display());
    eprintln!(
        "{} goldens: {} refused as on the JVM, {} models ({} text models identical); weights {} of {} identical, \
         largest difference {:e}; JVM model scores {} of {} identical, largest difference {:e}; trained model \
         scores {} of {} identical, largest difference {:e}; best outcomes {} (trained model differs on {})",
        totals.goldens,
        totals.refused,
        totals.models,
        totals.identical_text,
        totals.identical_weights,
        totals.weights,
        totals.max_weight_diff,
        totals.identical_scores,
        totals.scores,
        totals.max_score_diff,
        totals.identical_trained_scores,
        totals.trained_scores,
        totals.max_trained_score_diff,
        totals.best,
        totals.trained_best_differs,
    );
}
