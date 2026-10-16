//! Parity with OpenNLP's `DocumentCategorizerSVM` on a JVM: the goldens of
//! `SvmDoccatGolden.java` in `analysis/goldens-tool`, under
//! `analysis/goldens/data/svm-doccat/<run>`. Each run's model is read from
//! the golden of its training corpus (the LIBSVM model and the sidecar,
//! one line per row), and every document golden is compared: the feature
//! vector, the decision values, the vote and the categories layer.
//!
//! Tolerances, from `docs/doccat.md`:
//!
//! - The feature vector, the vote and the category order are exact.
//! - Decision values of the linear and polynomial kernels are exact: they
//!   are sums of products in LIBSVM's order. The RBF and sigmoid kernels
//!   call `Math.exp` and `Math.tanh`, which the Rust standard library does
//!   not reproduce in the last place, so their decision values are within
//!   1e-12.
//! - Probabilities are within 1e-9: Platt's sigmoid calls `Math.exp` for
//!   every kernel, and the pairwise coupling iterates on what it returns.

use std::path::{Path, PathBuf};

use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_layers::Tokens;
use turbo_doccat::svm::libsvm::Kernel;
use turbo_doccat::svm::{Scratch, SvmCategorizer, SvmDoccatModel};
use turbo_goldens::{
    ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape, Tolerance, known_shapes, load_layer,
    matches_golden,
};

const LINE: &[ColumnSpec] = &[ColumnSpec::new("line", ColumnType::Str)];
const VECTOR: &[ColumnSpec] = &[ColumnSpec::new("index", ColumnType::U32), ColumnSpec::new("value", ColumnType::F64)];
const DECISIONS: &[ColumnSpec] = &[
    ColumnSpec::new("first", ColumnType::Str),
    ColumnSpec::new("second", ColumnType::Str),
    ColumnSpec::new("value", ColumnType::F64),
];
const VOTE: &[ColumnSpec] = &[ColumnSpec::new("category", ColumnType::Str)];

const DECISION_TOLERANCE: f64 = 1e-12;
const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// The command that writes the goldens.
const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

fn shapes() -> Vec<Shape> {
    let mut shapes = known_shapes();
    for (id, columns) in [
        ("turbo:svm-model", LINE),
        ("turbo:svm-sidecar", LINE),
        ("turbo:svm-vector", VECTOR),
        ("turbo:svm-decisions", DECISIONS),
        ("turbo:svm-vote", VOTE),
    ] {
        shapes.push(Shape { id: id.into(), anchor: GoldenAnchor::Document, columns });
    }
    shapes
}

fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("no layer {id}"))
}

/// The files in `dir`, sorted.
fn sorted(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    out.sort();
    out
}

/// The text of a file exported one line per row.
fn file(g: &Golden, id: &str) -> String {
    let mut text = String::new();
    for line in layer(g, id).strs("line").expect("a line column").iter() {
        text.push_str(line);
        text.push('\n');
    }
    text
}

/// The run's model, from the one golden in its training set directory.
fn model(run: &Path, shapes: &[Shape]) -> SvmDoccatModel {
    let train = sorted(&run.join("svm-doccat-train"));
    assert_eq!(train.len(), 1, "{}: one training corpus golden", run.display());
    let g = Golden::read_with(&train[0], shapes).unwrap_or_else(|e| panic!("{e}"));
    // The golden was made from the corpus that is checked in.
    g.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
    SvmDoccatModel::from_text(&file(&g, "turbo:svm-model"), &file(&g, "turbo:svm-sidecar"))
        .unwrap_or_else(|e| panic!("{}: {e}", train[0].display()))
}

/// Counts over every document of every run.
#[derive(Default)]
struct Totals {
    runs: usize,
    documents: usize,
    decisions: usize,
    exact_decisions: usize,
    max_decision_diff: f64,
}

fn check_run(run: &Path, shapes: &[Shape], totals: &mut Totals) {
    let name = run.file_name().unwrap().to_string_lossy().into_owned();
    let model = model(run, shapes);
    let exact = matches!(model.libsvm().kernel(), Kernel::Linear | Kernel::Polynomial { .. });
    let categorizer = SvmCategorizer::new(model.clone()).unwrap_or_else(|e| panic!("{name}: {e}"));
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).add(categorizer).build().expect("the analyzer");
    let mut scratch = Scratch::new();
    let documents = sorted(&run.join("svm-doccat"));
    assert!(!documents.is_empty(), "{name}: no document goldens");
    for path in documents {
        let what = format!("{name}/{}", path.file_stem().unwrap().to_string_lossy());
        let g = Golden::read_with(&path, shapes).unwrap_or_else(|e| panic!("{e}"));
        let text = g.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let mut doc = Document::new(text.as_str());
        load_layer(&mut doc, &g, &Key::<Tokens>::DEFAULT).unwrap_or_else(|e| panic!("{what}: {e}"));

        let tokens = doc.get::<Tokens>().unwrap();
        let words: Vec<&str> = tokens.spans.iter().map(|(_, s)| &text[s.range()]).collect();
        let vote = model.decision_values(&words, &mut scratch);

        let v = layer(&g, "turbo:svm-vector");
        let (index, value) = scratch.vector();
        let want: Vec<i32> = v.u32s("index").unwrap().iter().map(|&i| i as i32).collect();
        assert_eq!(index, want, "{what}: feature indexes");
        let bits = |xs: &[f64]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(value), bits(v.f64s("value").unwrap()), "{what}: feature values");

        let d = layer(&g, "turbo:svm-decisions");
        let labels = model.libsvm().labels();
        let (first, second) = (d.strs("first").unwrap(), d.strs("second").unwrap());
        let mut pair = 0;
        for i in 0..labels.len() {
            for j in i + 1..labels.len() {
                assert_eq!(first.get(pair), model.categories()[labels[i] as usize].as_str(), "{what}");
                assert_eq!(second.get(pair), model.categories()[labels[j] as usize].as_str(), "{what}");
                pair += 1;
            }
        }
        let golden = d.f64s("value").unwrap();
        assert_eq!(scratch.decision_values().len(), golden.len(), "{what}: decision values");
        for (k, (&rust, &jvm)) in scratch.decision_values().iter().zip(golden).enumerate() {
            let diff = (rust - jvm).abs();
            if exact {
                assert_eq!(rust.to_bits(), jvm.to_bits(), "{what}: decision value {k}: {rust} against {jvm}");
            } else {
                assert!(diff <= DECISION_TOLERANCE, "{what}: decision value {k}: {rust} against {jvm}");
            }
            totals.decisions += 1;
            totals.exact_decisions += usize::from(rust.to_bits() == jvm.to_bits());
            totals.max_decision_diff = totals.max_decision_diff.max(diff);
        }
        let voted = layer(&g, "turbo:svm-vote").strs("category").unwrap();
        assert_eq!(voted.len(), 1, "{what}: vote");
        assert_eq!(voted.get(0), model.categories()[vote.category].as_str(), "{what}: vote");

        let mut docs = [doc];
        for r in analyzer.run(&mut docs) {
            r.unwrap_or_else(|e| panic!("{what}: {e}"));
        }
        if let Err(m) =
            matches_golden::<turbo_analysis_layers::Categories>(&docs[0], &g, Tolerance::abs(PROBABILITY_TOLERANCE))
        {
            panic!("{what}: {m}");
        }
        totals.documents += 1;
    }
    totals.runs += 1;
}

#[test]
fn every_run_matches_the_jvm() {
    let data = turbo_goldens::data_dir();
    let root = data.join("svm-doccat");
    if !root.is_dir() {
        eprintln!("skipped: {} is absent; write the goldens with `{GENERATE}`", root.display());
        return;
    }
    let shapes = shapes();
    let mut totals = Totals::default();
    for run in sorted(&root) {
        check_run(&run, &shapes, &mut totals);
    }
    assert!(totals.runs > 0, "{}: no runs", root.display());
    eprintln!(
        "{} runs, {} documents, {} of {} decision values bit for bit, largest difference {:e}",
        totals.runs, totals.documents, totals.exact_decisions, totals.decisions, totals.max_decision_diff
    );
}
