//! Reading the training goldens (`analysis/goldens/data/train/`): their
//! layer shapes, the events and the run's settings.

#![allow(dead_code)]

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_jvm::text::parse_float;
use turbo_train::{Event, IndexParams, IndexerKind, JavaString};

/// A predicate as `TrainGolden.lossless` wrote it: a doubled backslash is
/// one, `\uXXXX` is a lone surrogate, everything else is itself.
pub fn java_string(field: &str) -> JavaString {
    let mut units: Vec<u16> = Vec::with_capacity(field.len());
    let mut chars = field.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            units.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(u16::from(b'\\')),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).unwrap_or_else(|_| panic!("{field:?}: bad escape")));
            }
            other => panic!("{field:?}: unexpected {other:?} after a backslash"),
        }
    }
    JavaString::from_utf16(&units)
}

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;

/// The layer blocks a training golden has, all anchored on the document.
const TRAIN_LAYERS: &[(&str, &[ColumnSpec])] = &[
    ("turbo:events", &[col("outcome", STR), col("context", STR), col("values", STR)]),
    ("turbo:indexer-error", &[col("class", STR), col("message", STR)]),
    ("turbo:event-checksum", &[col("crc32c", U32)]),
    ("turbo:indexer-outcomes", &[col("label", STR)]),
    ("turbo:indexer-predicates", &[col("label", STR), col("count", U32)]),
    ("turbo:indexer-events", &[col("outcome", U32), col("context", STR), col("values", STR), col("seen", U32)]),
    ("turbo:gis-training", &[col("iterations", U32), col("message", STR)]),
    (
        "turbo:gis-iterations",
        &[
            col("iteration", U32),
            col("likelihood", F64),
            col("correct", U32),
            col("events", U32),
            col("parameters", STR),
        ],
    ),
    ("turbo:gis-parameters", &[col("iteration", U32), col("predicate", U32), col("outcome", U32), col("value", F64)]),
    ("turbo:gis-model", &[col("sha256", STR), col("length", U32)]),
    ("turbo:qn-training", &[col("iterations", U32), col("stop", STR), col("message", STR)]),
    (
        "turbo:qn-iterations",
        &[col("iteration", U32), col("value", F64), col("rate", F64), col("accuracy", F64), col("parameters", STR)],
    ),
    ("turbo:qn-parameters", &[col("iteration", U32), col("predicate", U32), col("outcome", U32), col("value", F64)]),
    ("turbo:qn-model", &[col("sha256", STR), col("length", U32)]),
    ("turbo:perceptron-training", &[col("iterations", U32), col("message", STR)]),
    (
        "turbo:perceptron-monitor",
        &[col("iteration", U32), col("correct", U32), col("events", U32), col("accuracy", F64)],
    ),
    ("turbo:perceptron-iterations", &[col("iteration", U32), col("parameters", STR)]),
    (
        "turbo:perceptron-parameters",
        &[col("iteration", U32), col("predicate", U32), col("outcome", U32), col("value", F64)],
    ),
    ("turbo:perceptron-model", &[col("sha256", STR), col("length", U32), col("error", STR)]),
];

/// `QNTrainer`'s class name: the annotator of the QN training goldens.
pub const QN_TRAINER: &str = "opennlp.tools.ml.maxent.quasinewton.QNTrainer";

/// The run's QN settings from the config records.
pub fn qn_params(g: &Golden) -> turbo_train::QnParams {
    turbo_train::QnParams {
        iterations: config(g, "Iterations").parse().unwrap(),
        threads: config(g, "Threads").parse().unwrap(),
        l1_cost: turbo_jvm::text::parse_double(config(g, "L1Cost")).unwrap(),
        l2_cost: turbo_jvm::text::parse_double(config(g, "L2Cost")).unwrap(),
        updates: config(g, "NumOfUpdates").parse().unwrap(),
        max_fct_eval: config(g, "MaxFctEval").parse().unwrap(),
    }
}

/// `PerceptronTrainer`'s class name: the annotator of the perceptron
/// training goldens.
pub const PERCEPTRON_TRAINER: &str = "opennlp.tools.ml.perceptron.PerceptronTrainer";

/// `SimplePerceptronSequenceTrainer`'s class name: the annotator of the
/// sequence perceptron goldens.
pub const SEQUENCE_PERCEPTRON_TRAINER: &str = "opennlp.tools.ml.perceptron.SimplePerceptronSequenceTrainer";

/// The run's perceptron settings from the config records, read as the
/// trainer reads them from its training parameters.
pub fn perceptron_params(g: &Golden) -> turbo_train::TrainingParameters {
    let mut p = turbo_train::TrainingParameters::new();
    p.put(turbo_train::Key::ITERATIONS, config(g, "Iterations").parse::<i32>().unwrap());
    p.put(turbo_train::Key::USE_AVERAGE, config(g, "UseAverage").parse::<bool>().unwrap());
    if g.header.annotator.class == PERCEPTRON_TRAINER {
        p.put(turbo_train::Key::USE_SKIPPED_AVERAGING, config(g, "UseSkippedAveraging").parse::<bool>().unwrap());
        p.put(turbo_train::Key::STEP_SIZE_DECREASE, config(g, "StepSizeDecrease").to_string());
        p.put(turbo_train::Key::TOLERANCE, config(g, "Tolerance").to_string());
    }
    p
}

pub fn train_shapes() -> Vec<Shape> {
    TRAIN_LAYERS
        .iter()
        .map(|(id, columns)| Shape { id: id.to_string(), anchor: GoldenAnchor::Document, columns })
        .collect()
}

pub fn config<'g>(g: &'g Golden, key: &str) -> &'g str {
    config_opt(g, key).unwrap_or_else(|| panic!("no config {key}"))
}

/// A config record the run may not have.
pub fn config_opt<'g>(g: &'g Golden, key: &str) -> Option<&'g str> {
    g.header.config.iter().find(|c| c.key == key).map(|c| c.value.as_str())
}

pub fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layers.iter().find(|l| l.id == id).unwrap_or_else(|| panic!("no layer {id}"))
}

/// The run's indexer settings from the config records.
pub fn params(g: &Golden) -> IndexParams {
    let kind = IndexerKind::parse(config(g, "DataIndexer")).unwrap();
    let cutoff: i32 = config(g, "Cutoff").parse().unwrap();
    let sort: bool = config(g, "sort").parse().unwrap();
    IndexParams::new(kind, cutoff, sort)
}

/// The events of the golden: the outcome, the predicates read back
/// through [`java_string`], and the values read with `Float.parseFloat`
/// from their `Float.toString` form, which round trips.
pub fn events(g: &Golden) -> Vec<Event> {
    let l = layer(g, "turbo:events");
    let (outcomes, contexts, values) =
        (l.strs("outcome").unwrap(), l.strs("context").unwrap(), l.strs("values").unwrap());
    (0..l.len())
        .map(|i| {
            let (outcome, context, values) = (outcomes.get(i), contexts.get(i), values.get(i));
            let context: Vec<JavaString> =
                if context.is_empty() { Vec::new() } else { context.split(' ').map(java_string).collect() };
            let values: Option<Vec<f32>> =
                (!values.is_empty()).then(|| values.split(' ').map(|v| parse_float(v).unwrap()).collect());
            Event::with_values(outcome, context, values).unwrap()
        })
        .collect()
}
