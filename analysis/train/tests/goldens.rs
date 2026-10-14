//! The indexers against the JVM: `analysis/goldens/data/train/` holds, for
//! events the JVM's sentence detector, tokenizer and a real valued source
//! made from the input texts, what `OnePassDataIndexer`,
//! `TwoPassDataIndexer` and `OnePassRealValueDataIndexer` produced with
//! each run's cutoff and sorting (`analysis/goldens-tool`, `TrainGolden`).
//! This test reads the events back through the event stream parsers,
//! indexes them, and compares every label, count, context and seen count,
//! and the event checksum, with the golden's.

use std::path::{Path, PathBuf};

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape, data_dir};
use turbo_jvm::text::{float_to_string, parse_float};
use turbo_train::error::NO_EVENTS;
use turbo_train::event::{checksum, parse_event, parse_real_value_event};
use turbo_train::{DataIndex, Error, Event, IndexParams, IndexerKind, JavaString};

/// A predicate as `TrainGolden.lossless` wrote it: a doubled backslash is
/// one, `\uXXXX` is a lone surrogate, everything else is itself.
fn java_string(field: &str) -> JavaString {
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
];

pub fn train_shapes() -> Vec<Shape> {
    TRAIN_LAYERS
        .iter()
        .map(|(id, columns)| Shape { id: id.to_string(), anchor: GoldenAnchor::Document, columns })
        .collect()
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

fn config<'g>(g: &'g Golden, key: &str) -> &'g str {
    &g.header.config.iter().find(|c| c.key == key).unwrap_or_else(|| panic!("no config {key}")).value
}

fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layers.iter().find(|l| l.id == id).unwrap_or_else(|| panic!("no layer {id}"))
}

/// The run's indexer settings from the config records.
fn params(g: &Golden) -> IndexParams {
    let kind = IndexerKind::parse(config(g, "DataIndexer")).unwrap();
    let cutoff: i32 = config(g, "Cutoff").parse().unwrap();
    let sort: bool = config(g, "sort").parse().unwrap();
    IndexParams::new(kind, cutoff, sort)
}

/// The events of the golden: the outcome, the predicates read back
/// through [`java_string`], and the values read with `Float.parseFloat`
/// from their `Float.toString` form, which round trips.
fn events(g: &Golden) -> Vec<Event> {
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

fn joined<T: ToString>(items: impl Iterator<Item = T>) -> String {
    items.map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

/// What the run has so the test can tell the goldens cover every shape.
#[derive(Default, Debug)]
struct Coverage {
    goldens: usize,
    errors: usize,
    one_outcome: usize,
    unsorted: usize,
    merged: usize,
    with_values: usize,
    dropped_events: usize,
    lone_surrogates: usize,
    kinds: [usize; 3],
}

fn check(path: &Path, g: &Golden, cov: &mut Coverage) {
    let at = path.display();
    let params = params(g);
    cov.kinds[match params.kind {
        IndexerKind::OnePass => 0,
        IndexerKind::TwoPass => 1,
        IndexerKind::OnePassRealValue => 2,
    }] += 1;
    if !params.sort {
        cov.unsorted += 1;
    }
    let events = events(g);
    // An event without a lone surrogate round trips through its line
    // format; one with a lone surrogate cannot be written as text.
    for e in &events {
        if e.context().iter().any(|c| c.to_str().is_none()) {
            cov.lone_surrogates += 1;
            continue;
        }
        let line = e.real_value_line();
        let back = if e.values().is_some() { parse_real_value_event(&line, 1) } else { parse_event(&line, 1) };
        assert_eq!(back.unwrap(), *e, "{at}");
    }
    let index = DataIndex::index(&events, &params);
    assert!(g.header.error.is_none(), "{at}");
    if let Some(error) = g.layers.iter().find(|l| l.id == "turbo:indexer-error") {
        assert_eq!(error.strs("class").unwrap().get(0), "opennlp.tools.util.InsufficientTrainingDataException", "{at}");
        assert_eq!(error.strs("message").unwrap().get(0), NO_EVENTS, "{at}");
        assert_eq!(index, Err(Error::InsufficientTrainingData(NO_EVENTS)), "{at}");
        cov.errors += 1;
        return;
    }
    let index = index.unwrap_or_else(|e| panic!("{at}: {e}"));

    let crc = layer(g, "turbo:event-checksum").u32s("crc32c").unwrap();
    assert_eq!(checksum(&events), crc[0], "{at}: event checksum");

    let outcomes = layer(g, "turbo:indexer-outcomes").strs("label").unwrap();
    assert_eq!(index.outcome_labels(), outcomes.iter().collect::<Vec<_>>(), "{at}: outcome labels");
    let preds = layer(g, "turbo:indexer-predicates");
    let labels: Vec<JavaString> = preds.strs("label").unwrap().iter().map(java_string).collect();
    assert_eq!(index.pred_labels(), labels, "{at}: predicate labels");
    assert_eq!(index.pred_counts(), preds.u32s("count").unwrap(), "{at}: predicate counts");

    let indexed = layer(g, "turbo:indexer-events");
    assert_eq!(index.num_unique_events(), indexed.len(), "{at}: unique events");
    let (outcome, context, values, seen) = (
        indexed.u32s("outcome").unwrap(),
        indexed.strs("context").unwrap(),
        indexed.strs("values").unwrap(),
        indexed.u32s("seen").unwrap(),
    );
    for i in 0..indexed.len() {
        assert_eq!(index.outcome(i), outcome[i], "{at}: event {i} outcome");
        assert_eq!(joined(index.context(i).iter()), context.get(i), "{at}: event {i} context");
        let v = index.values(i).map_or(String::new(), |v| joined(v.iter().map(|x| float_to_string(*x))));
        assert_eq!(v, values.get(i), "{at}: event {i} values");
        assert_eq!(index.seen(i), seen[i], "{at}: event {i} seen");
        if seen[i] > 1 {
            cov.merged += 1;
        }
        if !values.get(i).is_empty() {
            cov.with_values += 1;
        }
    }
    assert_eq!(index.has_values(), params.kind.keeps_values(), "{at}");
    if (index.num_events() as usize) < events.len() {
        cov.dropped_events += 1;
    }

    // The trainer refuses an index with one outcome before it starts.
    let training = layer(g, "turbo:gis-training");
    let iterations = training.u32s("iterations").unwrap()[0];
    if iterations == 0 {
        assert!(index.outcome_labels().len() <= 1, "{at}");
        assert_eq!(
            training.strs("message").unwrap().get(0),
            "opennlp.tools.util.InsufficientTrainingDataException: Training data must contain more than one outcome",
            "{at}"
        );
        cov.one_outcome += 1;
    } else {
        assert!(index.outcome_labels().len() > 1, "{at}");
        assert_eq!(layer(g, "turbo:gis-iterations").len(), iterations as usize, "{at}");
    }
}

#[test]
fn indexers_match_the_jvm_on_every_training_golden() {
    let shapes = train_shapes();
    let mut paths = Vec::new();
    golden_files(&data_dir().join("train"), &mut paths);
    paths.sort();
    let mut cov = Coverage::default();
    for path in &paths {
        let g = Golden::read_with(path, &shapes).unwrap_or_else(|e| panic!("{e}"));
        g.input_text(data_dir()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        check(path, &g, &mut cov);
        cov.goldens += 1;
    }
    eprintln!("{cov:?}");
    assert!(cov.goldens >= 100, "{cov:?}");
    assert!(cov.errors > 0 && cov.one_outcome > 0 && cov.unsorted > 0, "{cov:?}");
    assert!(cov.merged > 0 && cov.with_values > 0 && cov.dropped_events > 0, "{cov:?}");
    assert!(cov.lone_surrogates > 0, "{cov:?}");
    assert!(cov.kinds.iter().all(|&n| n > 0), "{cov:?}");
}

/// The run that trained the fixture model en-plain.bin: TwoPass, cutoff 1,
/// on the sentences it was trained from. Its event checksum is the
/// `Training-Eventhash` in the model's manifest.
#[test]
fn the_sentence_fixture_run_has_the_fixture_models_event_hash() {
    let path = data_dir().join("train/sentdetect-twopass-cutoff1/train/sentences.golden");
    let g = Golden::read_with(&path, &train_shapes()).unwrap_or_else(|e| panic!("{e}"));
    let events = events(&g);
    let crc = layer(&g, "turbo:event-checksum").u32s("crc32c").unwrap()[0];
    assert_eq!(checksum(&events), crc);
    let model = turbo_maxent::ModelFile::read(data_dir().join("fixtures/en-plain.bin")).unwrap();
    assert_eq!(model.manifest().get("Training-Eventhash"), Some(crc.to_string().as_str()));
    let index = DataIndex::index(&events, &params(&g)).unwrap();
    let maxent = model.maxent("sent.model").unwrap();
    assert_eq!(index.outcome_labels(), maxent.outcomes().collect::<Vec<_>>());
    assert_eq!(index.pred_labels().len(), maxent.num_predicates());
}
