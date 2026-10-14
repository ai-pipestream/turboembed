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
];

pub fn train_shapes() -> Vec<Shape> {
    TRAIN_LAYERS
        .iter()
        .map(|(id, columns)| Shape { id: id.to_string(), anchor: GoldenAnchor::Document, columns })
        .collect()
}

pub fn config<'g>(g: &'g Golden, key: &str) -> &'g str {
    &g.header.config.iter().find(|c| c.key == key).unwrap_or_else(|| panic!("no config {key}")).value
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
