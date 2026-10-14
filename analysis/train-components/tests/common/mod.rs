//! Reading the training goldens' events (`analysis/goldens/data/train/`
//! and `train-events/`).

#![allow(dead_code)]

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_train::{Event, JavaString};

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

/// A `str` field written with `lossless`, read back as text; a lone
/// surrogate stays as its six characters.
pub fn lossless_text(field: &str) -> String {
    java_string(field).to_string_lossy()
}

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;

/// The layers of the `train` and `train-events` goldens. Only the events,
/// the checksum and the `train-events` layers are read here; the
/// indexer's and the trainer's layers are listed so that the reader
/// accepts the `train` files (`analysis/train/tests` compares those).
const LAYERS: &[(&str, &[ColumnSpec])] = &[
    ("turbo:events", &[col("outcome", STR), col("context", STR), col("values", STR)]),
    ("turbo:stream-errors", &[col("after", U32), col("class", STR), col("message", STR)]),
    ("turbo:event-checksum", &[col("crc32c", U32)]),
    ("turbo:indexer-error", &[col("class", STR), col("message", STR)]),
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
    ("turbo:manifest", &[col("key", STR), col("value", STR)]),
    ("turbo:package-entries", &[col("name", STR), col("length", U32)]),
    ("turbo:dictionary", &[col("xml", STR)]),
    ("turbo:dictionary-cases", &[col("name", STR), col("xml", STR)]),
    ("turbo:tagdict", &[col("xml", STR)]),
    ("turbo:tagdict-cases", &[col("name", STR), col("xml", STR)]),
];

pub fn shapes() -> Vec<Shape> {
    LAYERS.iter().map(|(id, columns)| Shape { id: id.to_string(), anchor: GoldenAnchor::Document, columns }).collect()
}

pub fn read(path: &std::path::Path) -> Golden {
    Golden::read_with(path, &shapes()).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn config<'g>(g: &'g Golden, key: &str) -> Option<&'g str> {
    g.header.config.iter().find(|c| c.key == key).map(|c| c.value.as_str())
}

pub fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layers.iter().find(|l| l.id == id).unwrap_or_else(|| panic!("no layer {id}"))
}

pub fn layer_opt<'g>(g: &'g Golden, id: &str) -> Option<&'g GoldenLayer> {
    g.layers.iter().find(|l| l.id == id)
}

/// The events of the golden: the outcome and the predicates read back
/// through [`java_string`]. The sources this crate ports give no values.
pub fn events(g: &Golden) -> Vec<Event> {
    let l = layer(g, "turbo:events");
    let (outcomes, contexts, values) =
        (l.strs("outcome").unwrap(), l.strs("context").unwrap(), l.strs("values").unwrap());
    (0..l.len())
        .map(|i| {
            assert!(values.get(i).is_empty(), "an event with values");
            let context = contexts.get(i);
            Event::new(outcomes.get(i), split_context(context))
        })
        .collect()
}

/// The predicates of a context field, which the generator joined with a
/// space. The only predicate that holds a space is the sentence
/// detector's `eos= ` when a space is an end character (Thai), so an
/// `eos=` field followed by an empty field is that predicate.
fn split_context(context: &str) -> Vec<JavaString> {
    if context.is_empty() {
        return Vec::new();
    }
    let fields: Vec<&str> = context.split(' ').collect();
    let mut out = Vec::with_capacity(fields.len());
    let mut i = 0;
    while i < fields.len() {
        if fields[i] == "eos=" && fields.get(i + 1) == Some(&"") {
            out.push(java_string("eos= "));
            i += 2;
        } else {
            out.push(java_string(fields[i]));
            i += 1;
        }
    }
    out
}

pub fn checksum(g: &Golden) -> u32 {
    layer(g, "turbo:event-checksum").u32s("crc32c").unwrap()[0]
}

/// Every golden file under `dir`, recursively, sorted.
pub fn golden_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display())) {
            let p = entry.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "golden") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// The differences between two event lists, for a message.
pub fn describe_difference(expected: &[Event], actual: &[Event]) -> String {
    if expected.len() != actual.len() {
        return format!("{} events expected, {} made", expected.len(), actual.len());
    }
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        if e != a {
            return format!(
                "event {i}: expected {} {:?}, made {} {:?}",
                e.outcome(),
                e.context().iter().map(JavaString::to_string_lossy).collect::<Vec<_>>(),
                a.outcome(),
                a.context().iter().map(JavaString::to_string_lossy).collect::<Vec<_>>()
            );
        }
    }
    String::from("no difference")
}
