//! Reading the sequence goldens under `analysis/goldens/data`, which the
//! generator in `analysis/goldens-tool` writes (see the crate README): their
//! layer shapes, the strings written as `TrainGolden.lossless` writes them,
//! and the fixtures the runs read.

#![allow(dead_code)]

use std::borrow::Cow;
use std::path::PathBuf;

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_sequence::featuregen::descriptor::Resources;

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;

const SPAN: &[ColumnSpec] =
    &[col("case", U32), col("start", U32), col("end", U32), col("type", STR), col("typed", U32)];
const CASE_OUTCOME: &[ColumnSpec] = &[col("case", U32), col("outcome", STR)];

/// The layers of the sequence goldens anchored on the document.
const DOCUMENT_LAYERS: &[(&str, &[ColumnSpec])] = &[
    (
        "turbo:string-patterns",
        &[
            col("token", STR),
            col("all-letter", U32),
            col("initial-capital", U32),
            col("all-capital", U32),
            col("all-lower-case", U32),
            col("all-digit", U32),
            col("all-hiragana", U32),
            col("all-katakana", U32),
            col("period", U32),
            col("comma", U32),
            col("slash", U32),
            col("digit", U32),
            col("hyphen", U32),
            col("letters", U32),
            col("digits", U32),
            col("class", STR),
        ],
    ),
    ("turbo:features", &[col("sentence", U32), col("token", U32), col("feature", STR)]),
    ("turbo:generator", &[col("class", STR)]),
    ("turbo:decode", &[col("case", U32), col("outcomes", U32)]),
    ("turbo:decode-outcomes", CASE_OUTCOME),
    ("turbo:decode-spans", SPAN),
    ("turbo:valid", &[col("case", U32), col("priors", U32), col("outcome", STR), col("valid", U32)]),
    ("turbo:valid-priors", CASE_OUTCOME),
    ("turbo:encode", &[col("case", U32), col("length", U32)]),
    ("turbo:encode-spans", SPAN),
    ("turbo:encode-outcomes", CASE_OUTCOME),
    ("turbo:compatible", &[col("case", U32), col("compatible", U32)]),
    ("turbo:compatible-outcomes", CASE_OUTCOME),
    ("turbo:beam-config", &[col("beam-size", U32), col("codec", STR)]),
    ("turbo:beam-calls", &[col("call", U32), col("sentence", U32), col("index", U32)]),
    ("turbo:beam-call-priors", &[col("call", U32), col("outcome", STR)]),
    ("turbo:beam-call-features", &[col("call", U32), col("feature", STR)]),
    ("turbo:beam-sequences", &[col("sequence", U32), col("sentence", U32), col("score", F64)]),
    ("turbo:beam-outcomes", &[col("sequence", U32), col("outcome", STR), col("probability", F64)]),
    (
        "turbo:beam-names",
        &[col("sentence", U32), col("start", U32), col("end", U32), col("type", STR), col("typed", U32)],
    ),
    ("turbo:tagdict", &[col("present", U32), col("case-sensitive", U32), col("words", U32), col("sha256", STR)]),
    ("turbo:tagdict-entries", &[col("entry", U32), col("word", STR), col("tags", U32)]),
    ("turbo:tagdict-entry-tags", &[col("entry", U32), col("tag", STR)]),
    ("turbo:tagdict-lookups", &[col("lookup", U32), col("word", STR), col("found", U32), col("tags", U32)]),
    ("turbo:tagdict-lookup-tags", &[col("lookup", U32), col("tag", STR)]),
    ("turbo:outcomes", &[col("outcome", STR)]),
    ("turbo:guessed-format", &[col("format", STR)]),
    ("turbo:converted-tags", &[col("tag", STR), col("converted", STR)]),
    ("turbo:requested-tags", &[col("format", STR), col("sentence", U32), col("raw", STR), col("tag", STR)]),
    ("turbo:guessed-formats", &[col("line", U32), col("format", STR)]),
];

/// The sentences of a feature golden, with the outcomes the generators
/// were given.
const SENTENCES_WITH_OUTCOMES: &[ColumnSpec] = &[col("sentence", U32), col("token", STR), col("outcome", STR)];

/// The sentences of a beam search golden.
const SENTENCES: &[ColumnSpec] = &[col("sentence", U32), col("token", STR)];

/// The tags the chunker runs on, aligned to the sentences' tokens.
const TAGS: &[ColumnSpec] = &[col("tag", STR)];

fn shapes(sentences: &'static [ColumnSpec]) -> Vec<Shape> {
    let mut shapes: Vec<Shape> = DOCUMENT_LAYERS
        .iter()
        .map(|&(id, columns)| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns })
        .collect();
    shapes.push(Shape { id: "turbo:sentence-tokens".into(), anchor: GoldenAnchor::Document, columns: sentences });
    shapes.push(Shape {
        id: "turbo:sentence-tags".into(),
        anchor: GoldenAnchor::Aligned("turbo:sentence-tokens".into()),
        columns: TAGS,
    });
    shapes
}

/// The golden `<annotator>/<run>/<set>/<id>.golden`.
pub fn read(annotator: &str, run: &str, set: &str, id: &str) -> Golden {
    let sentences = if annotator == "features" { SENTENCES_WITH_OUTCOMES } else { SENTENCES };
    let path = turbo_goldens::data_dir().join(annotator).join(run).join(set).join(format!("{id}.golden"));
    Golden::read_with(&path, &shapes(sentences)).unwrap_or_else(|e| panic!("{e}"))
}

/// The goldens of a run on an input set, by input id, in id order.
pub fn read_run(annotator: &str, run: &str, set: &str) -> Vec<(String, Golden)> {
    let dir = turbo_goldens::data_dir().join(annotator).join(run).join(set);
    let mut ids: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter_map(|name| name.strip_suffix(".golden").map(str::to_string))
        .collect();
    ids.sort();
    ids.into_iter().map(|id| (id.clone(), read(annotator, run, set, &id))).collect()
}

/// The layer `id` of a golden, which must have it.
pub fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("no layer {id}"))
}

/// The `u32` column `name` of a layer.
pub fn u32s<'l>(l: &'l GoldenLayer, name: &str) -> &'l [u32] {
    l.u32s(name).unwrap_or_else(|| panic!("{}: no u32 column {name}", l.id))
}

/// The `f64` column `name` of a layer.
pub fn f64s<'l>(l: &'l GoldenLayer, name: &str) -> &'l [f64] {
    l.f64s(name).unwrap_or_else(|| panic!("{}: no f64 column {name}", l.id))
}

/// The `str` column `name` of a layer, as owned strings.
pub fn strs(l: &GoldenLayer, name: &str) -> Vec<String> {
    l.strs(name).unwrap_or_else(|| panic!("{}: no str column {name}", l.id)).iter().map(str::to_string).collect()
}

/// The rows of a layer grouped by the `u32` column `key`, which numbers
/// `n` groups from 0: the row indices of each group, in row order.
pub fn grouped(l: &GoldenLayer, key: &str, n: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); n];
    for (row, &k) in u32s(l, key).iter().enumerate() {
        out[k as usize].push(row);
    }
    out
}

/// The UTF-16 units of a string `TrainGolden.lossless` wrote: a doubled
/// backslash is one, `\uXXXX` is one unit (half a surrogate pair where the
/// string held one), everything else is itself.
pub fn lossless_units(field: &str) -> Vec<u16> {
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
    units
}

/// A string `TrainGolden.lossless` wrote, which must be well formed UTF-16.
pub fn lossless(field: &str) -> String {
    String::from_utf16(&lossless_units(field)).unwrap_or_else(|_| panic!("lone surrogate in {field:?}"))
}

/// A string `TrainGolden.lossless` wrote, escaped as
/// [`turbo_sequence::Features::escaped`] writes a feature, so the two
/// compare as strings and print readably.
pub fn lossless_escaped(field: &str) -> String {
    let mut out = String::new();
    for u in lossless_units(field) {
        turbo_sequence::features::escape_unit(&mut out, u);
    }
    out
}

/// A string escaped as [`turbo_sequence::Features::escaped`] writes a
/// feature.
pub fn escape(s: &str) -> String {
    let mut out = String::new();
    for u in s.encode_utf16() {
        turbo_sequence::features::escape_unit(&mut out, u);
    }
    out
}

/// The checked in fixtures, `analysis/goldens/data/fixtures`.
pub fn fixtures() -> PathBuf {
    turbo_goldens::data_dir().join("fixtures")
}

/// The feature generator descriptors and their resources.
pub fn featuregen() -> PathBuf {
    fixtures().join("featuregen")
}

/// The resources a descriptor names, read from a directory.
pub struct DirResources(pub PathBuf);

impl Resources for DirResources {
    fn resource(&self, name: &str) -> turbo_sequence::Result<Option<Cow<'_, [u8]>>> {
        match std::fs::read(self.0.join(name)) {
            Ok(bytes) => Ok(Some(Cow::Owned(bytes))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => panic!("{name}: {e}"),
        }
    }
}

/// The escaped features of `out`, one string per feature.
pub fn escaped(out: &turbo_sequence::Features) -> Vec<String> {
    (0..out.len()).map(|i| out.escaped(i)).collect()
}

/// One sentence of a golden's `turbo:sentence-tokens`.
pub struct Sentence {
    pub tokens: Vec<String>,
    /// The outcomes the generators were given, where the layer has them.
    pub outcomes: Vec<String>,
}

/// The sentences of a golden's `turbo:sentence-tokens`, in order.
pub fn sentences(g: &Golden) -> Vec<Sentence> {
    let l = layer(g, "turbo:sentence-tokens");
    let tokens = strs(l, "token");
    let outcomes = l.strs("outcome").map(|s| s.iter().map(str::to_string).collect::<Vec<_>>());
    let mut out: Vec<Sentence> = Vec::new();
    for (row, &k) in u32s(l, "sentence").iter().enumerate() {
        while out.len() <= k as usize {
            out.push(Sentence { tokens: Vec::new(), outcomes: Vec::new() });
        }
        let s = &mut out[k as usize];
        s.tokens.push(tokens[row].clone());
        if let Some(o) = &outcomes {
            s.outcomes.push(o[row].clone());
        }
    }
    out
}
