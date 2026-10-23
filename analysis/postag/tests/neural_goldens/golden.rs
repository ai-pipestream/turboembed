//! The goldens' shapes and columns, and the counts of compared values.

use std::path::{Path, PathBuf};

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape, known_shapes, sha256};
use turbo_postag::neural::PosSample;

/// The largest difference allowed between a BiLSTM value computed in Rust
/// from the JVM's model and the JVM's value.
pub(crate) const BILSTM_TOLERANCE: f64 = 1e-13;

/// The largest difference allowed between an emission of a BiLSTM model
/// trained in Rust and one of the model the JVM trained.
pub(crate) const BILSTM_TRAINED_TOLERANCE: f64 = 1e-12;

/// The command that writes the goldens.
pub(crate) const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

pub(crate) const TOKENS: &[ColumnSpec] =
    &[ColumnSpec::new("sentence", ColumnType::U32), ColumnSpec::new("token", ColumnType::Str)];
pub(crate) const MODEL: &[ColumnSpec] = &[
    ColumnSpec::new("magic", ColumnType::Str),
    ColumnSpec::new("length", ColumnType::U32),
    ColumnSpec::new("sha256", ColumnType::Str),
];
pub(crate) const MODEL_BYTES: &[ColumnSpec] =
    &[ColumnSpec::new("offset", ColumnType::U32), ColumnSpec::new("hex", ColumnType::Str)];
pub(crate) const TAGS: &[ColumnSpec] = &[ColumnSpec::new("tag", ColumnType::Str)];
pub(crate) const SCORES: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("tag", ColumnType::U32),
    ColumnSpec::new("score", ColumnType::F64),
];
pub(crate) const VITERBI: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("tag", ColumnType::U32),
];
pub(crate) const MARGINALS: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("tag", ColumnType::U32),
    ColumnSpec::new("probability", ColumnType::F64),
];
pub(crate) const CONTEXT: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("slot", ColumnType::U32),
    ColumnSpec::new("present", ColumnType::U32),
    ColumnSpec::new("symbol", ColumnType::Str),
    ColumnSpec::new("row", ColumnType::U32),
];
pub(crate) const VECTOR_ROWS: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("slot", ColumnType::U32),
    ColumnSpec::new("present", ColumnType::U32),
    ColumnSpec::new("row", ColumnType::U32),
];
pub(crate) const TAGGED: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("tag", ColumnType::Str),
    ColumnSpec::new("probability", ColumnType::F64),
];
pub(crate) const SEQUENCES: &[ColumnSpec] =
    &[ColumnSpec::new("sentence", ColumnType::U32), ColumnSpec::new("score", ColumnType::F64)];

pub(crate) fn shapes() -> Vec<Shape> {
    let mut shapes = known_shapes();
    for (id, columns) in [
        ("turbo:sentence-tokens", TOKENS),
        ("turbo:neural-model", MODEL),
        ("turbo:neural-model-bytes", MODEL_BYTES),
        ("turbo:neural-tags", TAGS),
        ("turbo:neural-emissions", SCORES),
        ("turbo:crf-viterbi", VITERBI),
        ("turbo:crf-marginals", MARGINALS),
        ("turbo:neural-context", CONTEXT),
        ("turbo:neural-vector-rows", VECTOR_ROWS),
        ("turbo:neural-tagged", TAGGED),
        ("turbo:neural-sequences", SEQUENCES),
    ] {
        shapes.push(Shape { id: id.into(), anchor: GoldenAnchor::Document, columns });
    }
    shapes
}

pub(crate) fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("no layer {id}"))
}

pub(crate) fn u32s<'g>(l: &'g GoldenLayer, name: &str) -> &'g [u32] {
    l.u32s(name).unwrap_or_else(|| panic!("{}: no u32 column {name}", l.id))
}

pub(crate) fn f64s<'g>(l: &'g GoldenLayer, name: &str) -> &'g [f64] {
    l.f64s(name).unwrap_or_else(|| panic!("{}: no f64 column {name}", l.id))
}

pub(crate) fn strs(l: &GoldenLayer, name: &str) -> Vec<String> {
    l.strs(name).unwrap_or_else(|| panic!("{}: no str column {name}", l.id)).iter().map(str::to_owned).collect()
}

/// The directories in `dir`, sorted.
pub(crate) fn sorted(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

/// The sentences of `turbo:sentence-tokens`, in order.
pub(crate) fn sentences(g: &Golden) -> Vec<Vec<String>> {
    let l = layer(g, "turbo:sentence-tokens");
    let mut out: Vec<Vec<String>> = Vec::new();
    for (&s, token) in u32s(l, "sentence").iter().zip(strs(l, "token")) {
        if s as usize == out.len() {
            out.push(Vec::new());
        }
        assert_eq!(s as usize + 1, out.len(), "sentence numbers are consecutive");
        out.last_mut().unwrap().push(token);
    }
    out
}

/// A `[sentence][index][tag]` table of a layer with those three columns
/// and one `f64` column, checking that the rows come in that order and
/// cover every token and tag.
pub(crate) fn table(g: &Golden, id: &str, value: &str, sentences: &[Vec<String>], tags: usize) -> Vec<Vec<Vec<f64>>> {
    let l = layer(g, id);
    let (s, i, t, v) = (u32s(l, "sentence"), u32s(l, "index"), u32s(l, "tag"), f64s(l, value));
    let mut out = Vec::new();
    let mut row = 0;
    for (k, sentence) in sentences.iter().enumerate() {
        let mut rows = Vec::new();
        for index in 0..sentence.len() {
            let mut scores = Vec::new();
            for tag in 0..tags {
                assert_eq!((s[row], i[row], t[row]), (k as u32, index as u32, tag as u32), "{id}: row {row}");
                scores.push(v[row]);
                row += 1;
            }
            rows.push(scores);
        }
        out.push(rows);
    }
    assert_eq!(row, l.len(), "{id}: rows past the last token");
    out
}

/// The SHA-256 of `bytes` in lowercase hex.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    turbo_goldens::hex::to_hex(&sha256(bytes))
}

/// The model file of the training golden, checked against its recorded
/// magic, length and SHA-256.
pub(crate) fn model_bytes(g: &Golden) -> Vec<u8> {
    let rows = layer(g, "turbo:neural-model-bytes");
    let mut bytes = Vec::new();
    for (&offset, hex) in u32s(rows, "offset").iter().zip(strs(rows, "hex")) {
        assert_eq!(offset as usize, bytes.len(), "model bytes offset");
        assert!(hex.len().is_multiple_of(2), "model bytes at {offset}");
        for pair in hex.as_bytes().chunks(2) {
            bytes.push(u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).expect("hex"));
        }
    }
    let model = layer(g, "turbo:neural-model");
    let magic_length = usize::from(u16::from_be_bytes([bytes[0], bytes[1]]));
    assert_eq!(std::str::from_utf8(&bytes[2..2 + magic_length]).unwrap(), strs(model, "magic")[0]);
    assert_eq!(bytes.len(), u32s(model, "length")[0] as usize);
    assert_eq!(sha256_hex(&bytes), strs(model, "sha256")[0]);
    bytes
}

/// The training corpus as `WordTagSampleStream` reads it: one sample per
/// line, tokens split at whitespace and each at its last `_`, and a line
/// with a token without `_` read as a sample with no tokens.
pub(crate) fn corpus(g: &Golden) -> Vec<PosSample> {
    let text = g.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
    text.lines()
        .map(|line| {
            let pairs: Option<Vec<(&str, &str)>> = line.split_whitespace().map(|t| t.rsplit_once('_')).collect();
            let (words, tags): (Vec<&str>, Vec<&str>) = pairs.unwrap_or_default().into_iter().unzip();
            PosSample::new(words, tags).unwrap()
        })
        .collect()
}

/// The word vectors of the runs with vectors: the word's length in UTF-16
/// units times `0.1f`, and `0.5f`.
pub(crate) fn vectors(word: &str) -> Option<Vec<f32>> {
    Some(vec![word.encode_utf16().count() as f32 * 0.1f32, 0.5f32])
}

/// How many values were compared, how many were bit for bit, and the
/// largest difference.
#[derive(Default)]
pub(crate) struct Count {
    pub(crate) values: usize,
    pub(crate) exact: usize,
    pub(crate) max_diff: f64,
}

impl Count {
    /// Compares `rust` with `jvm`, within `tolerance` (0 for exact).
    pub(crate) fn compare(&mut self, rust: f64, jvm: f64, tolerance: f64, what: &dyn Fn() -> String) {
        let diff = (rust - jvm).abs();
        self.values += 1;
        self.exact += usize::from(rust.to_bits() == jvm.to_bits());
        self.max_diff = self.max_diff.max(diff);
        if tolerance == 0.0 {
            assert_eq!(rust.to_bits(), jvm.to_bits(), "{}: {rust} against {jvm}", what());
        } else {
            assert!(diff <= tolerance, "{}: {rust} against {jvm}", what());
        }
    }
}

impl std::ops::AddAssign for Count {
    fn add_assign(&mut self, other: Count) {
        self.values += other.values;
        self.exact += other.exact;
        self.max_diff = self.max_diff.max(other.max_diff);
    }
}

impl std::fmt::Display for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} of {} bit for bit, largest difference {:e}", self.exact, self.values, self.max_diff)
    }
}

/// Counts over every golden of every run.
#[derive(Default)]
pub(crate) struct Totals {
    pub(crate) runs: usize,
    pub(crate) goldens: usize,
    pub(crate) sentences: usize,
    pub(crate) tags: usize,
    pub(crate) contexts: usize,
    pub(crate) ff_emissions: Count,
    pub(crate) bilstm_emissions: Count,
    pub(crate) marginals_of_jvm_emissions: Count,
    pub(crate) marginals: Count,
    pub(crate) probabilities: Count,
    pub(crate) sequence_scores: Count,
    pub(crate) packed_tags: usize,
    pub(crate) packed_emissions: Count,
    pub(crate) packed_probabilities: Count,
}

impl std::ops::AddAssign for Totals {
    fn add_assign(&mut self, other: Totals) {
        self.runs += other.runs;
        self.goldens += other.goldens;
        self.sentences += other.sentences;
        self.tags += other.tags;
        self.contexts += other.contexts;
        self.ff_emissions += other.ff_emissions;
        self.bilstm_emissions += other.bilstm_emissions;
        self.marginals_of_jvm_emissions += other.marginals_of_jvm_emissions;
        self.marginals += other.marginals;
        self.probabilities += other.probabilities;
        self.sequence_scores += other.sequence_scores;
        self.packed_tags += other.packed_tags;
        self.packed_emissions += other.packed_emissions;
        self.packed_probabilities += other.packed_probabilities;
    }
}
