//! The bytes of a trained model: `GenericModelWriter` over a `GISModel`, a
//! `QNModel` or a `PerceptronModel`, which is `BinaryGISModelWriter` with
//! `GISModelWriter.persist`, `BinaryQNModelWriter` with
//! `QNModelWriter.persist` or `BinaryPerceptronModelWriter` with
//! `PerceptronModelWriter.persist`, writing through `DataOutputStream` and
//! `ModelParameterChunker`.
//!
//! The layout, read back by `turbo-maxent` and by OpenNLP's
//! `GISModelReader`, `QNModelReader` and `PerceptronModelReader`:
//!
//! 1. `writeUTF("GIS")`, `writeInt(1)` and `writeDouble(1)`: the model type
//!    and the correction constant and parameter, which nothing reads; or
//!    `writeUTF("QN")` or `writeUTF("Perceptron")` alone.
//! 2. The outcome labels: their count, then each with `writeUTF`.
//! 3. The outcome patterns: the predicates are sorted by their pattern
//!    (the outcome ids a predicate has parameters for, compared
//!    lexicographically, then by length; a stable sort, so predicates with
//!    one pattern keep their index order) and consecutive equal patterns
//!    are grouped. The count of groups, then each group as the text
//!    `"<size> <o1> <o2> ..."`.
//! 4. The predicate labels in the sorted order, their count first.
//! 5. The parameters: for every predicate in the sorted order, one
//!    `writeDouble` per entry of its pattern.
//!
//! The perceptron writer first drops every parameter that is zero from
//! its predicate's pattern, and every predicate left with none; a model
//! with no parameter left cannot be written ([`Error::NoParameters`]).
//!
//! Every string goes through `ModelParameterChunker.writeUTF`: a string of
//! at most 65535 bytes of modified UTF-8 is written as `writeUTF` writes
//! it; a longer one is written as `"CHUNKED-MODEL-PARAMS:<n>"` and then
//! `n` chunks, each a `writeUTF` of a run of the string that
//! `CharsetEncoder` for UTF-8 fit into 65535 bytes. Two consequences of
//! measuring the chunks in standard UTF-8 are kept, as they are what the
//! JVM writes: a chunk full of characters outside the BMP can exceed
//! 65535 bytes of modified UTF-8, which `writeUTF` then refuses
//! ([`Error::StringTooLong`] here, `UTFDataFormatException` there), and
//! the encoder stops at a lone surrogate, so a chunked string is cut
//! there and the rest of it is lost. U+0000 counts one byte in a chunk
//! and takes two when written, so a chunk of them can overflow as well.

use crate::error::{Error, Result};
use crate::gis::GisModel;
use crate::java_string::JavaString;
use crate::naive_bayes::NaiveBayesModel;
use crate::perceptron::PerceptronModel;
use crate::qn::QnModel;

/// The type string `GISModelWriter.persist` starts with.
pub const GIS_TYPE: &str = "GIS";
/// The type string `QNModelWriter.persist` starts with.
pub const QN_TYPE: &str = "QN";
/// The type string `PerceptronModelWriter.persist` starts with.
pub const PERCEPTRON_TYPE: &str = "Perceptron";
/// The type string `NaiveBayesModelWriter.persist` starts with.
pub const NAIVE_BAYES_TYPE: &str = "NaiveBayes";
/// `ModelParameterChunker.SIGNATURE_CHUNKED_PARAMS`.
pub const CHUNKED_SIGNATURE: &str = "CHUNKED-MODEL-PARAMS:";
/// `writeUTF`'s limit on a string's modified UTF-8 bytes.
pub const MAX_UTF_BYTES: usize = 65535;

/// What a model writer reads from a model: `AbstractModel.getDataStructures`
/// as `GISModelWriter`'s constructor takes it apart, the predicates in
/// `pmap` order with each one's outcome pattern and parameters.
pub trait WritableModel {
    fn outcome_labels(&self) -> &[JavaString];
    fn pred_labels(&self) -> &[JavaString];
    /// The outcome ids predicate `p` has parameters for.
    fn pattern(&self, p: usize) -> &[u32];
    /// Predicate `p`'s parameters, one per entry of its pattern.
    fn parameters(&self, p: usize) -> &[f64];

    fn num_predicates(&self) -> usize {
        self.pred_labels().len()
    }
}

impl WritableModel for GisModel {
    fn outcome_labels(&self) -> &[JavaString] {
        GisModel::outcome_labels(self)
    }

    fn pred_labels(&self) -> &[JavaString] {
        GisModel::pred_labels(self)
    }

    fn pattern(&self, p: usize) -> &[u32] {
        GisModel::pattern(self, p)
    }

    fn parameters(&self, p: usize) -> &[f64] {
        GisModel::parameters(self, p)
    }
}

impl WritableModel for QnModel {
    fn outcome_labels(&self) -> &[JavaString] {
        QnModel::outcome_labels(self)
    }

    fn pred_labels(&self) -> &[JavaString] {
        QnModel::pred_labels(self)
    }

    fn pattern(&self, p: usize) -> &[u32] {
        QnModel::pattern(self, p)
    }

    fn parameters(&self, p: usize) -> &[f64] {
        QnModel::parameters(self, p)
    }
}

/// The model as the trainer leaves it, zero parameters included. Writing
/// the model drops them ([`write_perceptron`]).
impl WritableModel for PerceptronModel {
    fn outcome_labels(&self) -> &[JavaString] {
        PerceptronModel::outcome_labels(self)
    }

    fn pred_labels(&self) -> &[JavaString] {
        PerceptronModel::pred_labels(self)
    }

    fn pattern(&self, p: usize) -> &[u32] {
        PerceptronModel::pattern(self, p)
    }

    fn parameters(&self, p: usize) -> &[f64] {
        PerceptronModel::parameters(self, p)
    }
}

impl WritableModel for NaiveBayesModel {
    fn outcome_labels(&self) -> &[JavaString] {
        NaiveBayesModel::outcome_labels(self)
    }

    fn pred_labels(&self) -> &[JavaString] {
        NaiveBayesModel::pred_labels(self)
    }

    fn pattern(&self, p: usize) -> &[u32] {
        NaiveBayesModel::pattern(self, p)
    }

    fn parameters(&self, p: usize) -> &[f64] {
        NaiveBayesModel::parameters(self, p)
    }
}

/// `GenericModelWriter(model, DataOutputStream).persist()` for a GIS
/// model: the bytes of the `.model` entry.
pub fn write_gis(model: &GisModel) -> Result<Vec<u8>> {
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&JavaString::new(GIS_TYPE))?;
    out.write_int(1);
    out.write_double(1.0);
    write_body(&mut out, model)?;
    Ok(out.bytes)
}

/// `GenericModelWriter(model, DataOutputStream).persist()` for a QN
/// model: the bytes of the `.model` entry.
pub fn write_qn(model: &QnModel) -> Result<Vec<u8>> {
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&JavaString::new(QN_TYPE))?;
    write_body(&mut out, model)?;
    Ok(out.bytes)
}

/// `GenericModelWriter(model, DataOutputStream).persist()` for a
/// perceptron model: the bytes of the `.model` entry, without the zero
/// parameters (`PerceptronModelWriter.sortValues`).
pub fn write_perceptron(model: &PerceptronModel) -> Result<Vec<u8>> {
    let pruned = Pruned::of(model);
    if pruned.pred_labels.is_empty() {
        return Err(Error::NoParameters);
    }
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&JavaString::new(PERCEPTRON_TYPE))?;
    write_body(&mut out, &pruned)?;
    Ok(out.bytes)
}

/// `GenericModelWriter(model, DataOutputStream).persist()` for a naive
/// Bayes model: the bytes of the `.model` entry (`NaiveBayesModelWriter.persist`).
pub fn write_naive_bayes(model: &NaiveBayesModel) -> Result<Vec<u8>> {
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&JavaString::new(NAIVE_BAYES_TYPE))?;
    write_body(&mut out, model)?;
    Ok(out.bytes)
}

/// A perceptron model as its writer sees it: each predicate's pattern and
/// parameters without the zero ones, and only the predicates that keep a
/// parameter, in model order.
struct Pruned<'m> {
    outcome_labels: &'m [JavaString],
    pred_labels: Vec<JavaString>,
    /// `starts[p]..starts[p + 1]` of `outcomes` and `parameters` is
    /// predicate `p`'s.
    starts: Vec<usize>,
    outcomes: Vec<u32>,
    parameters: Vec<f64>,
}

impl<'m> Pruned<'m> {
    fn of(model: &'m PerceptronModel) -> Pruned<'m> {
        let mut pruned = Pruned {
            outcome_labels: model.outcome_labels(),
            pred_labels: Vec::new(),
            starts: vec![0],
            outcomes: Vec::new(),
            parameters: Vec::new(),
        };
        for (p, label) in model.pred_labels().iter().enumerate() {
            let mut kept = 0;
            for (&o, &x) in model.pattern(p).iter().zip(model.parameters(p)) {
                if x != 0.0 {
                    pruned.outcomes.push(o);
                    pruned.parameters.push(x);
                    kept += 1;
                }
            }
            if kept > 0 {
                pruned.pred_labels.push(label.clone());
                pruned.starts.push(pruned.outcomes.len());
            }
        }
        pruned
    }
}

impl WritableModel for Pruned<'_> {
    fn outcome_labels(&self) -> &[JavaString] {
        self.outcome_labels
    }

    fn pred_labels(&self) -> &[JavaString] {
        &self.pred_labels
    }

    fn pattern(&self, p: usize) -> &[u32] {
        &self.outcomes[self.starts[p]..self.starts[p + 1]]
    }

    fn parameters(&self, p: usize) -> &[f64] {
        &self.parameters[self.starts[p]..self.starts[p + 1]]
    }
}

/// What follows the type: the outcomes, the patterns, the predicates and
/// the parameters, as `AbstractMLModelWriter.persist` writes them.
fn write_body(out: &mut ModelOutput, model: &dyn WritableModel) -> Result<()> {
    out.write_int(to_java_int(model.outcome_labels().len()));
    for label in model.outcome_labels() {
        out.write_utf(label)?;
    }
    let sorted = sort_by_pattern(model);
    let groups = pattern_groups(model, &sorted);
    out.write_int(to_java_int(groups.len()));
    for (size, first) in &groups {
        let mut text = size.to_string();
        for oid in model.pattern(*first) {
            text.push(' ');
            text.push_str(&oid.to_string());
        }
        out.write_utf(&JavaString::new(&text))?;
    }
    out.write_int(to_java_int(model.num_predicates()));
    for &p in &sorted {
        out.write_utf(&model.pred_labels()[p])?;
    }
    for &p in &sorted {
        for &param in model.parameters(p) {
            out.write_double(param);
        }
    }
    Ok(())
}

/// `GISModelWriter.sortValues`: the predicate indices sorted by
/// `ComparablePredicate.compareTo`, a stable sort as `Arrays.sort` on
/// objects is.
fn sort_by_pattern(model: &dyn WritableModel) -> Vec<usize> {
    let mut sorted: Vec<usize> = (0..model.num_predicates()).collect();
    sorted.sort_by(|&a, &b| model.pattern(a).cmp(model.pattern(b)));
    sorted
}

/// `compressOutcomes`: runs of predicates with equal patterns, as the
/// run's size and its first predicate.
fn pattern_groups(model: &dyn WritableModel, sorted: &[usize]) -> Vec<(usize, usize)> {
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for &p in sorted {
        match groups.last_mut() {
            Some((size, first)) if model.pattern(*first) == model.pattern(p) => *size += 1,
            _ => groups.push((1, p)),
        }
    }
    groups
}

fn to_java_int(n: usize) -> i32 {
    i32::try_from(n).expect("a model has fewer than 2^31 labels")
}

/// A `DataOutputStream` into memory.
struct ModelOutput {
    bytes: Vec<u8>,
}

impl ModelOutput {
    fn write_int(&mut self, i: i32) {
        self.bytes.extend_from_slice(&i.to_be_bytes());
    }

    fn write_double(&mut self, d: f64) {
        self.bytes.extend_from_slice(&d.to_bits().to_be_bytes());
    }

    /// `ModelParameterChunker.writeUTF`.
    fn write_utf(&mut self, s: &JavaString) -> Result<()> {
        if s.len() <= MAX_UTF_BYTES {
            self.write_utf_plain(s.as_bytes());
            return Ok(());
        }
        let chunks = split_by_utf8_length(s);
        self.write_utf_plain(format!("{CHUNKED_SIGNATURE}{}", chunks.len()).as_bytes());
        for chunk in chunks {
            if chunk.len() > MAX_UTF_BYTES {
                return Err(Error::StringTooLong { what: "chunk of a model parameter", bytes: chunk.len() });
            }
            self.write_utf_plain(chunk);
        }
        Ok(())
    }

    /// `DataOutputStream.writeUTF` of bytes known to fit.
    fn write_utf_plain(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        self.bytes.extend_from_slice(bytes);
    }
}

/// `ModelParameterChunker.splitByByteLength`: the string cut into runs
/// whose standard UTF-8 encoding fits 65535 bytes, each as the run's
/// modified UTF-8 bytes. The UTF-8 encoder stops at a lone surrogate, so
/// the run before it is the last chunk and the rest is dropped.
fn split_by_utf8_length(s: &JavaString) -> Vec<&[u8]> {
    let bytes = s.as_bytes();
    let mut chunks = Vec::new();
    let mut chunk_start = 0;
    let mut chunk_utf8 = 0;
    let mut pos = 0;
    while pos < bytes.len() {
        let (modified_len, utf8_len) = match next_char(bytes, pos) {
            Some(lens) => lens,
            None => {
                // A lone surrogate: malformed input ends the encoding.
                chunks.push(&bytes[chunk_start..pos]);
                return chunks;
            }
        };
        if chunk_utf8 + utf8_len > MAX_UTF_BYTES {
            chunks.push(&bytes[chunk_start..pos]);
            chunk_start = pos;
            chunk_utf8 = 0;
        }
        chunk_utf8 += utf8_len;
        pos += modified_len;
    }
    chunks.push(&bytes[chunk_start..]);
    chunks
}

/// The character at `pos` of canonical modified UTF-8: its length there
/// and in standard UTF-8, or `None` for a lone surrogate or truncated
/// multi-byte sequence.
fn next_char(bytes: &[u8], pos: usize) -> Option<(usize, usize)> {
    match *bytes.get(pos)? {
        0x00..=0x7F => return Some((1, 1)),
        // U+0000, which canonical modified UTF-8 writes as C0 80 and
        // standard UTF-8 as one zero byte.
        0xC0 => return (pos + 2 <= bytes.len()).then_some((2, 1)),
        0xC1..=0xDF => return (pos + 2 <= bytes.len()).then_some((2, 2)),
        _ => {}
    }
    if pos + 3 > bytes.len() {
        return None;
    }
    let unit = (u32::from(bytes[pos] & 0x0F) << 12)
        | (u32::from(bytes[pos + 1] & 0x3F) << 6)
        | u32::from(bytes[pos + 2] & 0x3F);
    match unit {
        0xD800..=0xDBFF => {
            let next = pos + 3;
            if next + 3 <= bytes.len() && bytes[next] == 0xED && (0xB0..=0xBF).contains(&bytes[next + 1]) {
                Some((6, 4))
            } else {
                None
            }
        }
        0xDC00..=0xDFFF => None,
        _ => Some((3, 3)),
    }
}

#[cfg(test)]
mod tests;
