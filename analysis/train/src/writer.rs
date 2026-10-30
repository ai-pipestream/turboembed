//! Native binary model writer for [`GisModel`], [`QnModel`],
//! [`PerceptronModel`], and [`NaiveBayesModel`].
//!
//! Layout read back by `turbo-maxent`'s [`MaxentModel::from_trained_bytes`]:
//!
//! 1. The model type header string (`"TURBO-GIS-1"`, `"TURBO-QN-1"`,
//!    `"TURBO-PERCEPTRON-1"`, or `"TURBO-NAIVE-BAYES-1"`).
//! 2. The outcome labels: their `i32` little-endian count, then each label.
//! 3. The outcome patterns: predicates are stably sorted by their outcome
//!    pattern (compared lexicographically, then by length) and consecutive
//!    equal patterns are grouped. The `i32` little-endian group count, then
//!    each group formatted as `"<size> <o1> <o2> ..."`.
//! 4. The predicate labels in the sorted order, their `i32` little-endian
//!    count first.
//! 5. The parameters: for every predicate in the sorted order, one
//!    little-endian `f64` per entry of its pattern.
//!
//! The perceptron writer drops every zero parameter from its predicate's
//! pattern, and every predicate left with none; a model with no parameter
//! left cannot be written ([`Error::NoParameters`]).
//!
//! All strings on the wire are length-prefixed (`u16` little-endian byte
//! length) standard UTF-8. In-memory [`JavaString`] labels are transcoded to
//! standard UTF-8 at the wire boundary; unpaired surrogates are refused with
//! [`Error::UnpairedSurrogate`] and strings exceeding [`MAX_UTF_BYTES`] bytes
//! of standard UTF-8 are refused with [`Error::StringTooLong`].

use crate::error::{Error, Result};
use crate::gis::GisModel;
use crate::java_string::JavaString;
use crate::naive_bayes::NaiveBayesModel;
use crate::perceptron::PerceptronModel;
use crate::qn::QnModel;

/// The type string `write_gis` starts with.
pub const GIS_TYPE: &str = "TURBO-GIS-1";
/// The type string `write_qn` starts with.
pub const QN_TYPE: &str = "TURBO-QN-1";
/// The type string `write_perceptron` starts with.
pub const PERCEPTRON_TYPE: &str = "TURBO-PERCEPTRON-1";
/// The type string `write_naive_bayes` starts with.
pub const NAIVE_BAYES_TYPE: &str = "TURBO-NAIVE-BAYES-1";
/// Maximum standard UTF-8 byte length of a single wire string.
pub const MAX_UTF_BYTES: usize = u16::MAX as usize;

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
    write_gis_writable(model)
}

/// [`write_gis`] for any [`WritableModel`]: the `.model` entry of a GIS
/// model built outside the GIS trainer, such as OpenNLP's
/// `new GISModel(parameters, predicates, outcomes)` that a ranking
/// trainer assembles from its own weights.
pub fn write_gis_writable(model: &dyn WritableModel) -> Result<Vec<u8>> {
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_str(GIS_TYPE)?;
    write_body(&mut out, model)?;
    Ok(out.bytes)
}

/// `GenericModelWriter(model, DataOutputStream).persist()` for a QN
/// model: the bytes of the `.model` entry.
pub fn write_qn(model: &QnModel) -> Result<Vec<u8>> {
    write_qn_writable(model)
}

/// [`write_qn`] for any [`WritableModel`]: the `.model` entry of a QN
/// model built outside the QN trainer, such as OpenNLP's
/// `new QNModel(parameters, predicates, outcomes)` that a ranking
/// trainer assembles from its own weights.
pub fn write_qn_writable(model: &dyn WritableModel) -> Result<Vec<u8>> {
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_str(QN_TYPE)?;
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
    out.write_str(PERCEPTRON_TYPE)?;
    write_body(&mut out, &pruned)?;
    Ok(out.bytes)
}

/// `GenericModelWriter(model, DataOutputStream).persist()` for a naive
/// Bayes model: the bytes of the `.model` entry (`NaiveBayesModelWriter.persist`).
pub fn write_naive_bayes(model: &NaiveBayesModel) -> Result<Vec<u8>> {
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_str(NAIVE_BAYES_TYPE)?;
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
        out.write_str(&text)?;
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

/// Little-endian native binary model output buffer.
struct ModelOutput {
    bytes: Vec<u8>,
}

impl ModelOutput {
    fn write_int(&mut self, i: i32) {
        self.bytes.extend_from_slice(&i.to_le_bytes());
    }

    fn write_double(&mut self, d: f64) {
        self.bytes.extend_from_slice(&d.to_bits().to_le_bytes());
    }

    fn write_utf(&mut self, s: &JavaString) -> Result<()> {
        let utf8 = s.to_str().ok_or(Error::UnpairedSurrogate)?;
        self.write_str(&utf8)
    }

    fn write_str(&mut self, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        if bytes.len() > MAX_UTF_BYTES {
            return Err(Error::StringTooLong { what: "model parameter", bytes: bytes.len() });
        }
        self.bytes.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
