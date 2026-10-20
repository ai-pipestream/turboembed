//! The multi-label document categorizer model: one linear head per
//! label, a validated per-label threshold vector, and the model entry's
//! bytes, which `MultiLabelDoccatModel::write` and `::open` round-trip
//! with a trailing CRC-32C.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use turbo_analysis::{ModelRef, Tag, TagSet};
use turbo_train::crc32c;

use super::bytes::{MAX_UTF_BYTES, Reader, put_utf};
use super::error::MultiLabelError;
use crate::features::{BuiltinFeatureGenerator, DoccatContextGenerator};
use crate::model::parse_feature_generators;

/// The model entry's type string, `writeUTF` first, as `writeUTF("GIS")`
/// is first in a maxent model's bytes.
pub const MODEL_TYPE: &str = "MultiLabel";
/// The format version this reader and writer implement.
pub const FORMAT_VERSION: i32 = 1;
/// The model's entry in its package.
pub const MODEL_ENTRY: &str = "doccat.multilabel.model";
/// The name of the model's category `TagSet`.
pub const TAGSET_NAME: &str = "turbo:multilabel-doccat";
/// The threshold every label gets until calibration moves it.
pub const DEFAULT_THRESHOLD: f64 = 0.5;

/// The logistic sigmoid, into `0.0..=1.0`, stable for either sign.
pub(crate) fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// One label's One-vs-Rest head: a linear scorer (`bias` plus a weight
/// per document feature present) that the sigmoid turns into a score in
/// `0.0..=1.0`. The weights are the nonzero ones, sorted by feature name
/// so the bytes a model writes are deterministic.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearHead {
    bias: f64,
    weights: Vec<(String, f64)>,
}

impl LinearHead {
    /// Builds a head from its `bias` and `(feature, weight)` pairs.
    ///
    /// Returns [`MultiLabelError::Invalid`] when the bias or a weight is
    /// NaN or infinite, a feature name is empty, over `writeUTF`'s 65535
    /// bytes, or the names are not strictly ascending (so unique).
    pub fn new(bias: f64, weights: Vec<(String, f64)>) -> Result<LinearHead, MultiLabelError> {
        if !bias.is_finite() {
            return Err(MultiLabelError::Invalid(format!("a head bias of {bias}")));
        }
        for (i, (name, weight)) in weights.iter().enumerate() {
            if name.is_empty() {
                return Err(MultiLabelError::Invalid("a head with an empty feature name".into()));
            }
            if name.len() > MAX_UTF_BYTES {
                return Err(MultiLabelError::Invalid(format!(
                    "feature {name:?} is {0} bytes, over writeUTF's {MAX_UTF_BYTES}",
                    name.len()
                )));
            }
            if !weight.is_finite() {
                return Err(MultiLabelError::Invalid(format!("weight {weight} of feature {name:?}")));
            }
            if i > 0 && weights[i - 1].0.as_str() >= name.as_str() {
                return Err(MultiLabelError::Invalid(format!(
                    "head features not strictly ascending: {:?} then {name:?}",
                    weights[i - 1].0
                )));
            }
        }
        Ok(LinearHead { bias, weights })
    }

    /// The bias: the log-odds of a document with no weighted feature.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// The nonzero weights, sorted by feature name.
    pub fn weights(&self) -> &[(String, f64)] {
        &self.weights
    }

    /// The linear score for a document whose features are `features`,
    /// before the sigmoid: the bias plus every weight whose feature is
    /// present.
    pub fn score(&self, features: &HashSet<&str>) -> f64 {
        self.bias + self.weights.iter().filter(|(n, _)| features.contains(n.as_str())).map(|(_, w)| w).sum::<f64>()
    }

    /// The sigmoid score in `0.0..=1.0`.
    pub fn probability(&self, features: &HashSet<&str>) -> f64 {
        sigmoid(self.score(features))
    }
}

/// A multi-label document categorizer model: every known category label
/// scored independently in `0.0..=1.0` by its own [`LinearHead`] through
/// the sigmoid (One-vs-Rest; the scores do not sum to 1.0), emitted where
/// they reach the label's own threshold.
#[derive(Clone, Debug)]
pub struct MultiLabelDoccatModel {
    language: String,
    categories: Arc<TagSet>,
    heads: Vec<LinearHead>,
    thresholds: Vec<f64>,
    max_labels: Option<usize>,
    min_score: f64,
    context_generator: DoccatContextGenerator,
    model_ref: ModelRef,
}

impl MultiLabelDoccatModel {
    /// Builds a model from its labels, one head per label in label order,
    /// and one threshold per label (each finite, in the open interval
    /// `(0.0, 1.0)`; [`DEFAULT_THRESHOLD`] until calibration).
    ///
    /// `max_labels` is `None` and `min_score` 0.0; set them with
    /// [`MultiLabelDoccatModel::with_max_labels`] and
    /// [`MultiLabelDoccatModel::with_min_score`].
    pub fn new(
        language: impl Into<String>,
        labels: &[&str],
        heads: Vec<LinearHead>,
        thresholds: Vec<f64>,
        context_generator: DoccatContextGenerator,
    ) -> Result<MultiLabelDoccatModel, MultiLabelError> {
        let categories = TagSet::new(TAGSET_NAME, labels);
        Self::assemble(language.into(), categories, heads, thresholds, None, 0.0, context_generator)
    }

    /// Validates the parts every constructor and `open` share, and records
    /// the model reference as the SHA-256 of the entry's bytes.
    fn assemble(
        language: String,
        categories: Arc<TagSet>,
        heads: Vec<LinearHead>,
        thresholds: Vec<f64>,
        max_labels: Option<usize>,
        min_score: f64,
        context_generator: DoccatContextGenerator,
    ) -> Result<MultiLabelDoccatModel, MultiLabelError> {
        let k = categories.len();
        if k == 0 || k >= Tag::NONE.0 as usize {
            return Err(MultiLabelError::Invalid(format!("{k} labels, between 1 and {}", Tag::NONE.0 as usize - 1)));
        }
        if language.len() > MAX_UTF_BYTES {
            return Err(MultiLabelError::Invalid(format!(
                "language {language:?} is {0} bytes, over writeUTF's {MAX_UTF_BYTES}",
                language.len()
            )));
        }
        if heads.len() != k {
            return Err(MultiLabelError::Invalid(format!("{} heads for {k} labels", heads.len())));
        }
        check_thresholds(&thresholds, k)?;
        if let Some(max) = max_labels
            && max == 0
        {
            return Err(MultiLabelError::Invalid("max_labels of 0: at least one label must be able to emit".into()));
        }
        if !min_score.is_finite() || !(0.0..=1.0).contains(&min_score) {
            return Err(MultiLabelError::Invalid(format!("min_score {min_score}, outside 0.0..=1.0")));
        }
        let mut model = MultiLabelDoccatModel {
            language,
            categories,
            heads,
            thresholds,
            max_labels,
            min_score,
            context_generator,
            model_ref: ModelRef { id: String::new(), revision: String::new(), sha256: String::new() },
        };
        model.refresh_ref();
        Ok(model)
    }

    /// Recomputes the model reference from the entry's bytes.
    fn refresh_ref(&mut self) {
        let bytes = self.write();
        self.model_ref = ModelRef {
            id: format!("turbo:MultiLabelDoccat:{}", self.language),
            revision: FORMAT_VERSION.to_string(),
            sha256: turbo_maxent::sha256(&bytes).to_hex(),
        };
    }

    /// The same model emitting at most `max_labels` labels per item (the
    /// highest scorers; `None` keeps every label that clears its
    /// threshold).
    pub fn with_max_labels(mut self, max_labels: Option<usize>) -> Result<Self, MultiLabelError> {
        if max_labels == Some(0) {
            return Err(MultiLabelError::Invalid("max_labels of 0: at least one label must be able to emit".into()));
        }
        self.max_labels = max_labels;
        self.refresh_ref();
        Ok(self)
    }

    /// The same model where a label's score must also reach `min_score`
    /// (in `0.0..=1.0`) to be emitted.
    pub fn with_min_score(mut self, min_score: f64) -> Result<Self, MultiLabelError> {
        if !min_score.is_finite() || !(0.0..=1.0).contains(&min_score) {
            return Err(MultiLabelError::Invalid(format!("min_score {min_score}, outside 0.0..=1.0")));
        }
        self.min_score = min_score;
        self.refresh_ref();
        Ok(self)
    }

    /// The same model with a new per-label threshold vector, validated as
    /// in [`MultiLabelDoccatModel::new`].
    pub fn with_thresholds(mut self, thresholds: Vec<f64>) -> Result<Self, MultiLabelError> {
        check_thresholds(&thresholds, self.num_labels())?;
        self.thresholds = thresholds;
        self.refresh_ref();
        Ok(self)
    }

    /// The model's language code.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// The closed category set, in head order.
    pub fn categories(&self) -> &Arc<TagSet> {
        &self.categories
    }

    /// The number of labels.
    pub fn num_labels(&self) -> usize {
        self.categories.len()
    }

    /// The label name at `index`, or `None` past the end.
    pub fn label(&self, index: usize) -> Option<&str> {
        (index < self.categories.len()).then(|| self.categories.label(Tag(index as u16)))
    }

    /// The index of `label`, or `None` when the model has no such label.
    pub fn index(&self, label: &str) -> Option<usize> {
        self.categories.tag(label).map(|t| t.0 as usize)
    }

    /// The head scoring label `index`.
    pub fn head(&self, index: usize) -> Option<&LinearHead> {
        self.heads.get(index)
    }

    /// The per-label thresholds, in label order.
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    /// The most labels one item emits, or `None` for no cap.
    pub fn max_labels(&self) -> Option<usize> {
        self.max_labels
    }

    /// The score floor under which no label is emitted.
    pub fn min_score(&self) -> f64 {
        self.min_score
    }

    /// The configured [`DoccatContextGenerator`].
    pub fn context_generator(&self) -> &DoccatContextGenerator {
        &self.context_generator
    }

    /// The model reference recorded in layer provenance.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// The same model with the package-level identity recorded: the
    /// archive's OpenNLP-Version as the revision and the SHA-256 of the
    /// package's bytes, as [`super::open_package`] reads it.
    pub(crate) fn with_package_identity(mut self, revision: &str, sha256: &str) -> Self {
        self.model_ref.revision = revision.to_owned();
        self.model_ref.sha256 = sha256.to_owned();
        self
    }

    /// Scores `text` with empty extra information: every label's sigmoid
    /// score, in label order.
    pub fn scores(&self, text: &[&str]) -> Vec<f64> {
        let extra = BTreeMap::new();
        let mut out = Vec::with_capacity(self.num_labels());
        let mut scratch = Vec::new();
        self.scores_into(text, &extra, &mut scratch, &mut out);
        out
    }

    /// Scores `text` and `extra_information`: every label's sigmoid score
    /// in `0.0..=1.0`, in label order.
    pub fn scores_with_extra(&self, text: &[&str], extra_information: &BTreeMap<String, String>) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.num_labels());
        let mut scratch = Vec::new();
        self.scores_into(text, extra_information, &mut scratch, &mut out);
        out
    }

    /// Scores `text` and `extra_information` into `out`
    /// (`out.len() == self.num_labels()`), reusing `feature_scratch`.
    pub fn scores_into(
        &self,
        text: &[&str],
        extra_information: &BTreeMap<String, String>,
        feature_scratch: &mut Vec<String>,
        out: &mut Vec<f64>,
    ) {
        self.context_generator.extract_features(text, extra_information, feature_scratch);
        let features: HashSet<&str> = feature_scratch.iter().map(String::as_str).collect();
        out.clear();
        out.extend(self.heads.iter().map(|h| h.probability(&features)));
    }

    /// The labels `scores` emits: every label whose score reaches its
    /// threshold and `min_score`, at most `max_labels` of them, in label
    /// order. Empty is clean abstention, never a forced argmax.
    pub fn predicted_tags(&self, scores: &[f64]) -> Vec<Tag> {
        debug_assert_eq!(scores.len(), self.num_labels());
        let mut hit: Vec<usize> =
            (0..scores.len()).filter(|&i| scores[i] >= self.thresholds[i] && scores[i] >= self.min_score).collect();
        hit.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
        hit.truncate(self.max_labels.unwrap_or(usize::MAX));
        hit.sort_unstable();
        hit.into_iter().map(|i| Tag(i as u16)).collect()
    }

    /// The entry's bytes: `writeUTF("MultiLabel")`, the format version,
    /// the language, the feature generator class names, `max_labels` (-1
    /// for none), `min_score`, and per label the name, threshold, bias,
    /// and `(feature, weight)` pairs, then the CRC-32C of all of it.
    pub fn write(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_utf(&mut out, MODEL_TYPE);
        out.extend_from_slice(&FORMAT_VERSION.to_be_bytes());
        put_utf(&mut out, &self.language);
        let generators: Vec<&str> =
            self.context_generator.builtins().iter().map(BuiltinFeatureGenerator::class_name).collect();
        put_utf(&mut out, &generators.join(","));
        out.extend_from_slice(&(self.max_labels.map_or(-1, |m| m as i32)).to_be_bytes());
        out.extend_from_slice(&self.min_score.to_be_bytes());
        out.extend_from_slice(&(self.num_labels() as i32).to_be_bytes());
        for (i, head) in self.heads.iter().enumerate() {
            put_utf(&mut out, self.categories.label(Tag(i as u16)));
            out.extend_from_slice(&self.thresholds[i].to_be_bytes());
            out.extend_from_slice(&head.bias.to_be_bytes());
            out.extend_from_slice(&(head.weights.len() as i32).to_be_bytes());
            for (name, weight) in &head.weights {
                put_utf(&mut out, name);
                out.extend_from_slice(&weight.to_be_bytes());
            }
        }
        let crc = crc32c(&out);
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }

    /// Opens a model from its entry's bytes, the mirror of
    /// [`MultiLabelDoccatModel::write`]. Truncated or corrupted bytes
    /// fail with [`MultiLabelError`]; reading never panics.
    pub fn open(bytes: &[u8]) -> Result<MultiLabelDoccatModel, MultiLabelError> {
        let mut r = Reader::new(bytes);
        let magic = r.utf8("the model type")?;
        if magic != MODEL_TYPE {
            return Err(MultiLabelError::BadMagic);
        }
        let version = r.i32("the format version")?;
        if version != FORMAT_VERSION {
            return Err(MultiLabelError::UnsupportedVersion(version));
        }
        let language = r.utf8("the language")?.to_owned();
        let generators = r.utf8("the feature generators")?;
        let context_generator = DoccatContextGenerator::from_builtins(
            parse_feature_generators(generators).map_err(|e| MultiLabelError::Invalid(e.to_string()))?,
        );
        let max_labels = match r.i32("max_labels")? {
            -1 => None,
            m if m < -1 => return Err(MultiLabelError::Invalid(format!("max_labels of {m}"))),
            m => Some(m as usize),
        };
        let min_score = r.f64("min_score")?;
        let n = r.count("the label count")?;
        if n == 0 || n >= Tag::NONE.0 as usize {
            return Err(MultiLabelError::Invalid(format!("{n} labels, between 1 and {}", Tag::NONE.0 as usize - 1)));
        }
        let mut names = Vec::with_capacity(n);
        let mut heads = Vec::with_capacity(n);
        let mut thresholds = Vec::with_capacity(n);
        for _ in 0..n {
            let name = r.utf8("a label")?;
            if names.iter().any(|x| x == name) {
                return Err(MultiLabelError::Invalid(format!("duplicate label {name:?}")));
            }
            names.push(name.to_owned());
            let threshold = r.f64("a threshold")?;
            let bias = r.f64("a head bias")?;
            let weights = r.count("a head's weight count")?;
            if weights > r.remaining() {
                return Err(MultiLabelError::Truncated("a head's weights"));
            }
            let mut pairs = Vec::with_capacity(weights);
            for _ in 0..weights {
                let feature = r.utf8("a weight's feature")?;
                let weight = r.f64("a weight")?;
                pairs.push((feature.to_owned(), weight));
            }
            thresholds.push(threshold);
            heads.push(LinearHead::new(bias, pairs)?);
        }
        let stored = r.i32("the checksum")? as u32;
        let got = crc32c(&bytes[..r.position() - 4]);
        if got != stored {
            return Err(MultiLabelError::Checksum { want: stored, got });
        }
        r.done()?;
        let categories = TagSet::new(TAGSET_NAME, &names.iter().map(String::as_str).collect::<Vec<_>>());
        Self::assemble(language, categories, heads, thresholds, max_labels, min_score, context_generator)
    }
}

/// The thresholds every constructor validates: `k` of them, each finite
/// and in the open interval `(0.0, 1.0)`.
fn check_thresholds(thresholds: &[f64], k: usize) -> Result<(), MultiLabelError> {
    if thresholds.len() != k {
        return Err(MultiLabelError::Invalid(format!("{} thresholds for {k} labels", thresholds.len())));
    }
    for (i, &t) in thresholds.iter().enumerate() {
        if !t.is_finite() || t <= 0.0 || t >= 1.0 {
            return Err(MultiLabelError::Invalid(format!("threshold {t} of label {i}, outside the open 0.0..1.0")));
        }
    }
    Ok(())
}
