//! The multi-label document categorizer model: one linear head per
//! label, a validated per-label threshold vector, and the model entry's
//! bytes, which `MultiLabelDoccatModel::write` and `::open` round-trip
//! with a trailing CRC-32C.

use std::collections::BTreeMap;
use std::sync::Arc;

use turbo_analysis::{ModelRef, Tag, TagSet};
use turbo_train::crc32c;

use super::bytes::{MAX_UTF_BYTES, Reader, put_utf};
use super::error::MultiLabelError;
use super::head::LinearHead;
use super::validate::{check_label_names, check_thresholds};
use crate::features::DoccatContextGenerator;
use crate::model::{encode_feature_generators, parse_feature_generators};

/// The model entry's type string, `writeUTF` first, as `writeUTF("GIS")`
/// is first in a maxent model's bytes.
pub const MODEL_TYPE: &str = "MultiLabel";
/// The format version this reader and writer implement. Version 2 added
/// `min_labels` after `min_score`; version 1 entries (which end their
/// header at `min_score`) still open, with `min_labels` `None`.
pub const FORMAT_VERSION: i32 = 2;
/// The first format version: no `min_labels` field.
pub const FORMAT_VERSION_V1: i32 = 1;
/// The model's entry in its package.
pub const MODEL_ENTRY: &str = "doccat.multilabel.model";
/// The name of the model's category `TagSet`.
pub const TAGSET_NAME: &str = "turbo:multilabel-doccat";
/// The threshold every label gets until calibration moves it.
pub const DEFAULT_THRESHOLD: f64 = 0.5;
/// The open interval a threshold validates against is `(0.0, 1.0)`;
/// calibrated thresholds clamp to this epsilon inside it, so a cut point
/// at a validation score of exactly 0.0 or 1.0 still applies.
pub const MIN_THRESHOLD: f64 = 1e-6;
pub const MAX_THRESHOLD: f64 = 1.0 - 1e-6;

/// A multi-label document categorizer model: every known category label
/// scored independently in `0.0..=1.0` by its own [`LinearHead`] through
/// the sigmoid (One-vs-Rest; the scores do not sum to 1.0), emitted where
/// they reach the label's own threshold.
///
/// Emission is cardinality bounded: at most `max_labels` labels, and at
/// least `min_labels`, which back-fills the highest-scoring labels that
/// were held under their thresholds until `min_labels` emit. `min_score`
/// is a hard floor throughout: a label under it never emits, so a forced
/// fill emits the top labels that pass `min_score`, and when fewer than
/// `min_labels` labels pass `min_score`, all that do emit.
#[derive(Clone, Debug)]
pub struct MultiLabelDoccatModel {
    language: String,
    categories: Arc<TagSet>,
    heads: Vec<LinearHead>,
    thresholds: Vec<f64>,
    bounds: Bounds,
    context_generator: DoccatContextGenerator,
    model_ref: ModelRef,
}

/// The emission cardinality bounds: at most `max_labels`, at least
/// `min_labels` (back-filled by score), every emitted label at least
/// `min_score`. `None` is no cap and no count floor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Bounds {
    max_labels: Option<usize>,
    min_labels: Option<usize>,
    min_score: f64,
}

impl Bounds {
    /// Validates the bounds against each other and `num_labels`:
    /// `max_labels` and `min_labels` are `None` or positive and at most
    /// `i32::MAX` (the wire field is 32 bits; a larger value would
    /// truncate), `min_labels` at most `num_labels` labels and at most
    /// `max_labels` where both are set, and `min_score` finite in
    /// `0.0..=1.0`.
    fn check(&self, num_labels: usize) -> Result<(), MultiLabelError> {
        for (what, value) in [("max_labels", self.max_labels), ("min_labels", self.min_labels)] {
            if let Some(value) = value {
                if value == 0 {
                    let detail = if what == "max_labels" {
                        "max_labels of 0: at least one label must be able to emit"
                    } else {
                        "min_labels of 0: no forced emission is the default, leave it None"
                    };
                    return Err(MultiLabelError::Invalid(detail.into()));
                }
                if value > i32::MAX as usize {
                    return Err(MultiLabelError::Invalid(format!(
                        "{what} of {value}, over the wire field's {} (i32::MAX)",
                        i32::MAX as usize
                    )));
                }
            }
        }
        if let Some(min) = self.min_labels {
            if min > num_labels {
                return Err(MultiLabelError::Invalid(format!(
                    "min_labels of {min} for {num_labels} labels: at most one emission per label"
                )));
            }
            if let Some(max) = self.max_labels
                && min > max
            {
                return Err(MultiLabelError::Invalid(format!(
                    "min_labels of {min} over max_labels of {max}: the bounds cross"
                )));
            }
        }
        if !self.min_score.is_finite() || !(0.0..=1.0).contains(&self.min_score) {
            return Err(MultiLabelError::Invalid(format!("min_score {}, outside 0.0..=1.0", self.min_score)));
        }
        Ok(())
    }
}

impl MultiLabelDoccatModel {
    /// Builds a model from its labels, one head per label in label order,
    /// and one threshold per label (each finite, in the open interval
    /// `(0.0, 1.0)`; [`DEFAULT_THRESHOLD`] until calibration).
    ///
    /// `max_labels` and `min_labels` are `None` and `min_score` 0.0; set
    /// them with [`MultiLabelDoccatModel::with_max_labels`],
    /// [`MultiLabelDoccatModel::with_min_labels`], and
    /// [`MultiLabelDoccatModel::with_min_score`].
    pub fn new(
        language: impl Into<String>,
        labels: &[&str],
        heads: Vec<LinearHead>,
        thresholds: Vec<f64>,
        context_generator: DoccatContextGenerator,
    ) -> Result<MultiLabelDoccatModel, MultiLabelError> {
        check_label_names(labels)?;
        let categories = TagSet::new(TAGSET_NAME, labels);
        Self::assemble(language.into(), categories, heads, thresholds, Bounds::default(), context_generator)
    }

    /// Validates the parts every constructor and `open` share, and records
    /// the model reference as the SHA-256 of the entry's bytes.
    fn assemble(
        language: String,
        categories: Arc<TagSet>,
        heads: Vec<LinearHead>,
        thresholds: Vec<f64>,
        bounds: Bounds,
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
        bounds.check(k)?;
        let mut model = MultiLabelDoccatModel {
            language,
            categories,
            heads,
            thresholds,
            bounds,
            context_generator,
            model_ref: ModelRef { id: String::new(), revision: String::new(), sha256: String::new() },
        };
        model.refresh_ref();
        Ok(model)
    }

    /// Recomputes the model reference from the entry's bytes.
    fn refresh_ref(&mut self) {
        let sha256 = self.write().map(|b| turbo_maxent::sha256(&b).to_hex()).unwrap_or_default();
        self.model_ref = ModelRef {
            id: format!("turbo:MultiLabelDoccat:{}", self.language),
            revision: FORMAT_VERSION.to_string(),
            sha256,
        };
    }

    /// The same model with a new [`DoccatContextGenerator`].
    pub fn with_context_generator(mut self, context_generator: DoccatContextGenerator) -> Self {
        self.context_generator = context_generator;
        self.refresh_ref();
        self
    }

    /// The same model emitting at most `max_labels` labels per item (the
    /// highest scorers; `None` keeps every label that clears its
    /// threshold). Where `min_labels` is also set, it must be at most
    /// `max_labels`.
    pub fn with_max_labels(mut self, max_labels: Option<usize>) -> Result<Self, MultiLabelError> {
        let mut bounds = self.bounds;
        bounds.max_labels = max_labels;
        bounds.check(self.num_labels())?;
        self.bounds = bounds;
        self.refresh_ref();
        Ok(self)
    }

    /// The same model emitting at least `min_labels` labels per item: when
    /// fewer clear their thresholds, the highest-scoring labels that pass
    /// `min_score` fill the item up to `min_labels`, in score order
    /// (`min_score` stays a hard floor). At most `num_labels`, and at most
    /// `max_labels` where that is set.
    pub fn with_min_labels(mut self, min_labels: Option<usize>) -> Result<Self, MultiLabelError> {
        let mut bounds = self.bounds;
        bounds.min_labels = min_labels;
        bounds.check(self.num_labels())?;
        self.bounds = bounds;
        self.refresh_ref();
        Ok(self)
    }

    /// The same model where a label's score must also reach `min_score`
    /// (in `0.0..=1.0`) to be emitted.
    pub fn with_min_score(mut self, min_score: f64) -> Result<Self, MultiLabelError> {
        let mut bounds = self.bounds;
        bounds.min_score = min_score;
        bounds.check(self.num_labels())?;
        self.bounds = bounds;
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
        self.bounds.max_labels
    }

    /// The fewest labels one item emits, back-filled by score where the
    /// thresholds alone emit fewer, or `None` for no floor on count.
    pub fn min_labels(&self) -> Option<usize> {
        self.bounds.min_labels
    }

    /// The score floor under which no label is emitted, forced fills
    /// included.
    pub fn min_score(&self) -> f64 {
        self.bounds.min_score
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
        // Feature counts, the bag-of-features representation training
        // fits the heads with: a repeated token counts once per
        // occurrence at inference as it did in training.
        let mut counts: BTreeMap<&str, f64> = BTreeMap::new();
        for feature in feature_scratch.iter().map(String::as_str) {
            *counts.entry(feature).or_default() += 1.0;
        }
        out.clear();
        out.extend(self.heads.iter().map(|h| h.probability(&counts)));
    }

    /// The labels `scores` emits, in label order: every label whose score
    /// reaches its threshold and `min_score`, at most `max_labels` of
    /// them; then, where fewer than `min_labels` emit, the
    /// highest-scoring labels that pass `min_score` fill the item up to
    /// `min_labels` (`min_score` a hard floor: a label under it never
    /// emits, so the fill stops early when fewer than `min_labels` labels
    /// pass it). Empty is clean abstention, never a forced argmax.
    pub fn predicted_tags(&self, scores: &[f64]) -> Vec<Tag> {
        debug_assert_eq!(scores.len(), self.num_labels());
        let by_score = |a: &usize, b: &usize| scores[*b].total_cmp(&scores[*a]).then_with(|| a.cmp(b));
        let mut hit: Vec<usize> = (0..scores.len())
            .filter(|&i| scores[i] >= self.thresholds[i] && scores[i] >= self.bounds.min_score)
            .collect();
        hit.sort_by(by_score);
        hit.truncate(self.bounds.max_labels.unwrap_or(usize::MAX));
        if let Some(min) = self.bounds.min_labels
            && hit.len() < min
        {
            let mut rest: Vec<usize> =
                (0..scores.len()).filter(|i| !hit.contains(i) && scores[*i] >= self.bounds.min_score).collect();
            rest.sort_by(by_score);
            hit.extend(rest.into_iter().take(min - hit.len()));
        }
        hit.sort_unstable();
        hit.into_iter().map(|i| Tag(i as u16)).collect()
    }

    /// The entry's bytes: `writeUTF("MultiLabel")`, the format version,
    /// the language, the feature generator specifications, `max_labels` (-1
    /// for none), `min_labels` (version 2 and up; -1 for none),
    /// `min_score`, and per label the name, threshold, bias, and
    /// `(feature, weight)` pairs, then the CRC-32C of all of it.
    pub fn write(&self) -> Result<Vec<u8>, MultiLabelError> {
        let generators = encode_feature_generators(&self.context_generator).map_err(MultiLabelError::Invalid)?;
        let mut out = Vec::new();
        put_utf(&mut out, MODEL_TYPE);
        out.extend_from_slice(&FORMAT_VERSION.to_be_bytes());
        put_utf(&mut out, &self.language);
        put_utf(&mut out, &generators);
        out.extend_from_slice(&(self.bounds.max_labels.map_or(-1, |m| m as i32)).to_be_bytes());
        out.extend_from_slice(&(self.bounds.min_labels.map_or(-1, |m| m as i32)).to_be_bytes());
        out.extend_from_slice(&self.bounds.min_score.to_be_bytes());
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
        Ok(out)
    }

    /// Opens a model from its entry's bytes, the mirror of
    /// [`MultiLabelDoccatModel::write`]. Truncated or corrupted bytes
    /// fail with [`MultiLabelError`]; reading never panics. Version 1
    /// entries, which hold no `min_labels`, open with it `None`.
    pub fn open(bytes: &[u8]) -> Result<MultiLabelDoccatModel, MultiLabelError> {
        let mut r = Reader::new(bytes);
        let magic = r.utf8("the model type")?;
        if magic != MODEL_TYPE {
            return Err(MultiLabelError::BadMagic);
        }
        let version = r.i32("the format version")?;
        if version != FORMAT_VERSION && version != FORMAT_VERSION_V1 {
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
        let min_labels = if version == FORMAT_VERSION_V1 {
            None
        } else {
            match r.i32("min_labels")? {
                -1 => None,
                m if m < -1 => return Err(MultiLabelError::Invalid(format!("min_labels of {m}"))),
                m => Some(m as usize),
            }
        };
        let bounds = Bounds { max_labels, min_labels, min_score: r.f64("min_score")? };
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
        Self::assemble(language, categories, heads, thresholds, bounds, context_generator)
    }
}
