//! The ranking model (`CorefModel`): a classifier over anaphor and
//! candidate antecedent pairs with the outcomes `link` and `apart`, kept
//! in an OpenNLP model package (component `CorefAnnotator`, entry
//! `coref.model`) whose manifest marks a ranker (`coref.ranking`) and the
//! contextual token-vector dimension it was trained with
//! (`coref.tokenVectorDimension`).

use turbo_analysis::ModelRef;
use turbo_jvm::strict_math;
use turbo_maxent::{MaxentModel, ModelFile, ModelKind};
use turbo_train::TrainingReport;
use turbo_train::package::{Manifest, ModelPackage};
use turbo_unicode::modified_utf8::push_str_modified_utf8;

/// The component name the package carries.
pub const COMPONENT_NAME: &str = "CorefAnnotator";
/// The archive entry of the pair model.
pub const MODEL_ENTRY: &str = "coref.model";
/// The manifest property that marks a model trained by ranking.
pub const RANKING_PROPERTY: &str = "coref.ranking";
/// The manifest property holding the token-vector dimension.
pub const TOKEN_VECTOR_DIMENSION_PROPERTY: &str = "coref.tokenVectorDimension";
/// The outcome of a pair the model links.
pub const LINK: &str = "link";
/// The outcome of a pair the model leaves apart.
pub const APART: &str = "apart";

/// A coreference model read from, or packaged as, an OpenNLP `.bin`.
#[derive(Debug, Clone)]
pub struct CorefModel {
    pair: MaxentModel,
    link: usize,
    ranking: bool,
    token_vector_dimension: usize,
    language: String,
    bytes: Vec<u8>,
    model_ref: ModelRef,
}

impl CorefModel {
    /// Reads a model package; `Err` names what is wrong with it.
    pub fn read(bytes: Vec<u8>) -> Result<CorefModel, String> {
        let file = ModelFile::open(bytes).map_err(|e| format!("invalid model package: {e}"))?;
        if file.component_name() != COMPONENT_NAME {
            return Err(format!("model is for component {}, not {COMPONENT_NAME}", file.component_name()));
        }
        let pair = file.maxent(MODEL_ENTRY).map_err(|_| "Coreference model is incomplete".to_owned())?;
        if pair.num_outcomes() != 2 || pair.outcome_index(LINK).is_none() || pair.outcome_index(APART).is_none() {
            return Err("Coreference model requires exactly the link and apart outcomes".into());
        }
        let manifest = file.manifest();
        let ranking = manifest.get(RANKING_PROPERTY).is_some_and(|v| v.eq_ignore_ascii_case("true"));
        let dimension = manifest.get(TOKEN_VECTOR_DIMENSION_PROPERTY);
        let token_vector_dimension = match dimension.map(str::parse::<i32>) {
            Some(Ok(d)) if d >= 0 => d as usize,
            Some(Ok(_)) | None => return Err("Coreference model requires a non-negative token-vector dimension".into()),
            Some(Err(_)) => {
                return Err(format!(
                    "Coreference model has an invalid token-vector dimension: {}",
                    dimension.unwrap_or_default()
                ));
            }
        };
        let link = pair.outcome_index(LINK).expect("checked above");
        let language = file.language().to_owned();
        let bytes = file.bytes().to_vec();
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}:{language}"),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(CorefModel { pair, link, ranking, token_vector_dimension, language, bytes, model_ref })
    }

    /// Packages a pair model's `.model` bytes as a coreference model,
    /// with the trainer's manifest entries when there is a report, and
    /// reads the package back.
    pub fn package(
        language: &str,
        model_entry: Vec<u8>,
        ranking: bool,
        token_vector_dimension: usize,
        report: Option<&TrainingReport>,
        timestamp_ms: i64,
    ) -> Result<CorefModel, String> {
        if language.chars().all(turbo_unicode::java::is_whitespace) {
            return Err("languageCode must not be null or blank".into());
        }
        let mut manifest = Manifest::new(COMPONENT_NAME, language, timestamp_ms);
        if let Some(report) = report {
            manifest.set_training(report);
        }
        manifest.set(RANKING_PROPERTY, if ranking { "true" } else { "false" });
        manifest.set(TOKEN_VECTOR_DIMENSION_PROPERTY, &token_vector_dimension.to_string());
        let mut package = ModelPackage::new(manifest);
        package.add_entry(MODEL_ENTRY, model_entry);
        CorefModel::read(package.to_bytes())
    }

    pub fn is_ranking(&self) -> bool {
        self.ranking
    }

    /// The contextual token-vector dimension used in training, zero for a
    /// model trained without a token encoder.
    pub fn token_vector_dimension(&self) -> usize {
        self.token_vector_dimension
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn pair_model(&self) -> &MaxentModel {
        &self.pair
    }

    /// The model as provenance records it.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// The package bytes, as read or written.
    pub fn to_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The probability of `link` given the features, each counting its
    /// value (one when `values` is `None`), as `MaxentModel.eval(context,
    /// values)` computes it. `Err` when `values` does not give one finite
    /// value per name, or when the probability is not finite.
    pub fn link_probability(&self, names: &[String], values: Option<&[f32]>) -> Result<f64, String> {
        let mut out = vec![0.0; self.pair.num_outcomes()];
        match values {
            None => {
                let context: Vec<&str> = names.iter().map(String::as_str).collect();
                self.pair.eval(&context, &mut out);
                // The JVM's eval overflows to NaN on large logits, and a
                // QN eval can stay finite but unnormalized when
                // max + ln(sum) rounds back to max; the real-valued path
                // with every value one normalizes them.
                let sum: f64 = out.iter().sum();
                let normalized = out.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                    && sum.is_finite()
                    && (sum - 1.0).abs() <= 1e-6;
                if !normalized && self.pair.kind() != ModelKind::NaiveBayes {
                    self.eval_values(names, &vec![1.0; names.len()], &mut out)?;
                }
            }
            Some(values) => {
                if values.len() != names.len() {
                    return Err(format!("{} feature values for {} features", values.len(), names.len()));
                }
                if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                    return Err(format!("feature value {i} ({}) is not finite: {}", names[i], values[i]));
                }
                self.eval_values(names, values, &mut out)?;
            }
        }
        let p = out[self.link];
        if !p.is_finite() {
            return Err(format!("link probability is not finite: {p}"));
        }
        Ok(p)
    }

    /// The real-valued evaluation of GIS, QN and perceptron models: each
    /// known predicate adds its parameters times its value.
    fn eval_values(&self, names: &[String], values: &[f32], out: &mut [f64]) -> Result<(), String> {
        let n = out.len();
        let start = match self.pair.kind() {
            ModelKind::Gis => strict_math::log(1.0 / n as f64),
            ModelKind::Qn | ModelKind::Perceptron => 0.0,
            ModelKind::NaiveBayes => return Err("a naive Bayes coreference model takes no real-valued features".into()),
        };
        out.fill(start);
        let mut key = Vec::new();
        for (name, &value) in names.iter().zip(values) {
            key.clear();
            push_str_modified_utf8(&mut key, name);
            if let Some(p) = self.pair.predicate_index(&key) {
                for (&o, &w) in self.pair.pattern(p).iter().zip(self.pair.parameters(p)) {
                    out[o as usize] += w * value as f64;
                }
            }
        }
        if let Some(x) = out.iter().find(|x| !x.is_finite()) {
            return Err(format!("a link logit is not finite: {x}"));
        }
        match self.pair.kind() {
            ModelKind::Qn => {
                // exp(x - max) / sum over the shifted weights: max +
                // ln(sum) rounds back to max on large logits, which would
                // leave every outcome at exp(0) = 1.
                let max = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for x in out.iter_mut() {
                    *x = strict_math::exp(*x - max);
                    sum += *x;
                }
                out.iter_mut().for_each(|x| *x /= sum);
            }
            kind => {
                if kind == ModelKind::Perceptron {
                    let divisor = out.iter().fold(1.0f64, |m, &x| if m < x.abs() { x.abs() } else { m });
                    if divisor != 1.0 {
                        out.iter_mut().for_each(|x| *x /= divisor);
                    }
                }
                softmax(out);
            }
        }
        Ok(())
    }
}

/// `exp(x) / sum(exp)` over finite logits as the JVM computes it, and,
/// where that overflows or underflows to a sum that is not a positive
/// finite number, with the largest logit subtracted first.
fn softmax(out: &mut [f64]) {
    let logits = out.to_vec();
    let mut sum = 0.0;
    for (x, &l) in out.iter_mut().zip(&logits) {
        *x = strict_math::exp(l);
        sum += *x;
    }
    if !(sum.is_finite() && sum > 0.0) {
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        sum = 0.0;
        for (x, &l) in out.iter_mut().zip(&logits) {
            *x = strict_math::exp(l - max);
            sum += *x;
        }
    }
    out.iter_mut().for_each(|x| *x /= sum);
}
