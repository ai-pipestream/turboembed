//! The detector, OpenNLP's `LanguageDetectorME`: `predictLanguages`,
//! `predictLanguage`, `probingPredictLanguages`, and the annotator that
//! writes `opennlp:language`.

use std::cell::RefCell;
use std::cmp::Ordering;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Language, Strs};

use crate::context::ContextGenerator;
use crate::model::LanguageModel;
use crate::probing::{ProbingConfig, ProbingContext, seen_enough, take_chars};

/// The reason the annotator gives when it declines a document.
pub const NO_EVIDENCE: &str = "no language evidence";

/// Buffers for one detection, kept between calls so a detector run on
/// many texts allocates for the n-grams and the probabilities only while
/// they grow. The normalizers allocate a new string for a text they
/// change.
#[derive(Debug, Default)]
pub struct Scratch {
    context: ContextGenerator,
    probing: ProbingContext,
    probing_history: Vec<(u32, f64)>,
    probabilities: Vec<f64>,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }

    /// The context of the last text detected with
    /// [`LanguageDetector::predict_languages`] or
    /// [`LanguageDetector::predict_language`]: its n-grams in the order the
    /// model received them.
    pub fn context(&self) -> &ContextGenerator {
        &self.context
    }

    /// The cumulative context of the last text detected with
    /// [`LanguageDetector::probing_predict_languages`] or
    /// [`LanguageDetector::probing_predict_languages_with`]: its n-grams in
    /// the order the model received them on the final evaluated chunk.
    pub fn probing_context(&self) -> &ProbingContext {
        &self.probing
    }

    /// The probability of each outcome for the last text, in the model's
    /// outcome order, as `eval` returned them.
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Releases an oversized context. Its feature iterators become empty;
    /// the last prediction probabilities remain available.
    pub fn shrink_excess(&mut self) {
        self.context.shrink_excess();
        self.probing.shrink_excess();
    }
}

/// Every language the model knows, best first, as `predictLanguages`
/// returns them.
#[derive(Debug, Default, Clone)]
pub struct Ranking {
    outcome: Vec<u32>,
    probability: Vec<f64>,
    evidence: bool,
}

impl Ranking {
    pub fn new() -> Ranking {
        Ranking::default()
    }

    /// The outcome indexes, best first.
    pub fn outcomes(&self) -> &[u32] {
        &self.outcome
    }

    /// The probabilities, best first: `probabilities()[i]` is that of
    /// `outcomes()[i]`.
    pub fn probabilities(&self) -> &[f64] {
        &self.probability
    }

    pub fn len(&self) -> usize {
        self.outcome.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outcome.is_empty()
    }

    /// Whether at least one n-gram of the text is a predicate of the
    /// model. Without one, every probability is the model's prior, equal
    /// for every language, and the ranking is the outcome order.
    pub fn has_evidence(&self) -> bool {
        self.evidence
    }
}

/// The best language, as `predictLanguage` returns it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction<'m> {
    /// The outcome index in the model.
    pub outcome: u32,
    /// The language code.
    pub code: &'m str,
    pub probability: f64,
    /// As [`Ranking::has_evidence`].
    pub evidence: bool,
}

/// `LanguageDetectorME` over a [`LanguageModel`].
#[derive(Debug)]
pub struct LanguageDetector {
    model: LanguageModel,
}

impl LanguageDetector {
    pub fn new(model: LanguageModel) -> LanguageDetector {
        LanguageDetector { model }
    }

    pub fn model(&self) -> &LanguageModel {
        &self.model
    }

    /// `predictLanguages`: every language of the model with its
    /// probability, best first. Languages of equal probability keep the
    /// model's outcome order, as `Arrays.sort` (a stable sort) keeps them,
    /// and probabilities are ordered as `Double.compare` orders them.
    pub fn predict_languages(&self, text: &str, scratch: &mut Scratch, out: &mut Ranking) {
        self.evaluate(text, scratch);
        self.rank(&scratch.probabilities, self.has_evidence(scratch.context.features()), out);
    }

    /// `probingPredictLanguages(content)` with
    /// [`ProbingConfig::default()`]: detects over growing chunks of `text`,
    /// stopping early when the top language is stable across
    /// `min_consec_improvements` consecutive improvements with a margin of at
    /// least `min_diff` over the second language, or when `max_length`
    /// Unicode code points have been processed.
    ///
    /// Writes the final ranking into `out` and returns the number of Unicode
    /// code points (`char`s) processed (`ProbingLanguageDetectionResult.getLength()`).
    pub fn probing_predict_languages(&self, text: &str, scratch: &mut Scratch, out: &mut Ranking) -> usize {
        self.probing_predict_languages_with(text, ProbingConfig::default(), scratch, out)
    }

    /// `probingPredictLanguages(content, config)`: as
    /// [`probing_predict_languages`](Self::probing_predict_languages), with a
    /// custom [`ProbingConfig`].
    pub fn probing_predict_languages_with(
        &self,
        text: &str,
        config: ProbingConfig,
        scratch: &mut Scratch,
        out: &mut Ranking,
    ) -> usize {
        scratch.probing.reset();
        scratch.probing_history.clear();
        let mut start = 0usize;
        let mut rest = text;
        let mut has_prediction = false;
        loop {
            let actual_chunk_size = if start.saturating_add(config.chunk_size) > config.max_length {
                config.max_length.saturating_sub(start)
            } else {
                config.chunk_size
            };
            let (chunk, chunk_chars) = take_chars(&mut rest, actual_chunk_size);
            if chunk_chars == 0 {
                if !has_prediction {
                    self.evaluate_probing(scratch, out);
                }
                return start;
            }
            start += chunk_chars;
            scratch.probing.add_chunk(chunk);
            self.evaluate_probing(scratch, out);
            has_prediction = true;
            if seen_enough(&mut scratch.probing_history, out, &config) {
                return start;
            }
        }
    }

    /// `predictLanguage`: the first language `predictLanguages` ranks,
    /// found without sorting.
    pub fn predict_language(&self, text: &str, scratch: &mut Scratch) -> Prediction<'_> {
        self.evaluate(text, scratch);
        let p = &scratch.probabilities;
        let mut best = 0;
        for i in 1..p.len() {
            if java_double_compare(p[i], p[best]) == Ordering::Greater {
                best = i;
            }
        }
        Prediction {
            outcome: best as u32,
            code: self.model.code(best),
            probability: p[best],
            evidence: self.has_evidence(scratch.context.features()),
        }
    }

    /// Builds the context and evaluates the model on it, every n-gram with
    /// the value 1 (`predict` passes 1 whatever the count, OPENNLP-1261),
    /// which adds each parameter as it is.
    fn evaluate(&self, text: &str, scratch: &mut Scratch) {
        scratch.context.generate(text);
        let maxent = self.model.maxent();
        scratch.probabilities.clear();
        scratch.probabilities.resize(maxent.num_outcomes(), 0.0);
        maxent.eval_features(scratch.context.features(), &mut scratch.probabilities);
    }

    fn evaluate_probing(&self, scratch: &mut Scratch, out: &mut Ranking) {
        let maxent = self.model.maxent();
        scratch.probabilities.clear();
        scratch.probabilities.resize(maxent.num_outcomes(), 0.0);
        maxent.eval_features(scratch.probing.features(), &mut scratch.probabilities);
        let evidence = self.has_evidence(scratch.probing.features());
        self.rank(&scratch.probabilities, evidence, out);
    }

    fn rank(&self, p: &[f64], evidence: bool, out: &mut Ranking) {
        out.outcome.clear();
        out.outcome.extend(0..p.len() as u32);
        out.outcome.sort_by(|&a, &b| java_double_compare(p[b as usize], p[a as usize]));
        out.probability.clear();
        out.probability.extend(out.outcome.iter().map(|&o| p[o as usize]));
        out.evidence = evidence;
    }

    fn has_evidence<'a>(&self, mut features: impl Iterator<Item = &'a [u8]>) -> bool {
        let maxent = self.model.maxent();
        features.any(|f| maxent.has_predicate_modified_utf8(f))
    }
}

/// `Double.compare`: numeric order, with -0.0 below 0.0 and every NaN
/// equal to every other and above positive infinity.
fn java_double_compare(a: f64, b: f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    }
    if a > b {
        return Ordering::Greater;
    }
    // `doubleToLongBits` makes every NaN the one canonical NaN.
    let bits = |x: f64| if x.is_nan() { 0x7ff8_0000_0000_0000_i64 } else { x.to_bits() as i64 };
    bits(a).cmp(&bits(b))
}

thread_local! {
    /// The annotator's buffers, one set per thread.
    static SCRATCH: RefCell<(Scratch, Ranking)> = RefCell::new((Scratch::new(), Ranking::new()));
}

/// Reads a view and writes `opennlp:language`: every language of the
/// model, best first, with its probability.
///
/// A text none of whose n-grams the model knows (an empty text, or one the
/// normalizers empty, such as a text of URLs, digits or emoji) is declined
/// with `Error::Skipped("no language evidence")`, where OpenNLP returns
/// every language at the same probability. [`LanguageDetector::predict_languages`]
/// returns what OpenNLP returns.
impl Annotator for LanguageDetector {
    const VERSION: &'static str = "2";
    const ID: &'static str = "turbo-langdetect:maxent";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Language,), Error> {
        SCRATCH.with_borrow_mut(|(scratch, ranking)| {
            self.predict_languages(view.text(), scratch, ranking);
            let res = if !ranking.has_evidence() {
                Err(Error::Skipped(NO_EVIDENCE.into()))
            } else {
                let mut code = Strs::default();
                for &o in ranking.outcomes() {
                    code.push(self.model.code(o as usize));
                }
                Ok((Language { code, score: ranking.probabilities().to_vec() },))
            };
            scratch.shrink_excess();
            res
        })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_compare_as_java_compares_them() {
        assert_eq!(java_double_compare(0.1, 0.2), Ordering::Less);
        assert_eq!(java_double_compare(0.2, 0.1), Ordering::Greater);
        assert_eq!(java_double_compare(0.5, 0.5), Ordering::Equal);
        assert_eq!(java_double_compare(-0.0, 0.0), Ordering::Less);
        assert_eq!(java_double_compare(0.0, -0.0), Ordering::Greater);
        assert_eq!(java_double_compare(f64::NAN, f64::INFINITY), Ordering::Greater);
        assert_eq!(java_double_compare(f64::NAN, -f64::NAN), Ordering::Equal);
        assert_eq!(java_double_compare(f64::NEG_INFINITY, f64::MIN), Ordering::Less);
    }
}
