//! `LanguageDetectorCrossValidator`: n-fold cross-validation of the
//! language detector.
//!
//! As `LanguageDetectorCrossValidator.evaluate` does, [`CrossValidation`]
//! partitions the samples; for each fold the detector is trained on the
//! training samples ([`train_language_model_from_events`] over the events
//! of [`language_events`]), read back from its model file, and asked for
//! the language of each test sample, which [`LanguageDetectorEvaluator`]
//! scores.
//!
//! The merged accuracy differs in one place. The JVM adds each fold's
//! accuracy weighted by its document count to a `Mean`
//! (`documentAccuracy.add(evaluator.getAccuracy(),
//! evaluator.getDocumentCount())`), so the sum is a sum of rounded
//! products. [`LanguageCrossValidation::accuracy`] is the exact share,
//! correct samples over all samples; [`LanguageCrossValidation::jvm_accuracy`]
//! is the JVM's figure bit for bit, for comparing with it.

use turbo_formats::LanguageSample;
use turbo_formats::evaluate::{CrossValidation, LanguageDetectorEvaluator, Mean, Verdict};
use turbo_langdetect::{LanguageDetector, LanguageModel, Scratch};
use turbo_maxent::ModelFile;
use turbo_train::event::checksum;
use turbo_train::{TrainingConfiguration, TrainingParameters};

use super::{LanguageDetectorFactory, language_events, train_language_model_from_events};
use crate::error::{Error, Result};
use crate::package::Stamp;

/// One fold: what it trained on, what it tested on, and how it scored.
#[derive(Clone, Debug, PartialEq)]
pub struct LanguageFold {
    /// The fold's number, from 0.
    pub index: usize,
    pub training_samples: usize,
    pub test_samples: usize,
    /// The training events `LanguageDetectorME.train` made.
    pub events: usize,
    /// Their CRC32C, the model's `Training-Eventhash`.
    pub event_checksum: u32,
    /// Training events whose predicates may not be in the JVM's order
    /// (equal-hash n-grams in a tree bucket); see the module
    /// [`langdetect`](super).
    pub inexact_events: usize,
    /// Test samples given their reference language.
    pub correct: usize,
    /// The fold's `LanguageDetectorEvaluator.getAccuracy()`.
    pub accuracy: f64,
    /// The language predicted for each test sample, in test order.
    pub predictions: Vec<String>,
}

/// Every fold, with the merged accuracy two ways.
#[derive(Clone, Debug, PartialEq)]
pub struct LanguageCrossValidation {
    pub folds: Vec<LanguageFold>,
}

impl LanguageCrossValidation {
    /// Every fold's test samples, `getDocumentCount()`.
    pub fn document_count(&self) -> usize {
        self.folds.iter().map(|f| f.test_samples).sum()
    }

    /// Correct samples over all samples, rounded once; 0 with no
    /// samples, as the JVM's mean of nothing is.
    pub fn accuracy(&self) -> f64 {
        let correct: usize = self.folds.iter().map(|f| f.correct).sum();
        match self.document_count() {
            0 => 0.0,
            n => correct as f64 / n as f64,
        }
    }

    /// `getDocumentAccuracy()`: each fold's accuracy times its document
    /// count summed, over the document count, as the JVM's `Mean` rounds
    /// it. Opt-in, for comparing with the JVM; it can differ from
    /// [`accuracy`](Self::accuracy) in the last bits.
    pub fn jvm_accuracy(&self) -> f64 {
        let mut mean = Mean::default();
        for f in &self.folds {
            mean.add_times(f.accuracy, f.test_samples as u64);
        }
        mean.mean()
    }
}

/// `new LanguageDetectorCrossValidator(params, factory)`.
#[derive(Debug)]
pub struct LanguageDetectorCrossValidator<'f> {
    factory: &'f LanguageDetectorFactory,
    params: TrainingParameters,
}

impl<'f> LanguageDetectorCrossValidator<'f> {
    pub fn new(factory: &'f LanguageDetectorFactory, params: TrainingParameters) -> LanguageDetectorCrossValidator<'f> {
        LanguageDetectorCrossValidator { factory, params }
    }

    /// `evaluate(samples, nFolds)`. Training runs under `config`; each
    /// fold's model is stamped with `stamp`, which only its manifest
    /// shows.
    ///
    /// # Errors
    ///
    /// What the trainer refuses for a fold (training samples of one
    /// language only, say), and a model that cannot be read back.
    pub fn evaluate(
        &self,
        samples: &[LanguageSample<'_>],
        folds: usize,
        config: &mut TrainingConfiguration,
        stamp: Stamp<'_>,
    ) -> Result<LanguageCrossValidation> {
        let mut out = Vec::new();
        for fold in CrossValidation::new(samples.len(), folds) {
            let (events, inexact_events) = language_events(fold.training.iter().map(|&i| &samples[i]), self.factory);
            let trained = train_language_model_from_events(&events, self.factory, &self.params, config, stamp)?;
            let file = ModelFile::open(trained.package.to_bytes()).map_err(Error::Model)?;
            let detector = LanguageDetector::new(LanguageModel::open(file)?);
            let mut scratch = Scratch::new();
            let mut evaluator = LanguageDetectorEvaluator::default();
            let mut predictions = Vec::with_capacity(fold.test.len());
            let mut correct = 0;
            for &i in &fold.test {
                let reference = &samples[i];
                let code = detector.predict_language(reference.context.as_str(), &mut scratch).code.to_owned();
                if evaluator.evaluate(reference, &code) == Verdict::Correct {
                    correct += 1;
                }
                predictions.push(code);
            }
            out.push(LanguageFold {
                index: fold.index,
                training_samples: fold.training.len(),
                test_samples: fold.test.len(),
                events: events.len(),
                event_checksum: trained.report.event_hash.unwrap_or_else(|| checksum(&events)),
                inexact_events,
                correct,
                accuracy: evaluator.accuracy(),
                predictions,
            });
        }
        Ok(LanguageCrossValidation { folds: out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fold(correct: usize, test_samples: usize) -> LanguageFold {
        LanguageFold {
            index: 0,
            training_samples: 0,
            test_samples,
            events: 0,
            event_checksum: 0,
            inexact_events: 0,
            correct,
            accuracy: correct as f64 / test_samples as f64,
            predictions: Vec::new(),
        }
    }

    #[test]
    fn the_jvm_sum_of_weighted_accuracies_can_round_away_from_the_exact_share() {
        // 15 / 22 * 22 is 14.999999999999998.
        let cv = LanguageCrossValidation { folds: vec![fold(15, 22), fold(1, 1)] };
        assert_eq!(cv.document_count(), 23);
        assert_eq!(cv.accuracy(), 16.0 / 23.0);
        assert_eq!(cv.jvm_accuracy(), (15.0f64 / 22.0 * 22.0 + 1.0) / 23.0);
        assert_ne!(cv.accuracy(), cv.jvm_accuracy());
    }

    #[test]
    fn no_folds_give_zero_as_the_jvm_mean_of_nothing() {
        let cv = LanguageCrossValidation { folds: Vec::new() };
        assert_eq!((cv.accuracy(), cv.jvm_accuracy()), (0.0, 0.0));
    }
}
