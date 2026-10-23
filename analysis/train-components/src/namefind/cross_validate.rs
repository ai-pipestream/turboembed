//! `TokenNameFinderCrossValidator`: n-fold cross-validation of the name
//! finder over whole documents.
//!
//! As `TokenNameFinderCrossValidator.evaluate` does, the samples are first
//! grouped into documents (`NameToDocumentSampleStream`): a document starts
//! at the first sample and at every later sample that clears adaptive
//! data. [`CrossValidation`] partitions the documents, so a fold never
//! splits one. For each fold the name finder is trained on the training
//! documents' samples ([`train_name_model_from_events`], the events of
//! [`NameEventStream`]), read back from its model file, and run over the
//! test documents' samples, its adaptive data cleared at every sample that
//! clears it (`TokenNameFinderEvaluator.processSample`); the predictions
//! are scored with [`NameFinderEvaluator`] and the folds' counts merged
//! (`FMeasure.mergeInto`). Counts merge exactly, so the merged precision,
//! recall and F-measure are the JVM's with no opt-in.
//!
//! Each fold also records what the JVM's cross-validator keeps to itself:
//! the documents and samples on each side, the training events and their
//! checksum (`Training-Eventhash`), the fold's counts and every predicted
//! name, so a fold can be compared with the JVM's one by one.

use std::ops::Range;

use turbo_formats::evaluate::{CrossValidation, FMeasure, NameFinderEvaluator, Verdict};
use turbo_formats::{Name, NameSample};
use turbo_maxent::ModelFile;
use turbo_sequence::PatternMode;
use turbo_sequence::namefind::NameFinder;
use turbo_train::event::checksum;
use turbo_train::{TrainingConfiguration, TrainingParameters};

use super::{MODEL_ENTRY, NameEventStream, TokenNameFinderFactory, train_name_model_from_events};
use crate::error::Result;
use crate::package::Stamp;

/// `NameToDocumentSampleStream`: the documents of `samples`, as ranges of
/// sample indices in order. The first document starts at the first sample
/// whatever its flag; every later sample that clears adaptive data starts
/// a new one.
pub fn documents(samples: &[NameSample<'_>]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, sample) in samples.iter().enumerate().skip(1) {
        if sample.clear_adaptive_data() {
            out.push(start..i);
            start = i;
        }
    }
    if !samples.is_empty() {
        out.push(start..samples.len());
    }
    out
}

/// A name the fold's finder predicted for one test sample: tokens
/// `start..end` and the type the model's outcomes give it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredictedName {
    pub start: usize,
    pub end: usize,
    pub kind: Option<String>,
}

/// One fold: what it trained on, what it tested on, and how it scored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameFold {
    /// The fold's number, from 0.
    pub index: usize,
    pub training_documents: usize,
    pub training_samples: usize,
    pub test_documents: usize,
    pub test_samples: usize,
    /// The training events `NameFinderME.train` made.
    pub events: usize,
    /// Their CRC32C, the model's `Training-Eventhash`.
    pub event_checksum: u32,
    /// The fold's counts, `TokenNameFinderEvaluator.getFMeasure()`.
    pub fmeasure: FMeasure,
    /// Test samples whose predicted names equal the reference's (what the
    /// evaluation monitors hear as correctly classified).
    pub correct_samples: usize,
    /// The predicted names of each test sample, in test order.
    pub predictions: Vec<Vec<PredictedName>>,
}

/// Every fold, and their counts merged (`getFMeasure()`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameCrossValidation {
    pub folds: Vec<NameFold>,
    pub fmeasure: FMeasure,
}

/// `new TokenNameFinderCrossValidator(languageCode, type, params,
/// factory)`: the factory gives the language, the feature generators and
/// the codec; `kind` overrides the samples' name types when given.
#[derive(Debug)]
pub struct TokenNameFinderCrossValidator<'f> {
    factory: &'f TokenNameFinderFactory,
    kind: Option<String>,
    params: TrainingParameters,
}

impl<'f> TokenNameFinderCrossValidator<'f> {
    pub fn new(
        factory: &'f TokenNameFinderFactory,
        kind: Option<&str>,
        params: TrainingParameters,
    ) -> TokenNameFinderCrossValidator<'f> {
        TokenNameFinderCrossValidator { factory, kind: kind.map(str::to_owned), params }
    }

    /// `evaluate(samples, nFolds)`. Training runs under `config`; each
    /// fold's model is stamped with `stamp`, which only its manifest
    /// shows.
    ///
    /// # Errors
    ///
    /// What [`train_name_model_from_events`] refuses for a fold (a name the
    /// codec cannot encode, a trainer that does not train from events),
    /// and a prediction the evaluator refuses.
    pub fn evaluate(
        &self,
        samples: &[NameSample<'_>],
        folds: usize,
        config: &mut TrainingConfiguration,
        stamp: Stamp<'_>,
    ) -> Result<NameCrossValidation> {
        let docs = documents(samples);
        let mut out = Vec::new();
        let mut merged = FMeasure::default();
        for fold in CrossValidation::new(docs.len(), folds) {
            let training = flatten(&docs, &fold.training);
            let test = flatten(&docs, &fold.test);
            let mut stream = NameEventStream::new(self.factory, self.kind.as_deref())?;
            let mut events = Vec::new();
            for &i in &training {
                stream.events(&samples[i], &mut events)?;
            }
            let trained = train_name_model_from_events(&events, self.factory, &self.params, config, stamp)?;
            let file = ModelFile::open(trained.package.to_bytes()).map_err(crate::Error::Model)?;
            let maxent = file.maxent(MODEL_ENTRY).map_err(crate::Error::Model)?;
            let mut finder = NameFinder::new(&file, &maxent, PatternMode::Utf16Units)?;
            let mut evaluator = NameFinderEvaluator::default();
            let mut predictions = Vec::with_capacity(test.len());
            let mut correct_samples = 0;
            for &i in &test {
                let reference = &samples[i];
                if reference.clear_adaptive_data() {
                    finder.clear_adaptive_data();
                }
                let tokens: Vec<&str> = reference.tokens().iter().map(|t| t.as_str()).collect();
                let found: Vec<PredictedName> = finder
                    .find(&tokens)
                    .into_iter()
                    .map(|f| PredictedName { start: f.span.start, end: f.span.end, kind: f.span.kind })
                    .collect();
                let names = found
                    .iter()
                    .map(|p| Name::new(p.start as u32, p.end as u32, p.kind.as_deref()))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                if evaluator.evaluate(reference, &names)? == Verdict::Correct {
                    correct_samples += 1;
                }
                predictions.push(found);
            }
            let fmeasure = *evaluator.fmeasure();
            merged.merge(&fmeasure);
            out.push(NameFold {
                index: fold.index,
                training_documents: fold.training.len(),
                training_samples: training.len(),
                test_documents: fold.test.len(),
                test_samples: test.len(),
                events: events.len(),
                event_checksum: trained.report.event_hash.unwrap_or_else(|| checksum(&events)),
                fmeasure,
                correct_samples,
                predictions,
            });
        }
        Ok(NameCrossValidation { folds: out, fmeasure: merged })
    }
}

/// The sample indices of `documents`, in document order.
fn flatten(docs: &[Range<usize>], documents: &[usize]) -> Vec<usize> {
    documents.iter().flat_map(|&d| docs[d].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_formats::native::NameReader;
    use turbo_unicode::WhitespaceMode;

    #[test]
    fn documents_start_at_the_first_sample_and_at_each_clear() {
        let text = "<START:person> A <END> b .\nc d .\n\n<START:person> E <END> f .\n\ng .\nh .\n";
        let samples: Vec<_> = NameReader::new(text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect();
        assert_eq!(documents(&samples), vec![0..2, 2..3, 3..5]);
        assert!(documents(&[]).is_empty());
    }
}
