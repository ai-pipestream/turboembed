//! The English sentence detector and tokenizer trained on MASC, and their
//! gates against the OpenNLP English UD models.
//!
//! Each recipe trains on the training documents of [`crate::masc::SPLIT`]
//! with the factory defaults and the parameters the published UD 1.3
//! models were trained with ([`turbo_train_components::ud`]: GIS,
//! 300 iterations, cutoff 5, 4 threads, OpenNLP 2.5.4's whitespace), so
//! the trained model and the reference differ in their data, not their
//! settings. Training uses no randomness, so the recipe has no seed.
//!
//! The samples are what OpenNLP's MASC streams read, with one change: the
//! token stream skips a sentence without a sample alone
//! ([`EmptySentence::Skipped`]) instead of ending the corpus there, so one
//! such sentence does not lose the rest of the training data. The
//! sentence stream joins [`SENTENCES_PER_SAMPLE`] sentences with a space,
//! as `SentenceDetectorTrainer` reads MASC.
//!
//! Both models are scored on the same held-out samples with OpenNLP's
//! evaluators (`SentenceDetectorEvaluator`, `TokenizerEvaluator`), one
//! unit per sample, and gated by [`turbo_eval::model_gate`].

use std::sync::Arc;

use turbo_analysis::Span;
use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, PairedUnit, Task};
use turbo_formats::evaluate::{SentenceDetectorEvaluator, TokenizerEvaluator};
use turbo_formats::masc::{EmptySentence, MascCorpus, MascSentenceReader, MascTokenReader};
use turbo_formats::sample::{SentenceSample, TokenSample};
use turbo_maxent::ModelFile;
use turbo_sentdetect::WhitespaceMode;
use turbo_sentdetect::{SentenceDetector, SentenceModel};
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::{TokenizerMe, TokenizerModel};
use turbo_train::TrainingConfiguration;
use turbo_train_components::ud;
use turbo_train_components::{SentenceDetectorFactory, Stamp, TokenizerFactory, sentdetect, tokenize};
use turbo_zoo::{Index, Sha256};

use crate::error::{Error, Result};

/// The segmentation component a recipe trains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    Sentence,
    Tokens,
}

/// The language of the models.
pub const LANGUAGE: &str = "en";
/// Sentences per sentence detector sample, as the UD recipes use.
pub const SENTENCES_PER_SAMPLE: usize = ud::SENTENCES_PER_SAMPLE;
/// The whitespace definition training and scoring use: OpenNLP 2.5.4's,
/// which the reference models were trained with.
pub const WHITESPACE: WhitespaceMode = ud::WHITESPACE;
/// The `Timestamp` written into the trained models' manifests, so a
/// recipe gives the same bytes on every run: 2026-10-14T00:00:00Z.
pub const TIMESTAMP_MS: i64 = 1_791_936_000_000;

/// One trained model and the OpenNLP model it is gated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentationRecipe {
    pub component: Component,
    /// The trained model's zoo id, also its file name without `.bin`.
    pub model_id: &'static str,
    /// The reference model's id in the zoo's built-in index.
    pub reference_id: &'static str,
}

pub const SENTENCE: SegmentationRecipe = SegmentationRecipe {
    component: Component::Sentence,
    model_id: "turbo-en-masc-3.0.0-sentence-1",
    reference_id: "opennlp-en-ud-ewt-sentence-1.3-2.5.4",
};

pub const TOKENS: SegmentationRecipe = SegmentationRecipe {
    component: Component::Tokens,
    model_id: "turbo-en-masc-3.0.0-tokens-1",
    reference_id: "opennlp-en-ud-ewt-tokens-1.3-2.5.4",
};

/// The recipes, sentence detector first.
pub const RECIPES: [SegmentationRecipe; 2] = [SENTENCE, TOKENS];

/// The held-out samples of one recipe, borrowed from the held-out corpus.
#[derive(Debug, Clone)]
pub enum HeldOut<'c> {
    Sentences(Vec<SentenceSample<'c>>),
    Tokens(Vec<TokenSample<'c>>),
}

impl HeldOut<'_> {
    pub fn len(&self) -> usize {
        match self {
            HeldOut::Sentences(s) => s.len(),
            HeldOut::Tokens(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The SHA-256 of the samples as scored: for each, its text's length
    /// in bytes as a little endian `u64`, the text, the number of
    /// reference spans as a `u64`, and each span's start and end as little
    /// endian `u32`.
    pub fn sha256(&self) -> String {
        let mut bytes = Vec::new();
        let mut put = |text: &str, spans: &[Span]| {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
            bytes.extend_from_slice(&(spans.len() as u64).to_le_bytes());
            for s in spans {
                bytes.extend_from_slice(&s.start.to_le_bytes());
                bytes.extend_from_slice(&s.end.to_le_bytes());
            }
        };
        match self {
            HeldOut::Sentences(s) => s.iter().for_each(|s| put(s.document(), s.sentences())),
            HeldOut::Tokens(t) => t.iter().for_each(|t| put(t.text(), t.tokens())),
        }
        Sha256::of(&bytes).to_string()
    }
}

/// A model a recipe trained, and what it was trained from.
#[derive(Debug, Clone)]
pub struct Trained {
    /// The `.bin` file.
    pub bytes: Vec<u8>,
    /// The samples read from the training documents.
    pub samples: usize,
    /// Sentences the token stream skipped for having no sample (always 0
    /// for the sentence detector).
    pub skipped_sentences: usize,
    pub events: usize,
    /// `Training-Eventhash`.
    pub event_hash: u32,
}

/// The factories of the recipes: the language, every other setting at its
/// default, as the trainer tools build them.
fn sentence_factory() -> Result<SentenceDetectorFactory> {
    Ok(SentenceDetectorFactory::new(LANGUAGE, true, None, None)?)
}

fn tokenizer_factory() -> Result<TokenizerFactory> {
    Ok(TokenizerFactory::new(LANGUAGE, None, false, None)?)
}

impl SegmentationRecipe {
    /// The gate's task.
    pub fn task(&self) -> Task {
        match self.component {
            Component::Sentence => Task::SentenceDetector,
            Component::Tokens => Task::Tokenizer,
        }
    }

    /// Trains the model from the training documents.
    ///
    /// # Errors
    ///
    /// A sample the corpus fails on, a corpus without a sample
    /// ([`Error::EmptySplit`]), or what training fails on.
    pub fn train(&self, train: &MascCorpus) -> Result<Trained> {
        let stamp = Stamp::at(TIMESTAMP_MS);
        let params = ud::training_parameters();
        let mut config = TrainingConfiguration::default();
        let mut events = Vec::new();
        let (samples, skipped_sentences, package, report) = match self.component {
            Component::Sentence => {
                let factory = sentence_factory()?;
                let mut stream = sentdetect::SdEventStream::new(&factory, WHITESPACE);
                let mut samples = 0;
                for sample in MascSentenceReader::new(train, SENTENCES_PER_SAMPLE) {
                    stream.events(&sample?.sample, &mut events);
                    samples += 1;
                }
                if samples == 0 {
                    return Err(Error::EmptySplit { what: "training sentence" });
                }
                let t = sentdetect::train_sentence_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (samples, 0, t.package, t.report)
            }
            Component::Tokens => {
                let factory = tokenizer_factory()?;
                let mut stream = tokenize::TokSpanEventStream::new(&factory, WHITESPACE);
                let mut reader = MascTokenReader::with_empty_sentences(train, EmptySentence::Skipped)?;
                let mut samples = 0;
                while let Some(sample) = reader.read()? {
                    stream.events(&sample.sample, &mut events)?;
                    samples += 1;
                }
                if samples == 0 {
                    return Err(Error::EmptySplit { what: "training token sample" });
                }
                let t = tokenize::train_tokenizer_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (samples, reader.skipped(), t.package, t.report)
            }
        };
        Ok(Trained {
            bytes: package.to_bytes(),
            samples,
            skipped_sentences,
            events: events.len(),
            event_hash: report.event_hash.expect("trained from events"),
        })
    }

    /// The held-out samples, read as the training samples are.
    ///
    /// # Errors
    ///
    /// A sample the corpus fails on, or no sample ([`Error::EmptySplit`]).
    pub fn held_out<'c>(&self, held_out: &'c MascCorpus) -> Result<HeldOut<'c>> {
        let samples = match self.component {
            Component::Sentence => HeldOut::Sentences(
                MascSentenceReader::new(held_out, SENTENCES_PER_SAMPLE)
                    .map(|s| s.map(|s| s.sample))
                    .collect::<std::result::Result<_, _>>()?,
            ),
            Component::Tokens => {
                let mut reader = MascTokenReader::with_empty_sentences(held_out, EmptySentence::Skipped)?;
                let mut tokens = Vec::new();
                while let Some(s) = reader.read()? {
                    tokens.push(s.sample);
                }
                HeldOut::Tokens(tokens)
            }
        };
        if samples.is_empty() {
            return Err(Error::EmptySplit { what: "held-out sample" });
        }
        Ok(samples)
    }

    /// Each held-out sample's counts for the model in `bytes`.
    ///
    /// # Errors
    ///
    /// A model file that is not a model of the recipe's component, or a
    /// prediction the evaluator throws on.
    pub fn score(&self, bytes: &[u8], held_out: &HeldOut<'_>) -> Result<Vec<Counts>> {
        let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
        let counts = |f: &turbo_formats::evaluate::FMeasure| Counts {
            true_positives: f.true_positives(),
            selected: f.selected(),
            target: f.target(),
        };
        let mut out = Vec::with_capacity(held_out.len());
        let mut predicted = Vec::new();
        let mut spans: Vec<Span> = Vec::new();
        match (self.component, held_out) {
            (Component::Sentence, HeldOut::Sentences(samples)) => {
                let model = SentenceModel::open(file).map_err(|e| Error::Model(e.to_string()))?;
                let detector = SentenceDetector::new(Arc::new(model), WHITESPACE);
                for sample in samples {
                    predicted.clear();
                    detector.detect(sample.document(), &mut predicted);
                    spans.clear();
                    spans.extend(predicted.iter().map(|&(s, _)| s));
                    let mut evaluator = SentenceDetectorEvaluator::new(WHITESPACE);
                    evaluator.evaluate(sample, &spans)?;
                    out.push(counts(evaluator.fmeasure()));
                }
            }
            (Component::Tokens, HeldOut::Tokens(samples)) => {
                let model = TokenizerModel::open(file).map_err(|e| Error::Model(e.to_string()))?;
                let tokenizer =
                    TokenizerMe::new(model, TokenizerOptions { whitespace: WHITESPACE, keep_newlines: false });
                for sample in samples {
                    predicted.clear();
                    tokenizer.tokenize(sample.text(), &mut predicted);
                    spans.clear();
                    spans.extend(predicted.iter().map(|&(s, _)| s));
                    let mut evaluator = TokenizerEvaluator::default();
                    evaluator.evaluate(sample, &spans)?;
                    out.push(counts(evaluator.fmeasure()));
                }
            }
            (component, samples) => {
                let name = |c: Component| match c {
                    Component::Sentence => "sentence detector",
                    Component::Tokens => "tokenizer",
                };
                let samples = match samples {
                    HeldOut::Sentences(_) => "sentence",
                    HeldOut::Tokens(_) => "token",
                };
                return Err(Error::ComponentMismatch { recipe: name(component), samples });
            }
        }
        Ok(out)
    }

    /// Scores the trained model and the reference on the same held-out
    /// samples and runs the gate. `reference_id` names the reference in
    /// the record ([`SegmentationRecipe::reference_id`] for the published
    /// model), and `set_name` and `corpus_sha256` the held-out set.
    ///
    /// # Errors
    ///
    /// What [`SegmentationRecipe::score`] fails on, or the gate's errors.
    pub fn gate(
        &self,
        trained: &[u8],
        reference_id: &str,
        reference: &[u8],
        held_out: &HeldOut<'_>,
        set_name: &str,
        corpus_sha256: &str,
    ) -> Result<ModelGateRecord> {
        let candidate = self.score(trained, held_out)?;
        let baseline = self.score(reference, held_out)?;
        let units = candidate
            .into_iter()
            .zip(baseline)
            .map(|(candidate, reference)| PairedUnit { candidate, reference })
            .collect();
        let set = HeldOutSet {
            name: set_name.to_owned(),
            corpus_sha256: corpus_sha256.to_owned(),
            samples_sha256: held_out.sha256(),
        };
        let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
        ModelGateRecord::new(self.task(), set, id(self.model_id, trained), id(reference_id, reference), units)
            .map_err(Error::Gate)
    }

    /// Checks that `bytes` are the reference model the zoo's built-in
    /// index lists.
    ///
    /// # Errors
    ///
    /// [`Error::ReferenceDigest`] for other bytes.
    pub fn check_reference(&self, bytes: &[u8]) -> Result<()> {
        let index = Index::builtin();
        let entry = index.get(self.reference_id)?;
        let expected = entry.files[0].sha256.to_string();
        let found = Sha256::of(bytes).to_string();
        if expected == found {
            Ok(())
        } else {
            Err(Error::ReferenceDigest { id: self.reference_id.to_owned(), expected, found })
        }
    }

    /// The reference model's file name in the zoo's built-in index.
    pub fn reference_file(&self) -> Result<String> {
        let index = Index::builtin();
        Ok(index.get(self.reference_id)?.files[0].path.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recipes_name_models_the_zoo_lists_as_references() {
        for r in RECIPES {
            assert!(r.reference_file().unwrap().ends_with(".bin"));
            assert!(matches!(r.check_reference(b"not the model"), Err(Error::ReferenceDigest { .. })));
            let id: turbo_zoo::ArtifactId = r.model_id.parse().unwrap();
            assert_eq!(id.as_str(), r.model_id);
        }
        assert_eq!(SENTENCE.task(), Task::SentenceDetector);
        assert_eq!(TOKENS.task(), Task::Tokenizer);
    }
}
