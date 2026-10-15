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
//! The samples are what OpenNLP's MASC streams read, with two changes
//! ([`read_token_samples`]): the token stream skips a sentence without a
//! sample alone ([`EmptySentence::Skipped`]) instead of ending the corpus
//! there, so one such sentence does not lose the rest of the training
//! data; and it skips a sentence whose tokens are out of text order
//! ([`in_text_order`]), which [`tokenize::TokSpanEventStream`] refuses. It
//! skips held-out sentences the same way; the card records both counts,
//! and a recipe fails when either is more than
//! [`crate::MAX_SKIPPED_SENTENCES`] of the sentences read. The sentence
//! stream joins [`SENTENCES_PER_SAMPLE`] sentences with a space, as
//! `SentenceDetectorTrainer` reads MASC.
//!
//! MASC 3.0.0 has one sentence whose tokens are out of text order, in
//! `data/spoken/face-to-face/Bed012`: its Penn layer has a token
//! `non-deterministic` over three quarks and a token `-` over the second
//! of them, inside the first. OpenNLP's `TokSpanEventStream` makes events
//! of such a sentence (for the inner token's end, a split inside the
//! outer token where it also gave no-split events); the port refuses it
//! (`TokensOutOfOrder`), and the recipe skips it.
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
use turbo_zoo::Sha256;

use crate::card::HeldOutCard;
use crate::error::{Error, LeftOut, Result};
use crate::{GateRun, Trained, check_skipped};

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
    Tokens {
        samples: Vec<TokenSample<'c>>,
        /// The held-out sentences [`read_token_samples`] skipped.
        skipped: usize,
    },
}

impl HeldOut<'_> {
    pub fn len(&self) -> usize {
        match self {
            HeldOut::Sentences(s) => s.len(),
            HeldOut::Tokens { samples, .. } => samples.len(),
        }
    }

    /// The held-out sentences the stream skipped: always 0 for the
    /// sentence detector's.
    pub fn skipped(&self) -> usize {
        match self {
            HeldOut::Sentences(_) => 0,
            HeldOut::Tokens { skipped, .. } => *skipped,
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
            HeldOut::Tokens { samples, .. } => samples.iter().for_each(|t| put(t.text(), t.tokens())),
        }
        Sha256::of(&bytes).to_string()
    }
}

/// Whether every token starts and ends at or after the token before it,
/// as [`tokenize::TokSpanEventStream`] asks of a sample: a token inside
/// the one before it, or before it, is out of text order.
pub fn in_text_order(tokens: &[Span]) -> bool {
    tokens.windows(2).all(|w| w[1].start >= w[0].start && w[1].end >= w[0].end)
}

/// Reads the token samples of `corpus` as `MascTokenSampleStream` reads
/// them, giving each to `each`, except that a sentence without a sample
/// ([`EmptySentence::Skipped`]) is skipped alone, and so is a sentence
/// whose tokens are out of text order ([`in_text_order`]). Returns how
/// many were skipped.
///
/// # Errors
///
/// Any other error of the stream, which ends the read, or what `each`
/// fails on.
pub fn read_token_samples<'c>(
    corpus: &'c MascCorpus,
    mut each: impl FnMut(TokenSample<'c>) -> Result<()>,
) -> Result<usize> {
    let mut reader = MascTokenReader::with_empty_sentences(corpus, EmptySentence::Skipped)?;
    let mut out_of_order = 0;
    while let Some(read) = reader.read()? {
        if in_text_order(read.sample.tokens()) {
            each(read.sample)?;
        } else {
            out_of_order += 1;
        }
    }
    Ok(reader.skipped() + out_of_order)
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
    /// ([`Error::EmptySplit`]), more skipped sentences than
    /// [`crate::MAX_SKIPPED_SENTENCES`] allows ([`Error::TooManyLeftOut`]),
    /// or what training fails on.
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
                let mut samples = 0;
                let skipped = read_token_samples(train, |sample| {
                    stream.events(&sample, &mut events)?;
                    samples += 1;
                    Ok(())
                })?;
                if samples == 0 {
                    return Err(Error::EmptySplit { what: "training token sample" });
                }
                check_skipped(LeftOut::TrainingSentences, skipped, samples)?;
                let t = tokenize::train_tokenizer_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (samples, skipped, t.package, t.report)
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
    /// A sample the corpus fails on, no sample ([`Error::EmptySplit`]), or
    /// more skipped sentences than [`crate::MAX_SKIPPED_SENTENCES`] allows
    /// ([`Error::TooManyLeftOut`]).
    pub fn held_out<'c>(&self, held_out: &'c MascCorpus) -> Result<HeldOut<'c>> {
        let samples = match self.component {
            Component::Sentence => HeldOut::Sentences(
                MascSentenceReader::new(held_out, SENTENCES_PER_SAMPLE)
                    .map(|s| s.map(|s| s.sample))
                    .collect::<std::result::Result<_, _>>()?,
            ),
            Component::Tokens => {
                let mut tokens = Vec::new();
                let skipped = read_token_samples(held_out, |s| {
                    tokens.push(s);
                    Ok(())
                })?;
                HeldOut::Tokens { samples: tokens, skipped }
            }
        };
        if samples.is_empty() {
            return Err(Error::EmptySplit { what: "held-out sample" });
        }
        check_skipped(LeftOut::HeldOutSentences, samples.skipped(), samples.len())?;
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
            (Component::Tokens, HeldOut::Tokens { samples, .. }) => {
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
                    HeldOut::Tokens { .. } => "token",
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
    ) -> Result<GateRun> {
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
        let record =
            ModelGateRecord::new(self.task(), set, id(self.model_id, trained), id(reference_id, reference), units)
                .map_err(Error::Gate)?;
        let held_out = HeldOutCard { samples: held_out.len(), skipped_sentences: held_out.skipped(), tokens: None };
        Ok(GateRun { record, held_out })
    }

    /// Checks that `bytes` are the reference model the zoo's built-in
    /// index lists.
    ///
    /// # Errors
    ///
    /// [`Error::ReferenceDigest`] for other bytes.
    pub fn check_reference(&self, bytes: &[u8]) -> Result<()> {
        crate::Reference::Zoo(self.reference_id).check(bytes)
    }

    /// The reference model's file name in the zoo's built-in index.
    pub fn reference_file(&self) -> Result<String> {
        crate::Reference::Zoo(self.reference_id).file()
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

    #[test]
    fn tokens_are_in_text_order_unless_one_starts_or_ends_before_the_one_before_it() {
        let spans = |s: &[(u32, u32)]| s.iter().map(|&(a, b)| Span::new(a, b)).collect::<Vec<_>>();
        assert!(in_text_order(&[]));
        assert!(in_text_order(&spans(&[(0, 3)])));
        assert!(in_text_order(&spans(&[(0, 3), (4, 5), (5, 5), (5, 6)])));
        // Overlapping tokens in order are taken, as the event stream takes them.
        assert!(in_text_order(&spans(&[(0, 3), (2, 5)])));
        // MASC 3.0.0's `non-deterministic` and the `-` inside it.
        assert!(!in_text_order(&spans(&[(45, 62), (48, 49), (62, 63)])));
        assert!(!in_text_order(&spans(&[(4, 5), (0, 3)])));
        assert!(!in_text_order(&spans(&[(3, 6), (2, 7)])));
    }
}
