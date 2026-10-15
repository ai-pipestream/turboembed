//! Recipes that train the sentence detector, tokenizer, part-of-speech
//! tagger and lemmatizer on the CC BY 4.0 treebanks of [`crate::treebank`]
//! (issue #415), and their gates against the OpenNLP UD 1.3 models.
//!
//! Each recipe trains on the treebank's UD `train` file and is gated on
//! its UD `test` file; the `dev` file is checked against its digest and
//! not read. Training is what OpenNLP 2.5.4's CoNLL-U trainer tools do,
//! with the parameters the published UD 1.3 models were trained with
//! ([`turbo_train_components::ud`]: GIS, 300 iterations, cutoff 5,
//! 4 threads, the factory defaults for the language, OpenNLP 2.5.4's
//! whitespace, 10 sentences per sentence detector sample, the universal
//! tags), so a trained model and its reference differ in their data, not
//! their settings. Training uses no randomness, so a recipe has no seed.
//!
//! The samples are those of OpenNLP's CoNLL-U streams, with one change:
//! the tagger's and the lemmatizer's samples leave out empty nodes (ids
//! like `6.1`), which `ConlluPOSSampleStream` and
//! `ConlluLemmaSampleStream` keep. An empty node is not a word of the
//! text (AnCora writes its form `_`, Porttinari `__NOFORM__`), and
//! scoring one would count a token no text has; a model trained with them
//! learns to tag `_`, which on a held-out set with them would favour a
//! model trained on the same treebank over one trained on another.
//!
//! A treebank has a recipe for each component it annotates. FTB and
//! MarkIT have no sentence detector and no tokenizer: their `text`
//! comments put a space between every two tokens and their words carry
//! no `SpaceAfter=No`, so they hold no tokenization or sentence ends of
//! running text.
//!
//! Each model is gated against the OpenNLP UD 1.3 model for its language
//! and component ([`UdReference::Zoo`]), both scored on the same `test`
//! samples with OpenNLP's evaluators. Except for Catalan, the reference
//! was trained on another treebank, whose tagger may not have every tag
//! the `test` file uses: FTB tags determiners `DET` and particles `PART`,
//! and the Finnish reference has neither. A token with a tag the
//! reference has no outcome for is one it cannot get right, so such tokens
//! are left out of both taggers' counts, and the card lists their tags and
//! how many there were ([`tags_left_out`]); when they are more than
//! [`LEFT_OUT_LIMIT_PERCENT`] of the tokens, the gate refuses to compare
//! the two.
//!
//! For Italian there is no reference the recipes may use
//! ([`UdReference::NoReference`]): OpenNLP's Italian UD models are trained
//! on UD_Italian-VIT, which is CC BY-NC-SA 3.0, the data policy uses no
//! non-commercial data, and the zoo refuses those models. Such a model gets
//! an [`UngatedRecord`] and ships only as experimental.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, PairedUnit, Task, UngatedRecord};
use turbo_formats::FormatError;
use turbo_formats::conllu::{ConlluReader, ConlluSentenceReader, ConlluTokenReader, Sentence, Tagset, WordLine};
use turbo_formats::evaluate::LemmatizerEvaluator;
use turbo_formats::sample::{LemmaSample, PosSample};
use turbo_lemmatizer::{LemmatizerModel, MaxentLemmatizer, TokenTag, edit_script::decode_lemma};
use turbo_maxent::ModelFile;
use turbo_postag::PosTagFormat;
use turbo_train::TrainingConfiguration;
use turbo_train_components::ud::{self as published, Component};
use turbo_train_components::{
    LemmaEventStream, LemmatizerFactory, PosEventStream, PosTaggerFactory, SdEventStream, SentenceDetectorFactory,
    Stamp, TokSpanEventStream, TokenizerFactory, lemmatize, postag, sentdetect, tokenize,
};
use turbo_zoo::Sha256;

use crate::error::{Error, Result};
use crate::segmentation::{self, HeldOut, TIMESTAMP_MS, WHITESPACE};
use crate::treebank::TreebankText;
use crate::treebank::{
    CATALAN_ANCORA, FINNISH_FTB, ITALIAN_MARKIT, PORTUGUESE_PORTTINARI, RELEASE, SPANISH_ANCORA, Treebank,
};
use crate::ud_card::UdModelCard;
use crate::{Reference, Trained};

/// Sentences per sentence detector sample.
pub const SENTENCES_PER_SAMPLE: usize = published::SENTENCES_PER_SAMPLE;
/// The tag column the tagger and the lemmatizer read: the universal tags.
pub const TAGSET: Tagset = published::TAGSET;
/// The tags both taggers are scored in: their outcomes as they are, the
/// universal tags with a multiword token's tags joined by `+`, which both
/// were trained on.
pub const TAG_FORMAT: PosTagFormat = PosTagFormat::Custom;

/// The share of a tagger's held-out tokens, in percent, that may be left
/// out of the gate for tags the reference has no outcome for; more and
/// the gate refuses ([`Error::ReferenceTags`]).
pub const LEFT_OUT_LIMIT_PERCENT: f64 = 5.0;

/// A held-out tag the reference tagger has no outcome for, and how many
/// tokens have it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeftOutTag {
    pub tag: String,
    pub tokens: u64,
}

/// The tags of `samples` the tagger in `reference` (asked for
/// [`TAG_FORMAT`]) has no outcome for, with their token counts, in the
/// order of the tags' bytes, and the number of tokens of the samples.
///
/// # Errors
///
/// A model file that is not a part-of-speech model.
pub fn tags_left_out(reference: &[u8], samples: &[PosSample<'_>]) -> Result<(Vec<LeftOutTag>, u64)> {
    let model = crate::postag::open_tagger(reference, TAG_FORMAT)?;
    let tagset = model.tagset();
    let mut left_out: BTreeMap<&str, u64> = BTreeMap::new();
    let mut tokens = 0;
    for tag in samples.iter().flat_map(|s| s.tags()) {
        tokens += 1;
        if tagset.tag(tag.as_str()).is_none() {
            *left_out.entry(tag.as_str()).or_default() += 1;
        }
    }
    Ok((left_out.into_iter().map(|(tag, tokens)| LeftOutTag { tag: tag.to_owned(), tokens }).collect(), tokens))
}

/// The model a UD recipe's model is gated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdReference {
    /// An OpenNLP UD 1.3 model of the zoo's built-in index, by its id.
    Zoo(&'static str),
    /// No model the recipe may be gated against, and why: OpenNLP
    /// publishes none, or the one it publishes is trained on data the
    /// policy does not use.
    NoReference(&'static str),
}

/// Why the Italian recipes have no reference.
pub const NO_ITALIAN_REFERENCE: &str = "OpenNLP's Italian UD 1.3 models are trained on UD_Italian-VIT, which is CC \
     BY-NC-SA 3.0; the data policy uses no non-commercial data and the zoo refuses those models, so the model is \
     scored alone and ships only as experimental";

/// One model trained on a treebank and its reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdRecipe {
    /// The trained model's zoo id, also its file name without `.bin`.
    pub model_id: &'static str,
    pub treebank: Treebank,
    pub component: Component,
    pub reference: UdReference,
}

const fn recipe(model_id: &'static str, treebank: Treebank, component: Component, reference: UdReference) -> UdRecipe {
    UdRecipe { model_id, treebank, component, reference }
}

use Component::{Lemmas, Pos, Sentence as Sentences, Tokens};
use UdReference::{NoReference, Zoo};

/// The recipes, by treebank, in the order the binary runs them.
pub const UD_RECIPES: [UdRecipe; 16] = [
    recipe(
        "turbo-es-ud-ancora-2.16-sentence-1",
        SPANISH_ANCORA,
        Sentences,
        Zoo("opennlp-es-ud-gsd-sentence-1.3-2.5.4"),
    ),
    recipe("turbo-es-ud-ancora-2.16-tokens-1", SPANISH_ANCORA, Tokens, Zoo("opennlp-es-ud-gsd-tokens-1.3-2.5.4")),
    recipe("turbo-es-ud-ancora-2.16-pos-1", SPANISH_ANCORA, Pos, Zoo("opennlp-es-ud-gsd-pos-1.3-2.5.4")),
    recipe("turbo-es-ud-ancora-2.16-lemmas-1", SPANISH_ANCORA, Lemmas, Zoo("opennlp-es-ud-gsd-lemmas-1.3-2.5.4")),
    recipe(
        "turbo-ca-ud-ancora-2.16-sentence-1",
        CATALAN_ANCORA,
        Sentences,
        Zoo("opennlp-ca-ud-ancora-sentence-1.3-2.5.4"),
    ),
    recipe("turbo-ca-ud-ancora-2.16-tokens-1", CATALAN_ANCORA, Tokens, Zoo("opennlp-ca-ud-ancora-tokens-1.3-2.5.4")),
    recipe("turbo-ca-ud-ancora-2.16-pos-1", CATALAN_ANCORA, Pos, Zoo("opennlp-ca-ud-ancora-pos-1.3-2.5.4")),
    recipe("turbo-ca-ud-ancora-2.16-lemmas-1", CATALAN_ANCORA, Lemmas, Zoo("opennlp-ca-ud-ancora-lemmas-1.3-2.5.4")),
    recipe(
        "turbo-pt-ud-porttinari-2.16-sentence-1",
        PORTUGUESE_PORTTINARI,
        Sentences,
        Zoo("opennlp-pt-ud-gsd-sentence-1.3-2.5.4"),
    ),
    recipe(
        "turbo-pt-ud-porttinari-2.16-tokens-1",
        PORTUGUESE_PORTTINARI,
        Tokens,
        Zoo("opennlp-pt-ud-gsd-tokens-1.3-2.5.4"),
    ),
    recipe("turbo-pt-ud-porttinari-2.16-pos-1", PORTUGUESE_PORTTINARI, Pos, Zoo("opennlp-pt-ud-gsd-pos-1.3-2.5.4")),
    recipe(
        "turbo-pt-ud-porttinari-2.16-lemmas-1",
        PORTUGUESE_PORTTINARI,
        Lemmas,
        Zoo("opennlp-pt-ud-gsd-lemmas-1.3-2.5.4"),
    ),
    recipe("turbo-fi-ud-ftb-2.16-pos-1", FINNISH_FTB, Pos, Zoo("opennlp-fi-ud-tdt-pos-1.3-2.5.4")),
    recipe("turbo-fi-ud-ftb-2.16-lemmas-1", FINNISH_FTB, Lemmas, Zoo("opennlp-fi-ud-tdt-lemmas-1.3-2.5.4")),
    recipe("turbo-it-ud-markit-2.16-pos-1", ITALIAN_MARKIT, Pos, NoReference(NO_ITALIAN_REFERENCE)),
    recipe("turbo-it-ud-markit-2.16-lemmas-1", ITALIAN_MARKIT, Lemmas, NoReference(NO_ITALIAN_REFERENCE)),
];

/// The words of a sentence the tagger and the lemmatizer read: its word
/// lines without the empty nodes.
fn words<'s, 'a>(sentence: &'s Sentence<'a>) -> impl Iterator<Item = &'s WordLine<'a>> {
    sentence.word_lines.iter().filter(|l| !l.id.as_str().contains('.'))
}

/// Reads the part-of-speech samples of a CoNLL-U text: one per sentence,
/// its words' forms and universal tags, empty nodes left out. A sentence
/// with no word is skipped and counted.
///
/// # Errors
///
/// A sentence the CoNLL-U reader refuses.
pub fn pos_samples(conllu: &str) -> std::result::Result<(Vec<PosSample<'_>>, usize), FormatError> {
    let mut samples = Vec::new();
    let mut skipped = 0;
    for sentence in ConlluReader::new(conllu)? {
        let sentence = sentence?;
        let tokens: Vec<_> = words(&sentence).map(|l| l.form.clone()).collect();
        if tokens.is_empty() {
            skipped += 1;
            continue;
        }
        let tags = words(&sentence).map(|l| l.tag(TAGSET).clone()).collect();
        samples.push(PosSample::new(tokens, tags, sentence.source)?);
    }
    Ok((samples, skipped))
}

/// Reads the lemma samples of a CoNLL-U text as [`pos_samples`] reads the
/// part-of-speech samples, with each word's lemma.
///
/// # Errors
///
/// A sentence the CoNLL-U reader refuses.
pub fn lemma_samples(conllu: &str) -> std::result::Result<(Vec<LemmaSample<'_>>, usize), FormatError> {
    let mut samples = Vec::new();
    let mut skipped = 0;
    for sentence in ConlluReader::new(conllu)? {
        let sentence = sentence?;
        let tokens: Vec<_> = words(&sentence).map(|l| l.form.clone()).collect();
        if tokens.is_empty() {
            skipped += 1;
            continue;
        }
        let tags = words(&sentence).map(|l| l.tag(TAGSET).clone()).collect();
        let lemmas = words(&sentence).map(|l| l.lemma.clone()).collect();
        samples.push(LemmaSample::new(tokens, tags, lemmas, sentence.source)?);
    }
    Ok((samples, skipped))
}

/// The samples of one component read from one CoNLL-U file, borrowed
/// from its text.
#[derive(Debug, Clone)]
pub enum UdSamples<'c> {
    /// Sentence or token samples.
    Segments(HeldOut<'c>),
    Tags(Vec<PosSample<'c>>),
    Lemmas(Vec<LemmaSample<'c>>),
}

impl UdSamples<'_> {
    pub fn len(&self) -> usize {
        match self {
            UdSamples::Segments(s) => s.len(),
            UdSamples::Tags(t) => t.len(),
            UdSamples::Lemmas(l) => l.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The SHA-256 of the samples as scored: the segmentation samples'
    /// digest ([`HeldOut::sha256`]); for tags, each sample's number of
    /// tokens as a little endian `u64`, then for each token its text and
    /// its tag, each as its length in bytes as a little endian `u64` and
    /// the bytes; for lemmas the same with the lemma after the tag.
    pub fn sha256(&self) -> String {
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        };
        match self {
            UdSamples::Segments(s) => return s.sha256(),
            UdSamples::Tags(samples) => {
                for s in samples {
                    bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    for (token, tag) in s.tokens().iter().zip(s.tags()) {
                        put(&mut bytes, token.as_str());
                        put(&mut bytes, tag.as_str());
                    }
                }
            }
            UdSamples::Lemmas(samples) => {
                for s in samples {
                    bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    for ((token, tag), lemma) in s.tokens().iter().zip(s.tags()).zip(s.lemmas()) {
                        put(&mut bytes, token.as_str());
                        put(&mut bytes, tag.as_str());
                        put(&mut bytes, lemma.as_str());
                    }
                }
            }
        }
        Sha256::of(&bytes).to_string()
    }
}

/// The samples a recipe read from one file, and how many sentences it
/// left out.
#[derive(Debug, Clone)]
pub struct ReadSamples<'c> {
    pub samples: UdSamples<'c>,
    /// Sentences with no word once the empty nodes are left out (0 for the
    /// sentence detector and the tokenizer).
    pub skipped: usize,
}

/// What one UD recipe's gate made: a gate record, or the record of a
/// model scored alone.
#[derive(Debug, Clone, PartialEq)]
pub enum UdGate {
    /// The gate record, and for a tagger the held-out tags left out of
    /// both models' counts ([`tags_left_out`]).
    Gated {
        record: ModelGateRecord,
        tags_left_out: Vec<LeftOutTag>,
    },
    Ungated(UngatedRecord),
}

impl UdGate {
    /// Whether the model passed a gate: `false` for one scored alone.
    pub fn passed(&self) -> bool {
        match self {
            UdGate::Gated { record, .. } => record.result.pass,
            UdGate::Ungated(_) => false,
        }
    }

    /// The record as written to the gate file.
    pub fn to_json(&self) -> String {
        match self {
            UdGate::Gated { record, .. } => record.to_json(),
            UdGate::Ungated(record) => record.to_json(),
        }
    }
}

impl std::fmt::Display for UdGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UdGate::Gated { record, tags_left_out } => {
                if !tags_left_out.is_empty() {
                    let tags: Vec<String> = tags_left_out.iter().map(|t| format!("{} ({})", t.tag, t.tokens)).collect();
                    writeln!(f, "tokens left out, tags the reference has no outcome for: {}", tags.join(", "))?;
                }
                write!(f, "{record}")
            }
            UdGate::Ungated(record) => write!(f, "{record}"),
        }
    }
}

impl UdRecipe {
    /// The gate's task.
    pub fn task(&self) -> Task {
        match self.component {
            Component::Sentence => Task::SentenceDetector,
            Component::Tokens => Task::Tokenizer,
            Component::Pos => Task::PartOfSpeechTagger,
            Component::Lemmas => Task::Lemmatizer,
        }
    }

    /// The component the zoo lists the trained model under.
    pub fn zoo_component(&self) -> turbo_zoo::Component {
        match self.component {
            Component::Sentence => turbo_zoo::Component::SentenceDetector,
            Component::Tokens => turbo_zoo::Component::Tokenizer,
            Component::Pos => turbo_zoo::Component::Pos,
            Component::Lemmas => turbo_zoo::Component::Lemmatizer,
        }
    }

    /// The reference as a [`Reference`] whose file a run reads and checks;
    /// `None` when there is none.
    pub fn reference_model(&self) -> Option<Reference> {
        match self.reference {
            UdReference::Zoo(id) => Some(Reference::Zoo(id)),
            UdReference::NoReference(_) => None,
        }
    }

    /// The name of the held-out set in the records.
    pub fn held_out_name(&self) -> String {
        format!("{} {RELEASE} test ({})", self.treebank.repository, self.treebank.test.name)
    }

    /// Reads the recipe's samples from a CoNLL-U text, as the trainer
    /// tool reads them (with the change the module documentation gives).
    ///
    /// # Errors
    ///
    /// A sentence the reader refuses ([`Error::Conllu`], naming `file`).
    pub fn samples<'c>(&self, conllu: &'c str, file: &'static str) -> Result<ReadSamples<'c>> {
        let conllu_error =
            |error: FormatError| Error::Conllu { treebank: self.treebank.repository, file, error: Box::new(error) };
        let (samples, skipped) = match self.component {
            Component::Sentence => (
                UdSamples::Segments(HeldOut::Sentences(
                    ConlluSentenceReader::new(conllu, SENTENCES_PER_SAMPLE)
                        .map_err(conllu_error)?
                        .collect::<std::result::Result<_, _>>()
                        .map_err(conllu_error)?,
                )),
                0,
            ),
            Component::Tokens => (
                UdSamples::Segments(HeldOut::Tokens(
                    ConlluTokenReader::new(conllu, WHITESPACE)
                        .map_err(conllu_error)?
                        .collect::<std::result::Result<_, _>>()
                        .map_err(conllu_error)?,
                )),
                0,
            ),
            Component::Pos => {
                let (s, skipped) = pos_samples(conllu).map_err(conllu_error)?;
                (UdSamples::Tags(s), skipped)
            }
            Component::Lemmas => {
                let (s, skipped) = lemma_samples(conllu).map_err(conllu_error)?;
                (UdSamples::Lemmas(s), skipped)
            }
        };
        if samples.is_empty() {
            return Err(Error::EmptySplit { what: "sample of the treebank file" });
        }
        Ok(ReadSamples { samples, skipped })
    }

    /// Trains the model from the treebank's `train` file.
    ///
    /// # Errors
    ///
    /// What reading the samples or training fails on.
    pub fn train(&self, train: &str) -> Result<Trained> {
        let read = self.samples(train, self.treebank.train.name)?;
        let language = self.treebank.language;
        let stamp = Stamp::at(TIMESTAMP_MS);
        let params = published::training_parameters();
        let mut config = TrainingConfiguration::default();
        let mut events = Vec::new();
        let (bytes, report) = match &read.samples {
            UdSamples::Segments(HeldOut::Sentences(samples)) => {
                let factory = SentenceDetectorFactory::new(language, true, None, None)?;
                let mut stream = SdEventStream::new(&factory, WHITESPACE);
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = sentdetect::train_sentence_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Segments(HeldOut::Tokens(samples)) => {
                let factory = TokenizerFactory::new(language, None, false, None)?;
                let mut stream = TokSpanEventStream::new(&factory, WHITESPACE);
                for s in samples {
                    stream.events(s, &mut events)?;
                }
                let t = tokenize::train_tokenizer_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Tags(samples) => {
                let factory = PosTaggerFactory::new(language, None, None)?;
                let mut stream = PosEventStream::new(&factory)?;
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = postag::train_pos_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Lemmas(samples) => {
                let factory = LemmatizerFactory::new(language);
                let mut stream = LemmaEventStream::new(&factory);
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = lemmatize::train_lemmatizer_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
        };
        Ok(Trained {
            bytes,
            samples: read.samples.len(),
            skipped_sentences: read.skipped,
            events: events.len(),
            event_hash: report.event_hash.expect("trained from events"),
        })
    }

    /// Each held-out sample's counts for the model in `bytes`, scored
    /// with OpenNLP's evaluator for the component.
    ///
    /// # Errors
    ///
    /// A model file that is not a model of the component, samples of
    /// another component, or a prediction the evaluator throws on.
    pub fn score(&self, bytes: &[u8], held_out: &UdSamples<'_>) -> Result<Vec<Counts>> {
        match (self.component, held_out) {
            (Component::Sentence, UdSamples::Segments(s @ HeldOut::Sentences(_)))
            | (Component::Tokens, UdSamples::Segments(s @ HeldOut::Tokens(_))) => {
                segmentation::score_segments(bytes, s)
            }
            (Component::Pos, UdSamples::Tags(samples)) => crate::postag::score_tags(bytes, TAG_FORMAT, samples),
            (Component::Lemmas, UdSamples::Lemmas(samples)) => score_lemmas(bytes, samples),
            (component, samples) => Err(Error::ComponentMismatch {
                recipe: component_name(component),
                samples: match samples {
                    UdSamples::Segments(HeldOut::Sentences(_)) => "sentence",
                    UdSamples::Segments(HeldOut::Tokens(_)) => "token",
                    UdSamples::Tags(_) => "part-of-speech",
                    UdSamples::Lemmas(_) => "lemma",
                },
            }),
        }
    }

    /// Scores the trained model and, where there is one, the reference
    /// (its file's bytes in `reference`) on the held-out samples, and
    /// makes the gate record, or the record of the model scored alone.
    /// `corpus_sha256` is the SHA-256 of the `test` file.
    ///
    /// # Errors
    ///
    /// A reference given to a recipe without one or missing for one with
    /// one ([`Error::Model`]), what [`UdRecipe::score`] fails on, or the
    /// gate's errors.
    pub fn gate(&self, trained: &[u8], reference: Option<&[u8]>, held_out: &UdSamples<'_>) -> Result<UdGate> {
        let set = HeldOutSet {
            name: self.held_out_name(),
            corpus_sha256: self.treebank.test.sha256.to_owned(),
            samples_sha256: held_out.sha256(),
        };
        let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
        match (self.reference, reference) {
            (UdReference::Zoo(reference_id), Some(reference)) => {
                let (candidate, baseline, tags_left_out) = match held_out {
                    UdSamples::Tags(samples) => {
                        let (left_out, tokens) = tags_left_out(reference, samples)?;
                        let n: u64 = left_out.iter().map(|t| t.tokens).sum();
                        if n as f64 > tokens as f64 * LEFT_OUT_LIMIT_PERCENT / 100.0 {
                            return Err(Error::ReferenceTags { id: reference_id.to_owned(), left_out: n, tokens });
                        }
                        let counted = |tag: &str| !left_out.iter().any(|t| t.tag == tag);
                        let score = |bytes| crate::postag::score_tags_where(bytes, TAG_FORMAT, samples, counted);
                        (score(trained)?, score(reference)?, left_out)
                    }
                    _ => (self.score(trained, held_out)?, self.score(reference, held_out)?, Vec::new()),
                };
                let units = candidate
                    .into_iter()
                    .zip(baseline)
                    .map(|(candidate, reference)| PairedUnit { candidate, reference })
                    .collect();
                let record = ModelGateRecord::new(
                    self.task(),
                    set,
                    id(self.model_id, trained),
                    id(reference_id, reference),
                    units,
                )?;
                Ok(UdGate::Gated { record, tags_left_out })
            }
            (UdReference::NoReference(why), None) => Ok(UdGate::Ungated(UngatedRecord::new(
                self.task(),
                set,
                id(self.model_id, trained),
                why.to_owned(),
                self.score(trained, held_out)?,
            )?)),
            (UdReference::Zoo(id), None) => Err(Error::Model(format!("{}: no file given for {id}", self.model_id))),
            (UdReference::NoReference(_), Some(_)) => {
                Err(Error::Model(format!("{}: a reference given to a recipe that has none", self.model_id)))
            }
        }
    }
}

/// What one UD recipe made.
#[derive(Debug)]
pub struct UdOutcome {
    pub trained: Trained,
    pub gate: UdGate,
    pub card: UdModelCard,
}

/// Trains `recipe` on the treebank's `train` file, scores it and the
/// reference (the bytes of its file, `None` for a recipe without one) on
/// the `test` file, and makes the record and the card.
///
/// # Errors
///
/// What training, reading the samples, scoring or the gate fail on.
pub fn run_ud(recipe: &UdRecipe, text: &TreebankText, reference: Option<&[u8]>) -> Result<UdOutcome> {
    let trained = recipe.train(&text.train)?;
    let held_out = recipe.samples(&text.test, recipe.treebank.test.name)?;
    let gate = recipe.gate(&trained.bytes, reference, &held_out.samples)?;
    let card = UdModelCard::new(recipe, &trained, held_out.samples.len(), held_out.skipped, &gate);
    Ok(UdOutcome { trained, gate, card })
}

fn component_name(component: Component) -> &'static str {
    match component {
        Component::Sentence => "sentence detector",
        Component::Tokens => "tokenizer",
        Component::Pos => "part-of-speech tagger",
        Component::Lemmas => "lemmatizer",
    }
}

/// Each sample's counts for the lemmatizer in `bytes`, given the
/// sample's tags, as `LemmatizerEvaluator` counts them: the tokens given
/// the sample's lemma, of all its tokens.
///
/// # Errors
///
/// A model file that is not a lemmatizer model, a sentence no class
/// sequence scores above the search's minimum for, a class the decoder
/// cannot read, or what the evaluator throws on.
pub fn score_lemmas(bytes: &[u8], samples: &[LemmaSample<'_>]) -> Result<Vec<Counts>> {
    let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
    let model = LemmatizerModel::open(file).map_err(|e| Error::Model(e.to_string()))?;
    let lemmatizer = MaxentLemmatizer::new(Arc::new(model));
    let mut out = Vec::with_capacity(samples.len());
    let (mut classes, mut probs) = (Vec::new(), Vec::new());
    let mut words: Vec<TokenTag<'_>> = Vec::new();
    let mut lemmas: Vec<String> = Vec::new();
    for sample in samples {
        words.clear();
        words.extend(
            sample.tokens().iter().zip(sample.tags()).map(|(t, g)| TokenTag { token: t.as_str(), tag: g.as_str() }),
        );
        classes.clear();
        probs.clear();
        if !lemmatizer.classes(&words, &mut classes, &mut probs) {
            return Err(Error::Model(format!("no lemma class sequence scored above the minimum for: {sample:?}")));
        }
        lemmas.clear();
        for (w, class) in words.iter().zip(&classes) {
            lemmas.push(decode_lemma(w.token, class).map_err(|e| Error::Model(format!("{:?}: {e}", w.token)))?);
        }
        let predicted: Vec<&str> = lemmas.iter().map(String::as_str).collect();
        let mut evaluator = LemmatizerEvaluator::default();
        evaluator.evaluate(sample, &predicted)?;
        let correct = sample.lemmas().iter().zip(&predicted).filter(|(r, p)| r.as_str() == **p).count();
        out.push(Counts::accuracy(correct as u64, evaluator.word_count()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::treebank::TREEBANKS;

    #[test]
    fn the_recipes_name_their_treebank_component_and_reference() {
        for r in UD_RECIPES {
            let t = r.treebank;
            let id = format!("turbo-{}-ud-{}-2.16-{}-1", t.language, t.name, r.component.name());
            assert_eq!(r.model_id, id);
            let parsed: turbo_zoo::ArtifactId = r.model_id.parse().unwrap();
            assert_eq!(parsed.as_str(), r.model_id);
            match r.reference {
                UdReference::Zoo(reference) => {
                    let index = turbo_zoo::Index::builtin();
                    let a = index.get(reference).unwrap();
                    assert_eq!(a.component, Some(r.zoo_component()), "{reference}");
                    assert_eq!(a.language.as_ref().unwrap().as_str(), t.language, "{reference}");
                    assert!(reference.ends_with(&format!("-{}-1.3-2.5.4", r.component.name())));
                    assert_eq!(r.reference_model(), Some(Reference::Zoo(reference)));
                }
                UdReference::NoReference(why) => {
                    assert_eq!(t.language, "it");
                    assert!(why.contains("CC BY-NC-SA 3.0"));
                    assert_eq!(r.reference_model(), None);
                    // OpenNLP's Italian models are not in the zoo.
                    assert_eq!(turbo_zoo::Index::builtin().find("it", r.zoo_component()).count(), 0);
                }
            }
        }
        // Every treebank has a tagger and a lemmatizer; the ones whose text
        // is running text have a sentence detector and a tokenizer.
        for t in TREEBANKS {
            let components: Vec<Component> =
                UD_RECIPES.iter().filter(|r| r.treebank == t).map(|r| r.component).collect();
            let running = t.name != "ftb" && t.name != "markit";
            let expected: &[Component] = if running { &[Sentences, Tokens, Pos, Lemmas] } else { &[Pos, Lemmas] };
            assert_eq!(components, expected, "{}", t.repository);
        }
        assert_eq!(UD_RECIPES[3].task(), Task::Lemmatizer);
    }
}
