//! Recipes that train models on permissive data, and the accuracy
//! gate each model passes before it ships (issues #401 and #415).
//!
//! - [`masc`]: MASC 3.0.0 (CC BY 3.0 US), the record of its release, the digest of
//!   an extracted copy, and the fixed split into training and held-out
//!   documents.
//! - [`segmentation`]: the English sentence detector and tokenizer
//!   trained on MASC, scored with OpenNLP's evaluators beside the OpenNLP
//!   English UD models on the same held-out samples, and gated by
//!   `turbo_eval::model_gate`.
//! - [`postag`]: the English part-of-speech tagger trained on MASC's Penn
//!   tags, gated the same way against OpenNLP 1.5's Penn tagger
//!   (`en-pos-maxent.bin`).
//! - [`chunk`]: the English chunker trained on MASC's noun and verb
//!   chunks, gated on those two chunk types against OpenNLP 1.5's
//!   chunker (`en-chunker.bin`).
//! - [`namefind`]: English name finders trained on MASC's named entities,
//!   one per type (person, location, organization, date), each gated
//!   against OpenNLP 1.5's name finder of that type.
//! - [`spans`]: a held-out unit's counts from OpenNLP's `FMeasure`, both
//!   models' counts paired, and the types a gate of typed spans scores.
//! - [`card`]: the model card kept beside the gate record, and the
//!   model's permissive zoo entry with its licence and attribution.
//! - [`treebank`]: the Universal Dependencies treebanks licensed CC BY
//!   4.0 at release 2.16, their files and digests, and the licence check
//!   that reads `LICENSE.txt` and the README.
//! - [`ud`]: the sentence detector, tokenizer, tagger and lemmatizer
//!   trained on those treebanks' `train` files, gated on their `test`
//!   files against the OpenNLP UD 1.3 models (the Italian ones local-only,
//!   recorded by digest and only scored).
//! - [`ud_card`]: the card of a model a UD recipe trained, with the
//!   treebank's licence, attribution and citation, and its zoo entry,
//!   which a treebank whose text rights are not settled (Porttinari) never
//!   gets.
//! - [`release`]: the releases of this repository that publish the MASC
//!   and UD models, each naming the models it publishes, and their entries
//!   in the zoo's built-in index.
//! - [`tatoeba`]: Tatoeba's sentence exports read with each sentence's
//!   licence (issue #429).
//! - [`langdetect`]: the language detector trained on Tatoeba's CC BY
//!   2.0 FR sentences and gated against OpenNLP's `langdetect-183.bin`
//!   on held-out sentences of the languages both cover, with its own
//!   card.
//! - [`langdetect_web`]: the language detector trained on those
//!   sentences and on web text from FineWeb-2 and FineWeb, gated against
//!   `langdetect-183.bin` on the held-out sentences and on held-out web
//!   text, and compared with the Tatoeba model on both.
//! - [`fineweb`]: FineWeb-2's and FineWeb's web text (ODC-By 1.0), read
//!   by range from the leading row groups of one file per language and
//!   side, and the windows of it the language detector trains on and
//!   holds out.
//!
//! [`run`] runs one MASC recipe from a split corpus to the model, the
//! card and the gate record, and [`ud::run_ud`] one UD recipe from a
//! treebank; the `turbo-train-recipes` binary runs the recipes of
//! [`RECIPES`] on a MASC directory, or those of [`ud::UD_RECIPES`] on the
//! treebanks, and writes each model's directory, runs
//! [`langdetect::run`] on a Tatoeba export, and its `zoo-index` command
//! makes the zoo's entries for a release of the MASC models
//! ([`release`]).

pub mod card;
pub mod chunk;
pub mod depparse;
pub mod error;
pub mod fineweb;
pub mod langdetect;
pub mod langdetect_web;
pub mod masc;
pub mod namefind;
pub mod namefind_eval;
pub mod postag;
mod reference;
pub mod release;
pub mod segmentation;
pub mod spans;
pub mod tatoeba;
pub mod treebank;
pub mod ud;
pub mod ud_card;

use std::fmt::{self, Write as _};

use turbo_eval::model_gate::{ModelGateRecord, Task};
use turbo_formats::masc::MascCorpus;

pub use error::{Error, LeftOut, Result};
pub use reference::{RecordedReference, Reference};

use card::{ChunksCard, CorpusCard, HeldOutCard, ModelCard, NamesCard, ParametersCard};
use chunk::ChunkRecipe;
use masc::{CorpusDigest, SPLIT, SplitCorpus};
use namefind::NameRecipe;
use postag::PosRecipe;
use segmentation::{Component, SegmentationRecipe};

/// A model a recipe trained, and what it was trained from.
#[derive(Debug, Clone)]
pub struct Trained {
    /// The `.bin` file.
    pub bytes: Vec<u8>,
    /// The samples read from the training documents.
    pub samples: usize,
    /// Sentences left out of training: by the token stream for having no
    /// sample or tokens out of text order, by the tagger's for a token
    /// without a tag or for having no token (always 0 for the sentence
    /// detector), by the chunker's for those and for chunks that chunk
    /// tags cannot express, by the name finder's for having no token. At
    /// most [`MAX_SKIPPED_SENTENCES`] of those read, or
    /// [`chunk::MAX_SKIPPED_SENTENCES`] for the chunker.
    pub skipped_sentences: usize,
    pub events: usize,
    /// `Training-Eventhash`.
    pub event_hash: u32,
    /// For a name finder, what its reader counted of the named entities;
    /// `None` for the other components.
    pub names: Option<NamesCard>,
    /// For the chunker, the training documents' chunks in no sentence its
    /// reader reads ([`chunk::chunk_counts`]); `None` for the others.
    pub chunks: Option<ChunksCard>,
}

/// A bound on how much a recipe may leave out: at most `count` in every
/// `of`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limit {
    pub count: u64,
    pub of: u64,
}

impl Limit {
    /// Whether leaving out `left_out` of `total` is within the limit.
    pub fn allows(&self, left_out: u64, total: u64) -> bool {
        left_out <= total
            && self.of > 0
            && u128::from(left_out) * u128::from(self.of) <= u128::from(total) * u128::from(self.count)
    }

    /// Checks that leaving out `count` of `of` is within the limit.
    ///
    /// # Errors
    ///
    /// [`Error::TooManyLeftOut`] when it is not.
    pub fn check(&self, what: LeftOut, count: u64, of: u64) -> Result<()> {
        if self.allows(count, of) { Ok(()) } else { Err(Error::TooManyLeftOut { what, count, of, limit: *self }) }
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} in {}", self.count, self.of)
    }
}

/// The most sentences a recipe's reader may skip, of those it reads from
/// the training documents and, apart, of those it reads from the held-out
/// documents: one in a thousand, so the gate cannot lose a held-out
/// document of more than a few sentences. Read with the recipes' readers
/// and split by [`masc::SPLIT`] (the Twitter genre left out), the recorded
/// release (corpus digest `445449f5...`) has one training sentence the
/// token recipe skips, of 24,958 (a token inside another, in
/// `data/spoken/face-to-face/Bed012`), and none the tagger or the
/// sentence detector skips; and one held-out sentence the token and
/// tagger recipes skip (a sentence of no tokens), of 6,385.
pub const MAX_SKIPPED_SENTENCES: Limit = Limit { count: 1, of: 1000 };

/// The most held-out tokens the tagger's gate may leave out of both
/// models' scores because the reference cannot give their tag, as
/// written or through the mapping of its tags ([`postag::ReferenceTags`]):
/// one in a thousand. Read with the recipe's reader and split by
/// [`masc::SPLIT`], the recorded release's held-out documents (corpus
/// digest `445449f5...`) have 6 such tokens of 85,292: ``UH` `` (3), `"`
/// (2) and `PRN` (1).
pub const MAX_TOKENS_LEFT_OUT: Limit = Limit { count: 1, of: 1000 };

/// Checks the sentences a reader skipped against [`MAX_SKIPPED_SENTENCES`]:
/// `skipped` of `skipped + read`.
pub(crate) fn check_skipped(what: LeftOut, skipped: usize, read: usize) -> Result<()> {
    let skipped = skipped as u64;
    MAX_SKIPPED_SENTENCES.check(what, skipped, skipped.saturating_add(read as u64))
}

/// One recipe: a model trained on MASC and the OpenNLP model it is gated
/// against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recipe {
    Segmentation(SegmentationRecipe),
    Pos(PosRecipe),
    Chunk(ChunkRecipe),
    Name(NameRecipe),
}

/// The recipes, in the order the binary runs them: the sentence detector,
/// the tokenizer, the part-of-speech tagger, the chunker, then the name
/// finders of [`namefind::RECIPES`].
pub const RECIPES: [Recipe; 8] = [
    Recipe::Segmentation(segmentation::SENTENCE),
    Recipe::Segmentation(segmentation::TOKENS),
    Recipe::Pos(postag::POS),
    Recipe::Chunk(chunk::CHUNKS),
    Recipe::Name(namefind::PERSON),
    Recipe::Name(namefind::LOCATION),
    Recipe::Name(namefind::ORGANIZATION),
    Recipe::Name(namefind::DATE),
];

impl Recipe {
    /// The recipe of [`RECIPES`] that trains the model `id`.
    pub fn trains(id: &str) -> Option<Recipe> {
        RECIPES.into_iter().find(|r| r.model_id() == id)
    }

    /// The trained model's zoo id, also its file name without `.bin`.
    pub fn model_id(&self) -> &'static str {
        match self {
            Recipe::Segmentation(r) => r.model_id,
            Recipe::Pos(r) => r.model_id,
            Recipe::Chunk(r) => r.model_id,
            Recipe::Name(r) => r.model_id,
        }
    }

    /// The model the recipe's model is gated against.
    pub fn reference(&self) -> Reference {
        match self {
            Recipe::Segmentation(r) => Reference::Zoo(r.reference_id),
            Recipe::Pos(r) => r.reference,
            Recipe::Chunk(r) => r.reference,
            Recipe::Name(r) => r.reference,
        }
    }

    /// The id the gate record names the reference by.
    pub fn reference_id(&self) -> &'static str {
        self.reference().id()
    }

    /// The gate's task.
    pub fn task(&self) -> Task {
        match self {
            Recipe::Segmentation(r) => r.task(),
            Recipe::Pos(r) => r.task(),
            Recipe::Chunk(r) => r.task(),
            Recipe::Name(r) => r.task(),
        }
    }

    /// The component the zoo lists the trained model under.
    pub fn component(&self) -> turbo_zoo::Component {
        match self {
            Recipe::Segmentation(r) => match r.component {
                Component::Sentence => turbo_zoo::Component::SentenceDetector,
                Component::Tokens => turbo_zoo::Component::Tokenizer,
            },
            Recipe::Pos(_) => turbo_zoo::Component::Pos,
            Recipe::Chunk(_) => turbo_zoo::Component::Chunker,
            Recipe::Name(_) => turbo_zoo::Component::NameFinder,
        }
    }

    /// Whether the recipe reads the corpus's noun and verb chunks, so the
    /// corpus is to be read with its chunk files
    /// ([`MascCorpus::read_with_chunks`]).
    pub fn reads_chunks(&self) -> bool {
        matches!(self, Recipe::Chunk(_))
    }

    /// The training parameters, as the card records them: the MASC settings
    /// for the segmentation recipes and the tagger, the chunker's, and
    /// `NameFinderME.train`'s defaults for the name finders
    /// ([`namefind::training_parameters`]).
    pub fn parameters(&self) -> ParametersCard {
        use turbo_train_components::{namefind as names, ud};
        let (iterations, cutoff, threads) = match self {
            Recipe::Segmentation(_) | Recipe::Pos(_) => (ud::ITERATIONS, ud::CUTOFF, masc::THREADS),
            Recipe::Chunk(_) => (chunk::ITERATIONS, chunk::CUTOFF, chunk::THREADS),
            Recipe::Name(_) => (names::DEFAULT_ITERATIONS, names::DEFAULT_CUTOFF, 1),
        };
        ParametersCard { algorithm: "MAXENT".to_owned(), iterations, cutoff, threads }
    }

    /// Trains the model from the training documents.
    ///
    /// # Errors
    ///
    /// What the recipe's `train` fails on.
    pub fn train(&self, train: &MascCorpus) -> Result<Trained> {
        match self {
            Recipe::Segmentation(r) => r.train(train),
            Recipe::Pos(r) => r.train(train),
            Recipe::Chunk(r) => r.train(train),
            Recipe::Name(r) => r.train(train),
        }
    }

    /// Reads the held-out samples of `held_out`, scores the model in
    /// `trained` and `reference` (named `reference_id` in the record) on
    /// them, and runs the gate; `set_name` and `corpus_sha256` name the
    /// held-out set.
    ///
    /// # Errors
    ///
    /// What reading the samples, scoring or the gate fail on.
    pub fn gate(
        &self,
        trained: &[u8],
        reference_id: &str,
        reference: &[u8],
        held_out: &MascCorpus,
        set_name: &str,
        corpus_sha256: &str,
    ) -> Result<GateRun> {
        match self {
            Recipe::Segmentation(r) => {
                r.gate(trained, reference_id, reference, &r.held_out(held_out)?, set_name, corpus_sha256)
            }
            Recipe::Pos(r) => r.gate(trained, reference_id, reference, &r.held_out(held_out)?, set_name, corpus_sha256),
            Recipe::Chunk(r) => {
                r.gate(trained, reference_id, reference, &r.held_out(held_out)?, set_name, corpus_sha256)
            }
            Recipe::Name(r) => {
                r.gate(trained, reference_id, reference, &r.held_out(held_out)?, set_name, corpus_sha256)
            }
        }
    }

    /// Reads the training and held-out samples the recipe reads from
    /// `split`, without training or scoring, and checks what [`run`]
    /// checks of them: each side has a sample, and the recipe's reader
    /// skips at most [`MAX_SKIPPED_SENTENCES`] of the sentences of either
    /// side, and on the recorded release (`recorded`) the counts the
    /// recipe records ([`Recipe::check_recorded_skips`]); and for the
    /// tagger, given the bytes of its reference (named `reference_id`),
    /// that the reference can give the tags of all but
    /// [`MAX_TOKENS_LEFT_OUT`] of the held-out tokens. The binary runs it
    /// for every recipe before it trains any, so a run that would fail on
    /// its data writes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::EmptySplit`], [`Error::TooManyLeftOut`],
    /// [`Error::SkippedNotRecorded`], [`Error::TagsOutsideReference`], a
    /// sample the corpus fails on, or a reference that is not a
    /// part-of-speech model.
    pub fn check_data(
        &self,
        split: &SplitCorpus,
        recorded: bool,
        reference_id: &str,
        reference: Option<&[u8]>,
    ) -> Result<()> {
        match self {
            Recipe::Segmentation(r) => {
                r.check_training(&split.train)?;
                r.held_out(&split.held_out)?;
            }
            Recipe::Pos(r) => {
                r.check_training(&split.train)?;
                let held_out = r.held_out(&split.held_out)?;
                if let Some(reference) = reference {
                    r.check_coverage(reference_id, &r.reference_tags(reference)?, &held_out)?;
                }
            }
            Recipe::Chunk(r) => {
                let skipped = r.check_training(&split.train)?;
                self.check_recorded_skips(recorded, LeftOut::TrainingSentences, skipped.total())?;
                let held_out = r.held_out(&split.held_out)?;
                self.check_recorded_skips(recorded, LeftOut::HeldOutSentences, held_out.skipped().total())?;
            }
            Recipe::Name(r) => {
                r.check_training(&split.train)?;
                r.held_out(&split.held_out)?;
            }
        }
        Ok(())
    }

    /// On the recorded release (`recorded`), checks `skipped`, the
    /// sentences of `what` the recipe's reader skipped, against the count
    /// the recipe records: the chunker's
    /// ([`chunk::RECORDED_SKIPPED_SENTENCES`]). The other recipes record
    /// none, and a corpus that is not the recorded release is checked by
    /// the bound on skipped sentences alone.
    ///
    /// # Errors
    ///
    /// [`Error::SkippedNotRecorded`].
    pub fn check_recorded_skips(&self, recorded: bool, what: LeftOut, skipped: usize) -> Result<()> {
        match self {
            Recipe::Chunk(_) if recorded => chunk::check_recorded_skips(what, skipped),
            _ => Ok(()),
        }
    }

    /// The reference model's file name ([`Reference::file`]).
    ///
    /// # Errors
    ///
    /// A zoo id the index does not list.
    pub fn reference_file(&self) -> Result<String> {
        self.reference().file()
    }

    /// Checks that `bytes` are the reference model ([`Reference::check`]).
    ///
    /// # Errors
    ///
    /// [`Error::ReferenceDigest`] for other bytes.
    pub fn check_reference(&self, bytes: &[u8]) -> Result<()> {
        self.reference().check(bytes)
    }
}

/// A gate run: its record, and what it read from the held-out documents.
#[derive(Debug)]
pub struct GateRun {
    pub record: ModelGateRecord,
    pub held_out: HeldOutCard,
}

/// What one recipe made.
#[derive(Debug)]
pub struct Outcome {
    pub trained: Trained,
    pub gate: ModelGateRecord,
    pub card: ModelCard,
}

/// The name of the held-out set in the gate records: the split, and what
/// [`masc::EXCLUSIONS`] leaves out.
pub fn held_out_name() -> String {
    let mut name =
        format!("{} held out (every {}th document from the {}th", masc::NAME, SPLIT.every(), SPLIT.held_out_at() + 1);
    for exclusion in masc::EXCLUSIONS {
        write!(name, ", {exclusion} left out").expect("writing to a String");
    }
    name.push(')');
    name
}

/// Trains `recipe` on the training documents of `split`, scores it and
/// `reference` (the bytes of the reference model's file, named
/// `reference_id` in the record) on the held-out documents, and makes
/// the gate record and the card. `digest` is the corpus directory's, and
/// `recorded` whether it is the recorded release's.
///
/// # Errors
///
/// What training, reading the samples, scoring or the gate fail on,
/// [`Error::TooManyLeftOut`] when the recipe's reader skips more than
/// [`MAX_SKIPPED_SENTENCES`] of the training or of the held-out
/// sentences, and [`Error::SkippedNotRecorded`] when, on the recorded
/// release, it skips other counts than the recipe records
/// ([`Recipe::check_recorded_skips`]).
pub fn run(
    recipe: &Recipe,
    split: &SplitCorpus,
    digest: &CorpusDigest,
    recorded: bool,
    reference_id: &str,
    reference: &[u8],
) -> Result<Outcome> {
    let trained = recipe.train(&split.train)?;
    recipe.check_recorded_skips(recorded, LeftOut::TrainingSentences, trained.skipped_sentences)?;
    let GateRun { record: gate, held_out } =
        recipe.gate(&trained.bytes, reference_id, reference, &split.held_out, &held_out_name(), &digest.sha256)?;
    recipe.check_recorded_skips(recorded, LeftOut::HeldOutSentences, held_out.skipped_sentences)?;
    let corpus = CorpusCard {
        name: masc::NAME.to_owned(),
        corpus_sha256: digest.sha256.clone(),
        release_recorded: recorded,
        split: SPLIT,
        training_documents: split.train.documents().len(),
        held_out_documents: split.held_out_paths.clone(),
        excluded_documents: split.excluded.clone(),
    };
    let card = ModelCard::new(recipe, &trained, corpus, held_out, &gate);
    Ok(Outcome { trained, gate, card })
}
