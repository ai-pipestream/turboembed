//! Recipes that train models on permissive data, and the accuracy
//! gate each model passes before it ships (issue #401).
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
//! - [`spans`]: a held-out unit's counts from OpenNLP's `FMeasure`, both
//!   models' counts paired, and the types a gate of typed spans scores.
//! - [`card`]: the model card kept beside the gate record, and the
//!   model's permissive zoo entry with its licence and attribution.
//!
//! [`run`] runs one recipe from a split corpus to the model, the card
//! and the gate record; the `turbo-train-recipes` binary runs every
//! recipe of [`RECIPES`] on a MASC directory and writes each model's
//! directory.

pub mod card;
pub mod error;
pub mod masc;
pub mod postag;
pub mod segmentation;
pub mod spans;

use std::fmt::{self, Write as _};

use turbo_eval::model_gate::{ModelGateRecord, Task};
use turbo_formats::masc::MascCorpus;
use turbo_zoo::{Index, Sha256};

pub use error::{Error, LeftOut, Result};

use card::{CorpusCard, HeldOutCard, ModelCard};
use masc::{CorpusDigest, SPLIT, SplitCorpus};
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
    /// detector). At most
    /// [`MAX_SKIPPED_SENTENCES`] of those read.
    pub skipped_sentences: usize,
    pub events: usize,
    /// `Training-Eventhash`.
    pub event_hash: u32,
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
        u128::from(left_out) * u128::from(self.of) <= u128::from(total) * u128::from(self.count)
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
    MAX_SKIPPED_SENTENCES.check(what, skipped as u64, (skipped + read) as u64)
}

/// The OpenNLP model a recipe's model is gated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reference {
    /// A model of the zoo's built-in index, by its id.
    Zoo(&'static str),
    /// A model the zoo does not list, recorded by the recipe.
    Recorded(RecordedReference),
}

/// A reference model the zoo does not list: its file as published, which
/// a run reads from its references directory and checks against the
/// recorded digest. Such a model is only read to score it; it is never
/// shipped or bundled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedReference {
    /// The id the gate record names it by.
    pub id: &'static str,
    /// The file name as published.
    pub file: &'static str,
    /// Where it is published.
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

impl Reference {
    /// The id the gate record names the model by: its zoo id, or the
    /// recorded one.
    pub fn id(&self) -> &'static str {
        match self {
            Reference::Zoo(id) => id,
            Reference::Recorded(r) => r.id,
        }
    }

    /// The model's file name: the one the zoo's built-in index gives it,
    /// or the recorded one.
    ///
    /// # Errors
    ///
    /// A zoo id the index does not list.
    pub fn file(&self) -> Result<String> {
        match self {
            Reference::Zoo(id) => Ok(Index::builtin().get(id)?.files[0].path.to_string()),
            Reference::Recorded(r) => Ok(r.file.to_owned()),
        }
    }

    /// The model file's SHA-256: the one the zoo's built-in index lists, or
    /// the recorded one.
    ///
    /// # Errors
    ///
    /// A zoo id the index does not list.
    pub fn sha256(&self) -> Result<String> {
        match self {
            Reference::Zoo(id) => Ok(Index::builtin().get(id)?.files[0].sha256.to_string()),
            Reference::Recorded(r) => Ok(r.sha256.to_owned()),
        }
    }

    /// Checks that `bytes` are the model: the SHA-256 the zoo's built-in
    /// index lists, or the recorded size and SHA-256.
    ///
    /// # Errors
    ///
    /// [`Error::ReferenceDigest`] for other bytes, or a zoo id the index
    /// does not list.
    pub fn check(&self, bytes: &[u8]) -> Result<()> {
        let expected = self.sha256()?;
        let size = match self {
            Reference::Zoo(_) => None,
            Reference::Recorded(r) => Some(r.size),
        };
        let found = Sha256::of(bytes).to_string();
        if found == expected && size.is_none_or(|n| n == bytes.len() as u64) {
            Ok(())
        } else {
            Err(Error::ReferenceDigest { id: self.id().to_owned(), expected, found })
        }
    }
}

/// One recipe: a model trained on MASC and the OpenNLP model it is gated
/// against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recipe {
    Segmentation(SegmentationRecipe),
    Pos(PosRecipe),
}

/// The recipes, in the order the binary runs them: the sentence detector,
/// the tokenizer, the part-of-speech tagger.
pub const RECIPES: [Recipe; 3] = [
    Recipe::Segmentation(segmentation::SENTENCE),
    Recipe::Segmentation(segmentation::TOKENS),
    Recipe::Pos(postag::POS),
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
        }
    }

    /// The model the recipe's model is gated against.
    pub fn reference(&self) -> Reference {
        match self {
            Recipe::Segmentation(r) => Reference::Zoo(r.reference_id),
            Recipe::Pos(r) => r.reference,
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
        }
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
        }
    }

    /// Reads the training and held-out samples the recipe reads from
    /// `split`, without training or scoring, and checks what [`run`]
    /// checks of them: each side has a sample, and the recipe's reader
    /// skips at most [`MAX_SKIPPED_SENTENCES`] of the sentences of either
    /// side; and for the tagger, given the bytes of its reference (named
    /// `reference_id`), that the reference can give the tags of all but
    /// [`MAX_TOKENS_LEFT_OUT`] of the held-out tokens. The binary runs it
    /// for every recipe before it trains any, so a run that would fail on
    /// its data writes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::EmptySplit`], [`Error::TooManyLeftOut`],
    /// [`Error::TagsOutsideReference`], a sample the corpus fails on, or a
    /// reference that is not a part-of-speech model.
    pub fn check_data(&self, split: &SplitCorpus, reference_id: &str, reference: Option<&[u8]>) -> Result<()> {
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
        }
        Ok(())
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
/// What training, reading the samples, scoring or the gate fail on, and
/// [`Error::TooManyLeftOut`] when the recipe's reader skips more than
/// [`MAX_SKIPPED_SENTENCES`] of the training or of the held-out sentences.
pub fn run(
    recipe: &Recipe,
    split: &SplitCorpus,
    digest: &CorpusDigest,
    recorded: bool,
    reference_id: &str,
    reference: &[u8],
) -> Result<Outcome> {
    let trained = recipe.train(&split.train)?;
    let GateRun { record: gate, held_out } =
        recipe.gate(&trained.bytes, reference_id, reference, &split.held_out, &held_out_name(), &digest.sha256)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_recipe_names_a_reference_and_a_valid_model_id() {
        let ids: Vec<&str> = RECIPES.iter().map(Recipe::model_id).collect();
        assert_eq!(
            ids,
            ["turbo-en-masc-3.0.0-sentence-1", "turbo-en-masc-3.0.0-tokens-1", "turbo-en-masc-3.0.0-pos-1"]
        );
        for r in RECIPES {
            assert!(matches!(r.check_reference(b"not the model"), Err(Error::ReferenceDigest { .. })));
            let id: turbo_zoo::ArtifactId = r.model_id().parse().unwrap();
            assert_eq!(id.as_str(), r.model_id());
            match r.reference() {
                // The reference is a model of the component the trained
                // one is, under its zoo file name.
                Reference::Zoo(id) => {
                    assert_eq!(r.reference_file().unwrap(), format!("{id}.bin"));
                    assert_eq!(Index::builtin().get(id).unwrap().component, Some(r.component()));
                }
                // A recorded reference is one the zoo does not list, so it
                // can never be fetched or shipped through it.
                Reference::Recorded(rec) => {
                    assert_eq!(r.reference_file().unwrap(), rec.file);
                    assert!(Index::builtin().get(rec.id).is_err());
                }
            }
        }
        assert_eq!(RECIPES[2].task(), Task::PartOfSpeechTagger);
        assert_eq!(RECIPES[2].component(), turbo_zoo::Component::Pos);
        assert_eq!(RECIPES[2].reference_id(), "opennlp-en-pos-maxent-1.5");
        assert_eq!(RECIPES[2].reference().sha256().unwrap(), postag::EN_POS_MAXENT.sha256);
        for r in RECIPES {
            assert_eq!(Recipe::trains(r.model_id()), Some(r));
        }
        assert_eq!(Recipe::trains("turbo-en-masc-3.0.0-chunks-1"), None);
        assert!(Reference::Zoo("not-in-the-zoo").file().is_err());
    }

    #[test]
    fn a_recorded_reference_is_checked_by_size_and_digest() {
        let bytes = b"abc";
        let rec = RecordedReference {
            id: "x",
            file: "x.bin",
            url: "https://example.org/x.bin",
            size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        Reference::Recorded(rec).check(bytes).unwrap();
        assert_eq!(Reference::Recorded(rec).sha256().unwrap(), rec.sha256);
        assert!(Reference::Recorded(RecordedReference { size: 4, ..rec }).check(bytes).is_err());
        let other = Reference::Recorded(rec).check(b"abd").unwrap_err();
        // The message says what was recorded, not where.
        assert_eq!(
            other.to_string(),
            format!("reference model x: SHA-256 {}, expected {}", Sha256::of(b"abd"), rec.sha256)
        );
    }
}
