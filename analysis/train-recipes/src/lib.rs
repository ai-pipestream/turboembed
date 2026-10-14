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
//!   tags, gated the same way against the OpenNLP English UD tagger asked
//!   for Penn tags.
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

use turbo_eval::model_gate::{ModelGateRecord, Task};
use turbo_formats::masc::MascCorpus;
use turbo_zoo::{Index, Sha256};

pub use error::{Error, Result};

use card::{CorpusCard, ModelCard};
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
    /// sample, by the tagger's for a token without a tag or for having no
    /// token (always 0 for the sentence detector).
    pub skipped_sentences: usize,
    pub events: usize,
    /// `Training-Eventhash`.
    pub event_hash: u32,
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
    /// The trained model's zoo id, also its file name without `.bin`.
    pub fn model_id(&self) -> &'static str {
        match self {
            Recipe::Segmentation(r) => r.model_id,
            Recipe::Pos(r) => r.model_id,
        }
    }

    /// The reference model's id in the zoo's built-in index.
    pub fn reference_id(&self) -> &'static str {
        match self {
            Recipe::Segmentation(r) => r.reference_id,
            Recipe::Pos(r) => r.reference_id,
        }
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
    ) -> Result<ModelGateRecord> {
        match self {
            Recipe::Segmentation(r) => {
                r.gate(trained, reference_id, reference, &r.held_out(held_out)?, set_name, corpus_sha256)
            }
            Recipe::Pos(r) => r.gate(trained, reference_id, reference, &r.held_out(held_out)?, set_name, corpus_sha256),
        }
    }

    /// The reference model's file name in the zoo's built-in index.
    ///
    /// # Errors
    ///
    /// A reference the index does not list.
    pub fn reference_file(&self) -> Result<String> {
        reference_file(self.reference_id())
    }

    /// Checks that `bytes` are the reference model the zoo's built-in
    /// index lists.
    ///
    /// # Errors
    ///
    /// [`Error::ReferenceDigest`] for other bytes.
    pub fn check_reference(&self, bytes: &[u8]) -> Result<()> {
        check_reference(self.reference_id(), bytes)
    }
}

/// The file name of the model `id` in the zoo's built-in index.
///
/// # Errors
///
/// A model the index does not list.
pub fn reference_file(id: &str) -> Result<String> {
    Ok(Index::builtin().get(id)?.files[0].path.to_string())
}

/// Checks that `bytes` are the model `id` of the zoo's built-in index.
///
/// # Errors
///
/// [`Error::ReferenceDigest`] for other bytes, or a model the index does
/// not list.
pub fn check_reference(id: &str, bytes: &[u8]) -> Result<()> {
    let expected = Index::builtin().get(id)?.files[0].sha256.to_string();
    let found = Sha256::of(bytes).to_string();
    if expected == found { Ok(()) } else { Err(Error::ReferenceDigest { id: id.to_owned(), expected, found }) }
}

/// What one recipe made.
#[derive(Debug)]
pub struct Outcome {
    pub trained: Trained,
    pub gate: ModelGateRecord,
    pub card: ModelCard,
}

/// The name of the held-out set in the gate records.
pub fn held_out_name() -> String {
    format!("{} held out (every {}th document from the {}th)", masc::NAME, SPLIT.every(), SPLIT.held_out_at() + 1)
}

/// Trains `recipe` on the training documents of `split`, scores it and
/// `reference` (the bytes of the reference model's file, named
/// `reference_id` in the record) on the held-out documents, and makes
/// the gate record and the card. `digest` is the corpus directory's, and
/// `recorded` whether it is the recorded release's.
///
/// # Errors
///
/// What training, reading the samples, scoring or the gate fail on.
pub fn run(
    recipe: &Recipe,
    split: &SplitCorpus,
    digest: &CorpusDigest,
    recorded: bool,
    reference_id: &str,
    reference: &[u8],
) -> Result<Outcome> {
    let trained = recipe.train(&split.train)?;
    let gate =
        recipe.gate(&trained.bytes, reference_id, reference, &split.held_out, &held_out_name(), &digest.sha256)?;
    let corpus = CorpusCard {
        name: masc::NAME.to_owned(),
        corpus_sha256: digest.sha256.clone(),
        release_recorded: recorded,
        split: SPLIT,
        training_documents: split.train.documents().len(),
        held_out_documents: split.held_out_paths.clone(),
        skipped_documents: split.skipped_paths.clone(),
    };
    let card = ModelCard::new(recipe, &trained, corpus, &gate);
    Ok(Outcome { trained, gate, card })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_recipe_names_a_reference_the_zoo_lists_and_a_valid_model_id() {
        let ids: Vec<&str> = RECIPES.iter().map(Recipe::model_id).collect();
        assert_eq!(
            ids,
            ["turbo-en-masc-3.0.0-sentence-1", "turbo-en-masc-3.0.0-tokens-1", "turbo-en-masc-3.0.0-pos-1"]
        );
        for r in RECIPES {
            assert_eq!(r.reference_file().unwrap(), format!("{}.bin", r.reference_id()));
            assert!(matches!(r.check_reference(b"not the model"), Err(Error::ReferenceDigest { .. })));
            let id: turbo_zoo::ArtifactId = r.model_id().parse().unwrap();
            assert_eq!(id.as_str(), r.model_id());
            // The reference is a model of the component the trained one is.
            assert_eq!(Index::builtin().get(r.reference_id()).unwrap().component, Some(r.component()));
        }
        assert_eq!(RECIPES[2].task(), Task::PartOfSpeechTagger);
        assert_eq!(RECIPES[2].component(), turbo_zoo::Component::Pos);
        assert!(reference_file("not-in-the-zoo").is_err());
    }
}
