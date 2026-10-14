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
//! - [`card`]: the model card kept beside the gate record, and the
//!   model's permissive zoo entry with its licence and attribution.
//!
//! [`run`] runs one recipe from a split corpus to the model, the card
//! and the gate record; the `turbo-train-recipes` binary runs every
//! recipe on a MASC directory and writes each model's directory.
//!
//! The part-of-speech tagger recipe is not here: it needs the
//! part-of-speech event stream and trainer, which are not in the
//! workspace yet.

pub mod card;
pub mod error;
pub mod masc;
pub mod segmentation;

use turbo_eval::model_gate::ModelGateRecord;

pub use error::{Error, Result};

use card::{CorpusCard, ModelCard};
use masc::{CorpusDigest, SPLIT, SplitCorpus};
use segmentation::{SegmentationRecipe, Trained};

/// What one recipe made.
#[derive(Debug)]
pub struct Outcome {
    pub trained: Trained,
    pub gate: ModelGateRecord,
    pub card: ModelCard,
}

/// The name of the held-out set in the gate records.
pub fn held_out_name() -> String {
    format!("{} held out (every {}th document from the {}th)", masc::NAME, SPLIT.every, SPLIT.held_out_at + 1)
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
    recipe: &SegmentationRecipe,
    split: &SplitCorpus,
    digest: &CorpusDigest,
    recorded: bool,
    reference_id: &str,
    reference: &[u8],
) -> Result<Outcome> {
    let trained = recipe.train(&split.train)?;
    let held_out = recipe.held_out(&split.held_out)?;
    let gate = recipe.gate(&trained.bytes, reference_id, reference, &held_out, &held_out_name(), &digest.sha256)?;
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
