//! Web data planning for native language identification (#1525).
//!
//! - Data: the training sentences of [`crate::langdetect::LANGDETECT`],
//!   read, checked and split exactly as that recipe does, and the
//!   training windows of [`crate::fineweb`] ([`fineweb::WEB`]): at most
//!   1,000 web windows of at most 300 characters per language beside at
//!   most 5,000 sentences. No training window holds a held-out sentence of
//!   its language of at least [`fineweb::WebSplit::overlap_min_chars`]
//!   characters, byte for byte, and no held-out window such a training
//!   sentence ([`fineweb::Avoid`]). One event each, the sentences first,
//!   then the windows, each language by language in the order of
//!   [`crate::langdetect::LANGUAGES`].
//! - Native model: `langid` trains on the checked sentences and windows
//!   and scores exact accuracy on their held-out samples. The classical
//!   model comparisons are retired (#1648).
//!
//! The web windows' languages are the labels FineWeb-2's and FineWeb's
//! classifiers gave their documents, so web evaluation measures agreement
//! with those classifiers on text much like the text trained on.

mod card;
mod score;
#[cfg(test)]
mod tests;

use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::ModelGateRecord;

pub use card::{
    ComparedSentences, ComparisonCard, DatasetCard, DocumentsFileCard, WebDataCard, WebLanguageCard, WebTrainingCard,
    training_data,
};

use crate::Reference;
use crate::card::GATE_FILE;
use crate::error::{Error, Result, io};
use crate::fineweb::{self, Store, WEB, WebFile, WebRange, WebRecipe};
use crate::langdetect::{self, CARD_FILE, LANGDETECT, LanguageRecipe};
use crate::tatoeba::Export;

pub const MODEL_ID: &str = "turbo-mul-tatoeba-fineweb-langdetect-1";

/// The version of [`WebLanguageCard`] this build writes and reads.
pub const CARD_VERSION: u32 = 1;

/// The list of web documents trained on, shipped with the model.
pub const DOCUMENTS_FILE: &str = "training-documents.txt";

/// The gate record on the held-out web windows.
pub const WEB_GATE_FILE: &str = "gate-web.json";

/// The comparison with [`COMPARED`] on the held-out sentences.
pub const COMPARE_TATOEBA_FILE: &str = "compare-tatoeba.json";

/// The comparison with [`COMPARED`] on the held-out web windows of the
/// languages it has.
pub const COMPARE_WEB_FILE: &str = "compare-web.json";

/// The Tatoeba recipe's model as the zoo lists it, which both held-out
/// sets are scored with beside the trained model.
pub const COMPARED: Reference = Reference::Zoo(langdetect::MODEL_ID);

/// The recipe: the Tatoeba recipe's languages, split, parameters and
/// reference under this model's id, the web side for the same languages
/// in the same order, and the model scored beside it.
#[derive(Copy, Clone, Debug)]
pub struct WebLanguageRecipe {
    pub model_id: &'static str,
    pub tatoeba: LanguageRecipe,
    pub web: WebRecipe,
    pub compared: Reference,
}

pub const LANGDETECT_WEB: WebLanguageRecipe = WebLanguageRecipe {
    model_id: MODEL_ID,
    tatoeba: LanguageRecipe { model_id: MODEL_ID, ..LANGDETECT },
    web: WEB,
    compared: COMPARED,
};

/// One language's web windows, and once gated, how many held-out ones
/// each model got right.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebLanguageCount {
    pub tatoeba: String,
    pub training: usize,
    pub held_out: usize,
    /// Held-out windows the trained model got right.
    pub candidate_correct: usize,
    /// Held-out windows the reference got right.
    pub reference_correct: usize,
    /// Held-out windows [`COMPARED`] got right: none of a language it
    /// does not have.
    pub compared_correct: usize,
}

/// A model the recipe trained.
#[derive(Debug, Clone)]
pub struct TrainedWeb {
    /// The `.bin` file.
    pub bytes: Vec<u8>,
    /// The training sentences.
    pub sentences: usize,
    /// The training windows.
    pub windows: usize,
    pub events: usize,
    /// `Training-Eventhash`.
    pub event_hash: u32,
    /// Events whose predicates are not in the JVM's order
    /// (`turbo_train_components::langdetect`).
    pub inexact_events: usize,
    /// [`SENTENCES_FILE`](crate::langdetect::SENTENCES_FILE) of the
    /// sentences trained on.
    pub sentences_file: Vec<u8>,
    /// [`DOCUMENTS_FILE`] of the windows trained on.
    pub documents_file: Vec<u8>,
}

/// What the gates and the comparisons scored and found.
#[derive(Debug)]
pub struct Scored {
    pub tatoeba_gate: ModelGateRecord,
    pub web_gate: ModelGateRecord,
    pub compare_tatoeba: ModelGateRecord,
    /// On the held-out windows of the languages [`COMPARED`] has.
    pub compare_web: ModelGateRecord,
    /// For each language of the plan, the held-out sentences
    /// [`COMPARED`] got right.
    pub compared_sentences: Vec<usize>,
    pub web_counts: Vec<WebLanguageCount>,
}

/// What the recipe made.
#[derive(Debug)]
pub struct WebOutcome {
    pub trained: TrainedWeb,
    pub scored: Scored,
    pub card: WebLanguageCard,
}

/// The web data a run reads: the files and ranges lists, whether they
/// are the recorded ones ([`fineweb::FILES`] and [`fineweb::RANGES`]),
/// and the directory the ranges were fetched into, made by
/// [`WebInput::checked`].
#[non_exhaustive]
pub struct WebInput<'a> {
    pub web: WebRecipe,
    pub files: &'a [WebFile],
    pub ranges: &'a [WebRange],
    pub recorded: bool,
    pub store: &'a Store<'a>,
}

impl<'a> WebInput<'a> {
    /// The web data of `files` and `ranges` in `store`, once the ranges
    /// listed are checked to be the ones `web` reads of the files, from
    /// the bytes the directory holds ([`fineweb::record`]).
    ///
    /// # Errors
    ///
    /// A range missing or not the listed bytes, a file not read as the
    /// recipe needs, or ranges listed that are not the ones the files give.
    pub fn checked(
        web: WebRecipe,
        files: &'a [WebFile],
        ranges: &'a [WebRange],
        recorded: bool,
        store: &'a Store<'a>,
    ) -> Result<WebInput<'a>> {
        let made = fineweb::record(files, &web, store)?;
        fineweb::check_ranges(ranges, &made)?;
        Ok(WebInput { web, files, ranges, recorded, store })
    }
}

/// Reads the recipe's sentences from `export` and its windows from
/// `web`, plans, trains, scores the model against `reference` (the bytes
/// of the reference's file) and `compared` (those of [`COMPARED`]'s) and
/// makes the card. `recorded_export` is whether the export is the
/// recorded one.
///
/// # Errors
///
/// A reference or compared model that is not the recorded one, web data
/// checked for another web recipe, or what reading, training or scoring
/// fail on.
pub fn run(
    _recipe: &WebLanguageRecipe,
    _export: &Export,
    _recorded_export: bool,
    _web: &WebInput<'_>,
    _reference: &[u8],
    _compared: &[u8],
) -> Result<WebOutcome> {
    Err(Error::Model("classical web language detector training and gating is retired (#1648); use native n-gram language identification (#1525)".into()))
}

/// Writes the model's directory `dir/<model id>/`: the model file,
/// [`SENTENCES_FILE`](crate::langdetect::SENTENCES_FILE), [`DOCUMENTS_FILE`],
/// [`CARD_FILE`], [`GATE_FILE`], [`WEB_GATE_FILE`], [`COMPARE_TATOEBA_FILE`]
/// and [`COMPARE_WEB_FILE`]. The directory must not exist.
///
/// # Errors
///
/// An existing directory, or a write that fails.
pub fn write_web_language_dir(dir: &Path, outcome: &WebOutcome) -> Result<()> {
    let card = &outcome.card;
    let out = dir.join(&card.id);
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    std::fs::create_dir(&out).map_err(io(&out))?;
    let files: [(&str, &[u8]); 3] = [
        (&card.file.name, &outcome.trained.bytes),
        (&card.training_sentences.name, &outcome.trained.sentences_file),
        (&card.training_documents.name, &outcome.trained.documents_file),
    ];
    for (name, bytes) in files {
        let path = out.join(name);
        std::fs::write(&path, bytes).map_err(io(&path))?;
    }
    let card_path = out.join(CARD_FILE);
    std::fs::write(&card_path, card.to_json()).map_err(io(&card_path))?;
    let s = &outcome.scored;
    s.tatoeba_gate.write(&out.join(GATE_FILE))?;
    s.web_gate.write(&out.join(WEB_GATE_FILE))?;
    s.compare_tatoeba.write(&out.join(COMPARE_TATOEBA_FILE))?;
    Ok(s.compare_web.write(&out.join(COMPARE_WEB_FILE))?)
}
