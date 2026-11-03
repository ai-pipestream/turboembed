//! Registered training-only model selection for the MASC namefinder -2 attempt.
//!
//! Input is only the training corpus. Held-out data cannot affect selection.
//! The order in [`GRID`] also breaks equal merged F-measures, first wins.
mod record;
mod run;
pub use record::{Candidate, FoldCard, Scores, SelectionCard};
pub use run::{SelectedTraining, finish, train_selected, write_attempt};

use serde::{Deserialize, Serialize};
use turbo_formats::evaluate::CrossValidation;
use turbo_formats::masc::MascCorpus;
use turbo_train::{Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::namefind::TokenNameFinderCrossValidator;
use turbo_train_components::namefind::cross_validate::documents;

use super::{NameRecipe, NameSamples, check_training, factory, read_samples};
use crate::error::{Error, Result};
use crate::segmentation::TIMESTAMP_MS;

pub const FOLDS: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Clearing {
    Document,
    Sentence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Setting {
    pub iterations: i32,
    pub cutoff: i32,
    pub clearing: Clearing,
}

impl Setting {
    pub fn parameters(self) -> TrainingParameters {
        let mut parameters = super::training_parameters();
        parameters.put(Key::ITERATIONS, self.iterations);
        parameters.put(Key::CUTOFF, self.cutoff);
        parameters
    }
}

/// Fixed before any -2 held-out evaluation; no command-line tuning options.
pub const GRID: [Setting; 4] = [
    Setting { iterations: 300, cutoff: 0, clearing: Clearing::Document },
    Setting { iterations: 300, cutoff: 0, clearing: Clearing::Sentence },
    Setting { iterations: 100, cutoff: 5, clearing: Clearing::Document },
    Setting { iterations: 100, cutoff: 5, clearing: Clearing::Sentence },
];

pub const RECIPES: [NameRecipe; 4] = [
    NameRecipe { model_id: "turbo-en-masc-3.0.0-ner-person-2", ..super::PERSON },
    NameRecipe { model_id: "turbo-en-masc-3.0.0-ner-location-2", ..super::LOCATION },
    NameRecipe { model_id: "turbo-en-masc-3.0.0-ner-organization-2", ..super::ORGANIZATION },
    NameRecipe { model_id: "turbo-en-masc-3.0.0-ner-date-2", ..super::DATE },
];

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidCard { field: "namefinder_selection", message: message.into() }
}

fn training_samples<'a>(recipe: &NameRecipe, train: &'a MascCorpus) -> Result<NameSamples<'a>> {
    let mut samples = Vec::new();
    let (counts, skipped) = read_samples(train, &recipe.types, |sample| {
        samples.push(sample);
        Ok(())
    })?;
    check_training(samples.len(), &counts, skipped)?;
    Ok(NameSamples { samples, counts, skipped })
}

/// Runs only the registered grid against training documents. Every candidate
/// records all ten partitions and merged scores. No held-out argument exists.
pub fn select(recipe: &NameRecipe, train: &MascCorpus) -> Result<SelectionCard> {
    if !RECIPES.contains(recipe) {
        return Err(invalid("selection requires a registered -2 recipe"));
    }
    let samples = training_samples(recipe, train)?;
    let docs = documents(samples.samples());
    if docs.len() < FOLDS || docs.len() != samples.counts.documents {
        return Err(invalid("selection requires at least ten nonempty whole documents"));
    }
    let factory = factory()?;
    let mut candidates = Vec::new();
    for setting in GRID {
        let cv = TokenNameFinderCrossValidator::new(&factory, None, setting.parameters());
        let cv = if setting.clearing == Clearing::Sentence { cv.with_sentence_clearing() } else { cv };
        let result =
            cv.evaluate(samples.samples(), FOLDS, &mut TrainingConfiguration::default(), Stamp::at(TIMESTAMP_MS))?;
        let folds = result
            .folds
            .iter()
            .zip(CrossValidation::new(docs.len(), FOLDS))
            .map(|(fold, partition)| FoldCard::new(fold, partition))
            .collect();
        candidates.push(Candidate { setting, folds, merged: Scores::of(&result.fmeasure) });
    }
    let selected = record::best(&candidates)?;
    Ok(SelectionCard {
        version: 1,
        model_id: recipe.model_id.to_owned(),
        training_samples_sha256: samples.sha256(&recipe.types),
        training_documents: docs.len(),
        training_samples: samples.len(),
        folds: FOLDS,
        tie_rule: "first candidate in registered grid order".to_owned(),
        selected,
        candidates,
    })
}
