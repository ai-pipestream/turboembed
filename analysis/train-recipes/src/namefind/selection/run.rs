use std::io::Write;
use std::path::Path;

use turbo_formats::NameSample;
use turbo_formats::masc::MascCorpus;
use turbo_train::TrainingConfiguration;
use turbo_train_components::Stamp;
use turbo_train_components::namefind::{NameEventStream, train_name_model_from_events};

use super::{Clearing, GRID, RECIPES, SelectionCard, invalid, training_samples};
use crate::card::{CorpusCard, ModelCard, ParametersCard};
use crate::error::{Result, io};
use crate::masc::{self, CorpusDigest, SplitCorpus};
use crate::namefind::{NameRecipe, factory};
use crate::segmentation::TIMESTAMP_MS;
use crate::{GateRun, Outcome, Recipe, Trained, held_out_name};

/// A trained -2 candidate. Consumed by `finish`, which runs its gate once.
#[derive(Debug)]
pub struct SelectedTraining {
    selection: SelectionCard,
    trained: Trained,
    recipe: NameRecipe,
}

impl SelectedTraining {
    /// The deterministic model package, before held-out scoring.
    pub fn model_bytes(&self) -> &[u8] {
        &self.trained.bytes
    }
}

/// Train the selected configuration on all training documents. Reject a card
/// for different samples or a changed grid; never accept a held-out score here.
pub fn train_selected(recipe: &NameRecipe, train: &MascCorpus, selection: SelectionCard) -> Result<SelectedTraining> {
    let samples = training_samples(recipe, train)?;
    if !RECIPES.contains(recipe)
        || selection.version != 1
        || selection.folds != super::FOLDS
        || selection.training_documents != samples.counts().documents
        || selection.training_samples != samples.len()
        || selection.model_id != recipe.model_id
        || selection.training_samples_sha256 != samples.sha256(&recipe.types)
        || selection.candidates.iter().map(|c| c.setting).collect::<Vec<_>>() != GRID
        || selection.selected != super::record::best(&selection.candidates)?
    {
        return Err(invalid("selection card does not match this training input and registered grid"));
    }
    let setting = selection.candidates[selection.selected].setting;
    let factory = factory()?;
    let mut stream = NameEventStream::new(&factory, None)?;
    let mut events = Vec::new();
    for sample in samples.samples() {
        if setting.clearing == Clearing::Sentence {
            stream.clear_adaptive_data();
        }
        stream.events(sample, &mut events)?;
    }
    let trained = train_name_model_from_events(
        &events,
        &factory,
        &setting.parameters(),
        &mut TrainingConfiguration::default(),
        Stamp::at(TIMESTAMP_MS),
    )?;
    Ok(SelectedTraining {
        recipe: *recipe,
        selection,
        trained: Trained {
            bytes: trained.package.to_bytes(),
            samples: samples.len(),
            skipped_sentences: samples.skipped(),
            events: events.len(),
            event_hash: trained.report.event_hash.expect("trained from events"),
            names: Some(samples.counts().clone()),
            chunks: None,
        },
    })
}

/// Consumes the selected model and evaluates it and the pinned reference once
/// on the original held-out side. Selection and final training are already over.
pub fn finish(
    selected: SelectedTraining,
    split: &SplitCorpus,
    digest: &CorpusDigest,
    recorded: bool,
    reference: &[u8],
) -> Result<(Outcome, SelectionCard)> {
    let SelectedTraining { recipe, trained, selection } = selected;
    recipe.reference.check(reference)?;
    let setting = selection.candidates[selection.selected].setting;
    let mut held_out = recipe.held_out(&split.held_out)?;
    if setting.clearing == Clearing::Sentence {
        held_out.samples = held_out
            .samples
            .iter()
            .map(|s| NameSample::new(s.tokens().to_vec(), s.names().to_vec(), true, s.source(), 0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| invalid(e.to_string()))?;
    }
    let GateRun { record: gate, held_out } =
        recipe.gate(&trained.bytes, recipe.reference.id(), reference, &held_out, &held_out_name(), &digest.sha256)?;
    let corpus = CorpusCard {
        name: masc::NAME.to_owned(),
        corpus_sha256: digest.sha256.clone(),
        release_recorded: recorded,
        split: masc::SPLIT,
        training_documents: split.train.documents().len(),
        held_out_documents: split.held_out_paths.clone(),
        excluded_documents: split.excluded.clone(),
    };
    let mut card = ModelCard::new(&Recipe::Name(recipe), &trained, corpus, held_out, &gate);
    card.training.parameters = ParametersCard {
        algorithm: "MAXENT".to_owned(),
        iterations: setting.iterations,
        cutoff: setting.cutoff,
        threads: 1,
    };
    Ok((Outcome { trained, gate, card }, selection))
}

/// Creates one attempt directory before any CV, refusing an existing attempt.
/// Persists the complete selection before final training and held-out scoring.
/// Partial attempts remain on disk and are never automatically overwritten.
pub fn write_attempt(
    recipe: &NameRecipe,
    split: &SplitCorpus,
    digest: &CorpusDigest,
    recorded: bool,
    reference: &[u8],
    out: &Path,
) -> Result<bool> {
    if !RECIPES.contains(recipe) {
        return Err(invalid("attempt requires a registered -2 recipe"));
    }
    if recorded && (split.train.documents().len() != 351 || split.held_out.documents().len() != 39) {
        return Err(invalid("recorded -2 split must contain 351 training and 39 held-out documents"));
    }
    recipe.reference.check(reference)?;
    let directory = out.join(recipe.model_id);
    reserve_attempt(out, &directory)?;
    let selection = super::select(recipe, &split.train)?;
    let selection_json = serde_json::to_string_pretty(&selection).expect("finite CV scores");
    write_new(&directory.join("selection.json"), selection_json.as_bytes())?;
    let trained = train_selected(recipe, &split.train, selection)?;
    let (outcome, _) = finish(trained, split, digest, recorded, reference)?;
    write_new(&directory.join(&outcome.card.file.name), &outcome.trained.bytes)?;
    write_new(&directory.join("card.json"), outcome.card.to_json().as_bytes())?;
    write_new(&directory.join("gate.json"), outcome.gate.to_json().as_bytes())?;
    Ok(outcome.gate.result.pass)
}

fn reserve_attempt(out: &Path, directory: &Path) -> Result<()> {
    std::fs::create_dir_all(out).map_err(io(out))?;
    std::fs::create_dir(directory).map_err(io(directory))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io(path))?
        .write_all(bytes)
        .map_err(io(path))
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod tests;
