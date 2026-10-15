//! The releases that publish the recipes' models, and their entries in
//! the zoo's built-in index.
//!
//! The models are published as the assets of a release of this
//! repository on GitHub, as Kristian Rickert decided on 2026-10-15, and
//! the zoo's built-in index (`analysis/zoo/index/zoo.json`) lists each one
//! at its address there. A release names the models it publishes
//! ([`ModelsRelease`]), so a recipe added later goes out in a later
//! release and a published release keeps its assets. [`entries`] makes
//! a release's entries from the model directories `turbo-train-recipes
//! masc` writes, after checking each model file and gate record against
//! its card, and [`index_with`] is the built-in index with them added.
//! `turbo-train-recipes zoo-index` prints that index, and with `--check`
//! fails unless the zoo already carries it, which the `models` workflow
//! checks before it publishes anything.

use std::path::Path;

use turbo_eval::model_gate::ModelGateRecord;
use turbo_zoo::{Artifact, Index, verify};

use crate::card::{CARD_FILE, GATE_FILE, GateCard, ModelCard};
use crate::error::{Error, Result, io};
use crate::{RECIPES, Recipe, Reference};

/// Where this repository's releases publish their assets: asset `f` of
/// release `t` is at `{RELEASES}t/f`.
pub const RELEASES: &str = "https://github.com/ai-pipestream/turboembed/releases/download/";

/// A release of this repository whose assets are models the recipes of
/// [`RECIPES`] train.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelsRelease {
    /// The release's tag: ASCII letters, digits, `.`, `_` and `-`,
    /// starting with a letter or digit, so it is one segment of an address
    /// and a valid git tag.
    pub tag: &'static str,
    /// The ids of the models it publishes, in the order of [`RECIPES`].
    pub models: &'static [&'static str],
}

/// The release that publishes the models of the sentence detector,
/// tokenizer, part-of-speech tagger and chunker recipes.
pub const MASC_MODELS: ModelsRelease = ModelsRelease {
    tag: "models-masc-3.0.0-1",
    models: &[
        "turbo-en-masc-3.0.0-sentence-1",
        "turbo-en-masc-3.0.0-tokens-1",
        "turbo-en-masc-3.0.0-pos-1",
        "turbo-en-masc-3.0.0-chunks-1",
    ],
};

/// Every release of the recipes' models.
pub const MODELS_RELEASES: [ModelsRelease; 1] = [MASC_MODELS];

impl ModelsRelease {
    /// The release of [`MODELS_RELEASES`] tagged `tag`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownRelease`] when none is.
    pub fn tagged(tag: &str) -> Result<ModelsRelease> {
        MODELS_RELEASES.into_iter().find(|r| r.tag == tag).ok_or_else(|| Error::UnknownRelease {
            tag: tag.to_owned(),
            known: MODELS_RELEASES.iter().map(|r| r.tag).collect(),
        })
    }

    /// The address of the release's asset `file`.
    pub fn url(&self, file: &str) -> String {
        format!("{RELEASES}{}/{file}", self.tag)
    }

    /// The recipes that train the models it publishes, in the order of
    /// [`RECIPES`].
    pub fn recipes(&self) -> Vec<Recipe> {
        RECIPES.into_iter().filter(|r| self.models.contains(&r.model_id())).collect()
    }
}

/// The zoo entries of `release`'s models, in its order, made from the
/// model directories in `models` as `turbo-train-recipes masc` writes
/// them: each model file is checked against its card's size and digests,
/// and the card's gate against the gate record beside it.
///
/// # Errors
///
/// A model directory, card, gate record or model file that is missing or
/// cannot be read; a card in another model's directory, or a gate record
/// of another model or with another outcome than the card's
/// ([`Error::InvalidCard`]); a card that may not become a zoo entry
/// ([`ModelCard::artifact`]); or a model file that is not the one its card
/// describes ([`Error::Zoo`]).
pub fn entries(models: &Path, release: &ModelsRelease) -> Result<Vec<Artifact>> {
    release
        .recipes()
        .iter()
        .map(|recipe| {
            let dir = models.join(recipe.model_id());
            let card = ModelCard::read(&dir.join(CARD_FILE))?;
            if card.id != recipe.model_id() {
                return Err(Error::InvalidCard {
                    field: "id",
                    message: format!("{} in the directory of {}", card.id, recipe.model_id()),
                });
            }
            check_gate(&card, &ModelGateRecord::read(&dir.join(GATE_FILE))?)?;
            let artifact = card.artifact(&release.url(&card.file.name))?;
            let path = dir.join(&card.file.name);
            let bytes = std::fs::read(&path).map_err(io(&path))?;
            verify(&artifact, &artifact.files[0], &bytes)?;
            Ok(artifact)
        })
        .collect()
}

/// Checks that `record` is the gate of `card`'s model, with the outcome
/// the card gives, so the gate record published beside a model never
/// contradicts its card.
fn check_gate(card: &ModelCard, record: &ModelGateRecord) -> Result<()> {
    let mismatch = |message: String| Err(Error::InvalidCard { field: "gate", message });
    if record.candidate.id != card.id || record.candidate.sha256 != card.file.sha256 {
        return mismatch(format!(
            "{GATE_FILE} gates {} (SHA-256 {}), not {} (SHA-256 {})",
            record.candidate.id, record.candidate.sha256, card.id, card.file.sha256
        ));
    }
    let of_record = GateCard::of(record, card.gate.candidate_tag_format, card.gate.reference_tag_format);
    if of_record != card.gate {
        return mismatch(format!("{GATE_FILE} of {} has another outcome than its card", card.id));
    }
    Ok(())
}

/// `index` with `entries` added, as canonical JSON: an entry replaces the
/// listed artifact of its id where that one is, and the others follow the
/// listed artifacts in their order.
pub fn index_with(index: &Index, entries: &[Artifact]) -> String {
    let mut artifacts = index.list().to_vec();
    for entry in entries {
        match artifacts.iter_mut().find(|a| a.id == entry.id) {
            Some(listed) => *listed = entry.clone(),
            None => artifacts.push(entry.clone()),
        }
    }
    Index::to_canonical_json(&artifacts)
}

/// The reference models a run of `release`'s recipes reads, as
/// `(SHA-256, address)` pairs in the order of the recipes, each once:
/// what `analysis/train-recipes/masc-references.txt` lists for the
/// `models` workflow to fetch and check.
///
/// # Errors
///
/// A zoo reference the built-in index does not list.
pub fn references(release: &ModelsRelease) -> Result<Vec<(String, String)>> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for recipe in release.recipes() {
        let reference = recipe.reference();
        let url = match reference {
            Reference::Zoo(id) => Index::builtin().get(id)?.files[0].url.as_str().to_owned(),
            Reference::Recorded(r) => r.url.to_owned(),
        };
        let pair = (reference.sha256()?, url);
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use turbo_zoo::{Component, Format, Kind, LicenceClass};

    use super::*;
    use crate::card::{MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION};
    use crate::masc;

    #[test]
    fn every_release_is_tagged_once_and_names_recipes_in_their_order() {
        let tags: BTreeSet<&str> = MODELS_RELEASES.iter().map(|r| r.tag).collect();
        assert_eq!(tags.len(), MODELS_RELEASES.len());
        for release in MODELS_RELEASES {
            let tag = release.tag;
            assert!(tag.starts_with(|c: char| c.is_ascii_alphanumeric()), "{tag}");
            assert!(tag.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')), "{tag}");
            assert!(!tag.contains("..") && !tag.ends_with('.') && !tag.ends_with(".lock"), "{tag}");
            let ids: Vec<&str> = release.recipes().iter().map(Recipe::model_id).collect();
            assert_eq!(ids, release.models, "{tag}: a model no recipe trains, or out of order");
            assert_eq!(ModelsRelease::tagged(tag).unwrap(), release);
        }
        assert_eq!(
            MASC_MODELS.url("turbo-en-masc-3.0.0-pos-1.bin"),
            "https://github.com/ai-pipestream/turboembed/releases/download/models-masc-3.0.0-1/\
             turbo-en-masc-3.0.0-pos-1.bin"
        );
        for unknown in ["", "--check", "models-masc-3.0.0-2", "../models-masc-3.0.0-1"] {
            match ModelsRelease::tagged(unknown) {
                Err(Error::UnknownRelease { tag, known }) => {
                    assert_eq!(tag, unknown);
                    assert_eq!(known, [MASC_MODELS.tag]);
                }
                other => panic!("{unknown:?}: {other:?}"),
            }
        }
    }

    /// The zoo lists every model of [`MASC_MODELS`] at its address there,
    /// as its card makes the entry (permissive, needing no acceptance,
    /// trained on MASC alone), and nothing else at the release's
    /// addresses. That the bytes there are the recipes' is what
    /// `turbo-train-recipes zoo-index --check` checks on a run.
    #[test]
    fn the_builtin_index_lists_every_model_at_its_release_address() {
        let index = Index::builtin();
        for recipe in MASC_MODELS.recipes() {
            let id = recipe.model_id();
            let entry = index.get(id).unwrap();
            assert_eq!(entry.kind, Kind::Model, "{id}");
            assert_eq!(entry.component, Some(recipe.component()), "{id}");
            assert_eq!(entry.language.as_ref().unwrap().as_str(), "en", "{id}");
            assert_eq!(entry.format, Format::OpennlpModel, "{id}");
            assert_eq!(entry.version, MODEL_VERSION, "{id}");
            assert_eq!(entry.license, MODEL_LICENCE, "{id}");
            assert_eq!(entry.license_terms.as_str(), MODEL_LICENCE_TERMS, "{id}");
            assert_eq!(entry.class, LicenceClass::Permissive, "{id}");
            assert!(entry.licences_to_accept(turbo_zoo::Policy::default()).is_empty(), "{id}");
            assert_eq!(entry.trained_on, [masc::training_data()], "{id}");
            assert_eq!(entry.files.len(), 1, "{id}");
            let file = format!("{id}.bin");
            assert_eq!(entry.files[0].path.as_str(), file);
            assert_eq!(entry.files[0].url.as_str(), MASC_MODELS.url(&file));
            assert!(index.shippable().any(|a| a.id.as_str() == id), "{id}");
        }
        let prefix = MASC_MODELS.url("");
        let at_release: Vec<&str> = index
            .list()
            .iter()
            .filter(|a| a.files.iter().any(|f| f.url.as_str().starts_with(&prefix)))
            .map(|a| a.id.as_str())
            .collect();
        assert_eq!(at_release, MASC_MODELS.models);
        assert!(MASC_MODELS.recipes().iter().any(|r| r.component() == Component::Chunker));
    }

    /// The `models` workflow publishes [`MASC_MODELS`] under its tag at
    /// [`RELEASES`], and fetches the references the release's recipes
    /// read from the addresses `masc-references.txt` lists, checked
    /// against the SHA-256s it lists.
    #[test]
    fn the_models_workflow_publishes_the_release_and_fetches_its_references() {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workflow = std::fs::read_to_string(crate_dir.join("../../.github/workflows/models.yml")).unwrap();
        let lines: Vec<&str> = workflow.lines().map(str::trim).collect();
        assert!(lines.contains(&format!("TAG: {}", MASC_MODELS.tag).as_str()), "models.yml's TAG");
        assert!(lines.contains(&format!("RELEASES: {RELEASES}").as_str()), "models.yml's RELEASES");
        assert!(workflow.contains("analysis/train-recipes/masc-references.txt"));

        let listed = std::fs::read_to_string(crate_dir.join("masc-references.txt")).unwrap();
        let listed: Vec<(String, String)> = listed
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                let (sha256, url) = l.split_once(' ').unwrap();
                (sha256.to_owned(), url.to_owned())
            })
            .collect();
        let references = references(&MASC_MODELS).unwrap();
        assert_eq!(listed, references);
        // The workflow saves each file under the address's last segment,
        // the name a run reads it by.
        for recipe in MASC_MODELS.recipes() {
            let reference = recipe.reference();
            let (sha256, file) = (reference.sha256().unwrap(), reference.file().unwrap());
            assert!(
                listed.iter().any(|(s, url)| *s == sha256 && url.rsplit('/').next() == Some(file.as_str())),
                "{}",
                reference.id()
            );
        }
    }
}
