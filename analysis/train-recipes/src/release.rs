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
//! masc`, `ud` or `tatoeba` writes, after checking each file and gate
//! record against its card, and [`index_with`] is the built-in index with
//! them added. `turbo-train-recipes zoo-index` prints that index, and with
//! `--check` fails unless the zoo already carries it, which the `models`,
//! `models-ud` and `models-tatoeba` workflows check before they publish
//! anything.

use std::path::Path;

use turbo_eval::model_gate::ModelGateRecord;
use turbo_zoo::{Artifact, Index, verify};

use crate::card::{CARD_FILE, GATE_FILE, GateCard, ModelCard, ModelFileCard};
use crate::error::{Error, Result, io};
use crate::langdetect::{LANGDETECT, LanguageCard, LanguageRecipe};
use crate::ud::{UD_RECIPES, UdRecipe};
use crate::ud_card::UdModelCard;
use crate::{RECIPES, Recipe, Reference};

/// Where this repository's releases publish their assets: asset `f` of
/// release `t` is at `{RELEASES}t/f`.
pub const RELEASES: &str = "https://github.com/ai-pipestream/turboembed/releases/download/";

/// A release of this repository whose assets are models the recipes of
/// [`RECIPES`] and [`UD_RECIPES`] or the language detector recipe
/// ([`LANGDETECT`]) train.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelsRelease {
    /// The release's tag: ASCII letters, digits, `.`, `_` and `-`,
    /// starting with a letter or digit, so it is one segment of an address
    /// and a valid git tag.
    pub tag: &'static str,
    /// The ids of the models it publishes: those of [`RECIPES`] in its
    /// order, then those of [`UD_RECIPES`] in its order, then the language
    /// detector's.
    pub models: &'static [&'static str],
}

/// The release that publishes the models of the MASC sentence detector,
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

/// The release that publishes the models of the UD recipes that passed
/// their gates and may be shipped: Spanish and Catalan AnCora's four and
/// Finnish FTB's tagger and lemmatizer. Portuguese Porttinari's are not
/// in it, as the rights to its text are not settled, nor Italian
/// MarkIT's, which failed their gates.
pub const UD_MODELS: ModelsRelease = ModelsRelease {
    tag: "models-ud-2.16-1",
    models: &[
        "turbo-es-ud-ancora-2.16-sentence-1",
        "turbo-es-ud-ancora-2.16-tokens-1",
        "turbo-es-ud-ancora-2.16-pos-1",
        "turbo-es-ud-ancora-2.16-lemmas-1",
        "turbo-ca-ud-ancora-2.16-sentence-1",
        "turbo-ca-ud-ancora-2.16-tokens-1",
        "turbo-ca-ud-ancora-2.16-pos-1",
        "turbo-ca-ud-ancora-2.16-lemmas-1",
        "turbo-fi-ud-ftb-2.16-pos-1",
        "turbo-fi-ud-ftb-2.16-lemmas-1",
    ],
};

/// The release that publishes the language detector the Tatoeba recipe
/// trains on the export of 2026-10-10 (`langdetect::EXPORT`), with the ids
/// of the sentences it was trained on.
pub const TATOEBA_MODELS: ModelsRelease =
    ModelsRelease { tag: "models-tatoeba-2026-10-10-1", models: &[crate::langdetect::MODEL_ID] };

/// Every release of the recipes' models.
pub const MODELS_RELEASES: [ModelsRelease; 3] = [MASC_MODELS, UD_MODELS, TATOEBA_MODELS];

/// The release of this repository that keeps the Tatoeba export the
/// language detector is trained on (`langdetect::EXPORT`): its two
/// archives as Tatoeba published them, which Tatoeba replaces every week
/// at the same addresses. `analysis/train-recipes/tatoeba-export.txt`
/// lists their SHA-256s and addresses, which the `models-tatoeba`
/// workflow fetches them from.
pub const TATOEBA_EXPORT_RELEASE: &str = "data-tatoeba-2026-10-10";

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

    /// The MASC recipes that train models it publishes, in the order of
    /// [`RECIPES`].
    pub fn recipes(&self) -> Vec<Recipe> {
        RECIPES.into_iter().filter(|r| self.models.contains(&r.model_id())).collect()
    }

    /// The UD recipes that train models it publishes, in the order of
    /// [`UD_RECIPES`].
    pub fn ud_recipes(&self) -> Vec<UdRecipe> {
        UD_RECIPES.into_iter().filter(|r| self.models.contains(&r.model_id)).collect()
    }

    /// The language detector recipe, when it publishes its model.
    pub fn language_recipes(&self) -> Vec<LanguageRecipe> {
        [LANGDETECT].into_iter().filter(|r| self.models.contains(&r.model_id)).collect()
    }

    /// The references of the recipes that train its models, in the order
    /// of its models.
    fn recipe_references(&self) -> Vec<Reference> {
        let masc = self.recipes().into_iter().map(|r| r.reference());
        let ud = self.ud_recipes().into_iter().map(|r| r.reference);
        masc.chain(ud).chain(self.language_recipes().into_iter().map(|r| r.reference)).collect()
    }
}

/// What [`entries`] reads of a model's card: the MASC recipes' card, the
/// UD recipes' one or the language detector's.
trait Card: Sized {
    fn read(path: &Path) -> Result<Self>;
    fn id(&self) -> &str;
    fn file(&self) -> &ModelFileCard;
    fn gate(&self) -> &GateCard;
    /// The model's zoo entry, its files at their addresses in `release`.
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact>;
}

impl Card for ModelCard {
    fn read(path: &Path) -> Result<Self> {
        ModelCard::read(path)
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn file(&self) -> &ModelFileCard {
        &self.file
    }
    fn gate(&self) -> &GateCard {
        &self.gate
    }
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact> {
        ModelCard::artifact(self, &release.url(&self.file.name))
    }
}

impl Card for UdModelCard {
    fn read(path: &Path) -> Result<Self> {
        UdModelCard::read(path)
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn file(&self) -> &ModelFileCard {
        &self.file
    }
    fn gate(&self) -> &GateCard {
        &self.gate
    }
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact> {
        UdModelCard::artifact(self, &release.url(&self.file.name))
    }
}

impl Card for LanguageCard {
    fn read(path: &Path) -> Result<Self> {
        LanguageCard::read(path)
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn file(&self) -> &ModelFileCard {
        &self.file
    }
    fn gate(&self) -> &GateCard {
        &self.gate
    }
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact> {
        LanguageCard::artifact(self, &release.url(&self.file.name), &release.url(&self.training_sentences.name))
    }
}

/// The zoo entries of `release`'s models, in its order, made from the
/// model directories in `models` as `turbo-train-recipes masc`, `ud` and
/// `tatoeba` write them: each file of an entry (the model, and for the
/// language detector the ids of the sentences it was trained on) is
/// checked against its card's size and digests, and the card's gate
/// against the gate record beside it.
///
/// # Errors
///
/// A model directory, card, gate record or file that is missing or cannot
/// be read; a card in another model's directory, or a gate record of
/// another model or with another outcome than the card's
/// ([`Error::InvalidCard`]); a card that may not become a zoo entry
/// ([`ModelCard::artifact`], [`UdModelCard::artifact`],
/// [`LanguageCard::artifact`]); or a file that is not the one its card
/// describes ([`Error::Zoo`]).
pub fn entries(models: &Path, release: &ModelsRelease) -> Result<Vec<Artifact>> {
    release
        .models
        .iter()
        .map(|id| match Recipe::trains(id) {
            Some(_) => entry::<ModelCard>(models, id, release),
            None if *id == LANGDETECT.model_id => entry::<LanguageCard>(models, id, release),
            None => entry::<UdModelCard>(models, id, release),
        })
        .collect()
}

/// The zoo entry of model `id` of `release`, from its directory in
/// `models`.
fn entry<C: Card>(models: &Path, id: &str, release: &ModelsRelease) -> Result<Artifact> {
    let dir = models.join(id);
    let card = C::read(&dir.join(CARD_FILE))?;
    if card.id() != id {
        return Err(Error::InvalidCard { field: "id", message: format!("{} in the directory of {id}", card.id()) });
    }
    check_gate(card.id(), card.file(), card.gate(), &ModelGateRecord::read(&dir.join(GATE_FILE))?)?;
    let artifact = card.artifact(release)?;
    for file in &artifact.files {
        let path = dir.join(file.path.as_str());
        let bytes = std::fs::read(&path).map_err(io(&path))?;
        verify(&artifact, file, &bytes)?;
    }
    Ok(artifact)
}

/// Checks that `record` is the gate of model `id` (its file `file`), with
/// the outcome its card gives (`gate`), so the gate record published
/// beside a model never contradicts its card.
fn check_gate(id: &str, file: &ModelFileCard, gate: &GateCard, record: &ModelGateRecord) -> Result<()> {
    let mismatch = |message: String| Err(Error::InvalidCard { field: "gate", message });
    if record.candidate.id != id || record.candidate.sha256 != file.sha256 {
        return mismatch(format!(
            "{GATE_FILE} gates {} (SHA-256 {}), not {id} (SHA-256 {})",
            record.candidate.id, record.candidate.sha256, file.sha256
        ));
    }
    if GateCard::of(record, gate.candidate_tag_format, gate.reference_tag_format) != *gate {
        return mismatch(format!("{GATE_FILE} of {id} has another outcome than its card"));
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

/// The ids of the artifacts of `index` with a file at an address of
/// `release`, in the index's order.
pub fn listed_at(index: &Index, release: &ModelsRelease) -> Vec<String> {
    let prefix = release.url("");
    index
        .list()
        .iter()
        .filter(|a| a.files.iter().any(|f| f.url.as_str().starts_with(&prefix)))
        .map(|a| a.id.to_string())
        .collect()
}

/// Where `index` differs from the index `entries` of `release`'s models
/// give: a line for each entry that `index` lists otherwise, with the
/// size, SHA-256 and address of each file of both, or does not list, and
/// one for each artifact `index` lists at an address of the release that
/// is not one of its models. None when `index` lists `entries` and
/// nothing else there.
pub fn differences(index: &Index, entries: &[Artifact], release: &ModelsRelease) -> Vec<String> {
    let files = |a: &Artifact| {
        let files: Vec<String> =
            a.files.iter().map(|f| format!("{} bytes, SHA-256 {}, at {}", f.size, f.sha256, f.url)).collect();
        files.join("; ")
    };
    let mut lines = Vec::new();
    for entry in entries {
        match index.get(entry.id.as_str()) {
            Ok(listed) if listed == entry => {}
            Ok(listed) => {
                lines.push(format!(
                    "{}: the built-in index lists another entry ({}; the model has {})",
                    entry.id,
                    files(listed),
                    files(entry)
                ));
            }
            Err(_) => lines.push(format!("{}: the built-in index does not list it", entry.id)),
        }
    }
    for id in listed_at(index, release) {
        if !release.models.contains(&id.as_str()) {
            lines.push(format!(
                "{id}: the built-in index lists it at an address of release {}, which does not publish it",
                release.tag
            ));
        }
    }
    lines
}

/// The reference models a run of `release`'s recipes reads, as
/// `(SHA-256, address)` pairs in the order of its models, each once: what
/// `analysis/train-recipes/masc-references.txt`, `ud-references.txt` and
/// `tatoeba-references.txt` list for the `models`, `models-ud` and
/// `models-tatoeba` workflows to fetch and check.
///
/// # Errors
///
/// A zoo reference the built-in index does not list.
pub fn references(release: &ModelsRelease) -> Result<Vec<(String, String)>> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for reference in release.recipe_references() {
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

    use turbo_zoo::{Component, Format, Kind, LicenceClass, TrainingData};

    use super::*;
    use crate::card::{MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION};
    use crate::langdetect::{self, MODEL_ID, SENTENCES_FILE, ZOO_LANGUAGE};
    use crate::treebank::TextRights;
    use crate::{masc, tatoeba};

    #[test]
    fn every_release_is_tagged_once_and_names_recipes_in_their_order() {
        let tags: BTreeSet<&str> = MODELS_RELEASES.iter().map(|r| r.tag).collect();
        assert_eq!(tags.len(), MODELS_RELEASES.len());
        for release in MODELS_RELEASES {
            let tag = release.tag;
            assert!(tag.starts_with(|c: char| c.is_ascii_alphanumeric()), "{tag}");
            assert!(tag.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')), "{tag}");
            assert!(!tag.contains("..") && !tag.ends_with('.') && !tag.ends_with(".lock"), "{tag}");
            let masc = release.recipes().into_iter().map(|r| r.model_id());
            let ud = release.ud_recipes().into_iter().map(|r| r.model_id);
            let ids: Vec<&str> =
                masc.chain(ud).chain(release.language_recipes().into_iter().map(|r| r.model_id)).collect();
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
                    assert_eq!(known, [MASC_MODELS.tag, UD_MODELS.tag, TATOEBA_MODELS.tag]);
                }
                other => panic!("{unknown:?}: {other:?}"),
            }
        }
    }

    /// The UD release holds every UD recipe's model but those of a
    /// treebank whose text rights are not settled and those that failed
    /// their gates, the Italian ones (analysis/train-recipes/README.md).
    #[test]
    fn the_ud_release_leaves_out_porttinari_and_markit() {
        for recipe in UD_RECIPES {
            let released = UD_MODELS.models.contains(&recipe.model_id);
            let settled = !matches!(recipe.treebank.text_rights, TextRights::NotSettled { .. });
            let italian = recipe.treebank.language == "it";
            assert_eq!(released, settled && !italian, "{}", recipe.model_id);
        }
    }

    /// The zoo lists `id`'s model at its address in `release`, as its card
    /// makes the entry: permissive, needing no acceptance, trained on
    /// `trained_on` alone, its files the model and then `extra`, each at
    /// its address in the release.
    fn listed(
        index: &Index,
        release: &ModelsRelease,
        id: &str,
        component: Component,
        language: &str,
        trained_on: TrainingData,
        extra: &[&str],
    ) {
        let entry = index.get(id).unwrap();
        assert_eq!(entry.kind, Kind::Model, "{id}");
        assert_eq!(entry.component, Some(component), "{id}");
        assert_eq!(entry.language.as_ref().unwrap().as_str(), language, "{id}");
        assert_eq!(entry.format, Format::OpennlpModel, "{id}");
        assert_eq!(entry.version, MODEL_VERSION, "{id}");
        assert_eq!(entry.license, MODEL_LICENCE, "{id}");
        assert_eq!(entry.license_terms.as_str(), MODEL_LICENCE_TERMS, "{id}");
        assert_eq!(entry.class, LicenceClass::Permissive, "{id}");
        assert!(entry.licences_to_accept(turbo_zoo::Policy::default()).is_empty(), "{id}");
        assert_eq!(entry.trained_on, [trained_on], "{id}");
        let files: Vec<String> =
            std::iter::once(format!("{id}.bin")).chain(extra.iter().map(|f| (*f).to_owned())).collect();
        assert_eq!(entry.files.len(), files.len(), "{id}");
        for (listed, file) in entry.files.iter().zip(&files) {
            assert_eq!(listed.path.as_str(), file, "{id}");
            assert_eq!(listed.url.as_str(), release.url(file), "{id}");
        }
        assert!(index.shippable().any(|a| a.id.as_str() == id), "{id}");
    }

    /// The zoo lists every model of each release at its address there, and
    /// nothing else at the release's addresses. That the bytes there are
    /// the recipes' is what `turbo-train-recipes zoo-index --check` checks
    /// on a run.
    #[test]
    fn the_builtin_index_lists_every_model_at_its_release_address() {
        let index = Index::builtin();
        for recipe in MASC_MODELS.recipes() {
            listed(&index, &MASC_MODELS, recipe.model_id(), recipe.component(), "en", masc::training_data(), &[]);
        }
        for recipe in UD_MODELS.ud_recipes() {
            let (language, trained_on) = (recipe.treebank.language, recipe.treebank.training_data());
            listed(&index, &UD_MODELS, recipe.model_id, recipe.zoo_component(), language, trained_on, &[]);
        }
        let export = langdetect::EXPORT.digest().unwrap();
        let tatoeba = TrainingData {
            name: format!("Tatoeba export {}", &export[..16]),
            license: turbo_zoo::Licence::CcBy20Fr,
            license_terms: tatoeba::TERMS_URL.parse().unwrap(),
        };
        assert_eq!(langdetect::training_data(export), tatoeba);
        let (id, component) = (MODEL_ID, Component::LanguageDetector);
        listed(&index, &TATOEBA_MODELS, id, component, ZOO_LANGUAGE, tatoeba, &[SENTENCES_FILE]);
        for release in MODELS_RELEASES {
            assert_eq!(listed_at(&index, &release), release.models, "{}", release.tag);
        }
        assert!(MASC_MODELS.recipes().iter().any(|r| r.component() == Component::Chunker));
    }

    /// The built-in index with its own entries of a release has no
    /// differences; one with an entry changed, missing or added at the
    /// release's addresses has one line for each.
    #[test]
    fn differences_name_every_entry_listed_otherwise_and_every_other_model_at_the_release() {
        let builtin = Index::builtin();
        let entries: Vec<Artifact> = MASC_MODELS.models.iter().map(|id| builtin.get(id).unwrap().clone()).collect();
        assert!(differences(&builtin, &entries, &MASC_MODELS).is_empty());

        let mut changed = entries.clone();
        changed[1].files[0].size += 1;
        let lines = differences(&builtin, &changed, &MASC_MODELS);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with(&format!("{}: the built-in index lists another entry (", entries[1].id)));

        let mut extra = entries[0].clone();
        extra.id = "turbo-en-masc-3.0.0-extra-1".parse().unwrap();
        extra.files[0].path = "turbo-en-masc-3.0.0-extra-1.bin".parse().unwrap();
        extra.files[0].url = MASC_MODELS.url("turbo-en-masc-3.0.0-extra-1.bin").parse().unwrap();
        let listed: Vec<Artifact> =
            builtin.list().iter().filter(|a| a.id != entries[0].id).cloned().chain([extra]).collect();
        let index = Index::parse(Index::to_canonical_json(&listed).as_bytes()).unwrap();
        assert_eq!(
            differences(&index, &entries, &MASC_MODELS),
            [
                format!("{}: the built-in index does not list it", entries[0].id),
                "turbo-en-masc-3.0.0-extra-1: the built-in index lists it at an address of release \
                 models-masc-3.0.0-1, which does not publish it"
                    .to_owned(),
            ]
        );
    }

    /// The workflow that publishes `release` names its tag and [`RELEASES`],
    /// publishes through models-publish.yml, and fetches the references the
    /// release's recipes read from the addresses `references_file` lists,
    /// checked against the SHA-256s it lists.
    fn published_by(release: &ModelsRelease, workflow_file: &str, references_file: &str) {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workflow = std::fs::read_to_string(crate_dir.join("../../.github/workflows").join(workflow_file)).unwrap();
        let lines: Vec<&str> = workflow.lines().map(str::trim).collect();
        assert!(lines.contains(&format!("TAG: {}", release.tag).as_str()), "{workflow_file}'s TAG");
        assert!(lines.contains(&format!("RELEASES: {RELEASES}").as_str()), "{workflow_file}'s RELEASES");
        assert!(workflow.contains(&format!("analysis/train-recipes/{references_file}")), "{workflow_file}");
        assert!(lines.contains(&"uses: ./.github/workflows/models-publish.yml"), "{workflow_file} publishes");

        let listed = std::fs::read_to_string(crate_dir.join(references_file)).unwrap();
        let listed: Vec<(String, String)> = listed
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                let (sha256, url) = l.split_once(' ').unwrap();
                (sha256.to_owned(), url.to_owned())
            })
            .collect();
        assert_eq!(listed, references(release).unwrap(), "{references_file}");
        // The workflow saves each file under the address's last segment,
        // the name a run reads it by.
        for reference in release.recipe_references() {
            let (sha256, file) = (reference.sha256().unwrap(), reference.file().unwrap());
            assert!(
                listed.iter().any(|(s, url)| *s == sha256 && url.rsplit('/').next() == Some(file.as_str())),
                "{}",
                reference.id()
            );
        }
    }

    #[test]
    fn the_models_workflows_publish_the_releases_and_fetch_their_references() {
        published_by(&MASC_MODELS, "models.yml", "masc-references.txt");
        published_by(&UD_MODELS, "models-ud.yml", "ud-references.txt");
        published_by(&TATOEBA_MODELS, "models-tatoeba.yml", "tatoeba-references.txt");
    }

    /// The `models-tatoeba` workflow names the release that keeps the
    /// recorded export and reads `tatoeba-export.txt`, which lists the
    /// export's two archives at their addresses in that release, each
    /// under the name Tatoeba publishes it by, with a SHA-256 each; and it
    /// compares the model's four files between its two runs.
    #[test]
    fn the_tatoeba_workflow_trains_on_the_kept_export() {
        let tag = TATOEBA_EXPORT_RELEASE;
        assert!(tag.starts_with(|c: char| c.is_ascii_alphanumeric()), "{tag}");
        assert!(tag.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')), "{tag}");
        assert!(MODELS_RELEASES.iter().all(|r| r.tag != tag));

        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workflow = std::fs::read_to_string(crate_dir.join("../../.github/workflows/models-tatoeba.yml")).unwrap();
        let lines: Vec<&str> = workflow.lines().map(str::trim).collect();
        assert!(lines.contains(&format!("DATA_TAG: {tag}").as_str()));
        assert!(workflow.contains("analysis/train-recipes/tatoeba-export.txt"));
        let listed = std::fs::read_to_string(crate_dir.join("tatoeba-export.txt")).unwrap();
        let listed: Vec<(&str, &str)> = listed
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.split_once(' ').unwrap())
            .collect();
        let urls: Vec<String> = [tatoeba::DETAILED_ARCHIVE_URL, tatoeba::CC0_ARCHIVE_URL]
            .map(|u| format!("{RELEASES}{tag}/{}", u.rsplit('/').next().unwrap()))
            .to_vec();
        assert_eq!(listed.iter().map(|(_, url)| (*url).to_owned()).collect::<Vec<_>>(), urls);
        for (sha256, url) in listed {
            assert!(sha256.len() == 64 && sha256.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')), "{url}");
        }
        // The runs write the model's directory under its id.
        assert!(lines.contains(&format!("id={MODEL_ID}").as_str()));
        assert!(workflow.contains(&format!("\"$id.bin\" {SENTENCES_FILE} {CARD_FILE} {GATE_FILE}")));
    }
}
