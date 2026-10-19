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
//! masc`, `ud`, `tatoeba` or `tatoeba-fineweb` writes, after checking each
//! file, gate record and comparison record against its card, and
//! [`index_with`] is the built-in index with them added.
//! `turbo-train-recipes zoo-index` prints that index, and with `--check`
//! fails unless the zoo already carries it, which the `models`,
//! `models-ud`, `models-tatoeba` and `models-fineweb` workflows check
//! before they publish anything.

mod cards;
#[cfg(test)]
mod tests;

use std::path::Path;

use turbo_zoo::{Artifact, Index};

use crate::card::ModelCard;
use crate::error::{Error, Result};
use crate::langdetect::{LANGDETECT, LanguageCard, LanguageRecipe};
use crate::langdetect_web::{LANGDETECT_WEB, WebLanguageCard, WebLanguageRecipe};
use crate::ud::{UD_RECIPES, UdRecipe};
use crate::ud_card::UdModelCard;
use crate::{RECIPES, Recipe, Reference};

use self::cards::entry;

/// Where this repository's releases publish their assets: asset `f` of
/// release `t` is at `{RELEASES}t/f`.
pub const RELEASES: &str = "https://github.com/ai-pipestream/turboembed/releases/download/";

/// A release of this repository whose assets are models the recipes of
/// [`RECIPES`] and [`UD_RECIPES`] or the language detector recipes
/// ([`LANGDETECT`] and [`LANGDETECT_WEB`]) train.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelsRelease {
    /// The release's tag: ASCII letters, digits, `.`, `_` and `-`,
    /// starting with a letter or digit, so it is one segment of an address
    /// and a valid git tag.
    pub tag: &'static str,
    /// The ids of the models it publishes: those of [`RECIPES`] in its
    /// order, then those of [`UD_RECIPES`] in its order, then the language
    /// detector's, then the web language detector's.
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

/// The release that publishes the language detector the web recipe
/// trains on the same Tatoeba export and on the FineWeb-2 and FineWeb
/// documents of the recorded lists (`fineweb::FILES`), with the ids of
/// the sentences and the documents it was trained on.
pub const FINEWEB_MODELS: ModelsRelease =
    ModelsRelease { tag: "models-tatoeba-fineweb-2026-10-10-1", models: &[crate::langdetect_web::MODEL_ID] };

/// Every release of the recipes' models.
pub const MODELS_RELEASES: [ModelsRelease; 4] = [MASC_MODELS, UD_MODELS, TATOEBA_MODELS, FINEWEB_MODELS];

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

    /// The web language detector recipe, when it publishes its model.
    pub fn web_language_recipes(&self) -> Vec<WebLanguageRecipe> {
        [LANGDETECT_WEB].into_iter().filter(|r| self.models.contains(&r.model_id)).collect()
    }

    /// The references of the recipes that train its models, in the order
    /// of its models: for the web language detector, its gates' reference
    /// and then the model it is compared with.
    pub(super) fn recipe_references(&self) -> Vec<Reference> {
        let masc = self.recipes().into_iter().map(|r| r.reference());
        let ud = self.ud_recipes().into_iter().map(|r| r.reference);
        let language = self.language_recipes().into_iter().map(|r| r.reference);
        let web = self.web_language_recipes().into_iter().flat_map(|r| [r.tatoeba.reference, r.compared]);
        masc.chain(ud).chain(language).chain(web).collect()
    }
}

/// The zoo entries of `release`'s models, in its order, made from the
/// model directories in `models` as `turbo-train-recipes masc`, `ud`,
/// `tatoeba` and `tatoeba-fineweb` write them: each file of an entry (the
/// model, and for the language detectors the ids of the sentences, and of
/// the documents, they were trained on) is checked against its card's
/// size and digests, and the card's gates and comparisons against the
/// records beside it.
///
/// # Errors
///
/// A model directory, card, record or file that is missing or cannot be
/// read; a card in another model's directory, or a gate or comparison
/// record of another model or with another outcome than the card's
/// ([`Error::InvalidCard`]); a card that may not become a zoo entry
/// ([`ModelCard::artifact`], [`UdModelCard::artifact`],
/// [`LanguageCard::artifact`], [`WebLanguageCard::artifact`]); or a file
/// that is not the one its card describes ([`Error::Zoo`]).
pub fn entries(models: &Path, release: &ModelsRelease) -> Result<Vec<Artifact>> {
    release
        .models
        .iter()
        .map(|id| match Recipe::trains(id) {
            Some(_) => entry::<ModelCard>(models, id, release),
            None if *id == LANGDETECT.model_id => entry::<LanguageCard>(models, id, release),
            None if *id == LANGDETECT_WEB.model_id => entry::<WebLanguageCard>(models, id, release),
            None => entry::<UdModelCard>(models, id, release),
        })
        .collect()
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
/// give: a line for each model of `release` missing from `entries`, one
/// for each entry that `index` lists otherwise (with the size, SHA-256
/// and address of each file of both) or does not list, and one for each
/// artifact `index` lists at an address of the release that is not one
/// of its models. None when `entries` covers `release` and `index` lists
/// `entries` and nothing else there.
pub fn differences(index: &Index, entries: &[Artifact], release: &ModelsRelease) -> Vec<String> {
    let files = |a: &Artifact| {
        let files: Vec<String> =
            a.files.iter().map(|f| format!("{} bytes, SHA-256 {}, at {}", f.size, f.sha256, f.url)).collect();
        files.join("; ")
    };
    let mut lines = Vec::new();
    for &id in release.models {
        if !entries.iter().any(|e| e.id.as_str() == id) {
            lines.push(format!("{id}: release {} publishes it, but no entry was given", release.tag));
        }
    }
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
/// `analysis/train-recipes/masc-references.txt`, `ud-references.txt`,
/// `tatoeba-references.txt` and `fineweb-references.txt` list for the
/// `models`, `models-ud`, `models-tatoeba` and `models-fineweb` workflows
/// to fetch and check.
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
