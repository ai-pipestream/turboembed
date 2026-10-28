//! Historical recipe release identifiers and catalogue metadata helpers.
//!
//! Classical model publication is disabled under #1511. [`entries`]
//! returns an error before reading cards, gates or model files. These
//! identifiers remain until the training stack is removed under #1532.

#[cfg(test)]
mod tests;

use std::path::Path;

use turbo_zoo::{Artifact, Index};

use crate::depparse::{DEPPARSE_RECIPES, DepparseRecipe};
use crate::error::{Error, Result};
use crate::langdetect::{LANGDETECT, LanguageRecipe};
use crate::langdetect_web::{LANGDETECT_WEB, WebLanguageRecipe};
use crate::ud::{UD_RECIPES, UdRecipe};
use crate::{RECIPES, Recipe, Reference};

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
/// tokenizer and part-of-speech tagger recipes.
pub const MASC_MODELS: ModelsRelease = ModelsRelease {
    tag: "models-masc-3.0.0-1",
    models: &["turbo-en-masc-3.0.0-sentence-1", "turbo-en-masc-3.0.0-tokens-1", "turbo-en-masc-3.0.0-pos-1"],
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

/// The release that publishes the UD 2.16 dependency parser models
/// (both MaxEnt and Feedforward) for the four treebanks with settled text
/// rights that pass their LAS and UAS gates: Spanish AnCora, Catalan AnCora,
/// Finnish FTB, and Italian MarkIT. Portuguese Porttinari's are excluded
/// because the rights to its text are not settled.
pub const UD_DEPPARSE_MODELS: ModelsRelease = ModelsRelease {
    tag: "models-ud-depparse-2.16-1",
    models: &[
        "turbo-es-ud-ancora-2.16-depparse-maxent-1",
        "turbo-es-ud-ancora-2.16-depparse-feedforward-1",
        "turbo-ca-ud-ancora-2.16-depparse-maxent-1",
        "turbo-ca-ud-ancora-2.16-depparse-feedforward-1",
        "turbo-fi-ud-ftb-2.16-depparse-maxent-1",
        "turbo-fi-ud-ftb-2.16-depparse-feedforward-1",
        "turbo-it-ud-markit-2.16-depparse-maxent-1",
        "turbo-it-ud-markit-2.16-depparse-feedforward-1",
    ],
};

/// Every release of the recipes' models.
pub const MODELS_RELEASES: [ModelsRelease; 5] =
    [MASC_MODELS, UD_MODELS, TATOEBA_MODELS, FINEWEB_MODELS, UD_DEPPARSE_MODELS];

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

    /// The UD dependency parser recipes that train models it publishes, in
    /// the order of [`DEPPARSE_RECIPES`].
    pub fn depparse_recipes(&self) -> Vec<DepparseRecipe> {
        DEPPARSE_RECIPES.into_iter().filter(|r| self.models.contains(&r.model_id)).collect()
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

/// Classical recipe publication is disabled.
///
/// # Errors
///
/// Always returns [`Error::Model`] before any filesystem access,
/// including for an empty release.
pub fn entries(_models: &Path, _release: &ModelsRelease) -> Result<Vec<Artifact>> {
    Err(Error::Model(turbo_maxent::Error::Disabled.to_string()))
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
