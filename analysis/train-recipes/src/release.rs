//! The release that publishes the recipes' models, and their entries in
//! the zoo's built-in index.
//!
//! The models are published as the assets of a release of this
//! repository on GitHub, as Kristian Rickert decided on 2026-10-15, and
//! the zoo's built-in index (`analysis/zoo/index/zoo.json`) lists each one
//! at its address there. [`entries`] makes those entries from the model
//! directories `turbo-train-recipes masc` writes, after checking each
//! model file against its card, and [`index_with`] is the built-in index
//! with them added. `turbo-train-recipes zoo-index` prints that index,
//! and with `--check` fails unless the zoo already carries it, which the
//! `models` workflow checks before it publishes anything.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use turbo_zoo::{Artifact, Index, verify};

use crate::RECIPES;
use crate::card::{CARD_FILE, ModelCard};
use crate::error::{Error, Result, io};

/// Where this repository's releases publish their assets: asset `f` of
/// release `t` is at `{RELEASES}t/f`.
pub const RELEASES: &str = "https://github.com/ai-pipestream/turboembed/releases/download/";

/// The release that publishes the models of [`RECIPES`].
pub const MASC_MODELS: &str = "models-masc-3.0.0-1";

/// The tag of a release: ASCII letters, digits, `.`, `_` and `-`, and not
/// `.` or `..`, so it is one segment of an address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseTag(String);

impl ReleaseTag {
    /// The address of the release's asset `file`.
    pub fn url(&self, file: &str) -> String {
        format!("{RELEASES}{}/{file}", self.0)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for ReleaseTag {
    type Err = Error;

    fn from_str(s: &str) -> Result<ReleaseTag> {
        let allowed = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
        if s.is_empty() || s == "." || s == ".." || !s.chars().all(allowed) {
            return Err(Error::ReleaseTag { tag: s.to_owned() });
        }
        Ok(ReleaseTag(s.to_owned()))
    }
}

impl fmt::Display for ReleaseTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The zoo entries of the models in `models`, in the order of
/// [`RECIPES`], each published as an asset of release `tag`. `models`
/// holds a directory for every recipe's model, as
/// `turbo-train-recipes masc` writes them; each model file is checked
/// against its card's size and digests.
///
/// # Errors
///
/// A model directory, card or model file that is missing or cannot be
/// read, a card in another model's directory ([`Error::InvalidCard`]), a
/// card that may not become a zoo entry ([`ModelCard::artifact`]), or a
/// model file that is not the one its card describes ([`Error::Zoo`]).
pub fn entries(models: &Path, tag: &ReleaseTag) -> Result<Vec<Artifact>> {
    RECIPES
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
            let artifact = card.artifact(&tag.url(&card.file.name))?;
            let path = dir.join(&card.file.name);
            let bytes = std::fs::read(&path).map_err(io(&path))?;
            verify(&artifact, &artifact.files[0], &bytes)?;
            Ok(artifact)
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

#[cfg(test)]
mod tests {
    use turbo_zoo::LicenceClass;

    use super::*;
    use crate::masc;

    #[test]
    fn a_tag_is_one_segment_of_an_address() {
        let tag: ReleaseTag = MASC_MODELS.parse().unwrap();
        assert_eq!(
            tag.url("turbo-en-masc-3.0.0-pos-1.bin"),
            "https://github.com/ai-pipestream/turboembed/releases/download/models-masc-3.0.0-1/\
             turbo-en-masc-3.0.0-pos-1.bin"
        );
        for bad in ["", ".", "..", "a/b", "a b", "a%2Fb", "a\\b", "é", "v1?x"] {
            assert!(matches!(bad.parse::<ReleaseTag>(), Err(Error::ReleaseTag { .. })), "{bad:?}");
        }
    }

    /// The zoo carries an entry for every recipe's model at its address
    /// in [`MASC_MODELS`], permissive, needing no acceptance, and trained
    /// on MASC alone. That the bytes there are the recipes' is what
    /// `turbo-train-recipes zoo-index --check` checks on a run.
    #[test]
    fn the_builtin_index_lists_every_model_at_its_release_address() {
        let tag: ReleaseTag = MASC_MODELS.parse().unwrap();
        let index = Index::builtin();
        for recipe in RECIPES {
            let id = recipe.model_id();
            let entry = index.get(id).unwrap();
            assert_eq!(entry.component, Some(recipe.component()), "{id}");
            assert_eq!(entry.language.as_ref().unwrap().as_str(), "en", "{id}");
            assert_eq!(entry.class, LicenceClass::Permissive, "{id}");
            assert!(entry.licences_to_accept(turbo_zoo::Policy::default()).is_empty(), "{id}");
            assert_eq!(entry.trained_on, [masc::training_data()], "{id}");
            assert_eq!(entry.files.len(), 1, "{id}");
            let file = format!("{id}.bin");
            assert_eq!(entry.files[0].path.as_str(), file);
            assert_eq!(entry.files[0].url.as_str(), tag.url(&file));
            assert!(index.shippable().any(|a| a.id.as_str() == id), "{id}");
        }
    }
}
