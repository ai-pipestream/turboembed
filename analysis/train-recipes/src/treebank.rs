//! The Universal Dependencies treebanks licensed CC BY 4.0 that the UD
//! recipes train on (issue #415), at release [`RELEASE`]: each
//! repository's `LICENSE.txt`, `README.md` and UD's own `train`, `dev`
//! and `test` files, with their sizes and digests, and the licence check
//! that reads `LICENSE.txt` and the README's metadata and refuses anything
//! but CC BY 4.0.
//!
//! The files are fetched through the zoo by digest from
//! `raw.githubusercontent.com` at the release tag, as a permissive dataset
//! (no acceptance is asked for), and are never checked in.
//!
//! Each licence was read in the repository's own `LICENSE.txt` and README
//! at the release, on 2026-10-14:
//!
//! - Spanish AnCora, Catalan AnCora, Portuguese Porttinari and Italian
//!   MarkIT: UD's CC BY 4.0 statement, and `License: CC BY 4.0` in the
//!   README's metadata. Both AnCora READMEs still have a paragraph that
//!   says "The GNU license is inherited from the original dataset"; their
//!   changelogs' entry "2021-11-15 v2.9" says "The license changed to CC
//!   BY 4.0 (<https://doi.org/10.5281/zenodo.4762030>).", and `LICENSE.txt`
//!   and the metadata say CC BY 4.0.
//! - Finnish FTB: `LICENSE.txt` says "FTB1U is Copyright 2014 FIN-CLARIN"
//!   and offers LGPLv3 or later or, "at your option", CC BY 4.0, followed
//!   by the texts of the GPL 3 and the LGPL 3, and the README's metadata
//!   says CC BY 4.0 ([`Grant::CcBy40OrOther`]). The recipes take CC BY
//!   4.0, and the attribution keeps the copyright notice
//!   ([`Treebank::copyright`]).
//!
//! Porttinari's CC BY 4.0 grant is its authors', and its README says
//! "The texts in the treebank are from Folha de São Paulo newspaper,
//! which are publicly available at Kaggle website." The repository shows
//! no licence from Folha for the articles' text (the Kaggle dataset's own
//! terms were not read), so the rights to the text are not settled
//! ([`TextRights::NotSettled`]): the recipes stay, so a user can train the
//! models locally, and no Porttinari model is shipped
//! ([`crate::ud_card::UdModelCard::artifact`] refuses one).
//!
//! Portuguese DANTEStocks is CC BY 4.0 too, but its text is tweets
//! (`README.md`: "a collection of Brazilian Portuguese tweets"), and the
//! data policy refuses data under a platform's terms, as it does MASC's
//! Twitter genre, so it is refused and has no recipe.

mod catalog;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use turbo_zoo::{
    Acceptances, Artifact, File, Http, Index, Kind, Licence, Policy, Sha256, Sha512, TrainingData, UserId, Zoo,
    class_of, index::Format,
};

use crate::error::{Error, Result, io};

pub use self::catalog::{
    CATALAN_ANCORA, FINNISH_FTB, ITALIAN_MARKIT, PORTUGUESE_PORTTINARI, SPANISH_ANCORA, TREEBANKS,
};

/// The Universal Dependencies release the recipes train on: the tag of
/// each repository.
pub const RELEASE: &str = "r2.16";
/// The treebanks' licence.
pub const LICENCE: Licence = Licence::CcBy40;
/// The README metadata line that states the licence.
pub const README_LICENCE_LINE: &str = "License: CC BY 4.0";

/// One file of a treebank repository at the release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreebankFile {
    /// The file's name in the repository.
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub sha512: &'static str,
}

impl TreebankFile {
    /// Checks `bytes` against the recorded size, SHA-256 and SHA-512.
    fn check(&self, treebank: &Treebank, bytes: &[u8]) -> Result<()> {
        let sha256 = Sha256::of(bytes).to_string();
        let sha512 = Sha512::of(bytes).to_string();
        if bytes.len() as u64 == self.size && sha256 == self.sha256 && sha512 == self.sha512 {
            Ok(())
        } else {
            Err(Error::TreebankFile {
                treebank: treebank.repository,
                file: self.name,
                expected: format!("{} bytes, SHA-256 {}", self.size, self.sha256),
                found: format!("{} bytes, SHA-256 {sha256}", bytes.len()),
            })
        }
    }
}

/// How a treebank's `LICENSE.txt` grants CC BY 4.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grant {
    /// CC BY 4.0 alone, in UD's statement: "The treebank is licensed under
    /// the Creative Commons License Attribution 4.0 International." The
    /// text names no other licence.
    CcBy40,
    /// Another licence or, at the user's option, CC BY 4.0; the recipes
    /// take CC BY 4.0. `other` is the other licence.
    CcBy40OrOther { other: Licence },
}

/// Whether a treebank's licence reaches the text it annotates, as far as
/// its repository shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextRights {
    /// The repository names no owner of the text whose terms are other
    /// than the treebank's licence.
    Granted,
    /// The licence is the annotators' grant, and the repository says the
    /// text is another's (`source`) and shows no licence from its owner. A
    /// recipe may train a model for local use, and the model is not
    /// shipped ([`crate::ud_card::UdModelCard::artifact`] refuses it).
    NotSettled { source: &'static str },
}

/// One CC BY 4.0 treebank at the release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Treebank {
    /// The repository under `github.com/UniversalDependencies`.
    pub repository: &'static str,
    /// The language code the models are trained for.
    pub language: &'static str,
    /// The treebank's name in model ids (`ancora`).
    pub name: &'static str,
    pub grant: Grant,
    pub text_rights: TextRights,
    /// The copyright notice `LICENSE.txt` gives, which the attribution
    /// keeps as CC BY asks; `None` when it gives none.
    pub copyright: Option<&'static str>,
    pub licence_file: TreebankFile,
    pub readme: TreebankFile,
    pub train: TreebankFile,
    pub dev: TreebankFile,
    pub test: TreebankFile,
    /// The contributors the README's metadata names.
    pub contributors: &'static str,
    /// The work the README asks to be cited, or, where it names none, the
    /// treebank as the README and `LICENSE.txt` describe it.
    pub citation: &'static str,
}

/// The text of a treebank's `train` and `test` files, read and checked.
/// The `dev` file is checked but not kept: no recipe reads it.
#[derive(Debug, Clone)]
pub struct TreebankText {
    pub train: String,
    pub test: String,
}

/// `s` lowercased with each run of characters other than ASCII letters
/// and digits made one space, with a space at both ends, so a phrase is
/// found as whole words.
fn words(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push(' ');
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with(' ') {
            out.push(' ');
        }
    }
    if !out.ends_with(' ') {
        out.push(' ');
    }
    out
}

/// Phrases of Creative Commons licence elements other than Attribution,
/// as [`words`] writes them: in a licence's name or address.
const RESTRICTED: [&str; 10] = [
    " noncommercial ",
    " non commercial ",
    " sharealike ",
    " share alike ",
    " noderivatives ",
    " noderivs ",
    " no derivatives ",
    " by nc ",
    " by sa ",
    " by nd ",
];

/// Checks the text of a treebank's `LICENSE.txt` and README against
/// `grant`: the licence text must name the Creative Commons Attribution
/// 4.0 International licence and its address, must name no NonCommercial,
/// ShareAlike or NoDerivatives element, and must name a GNU licence only
/// where `grant` offers another licence beside CC BY 4.0, and then "at
/// your option"; the README must have the metadata line
/// [`README_LICENCE_LINE`]. Returns why it refuses.
pub fn check_licence_text(grant: Grant, licence: &str, readme: &str) -> std::result::Result<(), &'static str> {
    let text = words(licence);
    if !(text.contains(" attribution 4 0 international ") && text.contains(" creativecommons org licenses by 4 0 ")) {
        return Err("LICENSE.txt does not grant the Creative Commons Attribution 4.0 International licence");
    }
    if RESTRICTED.iter().any(|p| text.contains(p)) {
        return Err("LICENSE.txt names a NonCommercial, ShareAlike or NoDerivatives licence element");
    }
    let gnu = text.contains(" gnu ") || text.contains(" general public license ");
    match grant {
        Grant::CcBy40 if gnu => return Err("LICENSE.txt names a GNU licence beside CC BY 4.0"),
        Grant::CcBy40OrOther { other } => {
            if !gnu || other != Licence::Lgpl30OrLater || !text.contains(" alternatively at your option ") {
                return Err("LICENSE.txt does not offer CC BY 4.0 at the user's option");
            }
        }
        Grant::CcBy40 => {}
    }
    let mut found_licence_line = false;
    for line in readme.lines().map(str::trim) {
        if line.starts_with("License:") || line.starts_with("Licence:") {
            if line != README_LICENCE_LINE {
                return Err("the README's metadata does not say License: CC BY 4.0");
            }
            found_licence_line = true;
        }
    }
    if !found_licence_line {
        return Err("the README's metadata does not say License: CC BY 4.0");
    }
    Ok(())
}

impl Treebank {
    /// The five files, in the order the dataset lists them.
    pub fn files(&self) -> [TreebankFile; 5] {
        [self.licence_file, self.readme, self.train, self.dev, self.test]
    }

    /// A file's address at the release.
    pub fn url(&self, file: &TreebankFile) -> String {
        format!("https://raw.githubusercontent.com/UniversalDependencies/{}/{RELEASE}/{}", self.repository, file.name)
    }

    /// Where the licence is stated: `LICENSE.txt` at the release.
    pub fn licence_terms(&self) -> String {
        format!(
            "https://github.com/UniversalDependencies/{}/blob/{RELEASE}/{}",
            self.repository, self.licence_file.name
        )
    }

    /// The treebank and its release, as a model's training data names it.
    pub fn release_name(&self) -> String {
        format!("{} {RELEASE}", self.repository)
    }

    /// The zoo id of the dataset: the repository name lowercased with
    /// `-` for `_`, and the release (`ud-spanish-ancora-r2.16`).
    pub fn dataset_id(&self) -> String {
        format!("{}-{RELEASE}", self.repository.to_ascii_lowercase().replace('_', "-"))
    }

    /// The treebank as the zoo's permissive training data entry of a model.
    pub fn training_data(&self) -> TrainingData {
        TrainingData {
            name: self.release_name(),
            license: LICENCE,
            license_terms: self.licence_terms().parse().expect("a valid address"),
        }
    }

    /// What a model trained on the treebank credits, as CC BY asks: the
    /// treebank and its release, its contributors, where it is, its
    /// licence, its copyright notice where `LICENSE.txt` gives one, and
    /// that the model is made from it.
    pub fn attribution(&self) -> String {
        let copyright = self.copyright.map(|c| format!(" {c}.")).unwrap_or_default();
        format!(
            "Trained on {} of Universal Dependencies (https://github.com/UniversalDependencies/{}/tree/{RELEASE}), \
             by {}, under the Creative Commons Attribution 4.0 International licence \
             (https://creativecommons.org/licenses/by/4.0/).{copyright} The model is derived from the treebank; the \
             treebank itself is not included.",
            self.release_name(),
            self.repository,
            self.contributors
        )
    }

    /// The release's files as a zoo dataset artifact: `LICENSE.txt`, the
    /// README and the three CoNLL-U files.
    ///
    /// # Errors
    ///
    /// A recorded field the zoo index refuses.
    pub fn dataset(&self) -> Result<Artifact> {
        let bad =
            |field: &'static str, value: String| move |message: String| Error::InvalidRelease { field, value, message };
        let id = self.dataset_id();
        let terms = self.licence_terms();
        let mut files = Vec::with_capacity(5);
        for f in self.files() {
            let url = self.url(&f);
            files.push(File {
                path: f.name.parse().map_err(bad("file", f.name.to_owned()))?,
                url: url.parse().map_err(bad("url", url.clone()))?,
                size: f.size,
                sha256: f.sha256.parse().map_err(bad("sha256", f.sha256.to_owned()))?,
                sha512: f.sha512.parse().map_err(bad("sha512", f.sha512.to_owned()))?,
            });
        }
        Ok(Artifact {
            id: id.parse().map_err(bad("id", id.clone()))?,
            kind: Kind::Dataset,
            component: None,
            language: Some(self.language.parse().map_err(bad("language", self.language.to_owned()))?),
            format: Format::Conllu,
            version: RELEASE.to_owned(),
            license: LICENCE,
            license_terms: terms.parse().map_err(bad("terms", terms.clone()))?,
            trained_on: Vec::new(),
            class: class_of([LICENCE]),
            attribution: Some(self.attribution()),
            origin: None,
            files,
        })
    }

    /// The licence check: [`check_licence_text`] on the bytes of
    /// `LICENSE.txt` and the README.
    ///
    /// # Errors
    ///
    /// [`Error::Licence`] with the reason.
    pub fn check_licence(&self, licence: &[u8], readme: &[u8]) -> Result<()> {
        let refuse = |reason: &'static str| Error::Licence { treebank: self.repository, reason };
        let licence = std::str::from_utf8(licence).map_err(|_| refuse("LICENSE.txt is not UTF-8"))?;
        let readme = std::str::from_utf8(readme).map_err(|_| refuse("the README is not UTF-8"))?;
        check_licence_text(self.grant, licence, readme).map_err(refuse)
    }

    /// Reads the treebank's files from `dir`: the licence check first,
    /// then every file against its recorded size and digests.
    ///
    /// # Errors
    ///
    /// A file that cannot be read, [`Error::Licence`], or
    /// [`Error::TreebankFile`] for a file that is not the recorded one.
    pub fn read(&self, dir: &Path) -> Result<TreebankText> {
        let read = |f: &TreebankFile| {
            let path = dir.join(f.name);
            std::fs::read(&path).map_err(io(&path))
        };
        let (licence, readme) = (read(&self.licence_file)?, read(&self.readme)?);
        self.check_licence(&licence, &readme)?;
        self.licence_file.check(self, &licence)?;
        self.readme.check(self, &readme)?;
        let text = |f: &TreebankFile| -> Result<String> {
            let bytes = read(f)?;
            f.check(self, &bytes)?;
            String::from_utf8(bytes).map_err(|e| Error::Io { path: dir.join(f.name), message: e.to_string() })
        };
        let train = text(&self.train)?;
        text(&self.dev)?;
        let test = text(&self.test)?;
        Ok(TreebankText { train, test })
    }

    /// Fetches the release's files through the zoo into
    /// `home/artifacts/<dataset id>/`, by digest, and returns that
    /// directory. CC BY 4.0 is permissive, so no acceptance is asked for.
    ///
    /// # Errors
    ///
    /// What the zoo fails on.
    pub fn fetch(&self, home: &Path, user: &UserId) -> Result<PathBuf> {
        let artifact = self.dataset()?;
        let id = artifact.id.to_string();
        let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&artifact)).as_bytes())?;
        let zoo = Zoo::new(index, home.join("artifacts"), Policy::default(), &Http);
        Ok(zoo.fetch_with(&id, user, || Ok(Acceptances::default()))?.dir)
    }
}
