//! MASC, the Manually Annotated Sub-Corpus of the Open American National
//! Corpus (CC BY 3.0 US): the release the recipes train on, its digest,
//! and the fixed split into training and held-out documents.
//!
//! The release is fetched through the zoo by digest, as a permissive
//! dataset, and read from its extracted directory. It is not checked in:
//! CC BY allows it with attribution, but the policy prefers fetching by
//! digest, and the corpus is far larger than the repository's test data.
//!
//! [`RELEASE`] records the archive `MASC-3.0.0.zip`, a release asset of
//! the GitHub repository `nancyide/anc-website` (tag
//! `datasets-2026-09-25`), and the digest of the directory it extracts
//! to, `MASC-3.0.0` ([`corpus_digest`]). That anc.org's MASC download page
//! links to the asset was reported but not checked: anc.org could not be
//! reached from the machine the recipes were run on, where the archive's
//! size and digests and the corpus digest were checked. A run on another
//! directory is not a release of the models: the binary refuses it
//! unless given `--unrecorded-corpus`, and marks its cards
//! `release_recorded: false`.

mod split;
#[cfg(test)]
mod tests;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use turbo_zoo::{
    Acceptances, Artifact, File, Http, Index, Kind, Licence, LicenceClass, Policy, Sha256, TrainingData, UserId, Zoo,
    index::Format,
};

use crate::error::{Error, Result, io};

pub use self::split::{Categories, EXCLUSIONS, ExcludedDocument, Exclusion, SPLIT, Split, SplitCorpus};

/// The release, as MASC's publisher names it.
pub const NAME: &str = "MASC 3.0.0";
/// The release's version.
pub const VERSION: &str = "3.0.0";
/// MASC's licence.
pub const LICENCE: Licence = Licence::CcBy30Us;
/// Where MASC's publisher states the licence.
pub const TERMS: &str = "https://anc.org/data/masc/";
/// The zoo id of the release.
pub const DATASET_ID: &str = "masc-3.0.0";
/// Who made and publishes MASC, whom CC BY asks a model trained on it to
/// credit.
pub const AUTHOR: &str = "the American National Corpus project";
/// The author's site.
pub const AUTHOR_URL: &str = "https://anc.org/";
/// What a model trained on MASC credits, as CC BY asks: the work, its
/// author and publisher, where it is, and its licence. Whether the ANC
/// asks for a particular citation beyond this is not confirmed: their
/// site, where it would say so, could not be reached from the machine
/// the recipes were written and run on, and the text is to be checked
/// against it.
pub const ATTRIBUTION: &str = "Trained on MASC 3.0.0, the Manually Annotated Sub-Corpus of the American National \
     Corpus, by the American National Corpus project (https://anc.org/), published at https://anc.org/data/masc/ \
     under the Creative Commons Attribution 3.0 United States licence \
     (https://creativecommons.org/licenses/by/3.0/us/).";

/// The release archive as published, and the digest of the corpus it
/// extracts to ([`corpus_digest`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedRelease {
    /// The archive's file name.
    pub file: &'static str,
    /// The archive's format.
    pub format: Format,
    /// Where it is fetched from, an address that keeps serving the same
    /// bytes.
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub sha512: &'static str,
    /// The directory the archive extracts to, which holds every file of
    /// the release: the corpus directory a run is given.
    pub corpus_dir: &'static str,
    /// The [`corpus_digest`] of that directory.
    pub corpus_sha256: &'static str,
}

/// Whether the release's digests are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Release {
    /// A release whose archive has not been downloaded and checked: it
    /// can be neither fetched nor taken as the recorded corpus.
    /// [`RELEASE`] is recorded.
    NotRecorded,
    Recorded(RecordedRelease),
}

impl Release {
    /// Whether `digest` is the recorded release's corpus digest; never for
    /// a release not recorded.
    pub fn is(&self, digest: &CorpusDigest) -> bool {
        matches!(self, Release::Recorded(r) if r.corpus_sha256 == digest.sha256)
    }
}

/// The MASC release the recipes train on: the archive `MASC-3.0.0.zip`
/// released on GitHub as an asset of `nancyide/anc-website` (tag
/// `datasets-2026-09-25`), downloaded and checked on 2026-10-14; that
/// anc.org's MASC page links to it was reported, not checked (the
/// module's documentation). The archive's entries are the directory
/// `MASC-3.0.0` and 4,425 regular files under it (no links), its README
/// dates the release 2012-12-15, and its corpus digest covers those
/// files.
pub const RELEASE: Release = Release::Recorded(RecordedRelease {
    file: "MASC-3.0.0.zip",
    format: Format::Zip,
    url: "https://github.com/nancyide/anc-website/releases/download/datasets-2026-09-25/MASC-3.0.0.zip",
    size: 36_358_521,
    sha256: "0b300d96016233c7c79b43559b477a6bb0ed16bb416d5c517eccb108a6228493",
    sha512: "a88795055fab5c600f26215cbe6143c2325627adaf9ae80d06c7e815a76834debd4b7728a745aa67608ee1d028b35ca139b2b90760e69d0e7227af83bf05d6a9",
    corpus_dir: "MASC-3.0.0",
    corpus_sha256: "445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd",
});

/// MASC as the zoo's permissive training data entry of a model.
pub fn training_data() -> TrainingData {
    TrainingData { name: NAME.to_owned(), license: LICENCE, license_terms: TERMS.parse().expect("a valid address") }
}

/// The release as a zoo dataset artifact.
///
/// # Errors
///
/// [`Error::NotRecorded`] while the release is not recorded.
pub fn dataset(release: Release) -> Result<Artifact> {
    let Release::Recorded(r) = release else {
        return Err(Error::NotRecorded);
    };
    let parse = |field: &'static str, value: &'static str| {
        move |message: String| Error::InvalidRelease { field, value: value.to_owned(), message }
    };
    Ok(Artifact {
        id: DATASET_ID.parse().map_err(parse("id", DATASET_ID))?,
        kind: Kind::Dataset,
        component: None,
        language: Some("en".parse().map_err(parse("language", "en"))?),
        format: r.format,
        version: VERSION.to_owned(),
        license: LICENCE,
        license_terms: TERMS.parse().map_err(parse("terms", TERMS))?,
        trained_on: Vec::new(),
        class: LicenceClass::Permissive,
        attribution: Some(ATTRIBUTION.to_owned()),
        origin: None,
        files: vec![File {
            path: r.file.parse().map_err(parse("file", r.file))?,
            url: r.url.parse().map_err(parse("url", r.url))?,
            size: r.size,
            sha256: r.sha256.parse().map_err(parse("sha256", r.sha256))?,
            sha512: r.sha512.parse().map_err(parse("sha512", r.sha512))?,
            download: None,
        }],
    })
}

/// Fetches the release archive through the zoo into `home/artifacts`, by
/// digest. MASC is permissive, so no acceptance is asked for.
///
/// # Errors
///
/// [`Error::NotRecorded`] while the release is not recorded, or what the zoo
/// fails on.
pub fn fetch_release(release: Release, home: &Path, user: &UserId) -> Result<PathBuf> {
    let artifact = dataset(release)?;
    let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&artifact)).as_bytes())?;
    let zoo = Zoo::new(index, home.join("artifacts"), Policy::default(), &Http);
    let fetched = zoo.fetch_with(DATASET_ID, user, || Ok(Acceptances::default()))?;
    Ok(fetched.single().expect("one file in the release").to_path_buf())
}

/// The digest of a corpus directory: the SHA-256 of a `sha256sum` style
/// manifest of every regular file under it, one line per file, `<sha256>
/// <path>` with two spaces and the path relative to the directory, `/`
/// separated, sorted by the path's bytes. A symbolic link is refused, so
/// the digest covers only what is in the directory.
pub fn corpus_digest(dir: &Path) -> Result<CorpusDigest> {
    let mut files = Vec::new();
    collect(dir, "", &mut files)?;
    files.sort();
    let mut manifest = String::new();
    for rel in &files {
        let path = dir.join(rel);
        let bytes = std::fs::read(&path).map_err(io(&path))?;
        writeln!(manifest, "{}  {rel}", Sha256::of(&bytes)).expect("writing to a String");
    }
    Ok(CorpusDigest { sha256: Sha256::of(manifest.as_bytes()).to_string(), files: files.len() })
}

fn collect(dir: &Path, prefix: &str, out: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).map_err(io(dir))? {
        let entry = entry.map_err(io(dir))?;
        let path = entry.path();
        let kind = entry.file_type().map_err(io(&path))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|n| Error::Io { path: path.clone(), message: format!("the name {n:?} is not UTF-8") })?;
        let rel = format!("{prefix}{name}");
        if kind.is_symlink() {
            return Err(Error::Io { path, message: "a symbolic link in the corpus directory".into() });
        } else if kind.is_dir() {
            collect(&path, &format!("{rel}/"), out)?;
        } else if kind.is_file() {
            out.push(rel);
        } else {
            return Err(Error::Io { path, message: "not a regular file or directory in the corpus directory".into() });
        }
    }
    Ok(())
}

/// A corpus directory's digest and how many files it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusDigest {
    pub sha256: String,
    pub files: usize,
}

/// Checks that `dir` is the recorded release, or, when
/// `allow_unrecorded`, takes any corpus directory; [`Release::is`] on the
/// digest returned tells which it is.
///
/// # Errors
///
/// Unless `allow_unrecorded`: [`Error::CorpusDigest`] for a corpus that is
/// not the recorded release, and [`Error::NotRecorded`] while the release
/// is not recorded.
pub fn check_corpus(dir: &Path, release: Release, allow_unrecorded: bool) -> Result<CorpusDigest> {
    check_digest(corpus_digest(dir)?, release, allow_unrecorded)
}

/// [`check_corpus`] on a digest already made.
///
/// # Errors
///
/// As [`check_corpus`].
pub fn check_digest(digest: CorpusDigest, release: Release, allow_unrecorded: bool) -> Result<CorpusDigest> {
    match release {
        _ if release.is(&digest) || allow_unrecorded => Ok(digest),
        Release::Recorded(r) => Err(Error::CorpusDigest {
            expected: r.corpus_sha256.to_owned(),
            corpus_dir: r.corpus_dir,
            found: digest.sha256,
        }),
        Release::NotRecorded => Err(Error::NotRecorded),
    }
}
