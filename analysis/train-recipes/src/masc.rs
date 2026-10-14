//! MASC, the Manually Annotated Sub-Corpus of the Open American National
//! Corpus (CC BY 3.0 US): the release the recipes train on, its digest,
//! and the fixed split into training and held-out documents.
//!
//! The release is fetched through the zoo by digest, as a permissive
//! dataset, and read from its extracted directory. It is not checked in:
//! CC BY allows it with attribution, but the policy prefers fetching by
//! digest, and the corpus is far larger than the repository's test data.
//!
//! [`RELEASE`] has no digest yet. The release could not be downloaded
//! where the recipes were written (`anc.org` was not reachable), so its
//! size and digests are not known, and a recipe refuses to fetch it
//! ([`crate::Error::NotRecorded`]) or to treat a directory as the release.
//! Running the binary with `--unrecorded-corpus` on an extracted copy
//! prints the corpus digest to record here.

use std::path::{Path, PathBuf};

use turbo_formats::masc::{MascCorpus, MascDocument};
use turbo_zoo::{
    Acceptances, Artifact, File, Http, Index, Kind, Licence, LicenceClass, Policy, Sha256, TrainingData, UserId, Zoo,
    index::Format,
};

use crate::error::{Error, Result, io};

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
/// site, where it would say so, could not be reached when this was
/// written, and the text is to be checked against it.
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
    /// The [`corpus_digest`] of the extracted directory.
    pub corpus_sha256: &'static str,
}

/// Whether the release's digests are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Release {
    /// Not yet: the release has not been downloaded and checked.
    NotRecorded,
    Recorded(RecordedRelease),
}

/// The MASC release the recipes train on: not recorded until a copy of the
/// release has been downloaded and its digests recorded.
pub const RELEASE: Release = Release::NotRecorded;

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
        origin: None,
        files: vec![File {
            path: r.file.parse().map_err(parse("file", r.file))?,
            url: r.url.parse().map_err(parse("url", r.url))?,
            size: r.size,
            sha256: r.sha256.parse().map_err(parse("sha256", r.sha256))?,
            sha512: r.sha512.parse().map_err(parse("sha512", r.sha512))?,
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
        manifest.push_str(&format!("{}  {rel}\n", Sha256::of(&bytes)));
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

/// Checks that `dir` is the recorded release.
///
/// # Errors
///
/// [`Error::NotRecorded`] while the release is not recorded, unless
/// `allow_unrecorded`, and [`Error::CorpusDigest`] for another corpus.
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
        Release::Recorded(r) if r.corpus_sha256 != digest.sha256 => {
            Err(Error::CorpusDigest { expected: r.corpus_sha256.to_owned(), found: digest.sha256 })
        }
        Release::Recorded(_) => Ok(digest),
        Release::NotRecorded if allow_unrecorded => Ok(digest),
        Release::NotRecorded => Err(Error::NotRecorded),
    }
}

/// Which documents are held out: those whose place in the corpus, in the
/// order [`MascCorpus::read`] reads them (headers sorted by name in each
/// directory), leaves `held_out_at` when divided by `every`. The rule
/// uses no randomness, so no seed; with `every` 10 it holds out one
/// document in ten, spread over the corpus's genre directories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub every: usize,
    pub held_out_at: usize,
}

/// The split of the recipes: every tenth document, from the tenth.
pub const SPLIT: Split = Split { every: 10, held_out_at: 9 };

/// A corpus split into the documents trained on and those held out.
#[derive(Debug)]
pub struct SplitCorpus {
    pub train: MascCorpus,
    pub held_out: MascCorpus,
    /// The held-out documents' paths relative to the corpus directory.
    pub held_out_paths: Vec<String>,
    /// The documents the corpus read skipped (it logged and left them
    /// out), relative to the corpus directory.
    pub skipped_paths: Vec<String>,
}

impl Split {
    pub fn is_held_out(&self, position: usize) -> bool {
        position % self.every == self.held_out_at
    }

    /// Splits `corpus`, read from `root`, by the documents' places.
    pub fn apply(&self, corpus: MascCorpus, root: &Path) -> SplitCorpus {
        let root = std::path::absolute(root).unwrap_or_else(|_| root.to_owned());
        let rel = |p: &Path| p.strip_prefix(&root).unwrap_or(p).to_string_lossy().into_owned();
        let skipped_paths = corpus.skipped().iter().map(|s| rel(&s.path)).collect();
        let (mut train, mut held_out): (Vec<MascDocument>, Vec<MascDocument>) = (Vec::new(), Vec::new());
        for (i, doc) in corpus.into_documents().into_iter().enumerate() {
            if self.is_held_out(i) { held_out.push(doc) } else { train.push(doc) }
        }
        let held_out_paths = held_out.iter().map(|d| rel(d.path())).collect();
        SplitCorpus {
            train: MascCorpus::from_documents(train),
            held_out: MascCorpus::from_documents(held_out),
            held_out_paths,
            skipped_paths,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_holds_out_one_in_ten() {
        let held: Vec<usize> = (0..35).filter(|&i| SPLIT.is_held_out(i)).collect();
        assert_eq!(held, [9, 19, 29]);
    }

    #[test]
    fn an_unrecorded_release_is_neither_fetched_nor_taken_for_a_corpus() {
        assert_eq!(RELEASE, Release::NotRecorded);
        assert!(matches!(dataset(RELEASE), Err(Error::NotRecorded)));
        let home = std::env::temp_dir();
        let user: UserId = "tester".parse().unwrap();
        assert!(matches!(fetch_release(RELEASE, &home, &user), Err(Error::NotRecorded)));
    }

    const RECORDED: RecordedRelease = RecordedRelease {
        file: "MASC-3.0.0.zip",
        format: Format::Zip,
        url: "https://example.org/MASC-3.0.0.zip",
        size: 3,
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        sha512: "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        corpus_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    };

    #[test]
    fn a_recorded_release_is_a_permissive_dataset_the_index_takes() {
        let a = dataset(Release::Recorded(RECORDED)).unwrap();
        let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&a)).as_bytes()).unwrap();
        let got = index.get(DATASET_ID).unwrap();
        assert_eq!(got.class, LicenceClass::Permissive);
        assert_eq!(got.computed_class(), LicenceClass::Permissive);
        assert_eq!(index.shippable().count(), 1);
    }

    #[test]
    fn the_corpus_digest_covers_every_file_by_path_and_content() {
        let dir = std::env::temp_dir().join(format!("turbo-train-recipes-digest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("b")).unwrap();
        std::fs::write(dir.join("a.txt"), "abc").unwrap();
        std::fs::write(dir.join("b/c.txt"), "").unwrap();
        let d = corpus_digest(&dir).unwrap();
        // sha256sum's manifest of the two files.
        let manifest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  a.txt\n\
                        e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  b/c.txt\n";
        assert_eq!(d, CorpusDigest { sha256: Sha256::of(manifest.as_bytes()).to_string(), files: 2 });

        assert!(matches!(check_corpus(&dir, RELEASE, false), Err(Error::NotRecorded)));
        assert_eq!(check_corpus(&dir, RELEASE, true).unwrap(), d);
        assert!(matches!(check_corpus(&dir, Release::Recorded(RECORDED), true), Err(Error::CorpusDigest { .. })));
        let same = CorpusDigest { sha256: RECORDED.corpus_sha256.to_owned(), files: 2 };
        assert_eq!(check_digest(same.clone(), Release::Recorded(RECORDED), false).unwrap(), same);

        std::fs::write(dir.join("b/c.txt"), "x").unwrap();
        assert_ne!(corpus_digest(&dir).unwrap().sha256, d.sha256);
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.join("a.txt"), dir.join("link")).unwrap();
            assert!(corpus_digest(&dir).is_err());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
