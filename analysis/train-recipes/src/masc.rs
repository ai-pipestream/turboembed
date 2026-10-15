//! MASC, the Manually Annotated Sub-Corpus of the Open American National
//! Corpus (CC BY 3.0 US): the release the recipes train on, its digest,
//! and the fixed split into training and held-out documents.
//!
//! The release is fetched through the zoo by digest, as a permissive
//! dataset, and read from its extracted directory. It is not checked in:
//! CC BY allows it with attribution, but the policy prefers fetching by
//! digest, and the corpus is far larger than the repository's test data.
//!
//! [`RELEASE`] records the archive, `MASC-3.0.0.zip`, as anc.org's MASC
//! page links to it (a release asset of the ANC website's repository on
//! GitHub), and the digest of the directory it extracts to,
//! `MASC-3.0.0` ([`corpus_digest`]). A run on another directory is not a
//! release of the models: the binary refuses it unless given
//! `--unrecorded-corpus`, and marks its cards `release_recorded: false`.

use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};

use turbo_formats::masc::{MascCorpus, MascDocument};
use turbo_maxent::xml::{self, Attribute, Handler};
use turbo_zoo::{
    Acceptances, Artifact, File, Http, Index, Kind, Licence, LicenceClass, Policy, RefusalReason, Sha256, TrainingData,
    UserId, Zoo, index::Format,
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

/// The MASC release the recipes train on: MASC 3.0.0 as the archive
/// anc.org's MASC page links to, downloaded and checked on 2026-10-14.
/// The archive's entries are the directory `MASC-3.0.0` and 4,425
/// regular files under it (no links); its corpus digest covers those
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

/// A part of MASC the recipes leave out of both the training and the
/// held-out documents, wherever the split puts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exclusion {
    /// MASC's Twitter genre: in MASC 3.0.0, the two documents of
    /// `data/written/twitter` (`tweets1` and `tweets2`, 24,180 words),
    /// tweets from 2008 whose headers give their source as unknown. The
    /// licence policy refuses data under a platform's terms, and neither
    /// the release nor anc.org says whether Twitter's terms reach the
    /// tweets' text (docs/licences.md, "Datasets"); Kristian Rickert
    /// decided on 2026-10-15 to leave the genre out.
    TwitterGenre,
}

/// The exclusions the recipes apply ([`Split::apply`]).
pub const EXCLUSIONS: [Exclusion; 1] = [Exclusion::TwitterGenre];

impl Exclusion {
    /// The category of MASC's taxonomy (`resource-header.xml`) whose
    /// documents are left out, as a document header's `textClass` names
    /// it in its `catRef`: `TW`, "tweets", a genre of the written part
    /// (`WR`). In MASC 3.0.0 the two headers of `data/written/twitter`,
    /// and no other, name it.
    pub fn category(self) -> &'static str {
        match self {
            Exclusion::TwitterGenre => "TW",
        }
    }

    /// Why the documents are left out, in the licence policy's terms.
    pub fn reason(self) -> RefusalReason {
        match self {
            Exclusion::TwitterGenre => RefusalReason::PlatformTerms,
        }
    }

    /// Whether a document of `categories` is left out.
    pub fn covers(self, categories: &Categories) -> bool {
        categories.contains(self.category())
    }
}

impl fmt::Display for Exclusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Exclusion::TwitterGenre => "the Twitter genre",
        })
    }
}

/// The categories of MASC's taxonomy a document header's `textClass`
/// elements name in their `catRef` attributes, a list of codes separated
/// by whitespace (`WR TW`: written, tweets). A header without one has
/// none.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Categories(Vec<String>);

impl Categories {
    /// The categories of the header at `path`.
    ///
    /// # Errors
    ///
    /// A header that cannot be read, or is not XML ([`Error::Header`]).
    pub fn of_header(path: &Path) -> Result<Categories> {
        let bytes = std::fs::read(path).map_err(io(path))?;
        let header = |e: turbo_maxent::Error| Error::Header { path: path.to_owned(), message: e.to_string() };
        Categories::parse(&xml::decode(&bytes).map_err(header)?).map_err(header)
    }

    /// The categories of a header's decoded text.
    ///
    /// # Errors
    ///
    /// Text that is not XML.
    pub fn parse(text: &str) -> std::result::Result<Categories, turbo_maxent::Error> {
        struct TextClasses(Vec<String>);
        impl Handler for TextClasses {
            fn start_element(&mut self, name: &str, a: &[Attribute<'_>], _: usize) -> turbo_maxent::Result<()> {
                if name == "textClass"
                    && let Some(codes) = a.iter().find(|a| a.name == "catRef")
                {
                    self.0.extend(codes.value.split_whitespace().map(str::to_owned));
                }
                Ok(())
            }
            fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
                Ok(())
            }
            fn characters(&mut self, _: &str) {}
        }
        let mut found = TextClasses(Vec::new());
        xml::parse(text, &mut found)?;
        Ok(Categories(found.0))
    }

    /// Whether `code` is one of the categories.
    pub fn contains(&self, code: &str) -> bool {
        self.0.iter().any(|c| c == code)
    }

    /// The codes, in the order of the header.
    pub fn codes(&self) -> &[String] {
        &self.0
    }
}

/// A document left out by an exclusion.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedDocument {
    /// The document's header, relative to the corpus directory.
    pub path: String,
    pub exclusion: Exclusion,
}

/// Which documents are held out: those whose place in the corpus, in the
/// order [`MascCorpus::read`] reads them (headers sorted by name in each
/// directory), leaves `held_out_at` when divided by `every`. The rule
/// uses no randomness, so no seed; with `every` 10 it holds out one
/// document in ten, spread over the corpus's genre directories.
/// `held_out_at` is below `every`, so `every` is not 0: [`Split::new`]
/// and reading a card check it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "SplitFields", into = "SplitFields")]
pub struct Split {
    every: usize,
    held_out_at: usize,
}

/// A split as written in a card, checked by `Split::try_from`.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitFields {
    every: usize,
    held_out_at: usize,
}

impl TryFrom<SplitFields> for Split {
    type Error = String;
    fn try_from(f: SplitFields) -> std::result::Result<Split, String> {
        Split::new(f.every, f.held_out_at)
            .ok_or_else(|| format!("a split of every {} at {}: the place must be below every", f.every, f.held_out_at))
    }
}

impl From<Split> for SplitFields {
    fn from(s: Split) -> SplitFields {
        SplitFields { every: s.every, held_out_at: s.held_out_at }
    }
}

/// The split of the recipes: every tenth document, from the tenth.
pub const SPLIT: Split = match Split::new(10, 9) {
    Some(s) => s,
    None => panic!("a valid split"),
};

/// A corpus split into the documents trained on and those held out.
#[derive(Debug)]
pub struct SplitCorpus {
    pub train: MascCorpus,
    pub held_out: MascCorpus,
    /// The held-out documents' paths relative to the corpus directory.
    pub held_out_paths: Vec<String>,
    /// The documents left out of both sides by an exclusion, in the order
    /// read.
    pub excluded: Vec<ExcludedDocument>,
}

impl Split {
    /// The split that holds out the documents at `held_out_at` modulo
    /// `every`, or `None` unless `held_out_at < every`.
    pub const fn new(every: usize, held_out_at: usize) -> Option<Split> {
        if held_out_at < every { Some(Split { every, held_out_at }) } else { None }
    }

    pub fn every(&self) -> usize {
        self.every
    }

    pub fn held_out_at(&self) -> usize {
        self.held_out_at
    }

    pub fn is_held_out(&self, position: usize) -> bool {
        position % self.every == self.held_out_at
    }

    /// Splits `corpus`, read from `root`, by the documents' places, and
    /// leaves out of both sides the documents [`EXCLUSIONS`] covers
    /// ([`Split::apply_excluding`]).
    ///
    /// # Errors
    ///
    /// As [`Split::apply_excluding`].
    pub fn apply(&self, corpus: MascCorpus, root: &Path) -> Result<SplitCorpus> {
        self.apply_excluding(corpus, root, &EXCLUSIONS)
    }

    /// Splits `corpus`, read from `root`, by the documents' places, and
    /// leaves out of both sides each document an exclusion of
    /// `exclusions` covers, by the categories its header names
    /// ([`Categories`]). The places are those of every document read,
    /// excluded or not, so an exclusion takes its documents out of the
    /// side they fall on and moves no other document from one side to
    /// the other. A document the corpus read skipped (it logged its error
    /// and left it out, as `MascDocumentStream` does) has no place, so it
    /// would move every document read after it to the next place and
    /// change which are held out: a corpus with one is refused. MASC 3.0.0
    /// has none.
    ///
    /// # Errors
    ///
    /// [`Error::SkippedDocuments`] for a corpus whose read skipped a
    /// document, [`Error::OutsideCorpus`] for a document whose path is not
    /// under `root`, which a corpus read from `root` does not have, and
    /// [`Error::Header`] for a header that cannot be read again for its
    /// categories.
    pub fn apply_excluding(&self, corpus: MascCorpus, root: &Path, exclusions: &[Exclusion]) -> Result<SplitCorpus> {
        let root = std::path::absolute(root).map_err(io(root))?;
        let rel = |p: &Path| -> Result<String> {
            let inside = p.strip_prefix(&root).map_err(|_| Error::OutsideCorpus { path: p.to_owned() })?;
            Ok(inside.to_string_lossy().into_owned())
        };
        if !corpus.skipped().is_empty() {
            let documents = corpus
                .skipped()
                .iter()
                .map(|s| Ok(format!("{} ({})", rel(&s.path)?, s.error)))
                .collect::<Result<_>>()?;
            return Err(Error::SkippedDocuments { documents });
        }
        let (mut train, mut held_out): (Vec<MascDocument>, Vec<MascDocument>) = (Vec::new(), Vec::new());
        let (mut held_out_paths, mut excluded) = (Vec::new(), Vec::new());
        for (i, doc) in corpus.into_documents().into_iter().enumerate() {
            let path = rel(doc.path())?;
            if !exclusions.is_empty() {
                let categories = Categories::of_header(doc.path())?;
                if let Some(&exclusion) = exclusions.iter().find(|e| e.covers(&categories)) {
                    excluded.push(ExcludedDocument { path, exclusion });
                    continue;
                }
            }
            if self.is_held_out(i) {
                held_out.push(doc);
                held_out_paths.push(path);
            } else {
                train.push(doc);
            }
        }
        Ok(SplitCorpus {
            train: MascCorpus::from_documents(train),
            held_out: MascCorpus::from_documents(held_out),
            held_out_paths,
            excluded,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_holds_out_one_in_ten() {
        let held: Vec<usize> = (0..35).filter(|&i| SPLIT.is_held_out(i)).collect();
        assert_eq!(held, [9, 19, 29]);
        assert_eq!(Split::new(0, 0), None);
        assert_eq!(Split::new(10, 10), None);
        assert_eq!(serde_json::to_string(&SPLIT).unwrap(), r#"{"every":10,"held_out_at":9}"#);
        assert_eq!(serde_json::from_str::<Split>(r#"{"every":10,"held_out_at":9}"#).unwrap(), SPLIT);
        for bad in
            [r#"{"every":0,"held_out_at":0}"#, r#"{"every":3,"held_out_at":3}"#, r#"{"every":3,"held_out_at":1,"x":1}"#]
        {
            assert!(serde_json::from_str::<Split>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_twitter_genre_is_the_category_tw_of_a_header_s_text_class() {
        let header = |text_class: &str| {
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<documentHeader \
                 xmlns=\"http://www.xces.org/ns/GrAF/1.0/\"><profileDesc>{text_class}<primaryData \
                 loc=\"a.txt\" f.id=\"f.text\"/></profileDesc></documentHeader>"
            )
        };
        let of = |text_class: &str| Categories::parse(&header(text_class)).unwrap();
        let tweets = of("<textClass catRef=\"WR TW\"><domain>Leisure</domain></textClass>");
        assert_eq!(tweets.codes(), ["WR", "TW"]);
        assert!(Exclusion::TwitterGenre.covers(&tweets));
        // Codes are compared whole, after whitespace is normalized.
        assert!(Exclusion::TwitterGenre.covers(&of("<textClass catRef=\" SP\tTW \"/>")));
        for other in ["<textClass catRef=\"WR BL\"/>", "<textClass catRef=\"WR TWX\"/>", "<textClass/>", ""] {
            assert!(!Exclusion::TwitterGenre.covers(&of(other)), "{other}");
        }
        assert_eq!(of(""), Categories::default());
        assert!(Categories::parse("<documentHeader>").is_err());
        assert_eq!(Exclusion::TwitterGenre.reason(), RefusalReason::PlatformTerms);
        assert_eq!(EXCLUSIONS, [Exclusion::TwitterGenre]);
        assert_eq!(serde_json::to_string(&Exclusion::TwitterGenre).unwrap(), r#""twitter_genre""#);
        let doc =
            ExcludedDocument { path: "data/written/twitter/tweets1.hdr".into(), exclusion: Exclusion::TwitterGenre };
        let json = serde_json::to_string(&doc).unwrap();
        assert_eq!(json, r#"{"path":"data/written/twitter/tweets1.hdr","exclusion":"twitter_genre"}"#);
        assert_eq!(serde_json::from_str::<ExcludedDocument>(&json).unwrap(), doc);
    }

    #[test]
    fn a_document_outside_the_corpus_directory_is_refused() {
        let dir = std::env::temp_dir();
        let doc = turbo_formats::masc::MascDocument::parse(
            "/elsewhere/doc.hdr",
            turbo_formats::masc::MascFiles {
                text: b"A.",
                segments: Some(b"<graph><region xml:id=\"seg-r0\" anchors=\"0 2\"/></graph>"),
                sentences: Some(b"<graph><region xml:id=\"s-r0\" anchors=\"0 2\"/></graph>"),
                penn: None,
                named_entities: None,
            },
        )
        .unwrap();
        let corpus = MascCorpus::from_documents(vec![doc; 10]);
        assert!(matches!(SPLIT.apply(corpus, &dir), Err(Error::OutsideCorpus { .. })));
    }

    #[test]
    fn an_unrecorded_release_is_neither_fetched_nor_taken_for_a_corpus() {
        let unrecorded = Release::NotRecorded;
        assert!(matches!(dataset(unrecorded), Err(Error::NotRecorded)));
        let home = std::env::temp_dir();
        let user: UserId = "tester".parse().unwrap();
        assert!(matches!(fetch_release(unrecorded, &home, &user), Err(Error::NotRecorded)));
        let any = CorpusDigest { sha256: String::new(), files: 0 };
        assert!(!unrecorded.is(&any));
    }

    #[test]
    fn the_release_is_recorded_as_a_permissive_dataset_of_the_zoo() {
        let Release::Recorded(r) = RELEASE else { panic!("the release is recorded") };
        let a = dataset(RELEASE).unwrap();
        let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&a)).as_bytes()).unwrap();
        let got = index.get(DATASET_ID).unwrap();
        assert_eq!(got.class, LicenceClass::Permissive);
        assert_eq!((got.files[0].size, got.files[0].sha256.to_string()), (r.size, r.sha256.to_owned()));
        assert_eq!(r.corpus_sha256.len(), 64);
        assert!(r.corpus_sha256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        assert!(RELEASE.is(&CorpusDigest { sha256: r.corpus_sha256.to_owned(), files: 4425 }));
    }

    const RECORDED: RecordedRelease = RecordedRelease {
        file: "MASC-3.0.0.zip",
        format: Format::Zip,
        url: "https://example.org/MASC-3.0.0.zip",
        size: 3,
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        sha512: "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        corpus_dir: "MASC-3.0.0",
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

        assert!(matches!(check_corpus(&dir, Release::NotRecorded, false), Err(Error::NotRecorded)));
        assert_eq!(check_corpus(&dir, Release::NotRecorded, true).unwrap(), d);
        // Another corpus than the recorded release: refused, unless asked
        // for, and then not taken for the release.
        let other = check_corpus(&dir, Release::Recorded(RECORDED), false).unwrap_err();
        assert!(matches!(other, Error::CorpusDigest { .. }), "{other}");
        assert!(other.to_string().contains("MASC-3.0.0"), "{other}");
        assert_eq!(check_corpus(&dir, Release::Recorded(RECORDED), true).unwrap(), d);
        assert!(!Release::Recorded(RECORDED).is(&d));
        assert!(matches!(check_corpus(&dir, RELEASE, false), Err(Error::CorpusDigest { .. })));
        let same = CorpusDigest { sha256: RECORDED.corpus_sha256.to_owned(), files: 2 };
        assert_eq!(check_digest(same.clone(), Release::Recorded(RECORDED), false).unwrap(), same);
        assert!(Release::Recorded(RECORDED).is(&same));

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
