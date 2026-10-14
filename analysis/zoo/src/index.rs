//! The zoo index: every artifact the zoo knows, with its licence, the
//! licences of the data it was trained on, its class, and each file's
//! address, size and digests. docs/zoo.md describes the format.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::digest::{Sha256, Sha512};
use crate::error::ZooError;
use crate::licence::{Licence, LicenceClass};

/// The index this crate carries: the OpenNLP 1.3 UD models the zoo may
/// fetch (docs/licences.md lists them, and the ones it refuses).
pub const BUILTIN: &str = include_str!("../index/zoo.json");

/// The only `zoo_version` this build reads.
pub const ZOO_VERSION: u32 = 1;

macro_rules! constant_enum {
    ($(#[$m:meta])* $name:ident, $what:literal { $($(#[$vm:meta])* $variant:ident => $text:literal,)* }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum $name {
            $($(#[$vm])* $variant,)*
        }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant,)*];

            /// The constant the index writes.
            pub const fn name(self) -> &'static str {
                match self {
                    $($name::$variant => $text,)*
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.name())
            }
        }

        impl FromStr for $name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, String> {
                $name::ALL.iter().copied().find(|v| v.name() == s).ok_or_else(|| format!("{s:?} is not {}", $what))
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.name())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
            }
        }
    };
}

constant_enum! {
    /// What an artifact is.
    Kind, "an artifact kind" {
        Model => "KIND_MODEL",
        Head => "KIND_HEAD",
        Lexicon => "KIND_LEXICON",
        Centroids => "KIND_CENTROIDS",
        Dataset => "KIND_DATASET",
    }
}

constant_enum! {
    /// The analysis component a model serves. The first eight are
    /// OpenNLP's `opennlp.tools.models.ModelType` constants, prefixed.
    Component, "a component" {
        Lemmatizer => "COMPONENT_LEMMATIZER",
        Tokenizer => "COMPONENT_TOKENIZER",
        SentenceDetector => "COMPONENT_SENTENCE_DETECTOR",
        Pos => "COMPONENT_POS",
        PosGeneric => "COMPONENT_POS_GENERIC",
        NameFinder => "COMPONENT_NAME_FINDER",
        Chunker => "COMPONENT_CHUNKER",
        Parser => "COMPONENT_PARSER",
        LanguageDetector => "COMPONENT_LANGUAGE_DETECTOR",
        DocumentCategorizer => "COMPONENT_DOCUMENT_CATEGORIZER",
    }
}

constant_enum! {
    /// How an artifact's files are read.
    Format, "a format" {
        /// An OpenNLP model archive (`.bin`), as `BaseModel` writes it.
        OpennlpModel => "FORMAT_OPENNLP_MODEL",
        Safetensors => "FORMAT_SAFETENSORS",
        Conllu => "FORMAT_CONLLU",
        Text => "FORMAT_TEXT",
        Zip => "FORMAT_ZIP",
    }
}

/// An artifact's id: lowercase ASCII letters, digits, `.`, `_` and `-`,
/// starting with a letter or digit, at most 128 bytes. It names the
/// artifact's directory in the cache, so it can never be `.` or `..` or
/// hold a `/`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArtifactId(String);

impl ArtifactId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for ArtifactId {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let b = s.as_bytes();
        let ok = !b.is_empty()
            && b.len() <= 128
            && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
            && b.iter().all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'.' | b'_' | b'-'));
        if ok { Ok(ArtifactId(s.to_owned())) } else { Err(format!("{s:?} is not an artifact id")) }
    }
}

impl fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A language tag: two or three lowercase letters, then optional
/// subtags of one to eight ASCII letters or digits after `-` (`en`,
/// `pt-BR`). Only the shape is checked, not the registry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Language(String);

impl Language {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Language {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let mut parts = s.split('-');
        let primary = parts.next().unwrap_or("");
        let ok = (2..=3).contains(&primary.len())
            && primary.bytes().all(|c| c.is_ascii_lowercase())
            && parts.all(|p| (1..=8).contains(&p.len()) && p.bytes().all(|c| c.is_ascii_alphanumeric()));
        if ok { Ok(Language(s.to_owned())) } else { Err(format!("{s:?} is not a language tag")) }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// An `https://` address of printable ASCII with no spaces.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Url(String);

impl Url {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Url {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let ok = s.len() > "https://".len() && s.starts_with("https://") && s.bytes().all(|c| c.is_ascii_graphic());
        if ok { Ok(Url(s.to_owned())) } else { Err(format!("{s:?} is not an https:// address")) }
    }
}

impl fmt::Display for Url {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A file's path inside its artifact's directory: relative, `/`
/// separated, no empty, `.` or `..` component, no backslash and no
/// control character.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelPath(String);

impl RelPath {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for RelPath {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let ok = !s.is_empty()
            && !s.starts_with('/')
            && !s.chars().any(|c| c == '\\' || c.is_control())
            && s.split('/').all(|p| !p.is_empty() && p != "." && p != "..");
        if ok { Ok(RelPath(s.to_owned())) } else { Err(format!("{s:?} is not a relative path inside the directory")) }
    }
}

impl fmt::Display for RelPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

macro_rules! parsed_string_serde {
    ($($t:ty),*) => {$(
        impl Serialize for $t {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $t {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
            }
        }
    )*};
}

parsed_string_serde!(ArtifactId, Language, Url, RelPath);

/// Data an artifact was made from, and its licence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingData {
    /// The dataset and its release, as its publisher names them
    /// (`UD_English-EWT r2.16`).
    pub name: String,
    pub license: Licence,
    /// Where the dataset's publisher states the licence.
    pub license_terms: Url,
}

/// One file of an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct File {
    /// Where the file goes in the artifact's directory.
    pub path: RelPath,
    /// Where it is fetched from. The address must serve the same bytes for
    /// as long as the entry exists: a release archive, or a repository
    /// path at a commit.
    pub url: Url,
    pub size: u64,
    pub sha256: Sha256,
    /// The digest OpenNLP's `DownloadUtil` checks.
    pub sha512: Sha512,
}

/// One artifact of the index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub id: ArtifactId,
    pub kind: Kind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<Component>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<Language>,
    pub format: Format,
    /// The publisher's version of the artifact.
    pub version: String,
    /// The artifact's own licence.
    pub license: Licence,
    /// Where the artifact's publisher states that licence.
    pub license_terms: Url,
    /// What it was trained or derived from. Required, empty for an
    /// artifact made from no data (a dataset, a hand-written lexicon).
    pub trained_on: Vec<TrainingData>,
    /// The most restrictive class of `license` and every
    /// `trained_on[].license`. Written out so a reader that does not know
    /// a licence still sees the class; checked against the computed one.
    pub class: LicenceClass,
    pub files: Vec<File>,
}

impl Artifact {
    /// The class `license` and `trained_on` give, whatever `class` says.
    pub fn computed_class(&self) -> LicenceClass {
        self.licences().map(Licence::class).max().unwrap_or(LicenceClass::Permissive)
    }

    /// The artifact's own licence, then each training licence, in index
    /// order.
    pub fn licences(&self) -> impl Iterator<Item = Licence> + '_ {
        std::iter::once(self.license).chain(self.trained_on.iter().map(|t| t.license))
    }

    /// The distinct licences whose class needs the user's acceptance under
    /// `policy`, in `Licence` order.
    pub fn licences_to_accept(&self, policy: crate::licence::Policy) -> Vec<Licence> {
        let set: BTreeSet<Licence> = self.licences().filter(|l| l.class().needs_acceptance(policy)).collect();
        set.into_iter().collect()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexFile {
    zoo_version: u32,
    artifacts: Vec<Artifact>,
}

/// A refused artifact: what it is and why. The zoo neither lists nor
/// fetches it; it is kept only so that asking for it names the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub id: ArtifactId,
    /// Each refused licence among the artifact's own and its training
    /// data's.
    pub licences: Vec<Licence>,
}

/// A parsed, checked index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Index {
    artifacts: Vec<Artifact>,
    refused: Vec<Refusal>,
}

impl Index {
    /// The index this crate carries.
    pub fn builtin() -> Index {
        Index::parse(BUILTIN.as_bytes()).expect("the built-in index is checked by the crate's tests")
    }

    /// Parse and check an index. Any unknown field or constant, a missing
    /// required field, a malformed id, path, address or digest, a
    /// duplicate id or path, an artifact with no files, or a `class` that
    /// is not the one its licences give is `ZooError::Index`. Refused
    /// artifacts are set aside, not listed.
    pub fn parse(bytes: &[u8]) -> Result<Index, ZooError> {
        let file: IndexFile = serde_json::from_slice(bytes).map_err(|e| ZooError::Index(e.to_string()))?;
        if file.zoo_version != ZOO_VERSION {
            return Err(ZooError::Index(format!("zoo_version {} is not {ZOO_VERSION}", file.zoo_version)));
        }
        let mut ids = BTreeSet::new();
        let mut artifacts = Vec::new();
        let mut refused = Vec::new();
        for a in file.artifacts {
            if !ids.insert(a.id.clone()) {
                return Err(ZooError::Index(format!("{}: listed twice", a.id)));
            }
            let computed = a.computed_class();
            if a.class != computed {
                return Err(ZooError::Index(format!("{}: class is {}, its licences give {computed}", a.id, a.class)));
            }
            if a.files.is_empty() {
                return Err(ZooError::Index(format!("{}: no files", a.id)));
            }
            let mut paths = BTreeSet::new();
            for f in &a.files {
                if !paths.insert(f.path.as_str()) {
                    return Err(ZooError::Index(format!("{}: {} listed twice", a.id, f.path)));
                }
            }
            if a.version.is_empty() || a.version.chars().any(char::is_control) {
                return Err(ZooError::Index(format!(
                    "{}: version {:?} is empty or holds a control character",
                    a.id, a.version
                )));
            }
            for t in &a.trained_on {
                if t.name.is_empty() || t.name.chars().any(char::is_control) {
                    return Err(ZooError::Index(format!(
                        "{}: training data name {:?} is empty or holds a control character",
                        a.id, t.name
                    )));
                }
            }
            if computed == LicenceClass::Refused {
                let licences = a.licences().filter(|l| l.class() == LicenceClass::Refused).collect();
                refused.push(Refusal { id: a.id, licences });
            } else {
                artifacts.push(a);
            }
        }
        Ok(Index { artifacts, refused })
    }

    /// The artifacts the zoo lists, in index order. No refused artifact is
    /// among them.
    pub fn list(&self) -> &[Artifact] {
        &self.artifacts
    }

    /// The artifact `id`. A refused one is `ZooError::Refused`, naming its
    /// refused licences; one the index does not have is
    /// `ZooError::NotFound`.
    pub fn get(&self, id: &str) -> Result<&Artifact, ZooError> {
        if let Some(a) = self.artifacts.iter().find(|a| a.id.as_str() == id) {
            return Ok(a);
        }
        if let Some(r) = self.refused.iter().find(|r| r.id.as_str() == id) {
            return Err(ZooError::Refused { id: r.id.clone(), licences: r.licences.clone() });
        }
        Err(ZooError::NotFound(id.to_owned()))
    }

    /// The listed models for `component` in `language`, in index order, as
    /// OpenNLP's `DownloadUtil` looks a model up by language and
    /// `ModelType`.
    pub fn find(&self, language: &str, component: Component) -> impl Iterator<Item = &Artifact> + '_ {
        let language = language.to_owned();
        self.artifacts.iter().filter(move |a| {
            a.component == Some(component) && a.language.as_ref().is_some_and(|l| l.as_str() == language)
        })
    }

    /// `artifacts` written as an index in the canonical form: serde_json's
    /// pretty printer (two-space indent, fields in the order docs/zoo.md
    /// gives them) and a final newline.
    pub fn to_canonical_json(artifacts: &[Artifact]) -> String {
        let file = IndexFile { zoo_version: ZOO_VERSION, artifacts: artifacts.to_vec() };
        let mut s = serde_json::to_string_pretty(&file).expect("an index always serializes");
        s.push('\n');
        s
    }
}
