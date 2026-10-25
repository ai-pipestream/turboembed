//! The zoo index: every artifact the zoo knows, with its licence, the
//! licences of the data it was trained on, its class, and each file's
//! address, size and digests. docs/zoo.md describes the format.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::ZooError;
use crate::licence::{Licence, LicenceClass, Requirement};

/// The index this crate carries: the OpenNLP 1.3 UD models trained on
/// permissive and share-alike treebanks. docs/licences.md lists them, and
/// the published models the index leaves out: the refused ones, and the
/// local-only ones trained on non-commercial treebanks.
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
                match self { $($name::$variant => $text,)* }
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.name()) }
        }
        impl FromStr for $name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, String> {
                $name::ALL.iter().copied().find(|v| v.name() == s).ok_or_else(|| format!("{s:?} is not {}", $what))
            }
        }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> { s.serialize_str(self.name()) }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
            }
        }
    };
}

mod component;

pub use component::Component;

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
    /// How an artifact's files are read.
    Format, "a format" {
        /// An OpenNLP model archive (`.bin`), as `BaseModel` writes it.
        OpennlpModel => "FORMAT_OPENNLP_MODEL",
        Safetensors => "FORMAT_SAFETENSORS",
        Conllu => "FORMAT_CONLLU",
        Text => "FORMAT_TEXT",
        Zip => "FORMAT_ZIP",
        /// A PyTorch checkpoint in `torch.save`'s zip form only (the legacy
        /// non-zip form is refused): a stored zip of a pickled state dict
        /// and its tensor data. The zoo only fetches and checks its bytes.
        /// The bundle tool's reader takes the tensors without running the
        /// pickle: it allows only the opcodes a state dict uses, the
        /// persistent load of a storage and the tensor rebuild among them,
        /// and only the globals those name.
        Pytorch => "FORMAT_PYTORCH",
        /// A SentencePiece model (`spm.model`), with the model's JSON
        /// configuration beside it.
        Sentencepiece => "FORMAT_SENTENCEPIECE",
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

#[path = "file.rs"]
mod file;
#[path = "origin.rs"]
mod origin;
pub use file::File;
pub use origin::{RelPath, Url};
use origin::{origin_is_a_location, under_origin};

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

parsed_string_serde!(ArtifactId, Language);

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
    /// The credit or copyright notice the artifact's licence or training
    /// data's licences ask for. Required when any licence of the artifact
    /// requires attribution ([`Licence::requires_attribution`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<String>,
    /// For a `CLASS_LOCAL_ONLY` artifact, and only for one: the original
    /// publisher's own location, which every file's `url` must lie under.
    /// For a dataset it is the data's own place on its host, such as a
    /// treebank's repository path on `raw.githubusercontent.com`; for a
    /// model, the model publisher's. It has a path of at least one
    /// non-empty segment, so it is never a whole host; a file at the root
    /// of a host is named as the origin by its own address. The host is
    /// compared without regard to case. A file's path must be the origin's
    /// path or continue it after a `/`, so `.../UD_English-GUM` holds
    /// `.../UD_English-GUM/r2.16/...` but not `.../UD_English-GUM-copy/...`;
    /// a trailing `/` on the origin makes no difference. The origin has no
    /// user information, query or fragment, and neither its path nor a
    /// file's holds a `.` or `..` segment (plain or percent-encoded), a
    /// `\` or `;` (plain or percent-encoded), a percent-encoded `/`, `%` or
    /// NUL, a `%` not followed by two hex digits, or percent-encoded bytes
    /// that are not valid UTF-8, such as an overlong form.
    ///
    /// The rule covers the addresses the index names. The zoo's HTTP
    /// transport requests each file at its `url` and follows a redirect
    /// the server issues from there over HTTPS only, without checking it
    /// against the origin (a GitHub release, for one, redirects to another
    /// host). Every file's size, SHA-256 and SHA-512 are checked before it
    /// is kept, so the bytes are the ones the index records wherever a
    /// redirect leads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Url>,
    pub files: Vec<File>,
}

impl Artifact {
    /// The class `license` and `trained_on` give, whatever `class` says.
    pub fn computed_class(&self) -> LicenceClass {
        crate::licence::class_of(self.licences())
    }

    /// The artifact's own licence, then each training licence, in index
    /// order.
    pub fn licences(&self) -> impl Iterator<Item = Licence> + '_ {
        std::iter::once(self.license).chain(self.trained_on.iter().map(|t| t.license))
    }

    /// Whether the artifact's own licence or any training licence requires
    /// attribution ([`Licence::requires_attribution`]).
    pub fn requires_attribution(&self) -> bool {
        self.licences().any(Licence::requires_attribution)
    }

    /// The distinct terms the user must accept under `policy`, in
    /// `Licence` order, then by address: each licence among the artifact's
    /// own and its training data's whose class needs acceptance, with its
    /// `legal_url`, or, for a licence without one
    /// (`LicenseRef-RestrictedPublic`), the terms address the index gives
    /// next to it.
    pub fn licences_to_accept(&self, policy: crate::licence::Policy) -> Vec<Requirement> {
        let own = std::iter::once((self.license, &self.license_terms));
        let data = self.trained_on.iter().map(|t| (t.license, &t.license_terms));
        let set: BTreeSet<Requirement> = own
            .chain(data)
            .filter(|(l, _)| l.class().needs_acceptance(policy))
            .map(|(licence, terms)| Requirement {
                licence,
                text: licence.legal_url().unwrap_or(terms.as_str()).to_owned(),
            })
            .collect();
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
            match (&a.origin, computed == LicenceClass::LocalOnly) {
                (None, true) => {
                    return Err(ZooError::Index(format!(
                        "{}: {computed} needs origin, its publisher's own location",
                        a.id
                    )));
                }
                (Some(_), false) => {
                    return Err(ZooError::Index(format!("{}: origin is only for {}", a.id, LicenceClass::LocalOnly)));
                }
                (Some(origin), true) => {
                    if !origin_is_a_location(origin) {
                        return Err(ZooError::Index(format!(
                            "{}: origin {origin} is not a location: it is a whole host, or it has user \
                             information, a query or fragment, or a path form the zoo refuses (a `.` or `..` \
                             segment, a backslash, a `;`, or a malformed or refused percent-encoding)",
                            a.id
                        )));
                    }
                    if let Some(f) = a.files.iter().find(|f| !under_origin(origin, &f.url)) {
                        return Err(ZooError::Index(format!(
                            "{}: {} is fetched from {}, which is not under its origin {origin}",
                            a.id, f.path, f.url
                        )));
                    }
                }
                (None, false) => {}
            }
            let mut paths = BTreeSet::new();
            for f in &a.files {
                let p = f.path.as_str();
                if p.split('/').any(|part| part.ends_with(".partial")) {
                    return Err(ZooError::Index(format!(
                        "{}: {} uses a .partial name, which the cache reserves for temporary files",
                        a.id, f.path
                    )));
                }
                if !paths.insert(p) {
                    return Err(ZooError::Index(format!("{}: {} listed twice", a.id, f.path)));
                }
            }
            for &p in &paths {
                for (i, _) in p.match_indices('/') {
                    let dir = &p[..i];
                    if paths.contains(dir) {
                        return Err(ZooError::Index(format!("{}: {dir} is both a file and a directory of {p}", a.id)));
                    }
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
            if a.attribution.as_ref().is_some_and(|s| s.trim().is_empty() || s.chars().any(char::is_control)) {
                return Err(ZooError::Index(format!("{}: attribution is empty or holds a control character", a.id)));
            }
            if a.attribution.is_none() && a.requires_attribution() {
                return Err(ZooError::Index(format!("{}: requires attribution under its licence", a.id)));
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

    /// The artifacts that may be bundled and shipped: the permissive ones.
    /// A `CLASS_LOCAL_ONLY` artifact, or a model trained on one, is never
    /// among them.
    pub fn shippable(&self) -> impl Iterator<Item = &Artifact> + '_ {
        self.artifacts.iter().filter(|a| a.class.may_bundle())
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
