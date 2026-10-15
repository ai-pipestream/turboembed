//! The zoo index: every artifact the zoo knows, with its licence, the
//! licences of the data it was trained on, its class, and each file's
//! address, size and digests. docs/zoo.md describes the format.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::digest::{Sha256, Sha512};
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

/// An `https://` address of printable ASCII with no spaces and no
/// backslash, whose host is not empty: `https:///x` and
/// `https://user@/x` are refused.
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
        let Some(rest) = s.strip_prefix("https://") else {
            return Err(format!("{s:?} is not an https:// address"));
        };
        if !s.bytes().all(|c| c.is_ascii_graphic() && c != b'\\') {
            return Err(format!("{s:?} is not an https:// address of printable ASCII without a backslash"));
        }
        let authority = &rest[..rest.find(['/', '?', '#']).unwrap_or(rest.len())];
        let host = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
        if host.is_empty() || host.starts_with(':') {
            return Err(format!("{s:?} is not an https:// address: it names no host"));
        }
        Ok(Url(s.to_owned()))
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

/// An `https://` address split after its host: the host and the rest,
/// from the first `/`, `?` or `#` on (empty for a bare host).
/// The scheme is always `https`, which [`Url`] checks.
fn split_host(url: &Url) -> (&str, &str) {
    let rest = &url.as_str()["https://".len()..];
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    (&rest[..end], &rest[end..])
}

/// Strips one `.` or `%2e` (case-insensitive) from the front of `s`.
fn strip_dot(s: &[u8]) -> Option<&[u8]> {
    if let Some(rest) = s.strip_prefix(b".") {
        Some(rest)
    } else if s.len() >= 3 && s[..3].eq_ignore_ascii_case(b"%2e") {
        Some(&s[3..])
    } else {
        None
    }
}

/// The byte a `%` escape's two hex digits give, in either case; `None`
/// unless both are hex digits.
fn hex_byte(digits: &[u8]) -> Option<u8> {
    let [high, low] = digits else { return None };
    let value = |c: u8| (c as char).to_digit(16);
    Some((value(*high)? * 16 + value(*low)?) as u8)
}

/// Whether `path` is free of the bytes and escapes this check knows a
/// server may turn into a step out of a directory, and decodes to valid
/// UTF-8. Scans once, without allocating:
///
/// - every `%` starts an escape of two hex digits, so a malformed escape
///   a lenient decoder might repair into another escape (`%%32%65` or
///   `%2%65` into `%2e`) and IIS's `%u002e` form are refused;
/// - no `\` or `;` is in the path, plain or escaped (`%5c`, `%3b`): some
///   servers read `\` as `/` and strip a `;` with what follows it from a
///   segment (`..;` as `..`);
/// - no escape gives `/`, `%` (`%252e` decodes twice to `.`) or NUL;
/// - the decoded bytes are valid UTF-8 as RFC 3629 defines it, so there
///   is no overlong form (`%c0%ae` for `.`), no surrogate (`%ed%a0%80`),
///   no five- or six-byte form, no byte from `%f5` up, no lone or
///   missing continuation byte, and nothing past U+10FFFF. A lax decoder
///   reads an overlong form as the ASCII character it encodes; the other
///   forms are not UTF-8 at all.
///
/// A `.` or `..` segment is [`plain_path`]'s to refuse.
fn decodes_plainly(path: &[u8]) -> bool {
    // Continuation bytes the character being decoded still needs, and the
    // range the next one must fall in.
    let mut needed = 0u8;
    let (mut low, mut high) = (0x80u8, 0xBFu8);
    let mut i = 0;
    while i < path.len() {
        let byte = match path[i] {
            b'\\' | b';' => return false,
            b'%' => match path.get(i + 1..i + 3).and_then(hex_byte) {
                Some(b'/' | b'\\' | b'%' | b';' | 0) | None => return false,
                Some(decoded) => {
                    i += 3;
                    decoded
                }
            },
            plain => {
                i += 1;
                plain
            }
        };
        if needed > 0 {
            if !(low..=high).contains(&byte) {
                return false;
            }
            needed -= 1;
            (low, high) = (0x80, 0xBF);
            continue;
        }
        (needed, low, high) = match byte {
            0x00..=0x7F => (0, 0x80, 0xBF),
            0xC2..=0xDF => (1, 0x80, 0xBF),
            0xE0 => (2, 0xA0, 0xBF),
            0xE1..=0xEC | 0xEE..=0xEF => (2, 0x80, 0xBF),
            0xED => (2, 0x80, 0x9F),
            0xF0 => (3, 0x90, 0xBF),
            0xF1..=0xF3 => (3, 0x80, 0xBF),
            0xF4 => (3, 0x80, 0x8F),
            // A continuation byte with no lead, `%c0` and `%c1` (always
            // overlong), and `%f5` to `%ff`.
            _ => return false,
        };
    }
    needed == 0
}

/// Whether the path of `rest` (what [`split_host`] leaves, up to a `?` or
/// `#`) is free of the forms this check knows a server may read as a step
/// out of the directory the path names: a `.` or `..` segment, plain or
/// with its dots percent-encoded, and whatever [`decodes_plainly`]
/// refuses. A query or fragment is not checked.
fn plain_path(rest: &str) -> bool {
    let path = &rest[..rest.find(['?', '#']).unwrap_or(rest.len())];
    decodes_plainly(path.as_bytes())
        && path.split('/').all(|segment| match strip_dot(segment.as_bytes()) {
            None => true,
            Some(b"") => false,
            Some(after_first) => strip_dot(after_first) != Some(b""),
        })
}

/// Whether `origin` names a location a file can lie under: no user
/// information before its host (`https://a.org@b.net/` is on `b.net`), no
/// query or fragment, a plain path ([`plain_path`]), and at least one
/// non-empty path segment, so not a whole host (`https://example.org` or
/// `https://example.org/`).
fn origin_is_a_location(origin: &Url) -> bool {
    let (host, rest) = split_host(origin);
    !host.contains('@')
        && !rest.contains(['?', '#'])
        && plain_path(rest)
        && rest.split('/').any(|segment| !segment.is_empty())
}

/// Whether `file` lies under `origin` ([`Artifact::origin`]): the same
/// host, compared without regard to case, a plain path ([`plain_path`]),
/// and a path that is the origin's, without any trailing `/`, or continues
/// it after a `/`. Paths are compared byte for byte.
fn under_origin(origin: &Url, file: &Url) -> bool {
    let (origin_host, origin_rest) = split_host(origin);
    let (file_host, file_rest) = split_host(file);
    let prefix = origin_rest.trim_end_matches('/');
    origin_host.eq_ignore_ascii_case(file_host)
        && plain_path(file_rest)
        && file_rest.strip_prefix(prefix).is_some_and(|after| after.is_empty() || after.starts_with('/'))
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

#[cfg(test)]
mod tests {
    use super::plain_path;

    /// The percent-encoding of every byte of `bytes`, between `/a/x` and
    /// `y`, so no segment is a dot segment.
    fn encoded<'a>(bytes: &[u8], into: &'a mut String) -> &'a str {
        into.clear();
        into.push_str("/a/x");
        for b in bytes {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            into.extend(['%', HEX[usize::from(b >> 4)] as char, HEX[usize::from(b & 15)] as char]);
        }
        into.push('y');
        into
    }

    /// What the path check should say of these decoded bytes: valid UTF-8
    /// holding no `/`, `\`, `%`, `;` or NUL.
    fn expected(bytes: &[u8]) -> bool {
        std::str::from_utf8(bytes).is_ok() && !bytes.iter().any(|b| matches!(b, b'/' | b'\\' | b'%' | b';' | 0))
    }

    /// Every scalar value's UTF-8, percent-encoded, is accepted, except
    /// the five characters a path may not hold escaped.
    #[test]
    fn every_scalar_value_is_accepted_but_five() {
        let mut path = String::new();
        let mut utf8 = [0u8; 4];
        for c in (0..=0x10FFFF).filter_map(char::from_u32) {
            let bytes = c.encode_utf8(&mut utf8).as_bytes();
            let refused = matches!(c, '/' | '\\' | '%' | ';' | '\0');
            assert_eq!(plain_path(encoded(bytes, &mut path)), !refused, "U+{:04X}", c as u32);
        }
    }

    /// The check agrees with the standard library's UTF-8 validation on
    /// every sequence of one or two bytes, every three-byte sequence with
    /// a three-byte lead, and four-byte sequences over every lead and
    /// second byte with a few third and fourth bytes at the edges of the
    /// continuation range.
    #[test]
    fn decoded_bytes_are_checked_as_strictly_as_utf8() {
        let mut path = String::new();
        let mut check = |bytes: &[u8]| {
            assert_eq!(plain_path(encoded(bytes, &mut path)), expected(bytes), "{bytes:02x?}");
        };
        for a in 0..=255u8 {
            check(&[a]);
            for b in 0..=255u8 {
                check(&[a, b]);
                for c in [0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0] {
                    check(&[a, b, c, 0x80]);
                    check(&[a, b, 0x80, c]);
                }
            }
        }
        for a in 0xE0..=0xEFu8 {
            for b in 0..=255u8 {
                for c in 0..=255u8 {
                    check(&[a, b, c]);
                }
            }
        }
    }

    /// A character cut off by the end of the path, or by its query, is
    /// refused.
    #[test]
    fn a_character_cut_off_at_the_end_is_refused() {
        for path in ["/a/x%c3", "/a/x%e0%a0", "/a/x%f0%90%80", "/a/x%c3?%a9", "/a/x%c3#%a9"] {
            assert!(!plain_path(path), "{path}");
        }
        assert!(plain_path("/a/x%c3%a9?%c3"));
    }

    /// Escapes must be well formed: a `%` with fewer than two hex digits
    /// after it is refused, whatever a lenient decoder would make of it.
    #[test]
    fn malformed_escapes_are_refused() {
        for path in ["/a/%", "/a/%2", "/a/%zz", "/a/%2g", "/a/%%32%65", "/a/%2%65", "/a/%%35%63", "/a/%u002e", "/a/x%"]
        {
            assert!(!plain_path(path), "{path}");
        }
        for path in ["/a/%2E%2e.x", "/a/%41%7e", "/a/%3F", "/a/%23"] {
            assert!(plain_path(path), "{path}");
        }
    }
}
