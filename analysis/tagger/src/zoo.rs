//! Loading single-head and multi-task bundles from [`turbo_zoo::Zoo`]
//! (#1726).
//!
//! [`Zoo::ensure`] verifies every file's size, SHA-256 and SHA-512, enforces
//! `Acceptances` for `CLASS_LGPL`, `CLASS_SHARE_ALIKE` and `CLASS_GPL`, and
//! refuses `CLASS_LOCAL_ONLY` entries before touching the transport.

use std::fmt;
use std::path::PathBuf;

use turbo::status::Error as CoreError;
use turbo_analysis_embed::EncoderOptions;
use turbo_zoo::{Acceptances, Component, Format, Kind, UserId, Zoo, ZooError};

use crate::lemma::Lemmatizer;
use crate::multi::{MultiTask, Outputs};
use crate::parse::DependencyParser;
use crate::segment::{SentenceTagger, SpanTagger};
use crate::words_lang::{LanguageTagger, WordSplitter};
use crate::{EntityTagger, PosTagger};

/// Why a bundle or head could not be fetched or opened from the model zoo.
#[derive(Debug)]
pub enum TaggerZooError {
    /// The zoo refused the id or could not hand out its files.
    Zoo(ZooError),
    /// No artifact in the index serves `(language, component)`.
    NoComponent { language: String, component: Component },
    /// The artifact is not a `FORMAT_SAFETENSORS` (`KIND_MODEL` or `KIND_HEAD`) bundle.
    NotABundle { id: String, kind: Kind, format: Format },
    /// Opening the verified bundle or head failed.
    Bundle(CoreError),
}

impl fmt::Display for TaggerZooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaggerZooError::Zoo(e) => write!(f, "{e}"),
            TaggerZooError::NoComponent { language, component } => {
                write!(f, "no artifact in the zoo serves {component} for language {language:?}")
            }
            TaggerZooError::NotABundle { id, kind, format } => {
                write!(f, "{id} is {kind} {format}, not a FORMAT_SAFETENSORS model or head bundle with manifest.json")
            }
            TaggerZooError::Bundle(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TaggerZooError {}

impl From<ZooError> for TaggerZooError {
    fn from(e: ZooError) -> Self {
        TaggerZooError::Zoo(e)
    }
}

impl From<CoreError> for TaggerZooError {
    fn from(e: CoreError) -> Self {
        TaggerZooError::Bundle(e)
    }
}

fn is_bundle_artifact(a: &turbo_zoo::Artifact) -> bool {
    a.format == Format::Safetensors
        && matches!(a.kind, Kind::Model | Kind::Head)
        && a.files.iter().any(|f| f.path.as_str() == "manifest.json")
}

/// Ensures the bundle artifact `id` is verified in `zoo`'s cache and returns
/// its local bundle directory.
pub fn ensure_bundle(
    zoo: &Zoo<'_>,
    id: &str,
    user: &UserId,
    acceptances: &Acceptances,
) -> Result<PathBuf, TaggerZooError> {
    let a = zoo.index().get(id)?;
    if !is_bundle_artifact(a) {
        return Err(TaggerZooError::NotABundle { id: id.to_owned(), kind: a.kind, format: a.format });
    }
    let fetched = zoo.ensure(id, user, acceptances)?;
    Ok(fetched.dir)
}

/// Finds the first bundle artifact for `(language, component)` in `zoo`
/// (preferring `CLASS_PERMISSIVE` entries first via [`turbo_zoo::Index::find`])
/// and ensures its bundle directory is on disk.
pub fn ensure_component(
    zoo: &Zoo<'_>,
    language: &str,
    component: Component,
    user: &UserId,
    acceptances: &Acceptances,
) -> Result<PathBuf, TaggerZooError> {
    let a = zoo
        .index()
        .find(language, component)
        .find(|a| is_bundle_artifact(a))
        .ok_or_else(|| TaggerZooError::NoComponent { language: language.to_owned(), component })?;
    let id = a.id.as_str().to_owned();
    ensure_bundle(zoo, &id, user, acceptances)
}

impl PosTagger {
    /// Fetches and opens `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<PosTagger, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(PosTagger::open(&dir, name, o)?)
    }

    /// Opens the preferred `COMPONENT_POS` entry for `language` from `zoo`.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<PosTagger, TaggerZooError> {
        let dir = ensure_component(zoo, language, Component::Pos, user, acceptances)?;
        Ok(PosTagger::open(&dir, name, o)?)
    }
}

impl EntityTagger {
    /// Fetches and opens `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<EntityTagger, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(EntityTagger::open(&dir, name, o)?)
    }

    /// Opens the preferred `COMPONENT_NAME_FINDER` entry for `language` from `zoo`.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<EntityTagger, TaggerZooError> {
        let dir = ensure_component(zoo, language, Component::NameFinder, user, acceptances)?;
        Ok(EntityTagger::open(&dir, name, o)?)
    }
}

impl SpanTagger {
    /// Fetches and opens `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<SpanTagger, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(SpanTagger::open(&dir, name, o)?)
    }

    /// Opens the preferred `COMPONENT_NAME_FINDER` entry for `language` from `zoo`.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<SpanTagger, TaggerZooError> {
        let dir = ensure_component(zoo, language, Component::NameFinder, user, acceptances)?;
        Ok(SpanTagger::open(&dir, name, o)?)
    }
}

impl SentenceTagger {
    /// Fetches and opens `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<SentenceTagger, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(SentenceTagger::open(&dir, name, o)?)
    }

    /// Opens the preferred `COMPONENT_SENTENCE_DETECTOR` entry for `language` from `zoo`.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<SentenceTagger, TaggerZooError> {
        let dir = ensure_component(zoo, language, Component::SentenceDetector, user, acceptances)?;
        Ok(SentenceTagger::open(&dir, name, o)?)
    }
}

impl Lemmatizer {
    /// Fetches and opens `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<Lemmatizer, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(Lemmatizer::open(&dir, name, o)?)
    }

    /// Opens the preferred `COMPONENT_LEMMATIZER` entry for `language` from `zoo`.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<Lemmatizer, TaggerZooError> {
        let dir = ensure_component(zoo, language, Component::Lemmatizer, user, acceptances)?;
        Ok(Lemmatizer::open(&dir, name, o)?)
    }
}

impl DependencyParser {
    /// Fetches and opens `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<DependencyParser, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(DependencyParser::open(&dir, name, o)?)
    }

    /// Opens the preferred `COMPONENT_DEPENDENCY_PARSER` entry for `language` from `zoo`.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<DependencyParser, TaggerZooError> {
        let dir = ensure_component(zoo, language, Component::DependencyParser, user, acceptances)?;
        Ok(DependencyParser::open(&dir, name, o)?)
    }
}

impl WordSplitter {
    /// Fetches and opens `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<WordSplitter, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(WordSplitter::open(&dir, name, o)?)
    }

    /// Opens the preferred `COMPONENT_TOKENIZER` entry for `language` from `zoo`.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<WordSplitter, TaggerZooError> {
        let dir = ensure_component(zoo, language, Component::Tokenizer, user, acceptances)?;
        Ok(WordSplitter::open(&dir, name, o)?)
    }
}

impl LanguageTagger {
    /// Fetches and opens `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<LanguageTagger, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(LanguageTagger::open(&dir, name, o)?)
    }

    /// Opens the preferred `COMPONENT_LANGUAGE_DETECTOR` entry for `language` from `zoo`.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<LanguageTagger, TaggerZooError> {
        let dir = ensure_component(zoo, language, Component::LanguageDetector, user, acceptances)?;
        Ok(LanguageTagger::open(&dir, name, o)?)
    }
}

impl<W: Outputs> MultiTask<W> {
    /// Fetches and opens the multi-task bundle `id` from `zoo`.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        names: &[Option<&str>],
        o: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<MultiTask<W>, TaggerZooError> {
        let dir = ensure_bundle(zoo, id, user, acceptances)?;
        Ok(MultiTask::open(&dir, names, o)?)
    }
}
