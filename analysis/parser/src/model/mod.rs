//! Constituency parser models, parser types, and event types
//! (`opennlp.tools.parser.ParserModel`, `opennlp.tools.parser.ParserType`,
//! `opennlp.tools.parser.ParserEventTypeEnum`, and `opennlp.tools.parser.ParserChunkerFactory`).
//!
//! This module reads a model file. Writing one (`new ParserModel(...)` and the
//! `update*Model` copies) is `turbo_train_components::parser::model`, beside the trainers.

use std::collections::HashSet;
use std::fmt;

use crate::head_rules::{AncoraSpanishHeadRules, EnglishHeadRules, GapLabeler, HeadRules, HeadRulesError};
use crate::parse::{Constituent, Head, Parse, ParseError};

mod chunker;
mod parser_model;

pub use chunker::{ParserChunkerFactoryKind, ParserChunkerModel};
pub use parser_model::ParserModel;

/// `ParserModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "Parser";

/// `ParserModel.BUILD_MODEL_ENTRY_NAME`.
pub const BUILD_MODEL_ENTRY_NAME: &str = "build.model";

/// `ParserModel.CHECK_MODEL_ENTRY_NAME`.
pub const CHECK_MODEL_ENTRY_NAME: &str = "check.model";

/// `ParserModel.ATTACH_MODEL_ENTRY_NAME`.
pub const ATTACH_MODEL_ENTRY_NAME: &str = "attach.model";

/// `ParserModel.PARSER_TAGGER_MODEL_ENTRY_NAME`.
pub const PARSER_TAGGER_MODEL_ENTRY_NAME: &str = "parsertager.postagger";

/// `ParserModel.CHUNKER_TAGGER_MODEL_ENTRY_NAME`.
pub const CHUNKER_TAGGER_MODEL_ENTRY_NAME: &str = "parserchunker.chunker";

/// `ParserModel.HEAD_RULES_MODEL_ENTRY_NAME`.
pub const HEAD_RULES_MODEL_ENTRY_NAME: &str = "head-rules.headrules";

/// `ParserModel.PARSER_TYPE`: the manifest property naming the [`ParserType`].
pub const PARSER_TYPE: &str = "parser-type";

/// The manifest property `BaseModel.serialize` records the head rules' serializer class under,
/// `serializer-class-` and [`HEAD_RULES_MODEL_ENTRY_NAME`].
pub const HEAD_RULES_SERIALIZER_KEY: &str = "serializer-class-head-rules.headrules";

/// The serializer class `BaseModel.serialize` records for the tagger sub-model, a `POSModel`,
/// under `serializer-class-parsertager.postagger`. The chunker sub-model and the maxent models are
/// not `SerializableArtifact`s and get no such entry.
pub const POS_MODEL_SERIALIZER: &str = "opennlp.tools.util.model.POSModelSerializer";

/// `ParserChunkerFactory` class name stored in the chunker sub-model manifest.
pub const PARSER_CHUNKER_FACTORY: &str = "opennlp.tools.parser.ParserChunkerFactory";

/// Standard `ChunkerFactory` class name.
pub const DEFAULT_CHUNKER_FACTORY: &str = "opennlp.tools.chunker.ChunkerFactory";

/// Serializer class for `opennlp.tools.parser.lang.en.HeadRules`.
pub const EN_HEAD_RULES_SERIALIZER: &str = "opennlp.tools.parser.lang.en.HeadRules$HeadRulesSerializer";

/// Serializer class for `opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`.
pub const ES_HEAD_RULES_SERIALIZER: &str = "opennlp.tools.parser.lang.es.AncoraSpanishHeadRules$HeadRulesSerializer";

/// Fallback serializer class registered by `ParserModel.createArtifactSerializers`.
pub const PARSER_HEAD_RULES_SERIALIZER: &str = "opennlp.tools.parser.ParserModel$HeadRulesSerializer";

/// Constituency parser architecture (`opennlp.tools.parser.ParserType`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParserType {
    /// Bottom-up shift-reduce chunking parser (`opennlp.tools.parser.chunking.Parser`).
    Chunking,
    /// Bottom-up tree-insertion parser (`opennlp.tools.parser.treeinsert.Parser`).
    TreeInsert,
}

impl ParserType {
    /// Parses a `ParserType` name (`ParserType.parse`), returning `None` if unknown.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "CHUNKING" => Some(Self::Chunking),
            "TREEINSERT" => Some(Self::TreeInsert),
            _ => None,
        }
    }

    /// Returns the manifest representation (`"CHUNKING"` or `"TREEINSERT"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chunking => "CHUNKING",
            Self::TreeInsert => "TREEINSERT",
        }
    }
}

impl fmt::Display for ParserType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Enumerates the event types generated during parser training (`opennlp.tools.parser.ParserEventTypeEnum`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParserEventType {
    /// Constituent build decisions (`BUILD`).
    Build,
    /// Constituent completeness check decisions (`CHECK`).
    Check,
    /// Tree-insertion attachment decisions (`ATTACH`).
    Attach,
    /// Basal phrase chunking decisions (`CHUNK`).
    Chunk,
    /// Part-of-speech tagging decisions (`TAG`).
    Tag,
}

/// Head rules artifact stored inside a [`ParserModel`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadRulesArtifact {
    /// English Collins/Magerman head rules (`opennlp.tools.parser.lang.en.HeadRules`).
    English(EnglishHeadRules),
    /// AnCora Spanish regex head rules (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`).
    AncoraSpanish(AncoraSpanishHeadRules),
}

impl HeadRulesArtifact {
    /// Returns the Java `SerializableArtifact` serializer class name for this head rules implementation.
    pub fn serializer_class(&self) -> &'static str {
        match self {
            Self::English(_) => EN_HEAD_RULES_SERIALIZER,
            Self::AncoraSpanish(_) => ES_HEAD_RULES_SERIALIZER,
        }
    }

    /// Serializes the head rules in the text format Java writes.
    pub fn serialize_to_bytes(&self) -> Vec<u8> {
        match self {
            Self::English(en) => en.serialize_to_string().into_bytes(),
            Self::AncoraSpanish(es) => es.serialize_to_string().into_bytes(),
        }
    }
}

impl HeadRules for HeadRulesArtifact {
    fn get_head(&self, constituents: &[Parse], kind: &str) -> Option<Head> {
        match self {
            Self::English(en) => en.get_head(constituents, kind),
            Self::AncoraSpanish(es) => es.get_head(constituents, kind),
        }
    }

    fn punctuation_tags(&self) -> &HashSet<String> {
        match self {
            Self::English(en) => en.punctuation_tags(),
            Self::AncoraSpanish(es) => es.punctuation_tags(),
        }
    }
}

impl GapLabeler for HeadRulesArtifact {
    fn label_gaps(&self, stack: &mut [Constituent<'_>], at: usize) -> Result<(), ParseError> {
        match self {
            Self::English(en) => en.label_gaps(stack, at),
            Self::AncoraSpanish(es) => es.label_gaps(stack, at),
        }
    }
}

impl From<EnglishHeadRules> for HeadRulesArtifact {
    fn from(rules: EnglishHeadRules) -> Self {
        Self::English(rules)
    }
}

impl From<AncoraSpanishHeadRules> for HeadRulesArtifact {
    fn from(rules: AncoraSpanishHeadRules) -> Self {
        Self::AncoraSpanish(rules)
    }
}

/// Errors raised when opening or validating a [`ParserModel`].
#[derive(Debug)]
pub enum ParserModelError {
    /// Reading or parsing an archive entry failed.
    Maxent(turbo_maxent::Error),
    /// Sequence model configuration error.
    Sequence(turbo_sequence::Error),
    /// The `Component-Name` of the outer archive is not `Parser`.
    Component(String),
    /// Missing or unrecognized `parser-type` manifest property.
    InvalidParserType(Option<String>),
    /// Missing required model entry (`build.model`, `check.model`, `attach.model`, etc.).
    MissingArtifact(&'static str),
    /// `attach.model` was provided for a `CHUNKING` model.
    UnexpectedAttachModel,
    /// Invalid POS sub-model.
    PosModel(turbo_postag::ModelError),
    /// Invalid chunker sub-model.
    ChunkerModel(String),
    /// Invalid head rules entry.
    HeadRules(HeadRulesError),
    /// Unsupported head rules serializer class.
    UnsupportedHeadRulesSerializer(String),
}

impl fmt::Display for ParserModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Maxent(e) => write!(f, "parser model: {e}"),
            Self::Sequence(e) => write!(f, "parser model: {e}"),
            Self::Component(c) => write!(f, "parser model: component {c:?} is not {COMPONENT_NAME}"),
            Self::InvalidParserType(t) => write!(f, "parser model: invalid {PARSER_TYPE} {t:?}"),
            Self::MissingArtifact(a) => write!(f, "parser model: missing artifact {a}"),
            Self::UnexpectedAttachModel => f.write_str("attachModel must be null for chunking parser!"),
            Self::PosModel(e) => write!(f, "parser tagger model: {e}"),
            Self::ChunkerModel(e) => write!(f, "parser chunker model: {e}"),
            Self::HeadRules(e) => write!(f, "parser head rules: {e}"),
            Self::UnsupportedHeadRulesSerializer(s) => {
                write!(f, "parser head rules: unsupported serializer class {s:?}")
            }
        }
    }
}

impl std::error::Error for ParserModelError {}

impl From<turbo_maxent::Error> for ParserModelError {
    fn from(e: turbo_maxent::Error) -> Self {
        Self::Maxent(e)
    }
}

impl From<turbo_sequence::Error> for ParserModelError {
    fn from(e: turbo_sequence::Error) -> Self {
        Self::Sequence(e)
    }
}

impl From<turbo_postag::ModelError> for ParserModelError {
    fn from(e: turbo_postag::ModelError) -> Self {
        Self::PosModel(e)
    }
}

impl From<HeadRulesError> for ParserModelError {
    fn from(e: HeadRulesError) -> Self {
        Self::HeadRules(e)
    }
}
