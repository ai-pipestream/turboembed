//! Constituency parser models, parser types, and event types
//! (`opennlp.tools.parser.ParserModel`, `opennlp.tools.parser.ParserType`,
//! `opennlp.tools.parser.ParserEventTypeEnum`, and `opennlp.tools.parser.ParserChunkerFactory`).
//!
//! This module reads a model file. Writing one (`new ParserModel(...)` and the
//! `update*Model` copies) is `turbo_train_components::parser::model`, beside the trainers.

use std::collections::HashSet;
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use turbo_analysis::ModelRef;
use turbo_maxent::{Digest, MaxentModel, ModelFile};
use turbo_postag::PosModel;
use turbo_sequence::BeamSearch;
use turbo_sequence::chunker::{
    ChunkerValidator, DEFAULT_BEAM_SIZE as CHUNKER_DEFAULT_BEAM_SIZE, DefaultChunkerContext,
    MODEL_ENTRY_NAME as CHUNKER_MODEL_ENTRY_NAME, TokenTag,
};
use turbo_sequence::manifest_beam_size;

use crate::context::{ChunkContextGenerator, ParserChunkerSequenceValidator};
use crate::head_rules::{AncoraSpanishHeadRules, EnglishHeadRules, GapLabeler, HeadRules, HeadRulesError};
use crate::parse::{Constituent, Head, Parse, ParseError};

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

/// Which factory the chunker sub-model inside a [`ParserModel`] uses.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParserChunkerFactoryKind {
    /// `opennlp.tools.parser.ParserChunkerFactory` (`ChunkContextGenerator` + `ParserChunkerSequenceValidator`).
    ParserChunker,
    /// `opennlp.tools.chunker.ChunkerFactory` (`DefaultChunkerContext` + `ChunkerValidator`).
    DefaultChunker,
}

/// The chunker sub-model embedded in a [`ParserModel`] (`parserchunker.chunker`).
#[derive(Debug)]
pub struct ParserChunkerModel {
    file: ModelFile,
    maxent: MaxentModel,
    beam_size: usize,
    factory_kind: ParserChunkerFactoryKind,
    outcomes: Vec<String>,
}

impl ParserChunkerModel {
    /// Opens a chunker sub-model from a [`ModelFile`].
    pub fn open(file: ModelFile) -> Result<Self, ParserModelError> {
        if file.component_name() != "ChunkerME" {
            return Err(ParserModelError::ChunkerModel(format!(
                "component {:?} is not ChunkerME",
                file.component_name()
            )));
        }
        let factory_kind = match file.manifest().get("factory") {
            None | Some(DEFAULT_CHUNKER_FACTORY) => ParserChunkerFactoryKind::DefaultChunker,
            Some(PARSER_CHUNKER_FACTORY) => ParserChunkerFactoryKind::ParserChunker,
            Some(other) => {
                return Err(ParserModelError::ChunkerModel(format!("unsupported chunker factory {other:?}")));
            }
        };
        let maxent = file.maxent(CHUNKER_MODEL_ENTRY_NAME)?;
        if maxent.num_outcomes() == 0 {
            return Err(ParserModelError::ChunkerModel("classifier has no outcomes".into()));
        }
        let outcomes: Vec<String> = maxent
            .outcome_texts()
            .map_err(|o| ParserModelError::ChunkerModel(format!("outcome {o:?} is not valid UTF-8")))?
            .into_iter()
            .map(str::to_owned)
            .collect();
        let beam_size = manifest_beam_size(&file, CHUNKER_DEFAULT_BEAM_SIZE)?;
        if beam_size == 0 {
            return Err(ParserModelError::ChunkerModel("beam size must not be 0".into()));
        }
        Ok(Self { file, maxent, beam_size, factory_kind, outcomes })
    }

    /// Returns the underlying [`ModelFile`].
    pub fn file(&self) -> &ModelFile {
        &self.file
    }

    /// Returns the classifier (`chunker.model`).
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// Returns the beam size.
    pub fn beam_size(&self) -> usize {
        self.beam_size
    }

    /// Returns the factory kind.
    pub fn factory_kind(&self) -> ParserChunkerFactoryKind {
        self.factory_kind
    }

    /// Returns the outcome labels.
    pub fn outcomes(&self) -> &[String] {
        &self.outcomes
    }

    /// Runs `ChunkerME.topKSequences(words, tags, minSequenceScore)` over this chunker model.
    pub fn top_k_sequences<'m>(
        &'m self,
        words: &[&str],
        tags: &[&str],
        min_sequence_score: f64,
    ) -> Vec<turbo_sequence::Sequence<'m>> {
        let tuples: Vec<TokenTag<'_>> =
            words.iter().copied().zip(tags.iter().copied()).map(|(token, tag)| TokenTag { token, tag }).collect();
        let search = BeamSearch::new(self.beam_size, &self.maxent).expect("outcomes validated at open");
        match self.factory_kind {
            ParserChunkerFactoryKind::ParserChunker => {
                let mut cg = ChunkContextGenerator::with_cache(CHUNKER_DEFAULT_BEAM_SIZE);
                let validator = ParserChunkerSequenceValidator::new(&self.outcomes);
                search.best_sequences(CHUNKER_DEFAULT_BEAM_SIZE, &tuples, min_sequence_score, &mut cg, &validator)
            }
            ParserChunkerFactoryKind::DefaultChunker => {
                let mut cg = DefaultChunkerContext;
                let validator = ChunkerValidator;
                search.best_sequences(CHUNKER_DEFAULT_BEAM_SIZE, &tuples, min_sequence_score, &mut cg, &validator)
            }
        }
    }
}

/// An opened OpenNLP constituency [`ParserModel`] (`opennlp.tools.parser.ParserModel`).
#[derive(Debug)]
pub struct ParserModel {
    file: ModelFile,
    parser_type: ParserType,
    build_model: MaxentModel,
    check_model: MaxentModel,
    attach_model: Option<MaxentModel>,
    tagger_model: PosModel,
    chunker_model: ParserChunkerModel,
    head_rules: HeadRulesArtifact,
    model_ref: ModelRef,
}

impl ParserModel {
    /// Reads a [`ParserModel`] from a `.bin` file on disk.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, ParserModelError> {
        Self::open(ModelFile::read(path)?)
    }

    /// Reads a [`ParserModel`] from in-memory `.bin` bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, ParserModelError> {
        Self::open(ModelFile::open(bytes)?)
    }

    /// Opens a [`ParserModel`] from an already opened [`ModelFile`].
    pub fn open(file: ModelFile) -> Result<Self, ParserModelError> {
        if file.component_name() != COMPONENT_NAME {
            return Err(ParserModelError::Component(file.component_name().into()));
        }
        let type_str = file.manifest().get(PARSER_TYPE);
        let parser_type = type_str
            .and_then(ParserType::parse)
            .ok_or_else(|| ParserModelError::InvalidParserType(type_str.map(str::to_owned)))?;

        if !file.archive().contains(BUILD_MODEL_ENTRY_NAME) {
            return Err(ParserModelError::MissingArtifact(BUILD_MODEL_ENTRY_NAME));
        }
        let build_model = file.maxent(BUILD_MODEL_ENTRY_NAME)?;

        if !file.archive().contains(CHECK_MODEL_ENTRY_NAME) {
            return Err(ParserModelError::MissingArtifact(CHECK_MODEL_ENTRY_NAME));
        }
        let check_model = file.maxent(CHECK_MODEL_ENTRY_NAME)?;

        let attach_model = match parser_type {
            ParserType::Chunking => {
                if file.archive().contains(ATTACH_MODEL_ENTRY_NAME) {
                    return Err(ParserModelError::UnexpectedAttachModel);
                }
                None
            }
            ParserType::TreeInsert => {
                if !file.archive().contains(ATTACH_MODEL_ENTRY_NAME) {
                    return Err(ParserModelError::MissingArtifact(ATTACH_MODEL_ENTRY_NAME));
                }
                Some(file.maxent(ATTACH_MODEL_ENTRY_NAME)?)
            }
        };

        if !file.archive().contains(PARSER_TAGGER_MODEL_ENTRY_NAME) {
            return Err(ParserModelError::MissingArtifact(PARSER_TAGGER_MODEL_ENTRY_NAME));
        }
        let tagger_bytes = file.entry(PARSER_TAGGER_MODEL_ENTRY_NAME)?;
        let tagger_file = ModelFile::open(tagger_bytes.into_owned())?;
        let tagger_model = PosModel::open(tagger_file)?;

        if !file.archive().contains(CHUNKER_TAGGER_MODEL_ENTRY_NAME) {
            return Err(ParserModelError::MissingArtifact(CHUNKER_TAGGER_MODEL_ENTRY_NAME));
        }
        let chunker_bytes = file.entry(CHUNKER_TAGGER_MODEL_ENTRY_NAME)?;
        let chunker_file = ModelFile::open(chunker_bytes.into_owned())?;
        let chunker_model = ParserChunkerModel::open(chunker_file)?;

        if !file.archive().contains(HEAD_RULES_MODEL_ENTRY_NAME) {
            return Err(ParserModelError::MissingArtifact(HEAD_RULES_MODEL_ENTRY_NAME));
        }
        let head_rules_bytes = file.entry(HEAD_RULES_MODEL_ENTRY_NAME)?;
        let head_rules_str = std::str::from_utf8(&head_rules_bytes)
            .map_err(|e| HeadRulesError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        let head_rules = match file.manifest().get(HEAD_RULES_SERIALIZER_KEY) {
            None | Some(EN_HEAD_RULES_SERIALIZER) | Some(PARSER_HEAD_RULES_SERIALIZER) => {
                HeadRulesArtifact::English(EnglishHeadRules::from_str(head_rules_str)?)
            }
            Some(ES_HEAD_RULES_SERIALIZER) => {
                HeadRulesArtifact::AncoraSpanish(AncoraSpanishHeadRules::from_str(head_rules_str)?)
            }
            Some(other) => return Err(ParserModelError::UnsupportedHeadRulesSerializer(other.to_owned())),
        };

        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}:{}", file.language()),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };

        Ok(Self {
            file,
            parser_type,
            build_model,
            check_model,
            attach_model,
            tagger_model,
            chunker_model,
            head_rules,
            model_ref,
        })
    }

    /// Returns the parser architecture type (`ParserModel.getParserType`).
    pub fn parser_type(&self) -> ParserType {
        self.parser_type
    }

    /// Returns the `build.model` classifier (`ParserModel.getBuildModel`).
    pub fn build_model(&self) -> &MaxentModel {
        &self.build_model
    }

    /// Returns the `check.model` classifier (`ParserModel.getCheckModel`).
    pub fn check_model(&self) -> &MaxentModel {
        &self.check_model
    }

    /// Returns the `attach.model` classifier, present for `TREEINSERT` models (`ParserModel.getAttachModel`).
    pub fn attach_model(&self) -> Option<&MaxentModel> {
        self.attach_model.as_ref()
    }

    /// Returns the embedded POS tagger model (`ParserModel.getParserTaggerModel`).
    pub fn parser_tagger_model(&self) -> &PosModel {
        &self.tagger_model
    }

    /// Returns the embedded chunker model (`ParserModel.getParserChunkerModel`).
    pub fn parser_chunker_model(&self) -> &ParserChunkerModel {
        &self.chunker_model
    }

    /// Returns the embedded head rules (`ParserModel.getHeadRules`).
    pub fn head_rules(&self) -> &HeadRulesArtifact {
        &self.head_rules
    }

    /// Returns the model's language code.
    pub fn language(&self) -> &str {
        self.file.language()
    }

    /// Returns the OpenNLP version recorded in the manifest.
    pub fn opennlp_version(&self) -> &str {
        self.file.opennlp_version()
    }

    /// Returns the SHA-256 digest of the `.bin` archive.
    pub fn sha256(&self) -> Digest {
        self.file.sha256()
    }

    /// Returns the provenance [`ModelRef`].
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// Returns the underlying [`ModelFile`].
    pub fn file(&self) -> &ModelFile {
        &self.file
    }

    /// Returns the raw `.bin` archive bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.file.bytes()
    }
}
