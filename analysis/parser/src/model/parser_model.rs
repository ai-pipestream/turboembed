//! Opening an OpenNLP constituency [`ParserModel`] (`opennlp.tools.parser.ParserModel`) and
//! reading its sub-models, head rules, and manifest.

use std::path::Path;
use std::str::FromStr;

use turbo_analysis::ModelRef;
use turbo_maxent::{Digest, MaxentModel, ModelFile};
use turbo_postag::PosModel;

use crate::head_rules::{AncoraSpanishHeadRules, EnglishHeadRules, HeadRulesError};

use super::*;

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
