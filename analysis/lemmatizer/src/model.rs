//! A lemmatizer model: OpenNLP's `LemmatizerModel` with the choices
//! `LemmatizerFactory` and `LemmatizerME` make when they load it.

use std::path::Path;

use turbo_analysis::ModelRef;
use turbo_maxent::{Digest, MaxentModel, ModelFile};
use turbo_sequence::manifest_beam_size;

use crate::context::LemmaContext;
use crate::error::ModelError;
use crate::{CASE_SENSITIVE_FACTORY, LemmaEncoding};

/// The `Component-Name` of a lemmatizer model.
pub const COMPONENT_NAME: &str = "StatisticalLemmatizer";
/// The one factory class whose behaviour this port reproduces.
pub const FACTORY: &str = "opennlp.tools.lemmatizer.LemmatizerFactory";
/// The classifier's entry in the model archive.
pub const MODEL_ENTRY_NAME: &str = "lemmatizer.model";
/// `LemmatizerME.DEFAULT_BEAM_SIZE`: the beam without a `BeamSize` in
/// the manifest.
pub const DEFAULT_BEAM_SIZE: usize = 3;

/// An opened `LemmatizerME` model: the classifier whose outcomes are
/// lemma classes ([`crate::edit_script`]) and the beam size. The context
/// generator is [`LemmaContext`] and the sequence validator accepts all
/// classes. The native factory selects case-preserving classes; unknown
/// factories are refused.
#[derive(Debug)]
pub struct LemmatizerModel {
    file: ModelFile,
    maxent: MaxentModel,
    beam_size: usize,
    model_ref: ModelRef,
    encoding: LemmaEncoding,
}

impl LemmatizerModel {
    /// Reads a model file from disk.
    pub fn read(path: impl AsRef<Path>) -> Result<LemmatizerModel, ModelError> {
        LemmatizerModel::open(ModelFile::read(path)?)
    }

    /// Opens a trained `StatisticalLemmatizer` archive produced by `turbo-train-components`
    /// (`TrainedLemmatizerModel::package`).
    pub fn from_trained_archive(bytes: Vec<u8>) -> Result<LemmatizerModel, ModelError> {
        LemmatizerModel::open(ModelFile::from_trained_archive(bytes)?)
    }

    /// Reads the model from an opened model file. It checks what OpenNLP
    /// checks when it loads and runs a model, and what would fail later:
    /// the component is `StatisticalLemmatizer`, the factory is
    /// an OpenNLP or native factory (no factory means the OpenNLP default),
    /// the classifier is there with at
    /// least one outcome, and the manifest's `BeamSize` is a non-negative
    /// integer and not zero.
    pub fn open(file: ModelFile) -> Result<LemmatizerModel, ModelError> {
        if file.component_name() != COMPONENT_NAME {
            return Err(ModelError::Component(file.component_name().into()));
        }
        let encoding = match file.manifest().get("factory") {
            None | Some(FACTORY) => LemmaEncoding::OpenNlp,
            Some(CASE_SENSITIVE_FACTORY) => LemmaEncoding::CaseSensitiveV1,
            Some(factory) => return Err(ModelError::Factory(factory.into())),
        };
        let maxent = file.maxent(MODEL_ENTRY_NAME)?;
        if maxent.num_outcomes() == 0 {
            return Err(ModelError::NoOutcomes);
        }
        let beam_size = manifest_beam_size(&file, DEFAULT_BEAM_SIZE)?;
        if beam_size == 0 {
            return Err(ModelError::ZeroBeam);
        }
        let model_ref = ModelRef {
            id: format!(
                "{}:{COMPONENT_NAME}:{}",
                if encoding == LemmaEncoding::OpenNlp { "opennlp" } else { "turbo" },
                file.language()
            ),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(LemmatizerModel { file, maxent, beam_size, model_ref, encoding })
    }

    /// The same model searched with a beam of `size`, as a
    /// `LemmatizerModel` rebuilt with another `BeamSize` would be. Zero is
    /// refused.
    pub fn with_beam_size(mut self, size: usize) -> Result<LemmatizerModel, ModelError> {
        if size == 0 {
            return Err(ModelError::ZeroBeam);
        }
        self.beam_size = size;
        Ok(self)
    }

    /// The edit class format declared by the model factory.
    pub fn encoding(&self) -> LemmaEncoding {
        self.encoding
    }

    /// The manifest's `Language`.
    pub fn language(&self) -> &str {
        self.file.language()
    }

    /// The manifest's `OpenNLP-Version`: the version that trained it.
    pub fn opennlp_version(&self) -> &str {
        self.file.opennlp_version()
    }

    /// The classifier, `lemmatizer.model`; its outcomes are the lemma
    /// classes.
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// The model file.
    pub fn file(&self) -> &ModelFile {
        &self.file
    }

    /// The beam the search keeps: the manifest's `BeamSize`, 3 without
    /// one, or what [`with_beam_size`](Self::with_beam_size) set.
    pub fn beam_size(&self) -> usize {
        self.beam_size
    }

    /// The context generator, `DefaultLemmatizerContextGenerator`.
    pub fn context(&self) -> LemmaContext {
        LemmaContext
    }

    /// The SHA-256 of the model file.
    pub fn sha256(&self) -> Digest {
        self.file.sha256()
    }

    /// The model as provenance records it.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }
}
