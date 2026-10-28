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

#[cfg(test)]
mod tests {
    use super::*;

    const ENGLISH: &str = "opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin";
    const GERMAN: &str = "opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin";

    /// A published model, or `None` when it is not downloaded.
    fn published(file: &str) -> Option<LemmatizerModel> {
        Some(LemmatizerModel::read(turbo_goldens::published_model(file)?).unwrap())
    }

    #[test]
    fn the_english_model_opens() {
        let Some(m) = published(ENGLISH) else { return };
        assert_eq!(m.language(), "en");
        assert_eq!(m.opennlp_version(), "2.5.4");
        assert_eq!(m.beam_size(), 3);
        assert_eq!(m.sha256().to_hex(), "711ad051faba79865c5f9fc5b8c455d6ad8e6d3e8b92f42142af1ef0bc7b3a96");
        assert_eq!(m.model_ref().id, "opennlp:StatisticalLemmatizer:en");
        assert!(m.maxent().outcome_index("O").is_some());
        assert!(m.maxent().outcome_index("D0s").is_some());
        assert_eq!(m.with_beam_size(5).unwrap().beam_size(), 5);
    }

    #[test]
    fn the_german_model_opens() {
        let Some(m) = published(GERMAN) else { return };
        assert_eq!(m.language(), "de");
        assert_eq!(m.sha256().to_hex(), "8850986f02dfbe293046bfce93bc0c3f3fa010e89177250abca23d860033d4f6");
    }

    #[test]
    fn another_component_is_refused() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/pos-fixture.bin");
        let e = LemmatizerModel::read(path).unwrap_err();
        assert!(matches!(e, ModelError::Component(ref c) if c == "POSTaggerME"), "{e}");
        let Some(m) = published(ENGLISH) else { return };
        assert!(matches!(m.with_beam_size(0), Err(ModelError::ZeroBeam)));
    }
}
