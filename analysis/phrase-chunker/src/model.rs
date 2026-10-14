//! A chunker model: OpenNLP's `ChunkerModel` with the choices
//! `ChunkerFactory` and `ChunkerME` make when they load it.

use std::path::Path;
use std::sync::Arc;

use turbo_analysis::ModelRef;
use turbo_analysis_layers::{Tag, TagSet};
use turbo_maxent::{Digest, MaxentModel, ModelFile};
use turbo_sequence::chunker::{ChunkerValidator, DEFAULT_BEAM_SIZE, DefaultChunkerContext, MODEL_ENTRY_NAME};
use turbo_sequence::manifest_beam_size;

use crate::error::ModelError;
use crate::phrases::kind_of;

/// The `Component-Name` of a chunker model.
pub const COMPONENT_NAME: &str = "ChunkerME";
/// The one factory class whose behaviour this port reproduces.
pub const FACTORY: &str = "opennlp.tools.chunker.ChunkerFactory";

/// An opened `ChunkerME` model: the classifier and the beam size, with the
/// tag set of its outcomes and the tag set of the phrase kinds they name.
///
/// `ChunkerFactory` has one context generator and one validator, so there
/// is nothing else to choose; the beam size is the manifest's `BeamSize`,
/// or 10 without one, as `ChunkerModel.getChunkerSequenceModel` reads it.
#[derive(Debug)]
pub struct ChunkerModel {
    file: ModelFile,
    maxent: MaxentModel,
    beam_size: usize,
    tagset: Arc<TagSet>,
    kinds: Arc<TagSet>,
    model_ref: ModelRef,
}

impl ChunkerModel {
    /// Reads a model file from disk.
    pub fn read(path: impl AsRef<Path>) -> Result<ChunkerModel, ModelError> {
        ChunkerModel::open(ModelFile::read(path)?)
    }

    /// Reads the model from an opened model file. It checks what OpenNLP
    /// checks when it loads and runs a model, and what would fail later:
    /// the component is `ChunkerME`, the factory is `ChunkerFactory` (a
    /// manifest without one means the default factory, as `BaseModel`
    /// reads it), the classifier is there with at least one outcome, and
    /// the manifest's `BeamSize` is a non-negative integer and not zero.
    pub fn open(file: ModelFile) -> Result<ChunkerModel, ModelError> {
        if file.component_name() != COMPONENT_NAME {
            return Err(ModelError::Component(file.component_name().into()));
        }
        if let Some(factory) = file.manifest().get("factory")
            && factory != FACTORY
        {
            return Err(ModelError::Factory(factory.into()));
        }
        let maxent = file.maxent(MODEL_ENTRY_NAME)?;
        if maxent.num_outcomes() == 0 {
            return Err(ModelError::NoOutcomes);
        }
        let beam_size = manifest_beam_size(&file, DEFAULT_BEAM_SIZE)?;
        if beam_size == 0 {
            return Err(ModelError::ZeroBeam);
        }
        let outcomes = maxent.outcome_texts().map_err(|o| ModelError::OutcomeNotText(o.clone()))?;
        let tagset = TagSet::new("chunk", &outcomes);
        // The kinds the outcomes can start a phrase with, in outcome
        // order; an outcome without a kind is left out and refused when
        // it starts one.
        let mut kinds: Vec<&str> = Vec::new();
        for outcome in &outcomes {
            if let Some(kind) = kind_of(outcome)
                && !kinds.contains(&kind)
            {
                kinds.push(kind);
            }
        }
        let kinds = TagSet::new("phrase", &kinds);
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}:{}", file.language()),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(ChunkerModel { file, maxent, beam_size, tagset, kinds, model_ref })
    }

    /// The same model searched with a beam of `size`, as a `ChunkerModel`
    /// rebuilt with another `BeamSize` would be. Zero is refused.
    pub fn with_beam_size(mut self, size: usize) -> Result<ChunkerModel, ModelError> {
        if size == 0 {
            return Err(ModelError::ZeroBeam);
        }
        self.beam_size = size;
        Ok(self)
    }

    /// The manifest's `Language`.
    pub fn language(&self) -> &str {
        self.file.language()
    }

    /// The manifest's `OpenNLP-Version`: the version that trained it.
    pub fn opennlp_version(&self) -> &str {
        self.file.opennlp_version()
    }

    /// The classifier, `chunker.model`.
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// The model file.
    pub fn file(&self) -> &ModelFile {
        &self.file
    }

    /// The beam the search keeps: the manifest's `BeamSize`, 10 without
    /// one, or what [`with_beam_size`](Self::with_beam_size) set.
    pub fn beam_size(&self) -> usize {
        self.beam_size
    }

    /// The chunk tags the chunker writes: the classifier's outcomes, in
    /// outcome order, named `chunk`.
    pub fn tagset(&self) -> &Arc<TagSet> {
        &self.tagset
    }

    /// The tag of the outcome with this index.
    pub fn tag_of_outcome(&self, index: usize) -> Tag {
        Tag(index as u16)
    }

    /// The tag of this outcome, if the classifier has it.
    pub fn tag_of(&self, outcome: &str) -> Option<Tag> {
        self.tagset.tag(outcome)
    }

    /// The kinds of phrase the outcomes name (`NP` for `B-NP` and `I-NP`),
    /// distinct, in outcome order, named `phrase`.
    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    /// The context generator, `DefaultChunkerContextGenerator`.
    pub fn context(&self) -> DefaultChunkerContext {
        DefaultChunkerContext
    }

    /// The sequence validator, `DefaultChunkerSequenceValidator`.
    pub fn validator(&self) -> ChunkerValidator {
        ChunkerValidator
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

    fn fixtures() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures")
    }

    #[test]
    fn reads_the_fixture_model() {
        let m = ChunkerModel::read(fixtures().join("chunker-fixture.bin")).unwrap();
        assert_eq!(m.language(), "en");
        assert_eq!(m.opennlp_version(), "3.0.0-SNAPSHOT");
        assert_eq!(m.beam_size(), 10);
        assert_eq!(m.sha256().to_hex(), "cc68fdbac1ee51695e232e59ac02754dde51e5fb1dd50d15c565c805204f23b0");
        assert_eq!(m.model_ref().id, "opennlp:ChunkerME:en");
        assert_eq!(m.model_ref().revision, "3.0.0-SNAPSHOT");
        assert_eq!(m.tagset().name, "chunk");
        assert_eq!(m.tagset().len(), m.maxent().num_outcomes());
        for (i, outcome) in m.maxent().outcome_texts().unwrap().into_iter().enumerate() {
            assert_eq!(m.tagset().label(m.tag_of_outcome(i)), outcome);
            assert_eq!(m.tag_of(outcome), Some(m.tag_of_outcome(i)));
            assert!(outcome == "O" || outcome.starts_with("B-") || outcome.starts_with("I-"), "{outcome}");
        }
        assert!(m.tag_of("B-NP").is_some());
        assert_eq!(m.tag_of("NP"), None);
        // Every kind but the one `O` would name.
        assert_eq!(m.kinds().name, "phrase");
        let kinds: Vec<&str> = (0..m.kinds().len()).map(|i| m.kinds().label(Tag(i as u16))).collect();
        // The training data has B and I tags for NP and VP, and only B
        // tags for PP, SBAR and ADJP.
        assert_eq!(kinds, ["NP", "VP", "PP", "ADJP", "SBAR"]);
        assert_eq!(m.tagset().len(), 8);
    }

    #[test]
    fn beam_sizes() {
        let m = ChunkerModel::read(fixtures().join("chunker-fixture.bin")).unwrap();
        assert_eq!(m.with_beam_size(1).unwrap().beam_size(), 1);
        let m = ChunkerModel::read(fixtures().join("chunker-fixture.bin")).unwrap();
        assert!(matches!(m.with_beam_size(0), Err(ModelError::ZeroBeam)));
    }

    #[test]
    fn other_components_are_refused() {
        let e = ChunkerModel::read(fixtures().join("pos-fixture.bin")).unwrap_err();
        assert_eq!(e.to_string(), "chunker model: component \"POSTaggerME\" is not ChunkerME");
        let e = ChunkerModel::read(fixtures().join("en-plain.bin")).unwrap_err();
        assert_eq!(e.to_string(), "chunker model: component \"SentenceDetectorME\" is not ChunkerME");
    }
}
