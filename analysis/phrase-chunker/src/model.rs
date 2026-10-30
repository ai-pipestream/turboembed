//! A chunker model: `ChunkerModel` with the choices `ChunkerFactory` and
//! `ChunkerME` make when they load it.

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
    /// Reads a model file from disk (disabled for external `.bin` files under `#1511`).
    pub fn read(path: impl AsRef<Path>) -> Result<ChunkerModel, ModelError> {
        ChunkerModel::open(ModelFile::read(path)?)
    }

    /// Opens a trained `ChunkerME` archive produced by `turbo-train-components`
    /// (`TrainedChunkerModel::package`).
    pub fn from_trained_archive(bytes: Vec<u8>) -> Result<ChunkerModel, ModelError> {
        ChunkerModel::open(ModelFile::from_trained_archive(bytes)?)
    }

    /// Reads the model from an opened model file.
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
pub(crate) mod tests {
    use super::*;

    fn model_with_outcome_and_component(component: &str, outcome: turbo_train::JavaString) -> ModelFile {
        use turbo_train::{Event, Key, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters};
        let events = [Event::from_strs(outcome, &["w=a"]), Event::from_strs("O", &["w=b"])];
        let mut params = TrainingParameters::new();
        params.put(Key::ITERATIONS, 1);
        params.put(Key::CUTOFF, 0);
        let (model, _) =
            turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
        let mut package = ModelPackage::new(Manifest::new(component, "en", 0));
        package.add_entry(MODEL_ENTRY_NAME, turbo_train::write_gis(&model).unwrap());
        ModelFile::from_trained_archive(package.to_bytes()).unwrap()
    }

    fn model_with_outcome(outcome: turbo_train::JavaString) -> ModelFile {
        model_with_outcome_and_component(COMPONENT_NAME, outcome)
    }

    pub(crate) fn test_fixture_model() -> ChunkerModel {
        use turbo_sequence::Features;
        use turbo_sequence::chunker::TokenTag;
        use turbo_train::{Event, Key, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters};
        let sentences: &[&[(&str, &str, &str)]] = &[
            &[
                ("The", "DT", "B-NP"),
                ("cat", "NN", "I-NP"),
                ("sat", "VBD", "B-VP"),
                ("on", "IN", "B-PP"),
                ("the", "DT", "B-NP"),
                ("mat", "NN", "I-NP"),
                (".", ".", "O"),
            ],
            &[
                ("Very", "RB", "B-ADJP"),
                ("good", "JJ", "I-ADJP"),
                ("because", "IN", "B-SBAR"),
                ("dogs", "NN", "B-NP"),
                ("bark", "VBD", "B-VP"),
                (".", ".", "O"),
            ],
        ];
        let mut events = Vec::new();
        let mut feats = Features::new();
        for sent in sentences {
            let seq: Vec<TokenTag<'_>> = sent.iter().map(|&(token, tag, _)| TokenTag { token, tag }).collect();
            let priors: Vec<&str> = sent.iter().map(|&(_, _, out)| out).collect();
            for i in 0..seq.len() {
                feats.clear();
                DefaultChunkerContext.features(&mut feats, i, &seq, &priors[..i]);
                let ctx: Vec<String> = (0..feats.len()).map(|k| feats.to_string_lossy(k)).collect();
                let ctx_refs: Vec<&str> = ctx.iter().map(String::as_str).collect();
                events.push(Event::from_strs(priors[i], &ctx_refs));
            }
        }
        let mut params = TrainingParameters::new();
        params.put(Key::ITERATIONS, 20);
        params.put(Key::CUTOFF, 0);
        let (model, _) =
            turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
        let mut package = ModelPackage::new(Manifest::new(COMPONENT_NAME, "en", 0));
        package.add_entry(MODEL_ENTRY_NAME, turbo_train::write_gis(&model).unwrap());
        ChunkerModel::from_trained_archive(package.to_bytes()).unwrap()
    }

    #[test]
    fn a_model_with_an_outcome_holding_half_a_surrogate_pair_is_refused() {
        use turbo_train::{Event, Key, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters};
        let lone = turbo_train::JavaString::from_utf16(&[0x42, 0xD83D]);
        let events = [Event::from_strs(lone.clone(), &["w=a"]), Event::from_strs("O", &["w=b"])];
        let mut params = TrainingParameters::new();
        params.put(Key::ITERATIONS, 1);
        params.put(Key::CUTOFF, 0);
        let (model, _) =
            turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
        assert_eq!(turbo_train::write_gis(&model), Err(turbo_train::Error::UnpairedSurrogate));

        // Hand-crafted CESU-8 surrogate bytes on the wire are also refused by the reader.
        let valid_bytes = turbo_train::write_gis(
            &turbo_train::gis::train_events(
                &[Event::from_strs("BXYZ", &["w=a"]), Event::from_strs("O", &["w=b"])],
                &params,
                &mut TrainingConfiguration::default(),
            )
            .unwrap()
            .0,
        )
        .unwrap();
        let pos = valid_bytes.windows(4).position(|w| w == b"BXYZ").unwrap();
        let mut bad_bytes = valid_bytes;
        bad_bytes[pos..pos + 4].copy_from_slice(b"B\xED\xA0\xBD");
        let mut package = ModelPackage::new(Manifest::new(COMPONENT_NAME, "en", 0));
        package.add_entry(MODEL_ENTRY_NAME, bad_bytes);
        let file = ModelFile::from_trained_archive(package.to_bytes()).unwrap();
        assert!(ChunkerModel::open(file).is_err());

        let text = model_with_outcome(turbo_train::JavaString::new("B"));
        assert!(ChunkerModel::open(text).is_ok());
    }

    #[test]
    fn reads_the_trained_fixture_model_and_rejects_legacy_bin() {
        let legacy = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/chunker-fixture.bin");
        let legacy_bytes = std::fs::read(legacy).unwrap();
        assert!(ChunkerModel::from_trained_archive(legacy_bytes).is_err());

        let m = test_fixture_model();
        assert_eq!(m.language(), "en");
        assert_eq!(m.opennlp_version(), "1");
        assert_eq!(m.beam_size(), 10);
        assert_eq!(m.model_ref().id, "opennlp:ChunkerME:en");
        assert_eq!(m.model_ref().revision, "1");
        assert_eq!(m.tagset().name, "chunk");
        assert_eq!(m.tagset().len(), m.maxent().num_outcomes());
        for (i, outcome) in m.maxent().outcome_texts().unwrap().into_iter().enumerate() {
            assert_eq!(m.tagset().label(m.tag_of_outcome(i)), outcome);
            assert_eq!(m.tag_of(outcome), Some(m.tag_of_outcome(i)));
            assert!(outcome == "O" || outcome.starts_with("B-") || outcome.starts_with("I-"), "{outcome}");
        }
        assert!(m.tag_of("B-NP").is_some());
        assert_eq!(m.tag_of("NP"), None);
        assert_eq!(m.kinds().name, "phrase");
        let kinds: Vec<&str> = (0..m.kinds().len()).map(|i| m.kinds().label(Tag(i as u16))).collect();
        assert_eq!(kinds, ["NP", "VP", "PP", "ADJP", "SBAR"]);
        assert_eq!(m.tagset().len(), 8);
    }

    #[test]
    fn beam_sizes() {
        let m = test_fixture_model();
        assert_eq!(m.with_beam_size(1).unwrap().beam_size(), 1);
        let m = test_fixture_model();
        assert!(matches!(m.with_beam_size(0), Err(ModelError::ZeroBeam)));
    }

    #[test]
    fn other_components_are_refused() {
        let e = ChunkerModel::open(model_with_outcome_and_component("POSTaggerME", "B".into())).unwrap_err();
        assert_eq!(e.to_string(), "chunker model: component \"POSTaggerME\" is not ChunkerME");
        let e = ChunkerModel::open(model_with_outcome_and_component("SentenceDetectorME", "B".into())).unwrap_err();
        assert_eq!(e.to_string(), "chunker model: component \"SentenceDetectorME\" is not ChunkerME");
    }
}
