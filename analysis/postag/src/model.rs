//! A part-of-speech model: OpenNLP's `POSModel` with the choices
//! `POSTaggerFactory` and `POSTaggerME` make when they load it.

use std::path::Path;
use std::sync::Arc;

use turbo_analysis::ModelRef;
use turbo_analysis_layers::{Tag, TagSet};
use turbo_maxent::{Digest, MaxentModel, ModelFile};
use turbo_sequence::postag::{
    DEFAULT_BEAM_SIZE, DEFAULT_TAG_FORMAT, MODEL_ENTRY_NAME, PosContext, PosTagFormat, PosValidator, TagDictionary,
    TagFormatMapper,
};
use turbo_sequence::{PatternMode, manifest_beam_size};

use crate::error::ModelError;

/// The `Component-Name` of a part-of-speech model.
pub const COMPONENT_NAME: &str = "POSTaggerME";
/// The one factory class whose behaviour this port reproduces.
pub const FACTORY: &str = "opennlp.tools.postag.POSTaggerFactory";

/// An opened `POSTaggerME` model: the classifier, the beam size, the tag
/// dictionary, the context generator's descriptor and the tag format the
/// tagger returns.
///
/// The tag format is [`DEFAULT_TAG_FORMAT`] (Universal Dependencies)
/// unless [`with_format`](Self::with_format) asks for another, as
/// `POSTaggerME(POSModel)` and `POSTaggerME(POSModel, POSTagFormat)`
/// take it. The tags written are the model's outcomes converted into that
/// format by `POSTagFormatMapper` when the format guessed from the
/// outcomes differs from the one asked for, and the outcomes unchanged
/// otherwise ([`tag_format`](Self::tag_format) says which came out).
///
/// Features are built over UTF-16 code units ([`PatternMode::Utf16Units`]),
/// which is how OpenNLP trained every published model.
#[derive(Debug)]
pub struct PosModel {
    file: ModelFile,
    maxent: MaxentModel,
    beam_size: usize,
    /// The context generator's cache size when a caller set one; the beam
    /// size otherwise, as `POSTaggerME` defaults it.
    cache_size: Option<usize>,
    dictionary: Option<Arc<TagDictionary>>,
    requested: PosTagFormat,
    guessed: PosTagFormat,
    mapper: Option<TagFormatMapper>,
    tagset: Arc<TagSet>,
    /// The tag of each outcome, by outcome index.
    outcome_tags: Vec<Tag>,
    model_ref: ModelRef,
}

impl PosModel {
    /// Reads a model file from disk, in the default tag format.
    pub fn read(path: impl AsRef<Path>) -> Result<PosModel, ModelError> {
        PosModel::open(ModelFile::read(path)?)
    }

    /// Reads the model from an opened model file, in the default tag
    /// format.
    pub fn open(file: ModelFile) -> Result<PosModel, ModelError> {
        PosModel::open_with_format(file, DEFAULT_TAG_FORMAT)
    }

    /// Reads the model from an opened model file, to return tags in
    /// `format`. It checks what OpenNLP checks when it loads and runs a
    /// model, and what would fail later: the component is `POSTaggerME`,
    /// the factory is `POSTaggerFactory` (a manifest without one means
    /// the default factory, as `BaseModel` reads it), the classifier is
    /// there with at least one outcome, the manifest's `BeamSize` is a
    /// non-negative integer and not zero, the tag dictionary reads, and
    /// the descriptor builds its generators.
    pub fn open_with_format(file: ModelFile, format: PosTagFormat) -> Result<PosModel, ModelError> {
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
        let dictionary = TagDictionary::for_model(&file)?.map(Arc::new);
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}:{}", file.language()),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        let mut model = PosModel {
            file,
            maxent,
            beam_size,
            cache_size: None,
            dictionary,
            requested: format,
            guessed: PosTagFormat::Unknown,
            mapper: None,
            tagset: TagSet::new("", &[]),
            outcome_tags: Vec::new(),
            model_ref,
        };
        model.set_format(format);
        // The descriptor is read once here so that a model whose
        // generators cannot be built is refused when it is opened.
        model.new_context()?;
        Ok(model)
    }

    /// The same model returning tags in `format`.
    pub fn with_format(mut self, format: PosTagFormat) -> PosModel {
        self.set_format(format);
        self
    }

    /// The same model searched with a beam of `size`, as a `POSModel`
    /// rebuilt with another `BeamSize` would be. Zero is refused.
    pub fn with_beam_size(mut self, size: usize) -> Result<PosModel, ModelError> {
        if size == 0 {
            return Err(ModelError::ZeroBeam);
        }
        self.beam_size = size;
        Ok(self)
    }

    /// The same model with a context generator cache of `size` entries,
    /// as `POSTaggerME(model, format, contextCacheSize)` takes it; zero
    /// turns the cache off. Without this the cache is as large as the
    /// beam.
    pub fn with_context_cache(mut self, size: usize) -> PosModel {
        self.cache_size = Some(size);
        self
    }

    /// `POSTaggerME`'s choice of mapper for `format`, and the tag set of
    /// the tags it then returns: the outcomes converted into the format
    /// when a mapper is made, and the outcomes themselves otherwise.
    fn set_format(&mut self, format: PosTagFormat) {
        self.requested = format;
        self.guessed = TagFormatMapper::guess(self.maxent.outcomes()).guessed_format();
        self.mapper = TagFormatMapper::for_tagger(format, self.maxent.outcomes());
        let mut labels: Vec<&str> = Vec::with_capacity(self.maxent.num_outcomes());
        for outcome in self.maxent.outcomes() {
            let label = self.mapper.map_or(outcome, |m| m.convert_tag(outcome));
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
        self.tagset = TagSet::new(self.tag_format().as_str().to_ascii_lowercase().as_str(), &labels);
        self.outcome_tags = self
            .maxent
            .outcomes()
            .map(|outcome| {
                let label = self.mapper.map_or(outcome, |m| m.convert_tag(outcome));
                self.tagset.tag(label).expect("every converted outcome is in the tag set")
            })
            .collect();
    }

    /// The manifest's `Language`.
    pub fn language(&self) -> &str {
        self.file.language()
    }

    /// The manifest's `OpenNLP-Version`: the version that trained it.
    pub fn opennlp_version(&self) -> &str {
        self.file.opennlp_version()
    }

    /// The classifier, `pos.model`.
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// The model file, for the descriptor and resources.
    pub fn file(&self) -> &ModelFile {
        &self.file
    }

    /// The beam the search keeps: the manifest's `BeamSize`, 3 without
    /// one, or what [`with_beam_size`](Self::with_beam_size) set.
    pub fn beam_size(&self) -> usize {
        self.beam_size
    }

    /// The context generator's cache size: the beam size unless
    /// [`with_context_cache`](Self::with_context_cache) set one.
    pub fn context_cache_size(&self) -> usize {
        self.cache_size.unwrap_or(self.beam_size)
    }

    /// The tag dictionary, `tags.tagdict`, when the model has one.
    pub fn tag_dictionary(&self) -> Option<&Arc<TagDictionary>> {
        self.dictionary.as_ref()
    }

    /// The format the tagger was asked for.
    pub fn requested_format(&self) -> PosTagFormat {
        self.requested
    }

    /// The format `POSTagFormatMapper` guessed from the outcomes.
    pub fn guessed_format(&self) -> PosTagFormat {
        self.guessed
    }

    /// The format of the tags written: the guessed one when the outcomes
    /// are returned unchanged (the format asked for is the guessed one, or
    /// `Custom`), and otherwise the one the mapper converts to, which for
    /// a model of unknown format is still `Unknown`.
    pub fn tag_format(&self) -> PosTagFormat {
        match (self.requested, self.mapper) {
            (PosTagFormat::Custom, _) => PosTagFormat::Custom,
            (_, None) => self.guessed,
            (_, Some(m)) => match m.guessed_format() {
                PosTagFormat::Ud => PosTagFormat::Penn,
                PosTagFormat::Penn => PosTagFormat::Ud,
                other => other,
            },
        }
    }

    /// Whether the outcomes are converted before they are written.
    pub fn converts_tags(&self) -> bool {
        self.mapper.is_some()
    }

    /// The tags the tagger writes: the distinct converted outcomes, in
    /// outcome order, named after [`tag_format`](Self::tag_format) in
    /// lowercase (`ud`, `penn`, `custom`, `unknown`).
    pub fn tagset(&self) -> &Arc<TagSet> {
        &self.tagset
    }

    /// The tag written for the outcome with this index.
    pub fn tag_of_outcome(&self, index: usize) -> Tag {
        self.outcome_tags[index]
    }

    /// The tag written for this outcome, if the classifier has it.
    pub fn tag_of(&self, outcome: &str) -> Option<Tag> {
        self.maxent.outcome_index(outcome).map(|i| self.outcome_tags[i])
    }

    /// The label written for this outcome, converted when the format asks
    /// for it; the outcome itself otherwise.
    pub fn label_of<'a>(&self, outcome: &'a str) -> &'a str {
        self.mapper.map_or(outcome, |m| m.convert_tag(outcome))
    }

    /// A context generator for this model, with the cache size of
    /// [`context_cache_size`](Self::context_cache_size): the one
    /// `POSTaggerFactory.getPOSContextGenerator` makes. Each has its own
    /// cache, so one serves one search at a time.
    pub fn new_context(&self) -> Result<PosContext, ModelError> {
        Ok(PosContext::for_model(&self.file, self.context_cache_size(), PatternMode::Utf16Units)?)
    }

    /// The sequence validator, `DefaultPOSSequenceValidator` over the tag
    /// dictionary.
    pub fn validator(&self) -> PosValidator {
        PosValidator::new(self.dictionary.clone())
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

    /// The English model, or `None` when it is not downloaded.
    fn english() -> Option<PosModel> {
        Some(PosModel::read(turbo_goldens::published_model("opennlp-en-ud-ewt-pos-1.3-2.5.4.bin")?).unwrap())
    }

    #[test]
    fn reads_the_english_model() {
        let Some(m) = english() else { return };
        assert_eq!(m.language(), "en");
        assert_eq!(m.opennlp_version(), "2.5.4");
        assert_eq!(m.beam_size(), 3);
        assert_eq!(m.context_cache_size(), 3);
        assert!(m.tag_dictionary().is_none());
        assert_eq!(m.sha256().to_hex(), "510b89087e7ef7ffb8c50c63e3e999122404ac9fa64790dfa93b38d1938c52ce");
        assert_eq!(m.model_ref().id, "opennlp:POSTaggerME:en");
        assert_eq!(m.model_ref().revision, "2.5.4");
        // UD outcomes, asked for UD: unchanged, so the tag set is the
        // outcomes themselves, joined ones (such as AUX+PART) included.
        let ud = PosTagFormat::Ud;
        assert_eq!((m.requested_format(), m.guessed_format(), m.tag_format()), (ud, ud, ud));
        assert!(!m.converts_tags());
        assert_eq!(m.tagset().name, "ud");
        assert_eq!(m.tagset().len(), m.maxent().num_outcomes());
        for (i, outcome) in m.maxent().outcomes().enumerate() {
            assert_eq!(m.tagset().label(m.tag_of_outcome(i)), outcome);
            assert_eq!(m.tag_of(outcome), Some(m.tag_of_outcome(i)));
            assert_eq!(m.label_of(outcome), outcome);
        }
        assert!(m.tagset().tag("NOUN").is_some());
        assert_eq!(m.tag_of("NN"), None);
    }

    #[test]
    fn penn_tags_from_a_ud_model() {
        let Some(m) = english() else { return };
        let m = m.with_format(PosTagFormat::Penn);
        let (penn, ud) = (PosTagFormat::Penn, PosTagFormat::Ud);
        assert_eq!((m.requested_format(), m.guessed_format(), m.tag_format()), (penn, ud, penn));
        assert!(m.converts_tags());
        assert_eq!(m.tagset().name, "penn");
        assert_eq!(m.label_of("NOUN"), "NN");
        assert_eq!(m.label_of("AUX+PART"), "?");
        assert!(m.tagset().len() < m.maxent().num_outcomes(), "the joined outcomes share one tag");
        assert_eq!(m.tagset().label(m.tag_of("AUX+PART").unwrap()), "?");
    }

    #[test]
    fn custom_and_unknown_formats() {
        let Some(m) = english() else { return };
        let m = m.with_format(PosTagFormat::Custom);
        assert_eq!(m.tag_format(), PosTagFormat::Custom);
        assert!(!m.converts_tags());
        assert_eq!(m.tagset().name, "custom");
        assert_eq!(m.label_of("AUX+PART"), "AUX+PART");
        // Asking a UD model for the unknown format converts to Penn, as
        // Java does.
        let m = m.with_format(PosTagFormat::Unknown);
        assert_eq!(m.tag_format(), PosTagFormat::Penn);
        assert_eq!(m.label_of("NOUN"), "NN");
    }

    #[test]
    fn beam_and_cache_sizes() {
        let Some(m) = english() else { return };
        let m = m.with_beam_size(10).unwrap();
        assert_eq!((m.beam_size(), m.context_cache_size()), (10, 10));
        let m = m.with_context_cache(0);
        assert_eq!((m.beam_size(), m.context_cache_size()), (10, 0));
        assert!(matches!(english().unwrap().with_beam_size(0), Err(ModelError::ZeroBeam)));
    }

    #[test]
    fn the_fixture_model_has_a_tag_dictionary_and_penn_outcomes() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/pos-fixture.bin");
        let m = PosModel::read(path).unwrap();
        assert_eq!(m.language(), "en");
        let d = m.tag_dictionary().unwrap();
        assert!(!d.is_case_sensitive());
        assert_eq!(d.tags("Paris").unwrap(), ["NNP"]);
        assert_eq!(d.tags("RUN").unwrap(), ["NN", "VB", "VBP"]);
        assert_eq!((m.guessed_format(), m.tag_format()), (PosTagFormat::Penn, PosTagFormat::Ud));
        assert_eq!(m.label_of("NNP"), "PROPN");
        assert_eq!(m.tagset().name, "ud");
        // The validator refuses a tag the dictionary does not list.
        use turbo_sequence::SequenceValidator;
        let v = m.validator();
        assert!(v.valid_sequence(0, &["Paris"], &[], "NNP"));
        assert!(!v.valid_sequence(0, &["Paris"], &[], "NN"));
        assert!(v.valid_sequence(0, &["London"], &[], "NN"));
    }

    #[test]
    fn other_components_and_factories_are_refused() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/en-plain.bin");
        let e = PosModel::read(path).unwrap_err();
        assert_eq!(e.to_string(), "part-of-speech model: component \"SentenceDetectorME\" is not POSTaggerME");
    }
}
