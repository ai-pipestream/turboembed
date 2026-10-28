//! Statistical POS and NER helpers for the in-process analysis engine.

use std::path::Path;
use std::sync::Arc;

use turbo_maxent::{MaxentModel, ModelFile};
use turbo_sequence::namefind::{
    DEFAULT_BEAM_SIZE as NER_DEFAULT_BEAM, DefaultNameContext, MODEL_ENTRY_NAME as NER_MODEL_ENTRY, NameCodec,
    NameFinder,
};
use turbo_sequence::postag::{
    DEFAULT_BEAM_SIZE as POS_DEFAULT_BEAM, DEFAULT_TAG_FORMAT, MODEL_ENTRY_NAME as POS_MODEL_ENTRY, PosContext,
    PosValidator, TagDictionary, TagFormatMapper,
};
use turbo_sequence::{BeamSearch, PatternMode, manifest_beam_size};

use crate::status::{BUNDLE_INVALID, BUNDLE_NOT_FOUND, Error, Result};

pub(super) const DEFAULT_POS_MODEL: &[u8] =
    include_bytes!("../../../analysis/goldens/data/fixtures/pos-fixture-rust.bin");
pub(super) const DEFAULT_LEMMA_MODEL: &[u8] =
    include_bytes!("../../../analysis/goldens/data/fixtures/lemma-fixture-rust.bin");
pub(super) const DEFAULT_LEMMA_DICT: &[u8] =
    include_bytes!("../../../analysis/goldens/data/fixtures/lemmatizer-dictionary.txt");
pub(super) const DEFAULT_NER_MODEL: &[u8] =
    include_bytes!("../../../analysis/goldens/data/fixtures/ner-fixture-rust.bin");
pub(super) const DEFAULT_PEOPLE_XML: &[u8] =
    include_bytes!("../../../analysis/goldens/data/fixtures/namefind/people.xml");
pub(super) const DEFAULT_PLACES_XML: &[u8] =
    include_bytes!("../../../analysis/goldens/data/fixtures/namefind/places.xml");

pub(super) fn read_model_file(path: &str, what: &str) -> Result<ModelFile> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(Error::new(BUNDLE_NOT_FOUND, format!("{what}: file not found at {path}")));
    }
    ModelFile::read(p).map_err(|e| Error::new(BUNDLE_INVALID, format!("{what} ({path}): {e}")))
}

pub(super) struct PosEngine {
    file: ModelFile,
    maxent: MaxentModel,
    beam_size: usize,
    dictionary: Option<Arc<TagDictionary>>,
    mapper: Option<TagFormatMapper>,
}

impl PosEngine {
    pub(super) fn open(file: ModelFile) -> Result<Self> {
        let maxent =
            file.maxent(POS_MODEL_ENTRY).map_err(|e| Error::new(BUNDLE_INVALID, format!("POS model entry: {e}")))?;
        if maxent.num_outcomes() == 0 {
            return Err(Error::new(BUNDLE_INVALID, "POS model has zero outcomes"));
        }
        let outcomes = maxent
            .outcome_texts()
            .map_err(|o| Error::new(BUNDLE_INVALID, format!("POS outcome is not valid UTF-8: {o:?}")))?;
        let beam_size = manifest_beam_size(&file, POS_DEFAULT_BEAM)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("POS BeamSize: {e}")))?;
        if beam_size == 0 {
            return Err(Error::new(BUNDLE_INVALID, "POS BeamSize must be non-zero"));
        }
        let dictionary = TagDictionary::for_model(&file)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("POS tag dictionary: {e}")))?
            .map(Arc::new);
        let mapper = TagFormatMapper::for_tagger(DEFAULT_TAG_FORMAT, outcomes.iter().copied());
        let mut ctx = PosContext::for_model(&file, beam_size, PatternMode::Utf16Units)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("POS context descriptor: {e}")))?;
        let _ = &mut ctx;
        Ok(Self { file, maxent, beam_size, dictionary, mapper })
    }

    pub(super) fn tag_tokens(&self, tokens: &[&str]) -> Result<Vec<(String, f64)>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let mut ctx = PosContext::for_model(&self.file, self.beam_size, PatternMode::Utf16Units)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("POS context: {e}")))?;
        let search = BeamSearch::new(self.beam_size, &self.maxent)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("POS beam search: {e}")))?;
        let validator = PosValidator::new(self.dictionary.clone());
        let Some(best) = search.best_sequence(tokens, &mut ctx, &validator) else {
            return Ok(tokens.iter().map(|_| ("X".to_owned(), 0.0)).collect());
        };
        let mut out = Vec::with_capacity(tokens.len());
        for (&outcome, &prob) in best.outcomes().iter().zip(best.probs()) {
            let tag = match &self.mapper {
                Some(m) => m.convert_tag(outcome),
                None => outcome,
            };
            out.push((tag.to_owned(), prob));
        }
        Ok(out)
    }
}

pub(super) struct NerEngine {
    file: ModelFile,
    maxent: MaxentModel,
    beam_size: usize,
    codec: NameCodec,
}

impl NerEngine {
    pub(super) fn open(file: ModelFile) -> Result<Self> {
        let maxent = file
            .maxent(NER_MODEL_ENTRY)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("TokenNameFinder model entry: {e}")))?;
        let beam_size = manifest_beam_size(&file, NER_DEFAULT_BEAM)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("TokenNameFinder BeamSize: {e}")))?;
        let codec = NameCodec::for_model(&file);
        let _ = DefaultNameContext::for_model(&file, PatternMode::Utf16Units)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("TokenNameFinder context: {e}")))?;
        Ok(Self { file, maxent, beam_size, codec })
    }

    pub(super) fn find(&self, tokens: &[&str]) -> Result<Vec<(u32, u32, String, f64)>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let context = DefaultNameContext::for_model(&self.file, PatternMode::Utf16Units)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("TokenNameFinder context: {e}")))?;
        let mut finder = NameFinder::with_context(&self.maxent, self.beam_size, self.codec, context)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("TokenNameFinder init: {e}")))?;
        let found = finder.find(tokens);
        Ok(found
            .into_iter()
            .map(|n| (n.span.start as u32, n.span.end as u32, n.span.kind.unwrap_or_else(|| "default".into()), n.prob))
            .collect())
    }
}
