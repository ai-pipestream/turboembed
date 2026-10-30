//! Statistical POS and NER helpers for the in-process analysis engine.
//!
//! This module has no lexical defaults. A prediction comes from the
//! explicitly configured statistical model or is reported as unknown.

use std::path::Path;
use std::sync::Arc;

use turbo_analysis::TagSet;
use turbo_maxent::{MaxentModel, ModelFile};
use turbo_sequence::namefind::{
    DEFAULT_BEAM_SIZE as NER_DEFAULT_BEAM, DefaultNameContext, MODEL_ENTRY_NAME as NER_MODEL_ENTRY, NameCodec,
    NameFinder,
};
use turbo_sequence::postag::{
    DEFAULT_BEAM_SIZE as POS_DEFAULT_BEAM, DEFAULT_TAG_FORMAT, MODEL_ENTRY_NAME as POS_MODEL_ENTRY, PosContext,
    PosTagFormat, PosValidator, TagDictionary, TagFormatMapper,
};
use turbo_sequence::{BeamSearch, PatternMode, manifest_beam_size};

use crate::status::{BUNDLE_INVALID, BUNDLE_NOT_FOUND, Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PosOutcome {
    pub(super) tag: Option<String>,
    pub(super) prob: Option<f64>,
}

impl PosOutcome {
    fn unknown() -> Self {
        Self { tag: None, prob: None }
    }
}

pub(super) fn read_model_file(path: &str, what: &str) -> Result<ModelFile> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(Error::new(BUNDLE_NOT_FOUND, format!("{what}: file not found at {path}")));
    }
    let bytes = std::fs::read(p).map_err(|e| Error::new(BUNDLE_INVALID, format!("{what} ({path}): {e}")))?;
    ModelFile::from_trained_archive(bytes).map_err(|e| Error::new(BUNDLE_INVALID, format!("{what} ({path}): {e}")))
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LemmaOutcome {
    pub(super) lemma: Option<String>,
    pub(super) prob: Option<f64>,
}

impl LemmaOutcome {
    pub(super) fn unknown() -> Self {
        Self { lemma: None, prob: None }
    }
}

pub(super) struct PosEngine {
    file: ModelFile,
    maxent: MaxentModel,
    beam_size: usize,
    dictionary: Option<Arc<TagDictionary>>,
    mapper: Option<TagFormatMapper>,
    tagset: Arc<TagSet>,
}

impl PosEngine {
    pub(super) fn open(file: ModelFile) -> Result<Self> {
        if file.component_name() != "POSTaggerME" {
            return Err(Error::new(
                BUNDLE_INVALID,
                format!("POS model component is {}, expected POSTaggerME", file.component_name()),
            ));
        }
        if let Some(factory) = file.manifest().get("factory")
            && factory != "opennlp.tools.postag.POSTaggerFactory"
        {
            return Err(Error::new(BUNDLE_INVALID, format!("POS model factory is {factory}")));
        }
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
        let guessed = TagFormatMapper::guess(outcomes.iter().copied()).guessed_format();
        let mapper = TagFormatMapper::for_tagger(DEFAULT_TAG_FORMAT, outcomes.iter().copied());
        let mut labels: Vec<&str> = Vec::with_capacity(outcomes.len());
        for &outcome in &outcomes {
            let label = mapper.map_or(outcome, |mapper| mapper.convert_tag(outcome));
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
        let tag_format = match mapper {
            None => guessed,
            Some(mapper) => match mapper.guessed_format() {
                PosTagFormat::Ud => PosTagFormat::Penn,
                PosTagFormat::Penn => PosTagFormat::Ud,
                other => other,
            },
        };
        let tagset = TagSet::new(tag_format.as_str().to_ascii_lowercase().as_str(), &labels);
        let mut ctx = PosContext::for_model(&file, beam_size, PatternMode::Utf16Units)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("POS context descriptor: {e}")))?;
        let _ = &mut ctx;
        Ok(Self { file, maxent, beam_size, dictionary, mapper, tagset })
    }

    pub(super) fn tagset(&self) -> Arc<TagSet> {
        Arc::clone(&self.tagset)
    }

    pub(super) fn tag_tokens(&self, tokens: &[&str]) -> Result<Vec<PosOutcome>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let mut ctx = PosContext::for_model(&self.file, self.beam_size, PatternMode::Utf16Units)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("POS context: {e}")))?;
        let search = BeamSearch::new(self.beam_size, &self.maxent)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("POS beam search: {e}")))?;
        let validator = PosValidator::new(self.dictionary.clone());
        let Some(best) = search.best_sequence(tokens, &mut ctx, &validator) else {
            return Ok(tokens.iter().map(|_| PosOutcome::unknown()).collect());
        };
        if best.outcomes().len() != tokens.len() || best.probs().len() != tokens.len() {
            return Err(Error::new(BUNDLE_INVALID, "POS beam search returned an outcome count mismatch"));
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (&outcome, &prob) in best.outcomes().iter().zip(best.probs()) {
            let tag = match &self.mapper {
                Some(m) => m.convert_tag(outcome),
                None => outcome,
            };
            out.push(PosOutcome { tag: Some(tag.to_owned()), prob: Some(prob) });
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
        if beam_size == 0 {
            return Err(Error::new(BUNDLE_INVALID, "TokenNameFinder BeamSize must be non-zero"));
        }
        let codec = NameCodec::for_model(&file);
        let _ = DefaultNameContext::for_model(&file, PatternMode::Utf16Units)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("TokenNameFinder context: {e}")))?;
        Ok(Self { file, maxent, beam_size, codec })
    }

    /// Stateless per call: a fresh production `NameFinder` and context are
    /// constructed for this one token sequence, so no adaptive data from a
    /// previous call can affect a later one.
    pub(super) fn find(&self, tokens: &[&str]) -> Result<Vec<(u32, u32, String, f64)>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let context = DefaultNameContext::for_model(&self.file, PatternMode::Utf16Units)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("TokenNameFinder context: {e}")))?;
        let mut finder = NameFinder::with_context(&self.maxent, self.beam_size, self.codec, context)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("TokenNameFinder init: {e}")))?;
        let found = finder.find(tokens);
        let mut out = Vec::with_capacity(found.len());
        for n in found {
            let start = u32::try_from(n.span.start)
                .map_err(|_| Error::new(BUNDLE_INVALID, "TokenNameFinder returned an out-of-range span"))?;
            let end = u32::try_from(n.span.end)
                .map_err(|_| Error::new(BUNDLE_INVALID, "TokenNameFinder returned an out-of-range span"))?;
            if start > end || end as usize > tokens.len() {
                return Err(Error::new(BUNDLE_INVALID, "TokenNameFinder returned an invalid span"));
            }
            out.push((start, end, n.span.kind.unwrap_or_else(|| "default".into()), n.prob));
        }
        Ok(out)
    }
}
