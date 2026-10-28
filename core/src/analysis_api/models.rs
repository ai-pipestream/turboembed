//! Statistical POS and NER helpers and built-in lexical defaults for the
//! in-process analysis engine.

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

pub(super) const DEFAULT_LEMMA_DICT: &[u8] = b"\
the\tDET\tthe
a\tDET\ta
an\tDET\tan
is\tAUX\tbe
is\tVERB\tbe
was\tAUX\tbe
are\tAUX\tbe
were\tAUX\tbe
have\tVERB\thave
have\tAUX\thave
has\tAUX\thave
had\tAUX\thave
went\tVERB\tgo
wrote\tVERB\twrite
read\tVERB\tread
read\tNOUN\treading
worked\tVERB\twork
workers\tNOUN\tworker
rivers\tNOUN\triver
valleys\tNOUN\tvalley
plains\tNOUN\tplain
farms\tNOUN\tfarm
dams\tNOUN\tdam
ideas\tNOUN\tidea
patients\tNOUN\tpatient
years\tNOUN\tyear
moved\tVERB\tmove
opened\tVERB\topen
closed\tVERB\tclose
relied\tVERB\trely
held\tVERB\thold
brings\tVERB\tbring
stops\tVERB\tstop
sees\tVERB\tsee
carve\tVERB\tcarve
flew\tVERB\tfly
visited\tVERB\tvisit
emailed\tVERB\temail
The\tDET\tthe
U.S.\tPROPN\tU.S.
u.s.\tPROPN\tUnited States
mathematicians\tNOUN\tmathematician
sets\tNOUN\tset
";

pub(super) const DEFAULT_PEOPLE_XML: &[u8] = br#"<?xml version="1.0" encoding="UTF-8"?>
<dictionary case_sensitive="true">
  <entry><token>John</token><token>Smith</token></entry>
  <entry><token>john</token><token>smith</token></entry>
  <entry><token>Smith</token></entry>
  <entry><token>Dr.</token><token>Who</token></entry>
  <entry><token>JOHN</token></entry>
</dictionary>
"#;

pub(super) const DEFAULT_PLACES_XML: &[u8] = r#"<?xml version="1.0" encoding="UTF-8"?>
<dictionary case_sensitive="false">
  <entry><token>New</token><token>York</token></entry>
  <entry><token>New</token><token>York</token><token>City</token></entry>
  <entry><token>York</token><token>City</token><token>Hall</token></entry>
  <entry><token>City</token></entry>
  <entry><token>San</token><token>Francisco</token></entry>
  <entry><token>São</token><token>Paulo</token></entry>
  <entry><token>İstanbul</token></entry>
  <entry><token>strasse</token></entry>
  <entry><token>K</token></entry>
  <entry><token>😀</token><token>face</token></entry>
</dictionary>
"#
.as_bytes();

pub(super) fn default_tag_tokens(tokens: &[&str]) -> Vec<(String, f64)> {
    tokens
        .iter()
        .enumerate()
        .map(|(idx, &tok)| {
            let lower = tok.to_ascii_lowercase();
            let tag = if tok.chars().all(|c| c.is_ascii_punctuation()) {
                "PUNCT"
            } else if matches!(lower.as_str(), "the" | "a" | "an" | "this" | "that" | "these" | "those") {
                "DET"
            } else if matches!(lower.as_str(), "is" | "are" | "was" | "were" | "be" | "been" | "have" | "has" | "had") {
                "AUX"
            } else if matches!(lower.as_str(), "in" | "on" | "at" | "to" | "from" | "with" | "by" | "for" | "of") {
                "ADP"
            } else if matches!(lower.as_str(), "and" | "or" | "but") {
                "CCONJ"
            } else if matches!(lower.as_str(), "he" | "she" | "it" | "they" | "we" | "i" | "you") {
                "PRON"
            } else if matches!(lower.as_str(), "went" | "visited" | "flew" | "works" | "emailed" | "ran" | "wrote")
                || lower.ends_with("ed")
                || lower.ends_with("ing")
            {
                "VERB"
            } else if matches!(lower.as_str(), "fast" | "quickly" | "yesterday" | "always" | "never")
                || lower.ends_with("ly")
            {
                "ADV"
            } else if idx > 0 && tok.chars().next().is_some_and(char::is_uppercase) {
                "PROPN"
            } else {
                "NOUN"
            };
            (tag.to_owned(), 1.0)
        })
        .collect()
}

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
