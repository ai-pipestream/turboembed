//! Resolving stored dictionary and learned vocabulary artifacts for
//! `LoadBundle`, `AnalyzeStream`, and `AnalyzeDocumentProgressive` (#2013).
//!
//! Each [`VocabularyArtifactRef`] names an `artifact_id` and its 64-digit
//! lowercase hexadecimal `sha256`. References are validated with the same
//! rules as [`crate::pii_policy::resolve`], resolved against the server's
//! vocabulary [`Store`], and cached by `(artifact_id, sha256)` so repeated
//! requests reuse the parsed stopword list and compiled glossary automaton
//! without rereading disk. Errors never quote dictionary terms or file paths.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use turbo_analysis::{Annotator, Error, Fnv1a, ModelRef, Refs, View};
use turbo_analysis_layers::{Language, Tokens};
use turbo_glossary::{
    Case, DelimitedReader, Delimiter, FirstRecord, GlossaryAnnotator, GlossaryMatcher, ImageBuf, MappedGlossaryMatcher,
    TbxReader, TextMatcher,
};
use turbo_glossary_layers::{EntryIx, Glossary};
use turbo_keywords::{Stopwords as KeywordStopwords, YakeAnnotator};
use turbo_stopword::{StopwordAnnotator, StopwordLanguage, Stopwords};

use super::formats;
use super::store::{Store, Stored};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::pii_policy::{MAX_REF_ID_BYTES, ResolveError};
use crate::proto::result::{VocabularyArtifactRef, VocabularyBinding};

const LEARNED_VOCAB_REVISION: &str = "turbo-vocabulary-1";

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

fn ref_failure(field: &str, artifact_id: &str, err: ResolveError) -> Failure {
    bad(format!("{field} {artifact_id:?}: {}", err.reason()))
}

fn check_artifact_ref(field: &str, r: &VocabularyArtifactRef) -> Result<(), Failure> {
    let id = r.artifact_id.as_str();
    if id.is_empty() {
        return Err(bad(format!("{field}.artifact_id is required")));
    }
    if id.len() > MAX_REF_ID_BYTES {
        return Err(bad(format!("{field}.artifact_id: is longer than 128 bytes")));
    }
    if id.contains('/') || id.contains('\\') || id.contains("..") || id.starts_with('.') {
        return Err(bad(format!("{field}.artifact_id: looks like a file path; name a registered reference")));
    }
    if !id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b':')) {
        return Err(bad(format!(
            "{field}.artifact_id: has a character outside letters, digits, '.', '_', '-' and ':'"
        )));
    }
    let sha = r.sha256.as_str();
    if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(bad(format!("{field}.sha256: is not 64 lowercase hexadecimal digits")));
    }
    Ok(())
}

fn stopword_language(tag: &str) -> StopwordLanguage {
    let primary = tag.trim().split(['-', '_']).next().unwrap_or_default().to_ascii_lowercase();
    StopwordLanguage::from_code(&primary).unwrap_or(StopwordLanguage::English)
}

fn fold_stopword(word: &str) -> String {
    word.trim().to_lowercase()
}

fn sorted_stopwords(words: impl IntoIterator<Item = String>) -> Result<Arc<[Box<str>]>, ResolveError> {
    let mut list: Vec<Box<str>> =
        words.into_iter().map(|w| fold_stopword(&w)).filter(|w| !w.is_empty()).map(String::into_boxed_str).collect();
    list.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    list.dedup();
    if list.is_empty() {
        return Err(ResolveError::Unreadable);
    }
    Ok(Arc::from(list))
}

/// Parses a `# turbo-vocabulary 1` TSV artifact into `(term, kind)` rows.
fn parse_learned_vocabulary(bytes: &[u8]) -> Result<Vec<(&str, &str)>, ResolveError> {
    let text = std::str::from_utf8(bytes).map_err(|_| ResolveError::Unreadable)?;
    let mut rows = Vec::new();
    let mut saw_header = false;
    for line in text.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !saw_header {
            if !line.starts_with("term\tkind\t") {
                return Err(ResolveError::Unreadable);
            }
            saw_header = true;
            continue;
        }
        let mut cols = line.split('\t');
        let term = cols.next().map(str::trim).filter(|s| !s.is_empty()).ok_or(ResolveError::Unreadable)?;
        let kind = cols.next().map(str::trim).filter(|s| !s.is_empty()).ok_or(ResolveError::Unreadable)?;
        rows.push((term, kind));
    }
    if !saw_header || rows.is_empty() {
        return Err(ResolveError::Unreadable);
    }
    Ok(rows)
}

fn hash_with_model(base: u64, model: &ModelRef) -> u64 {
    let mut h = Fnv1a::new();
    h.u64(base).str(&model.id).str(&model.revision).str(&model.sha256);
    h.finish()
}

/// A resolved stopword artifact ready to build [`BoundStopwordAnnotator`]
/// and [`BoundYakeAnnotator`].
#[derive(Clone)]
pub(crate) struct ResolvedStopwords {
    pub(crate) reference: VocabularyArtifactRef,
    pub(crate) model: ModelRef,
    stopword_annotator: BoundStopwordAnnotator,
    yake_annotator: BoundYakeAnnotator,
}

impl std::fmt::Debug for ResolvedStopwords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedStopwords").field("model", &self.model).finish()
    }
}

impl ResolvedStopwords {
    fn new(language: StopwordLanguage, words: Arc<[Box<str>]>, model: ModelRef) -> Self {
        let reference = VocabularyArtifactRef { artifact_id: model.id.clone(), sha256: model.sha256.clone() };
        let kw_words = KeywordStopwords::List(words);
        let sw_config = hash_with_model(kw_words.config_hash() ^ (language as u64), &model);
        let stopword_annotator =
            BoundStopwordAnnotator { language, words: kw_words.clone(), model: model.clone(), config: sw_config };
        let kw_inner = YakeAnnotator::new(kw_words);
        let kw_config = hash_with_model(kw_inner.config_hash(), &model);
        let yake_annotator = BoundYakeAnnotator { inner: kw_inner, model: model.clone(), config: kw_config };
        Self { reference, model, stopword_annotator, yake_annotator }
    }

    pub(crate) fn stopword_annotator(&self) -> BoundStopwordAnnotator {
        self.stopword_annotator.clone()
    }

    pub(crate) fn yake_annotator(&self) -> BoundYakeAnnotator {
        self.yake_annotator.clone()
    }
}

/// A resolved glossary artifact ready to build [`BoundGlossaryAnnotator`].
#[derive(Clone)]
pub(crate) struct ResolvedGlossary {
    pub(crate) reference: VocabularyArtifactRef,
    pub(crate) model: ModelRef,
    annotator: BoundGlossaryAnnotator,
}

impl std::fmt::Debug for ResolvedGlossary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedGlossary").field("model", &self.model).finish()
    }
}

impl ResolvedGlossary {
    fn new(glossary: Arc<Glossary>, case: Case, model: ModelRef) -> Result<Self, ResolveError> {
        let reference = VocabularyArtifactRef { artifact_id: model.id.clone(), sha256: model.sha256.clone() };
        let matcher = GlossaryMatcher::Text(TextMatcher::new(&glossary, case).map_err(|_| ResolveError::Unreadable)?);
        let inner = Arc::new(GlossaryAnnotator::new(glossary, matcher));
        let config = hash_with_model(inner.config_hash(), &model);
        let annotator = BoundGlossaryAnnotator { inner, model: model.clone(), config };
        Ok(Self { reference, model, annotator })
    }

    pub(crate) fn annotator(&self) -> BoundGlossaryAnnotator {
        self.annotator.clone()
    }
}

/// Resolved vocabulary artifacts bound to a bundle or request.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResolvedVocabulary {
    pub(crate) stopwords: Option<Arc<ResolvedStopwords>>,
    pub(crate) glossary: Option<Arc<ResolvedGlossary>>,
}

impl ResolvedVocabulary {
    pub(crate) fn is_empty(&self) -> bool {
        self.stopwords.is_none() && self.glossary.is_none()
    }

    pub(crate) fn to_proto(&self) -> Option<VocabularyBinding> {
        (!self.is_empty()).then(|| VocabularyBinding {
            stopwords: self.stopwords.as_ref().map(|s| s.reference.clone()),
            glossary: self.glossary.as_ref().map(|g| g.reference.clone()),
        })
    }

    /// Complete effective binding identity for `AnalyzeStream` batch grouping:
    /// artifact ID, SHA-256, and compiled config hash for stopwords and glossary.
    pub(crate) fn batch_key(&self) -> (String, String, u64, String, String, u64) {
        let (s_id, s_sha, s_cfg) = self
            .stopwords
            .as_ref()
            .map(|s| (s.model.id.clone(), s.model.sha256.clone(), s.stopword_annotator.config_hash()))
            .unwrap_or_default();
        let (g_id, g_sha, g_cfg) = self
            .glossary
            .as_ref()
            .map(|g| (g.model.id.clone(), g.model.sha256.clone(), g.annotator.config_hash()))
            .unwrap_or_default();
        (s_id, s_sha, s_cfg, g_id, g_sha, g_cfg)
    }
}

/// Stopword annotator bound to a vocabulary store artifact, recording the
/// artifact's [`ModelRef`] and digest in the layer's provenance.
#[derive(Clone)]
pub(crate) struct BoundStopwordAnnotator {
    language: StopwordLanguage,
    words: KeywordStopwords,
    model: ModelRef,
    config: u64,
}

impl Annotator for BoundStopwordAnnotator {
    const ID: &'static str = StopwordAnnotator::ID;
    const VERSION: &'static str = StopwordAnnotator::VERSION;
    type Reads = (View, Tokens, Option<Language>);
    type Writes = (Stopwords,);

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }
    fn config_hash(&self) -> u64 {
        self.config
    }
    fn annotate(&self, (view, tokens, _detected): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the tokens are in another view".into() });
        }
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        if starts.len() != ends.len() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "mismatched token span columns".into() });
        }
        let mut stop = Vec::with_capacity(starts.len());
        for (&s, &e) in starts.iter().zip(ends) {
            let (s, e) = (s as usize, e as usize);
            if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                return Err(Error::Annotator { annotator: Self::ID, detail: "invalid token span".into() });
            }
            stop.push(self.words.contains(&fold_stopword(&text[s..e])));
        }
        Ok((Stopwords { language: self.language, stop },))
    }
}

/// YAKE keyword annotator using a bound stopword artifact, recording the
/// artifact's [`ModelRef`] and digest in the layer's provenance.
#[derive(Clone)]
pub(crate) struct BoundYakeAnnotator {
    inner: YakeAnnotator,
    model: ModelRef,
    config: u64,
}

impl Annotator for BoundYakeAnnotator {
    const ID: &'static str = YakeAnnotator::ID;
    const VERSION: &'static str = YakeAnnotator::VERSION;
    type Reads = <YakeAnnotator as Annotator>::Reads;
    type Writes = <YakeAnnotator as Annotator>::Writes;

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }
    fn config_hash(&self) -> u64 {
        self.config
    }
    fn annotate(&self, reads: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.inner.annotate(reads)
    }
}

/// Glossary annotator bound to a vocabulary store artifact, recording the
/// artifact's [`ModelRef`] and digest in the layer's provenance.
#[derive(Clone)]
pub(crate) struct BoundGlossaryAnnotator {
    inner: Arc<GlossaryAnnotator>,
    model: ModelRef,
    config: u64,
}

impl Annotator for BoundGlossaryAnnotator {
    const ID: &'static str = GlossaryAnnotator::ID;
    const VERSION: &'static str = GlossaryAnnotator::VERSION;
    type Reads = <GlossaryAnnotator as Annotator>::Reads;
    type Writes = <GlossaryAnnotator as Annotator>::Writes;
    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }
    fn config_hash(&self) -> u64 {
        self.config
    }
    fn annotate(&self, reads: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.inner.annotate(reads)
    }
}

/// Digest-pinned resolver and cache over the server's vocabulary [`Store`].
#[derive(Default)]
pub(crate) struct VocabularyResolver {
    store: Option<Arc<Store>>,
    stopwords: Mutex<HashMap<(String, String), Arc<ResolvedStopwords>>>,
    glossaries: Mutex<HashMap<(String, String), Arc<ResolvedGlossary>>>,
}

impl VocabularyResolver {
    pub(crate) fn new(store: Option<Arc<Store>>) -> Self {
        Self { store, stopwords: Mutex::new(HashMap::new()), glossaries: Mutex::new(HashMap::new()) }
    }

    pub(crate) fn resolve_binding(&self, binding: &VocabularyBinding) -> Result<ResolvedVocabulary, Failure> {
        self.resolve_merged(&ResolvedVocabulary::default(), Some(binding))
    }

    pub(crate) fn resolve_merged(
        &self,
        base: &ResolvedVocabulary,
        override_binding: Option<&VocabularyBinding>,
    ) -> Result<ResolvedVocabulary, Failure> {
        let Some(b) = override_binding else { return Ok(base.clone()) };
        let stopwords = match &b.stopwords {
            Some(r) => Some(self.resolve_stopwords("vocabulary.stopwords", r)?),
            None => base.stopwords.clone(),
        };
        let glossary = match &b.glossary {
            Some(r) => Some(self.resolve_glossary("vocabulary.glossary", r)?),
            None => base.glossary.clone(),
        };
        Ok(ResolvedVocabulary { stopwords, glossary })
    }

    fn read_checked(&self, field: &str, r: &VocabularyArtifactRef) -> Result<(&Arc<Store>, Stored), Failure> {
        check_artifact_ref(field, r)?;
        let store = self.store.as_ref().ok_or_else(|| ref_failure(field, &r.artifact_id, ResolveError::NoRegistry))?;
        match store.check_digest(&r.artifact_id, &r.sha256) {
            None => {
                let key = (r.artifact_id.clone(), r.sha256.clone());
                self.stopwords.lock().unwrap_or_else(|p| p.into_inner()).remove(&key);
                self.glossaries.lock().unwrap_or_else(|p| p.into_inner()).remove(&key);
                return Err(ref_failure(field, &r.artifact_id, ResolveError::NotRegistered));
            }
            Some(false) => return Err(ref_failure(field, &r.artifact_id, ResolveError::DigestMismatch)),
            Some(true) => {}
        }
        let stored = store
            .read(&r.artifact_id)
            .map_err(|_| ref_failure(field, &r.artifact_id, ResolveError::Unreadable))?
            .ok_or_else(|| ref_failure(field, &r.artifact_id, ResolveError::NotRegistered))?;
        Ok((store, stored))
    }

    pub(crate) fn resolve_stopwords(
        &self,
        field: &str,
        r: &VocabularyArtifactRef,
    ) -> Result<Arc<ResolvedStopwords>, Failure> {
        let (_, stored) = self.read_checked(field, r)?;
        let key = (r.artifact_id.clone(), r.sha256.clone());
        if let Some(hit) = self.stopwords.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() {
            return Ok(hit);
        }
        let arc = Arc::new(build_stopwords(&stored, &r.sha256).map_err(|e| ref_failure(field, &r.artifact_id, e))?);
        self.stopwords.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    pub(crate) fn resolve_glossary(
        &self,
        field: &str,
        r: &VocabularyArtifactRef,
    ) -> Result<Arc<ResolvedGlossary>, Failure> {
        let (store, stored) = self.read_checked(field, r)?;
        let key = (r.artifact_id.clone(), r.sha256.clone());
        if let Some(hit) = self.glossaries.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() {
            return Ok(hit);
        }
        let has_hdr = store.dictionary(&r.artifact_id).is_some_and(|d| d.has_header_row);
        let arc =
            Arc::new(build_glossary(&stored, &r.sha256, has_hdr).map_err(|e| ref_failure(field, &r.artifact_id, e))?);
        self.glossaries.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&arc));
        Ok(arc)
    }
}

fn build_stopwords(stored: &Stored, expected_sha256: &str) -> Result<ResolvedStopwords, ResolveError> {
    match stored {
        Stored::Dictionary(d, bytes) => {
            if d.sha256 != expected_sha256 {
                return Err(ResolveError::DigestMismatch);
            }
            if d.format_id != "plain-word-list" {
                return Err(ResolveError::WrongFormat { format: d.format_id.clone() });
            }
            let words = sorted_stopwords(formats::word_list(bytes).map_err(|_| ResolveError::Unreadable)?)?;
            let model = ModelRef { id: d.artifact_id.clone(), revision: d.format_id.clone(), sha256: d.sha256.clone() };
            Ok(ResolvedStopwords::new(stopword_language(&d.language), words, model))
        }
        Stored::Vocabulary(v, bytes) => {
            if v.sha256 != expected_sha256 {
                return Err(ResolveError::DigestMismatch);
            }
            let rows = parse_learned_vocabulary(bytes)?;
            let words = sorted_stopwords(rows.into_iter().filter(|(_, k)| *k == "term").map(|(t, _)| t.to_owned()))?;
            let rev = LEARNED_VOCAB_REVISION.to_owned();
            let model = ModelRef { id: v.artifact_id.clone(), revision: rev, sha256: v.sha256.clone() };
            Ok(ResolvedStopwords::new(stopword_language(&v.language), words, model))
        }
    }
}

fn build_glossary(
    stored: &Stored,
    expected_sha256: &str,
    has_header_row: bool,
) -> Result<ResolvedGlossary, ResolveError> {
    match stored {
        Stored::Dictionary(d, bytes) => {
            if d.sha256 != expected_sha256 {
                return Err(ResolveError::DigestMismatch);
            }
            let first = if has_header_row { FirstRecord::Header } else { FirstRecord::Data };
            let (glossary, case) = match d.format_id.as_str() {
                "glossary-csv" => {
                    let file = DelimitedReader::new(Delimiter::COMMA, first)
                        .read(bytes)
                        .map_err(|_| ResolveError::Unreadable)?;
                    (Arc::new(file.glossary), Case::Sensitive)
                }
                "glossary-tbx" => {
                    let reader = TbxReader::new(&d.language).ok_or(ResolveError::Unreadable)?;
                    let file = reader.read(bytes).map_err(|_| ResolveError::Unreadable)?;
                    (Arc::new(file.glossary), Case::Sensitive)
                }
                "glossary-mapped" => {
                    let image = ImageBuf::copy_from(bytes);
                    let m = MappedGlossaryMatcher::from_bytes(image.bytes()).map_err(|_| ResolveError::Unreadable)?;
                    let mut g = Glossary::new();
                    for i in 0..m.entries() {
                        let ix = EntryIx(i as u32);
                        let (id, term) =
                            (m.id(ix).ok_or(ResolveError::Unreadable)?, m.term(ix).ok_or(ResolveError::Unreadable)?);
                        g.push(id, term).map_err(|_| ResolveError::Unreadable)?;
                    }
                    (Arc::new(g), m.case())
                }
                _ => return Err(ResolveError::WrongFormat { format: d.format_id.clone() }),
            };
            if glossary.is_empty() {
                return Err(ResolveError::Unreadable);
            }
            let model = ModelRef { id: d.artifact_id.clone(), revision: d.format_id.clone(), sha256: d.sha256.clone() };
            ResolvedGlossary::new(glossary, case, model)
        }
        Stored::Vocabulary(v, bytes) => {
            if v.sha256 != expected_sha256 {
                return Err(ResolveError::DigestMismatch);
            }
            let mut g = Glossary::new();
            for (term, _) in parse_learned_vocabulary(bytes)? {
                g.push(term, term).map_err(|_| ResolveError::Unreadable)?;
            }
            if g.is_empty() {
                return Err(ResolveError::Unreadable);
            }
            let rev = LEARNED_VOCAB_REVISION.to_owned();
            let model = ModelRef { id: v.artifact_id.clone(), revision: rev, sha256: v.sha256.clone() };
            ResolvedGlossary::new(Arc::new(g), Case::Sensitive, model)
        }
    }
}
