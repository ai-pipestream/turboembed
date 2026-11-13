//! Zero-copy stem-to-surface / prefix lexicon serving and context-aware query
//! spellcheck RPCs on `TurboVocabularyService` (issue #2341).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status, Streaming};
use turbo_stem::{Algorithm, SnowballAlgorithm, SnowballStemmer};
use turbo_suggest::{
    MappedFile, MappedSurfaceLexicon, QuerySpellchecker, SpellcheckConfig, SurfaceLexiconBuilder,
    lexicon::DEFAULT_TOP_K,
};

use super::{
    MAX_LANGUAGE, MAX_LEARN_BYTES, Stored, Vocabulary, artifact_id, blocking, check_text, fail, first, internal,
    invalid, not_found,
};
use crate::api::TURBO_E_CAPACITY;
use crate::kserve::Reply;
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::vocabulary::*;

const MAX_SUGGEST_LIMIT: usize = 256;
const DEFAULT_SUGGEST_LIMIT: usize = 10;
const MAX_QUERY_BYTES: usize = 16 << 10;

#[derive(Default)]
pub(crate) struct SuggestCache {
    lexicons: Mutex<BTreeMap<(String, String), Arc<MappedFile>>>,
    spellcheckers: Mutex<BTreeMap<(String, String, u32), Arc<QuerySpellchecker>>>,
}

impl SuggestCache {
    pub(crate) fn evict(&self, id: &str) {
        self.lexicons.lock().unwrap_or_else(|p| p.into_inner()).retain(|(k, _), _| k != id);
        self.spellcheckers.lock().unwrap_or_else(|p| p.into_inner()).retain(|(k, _, _), _| k != id);
    }

    fn mapped(&self, store: &super::Store, id: &str) -> Result<Arc<MappedFile>, Status> {
        let (desc, path) = store.suggest_file(id).ok_or_else(|| not_found(id))?;
        let key = (desc.artifact_id, desc.sha256);
        if let Some(hit) = self.lexicons.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
            return Ok(Arc::clone(hit));
        }
        // SAFETY: Store writes `<id>.slex` via a synced temporary file and
        // atomically renames it into place before publishing `<id>.meta`; no
        // code mutates or truncates a stored `.slex` file in place.
        let mapped =
            unsafe { MappedFile::open(&path) }.map_err(|e| internal(format!("mmap {}: {e}", path.display())))?;
        if mapped.bytes().len() as u64 != desc.byte_size || turbo_maxent::sha256(mapped.bytes()).to_hex() != key.1 {
            return Err(internal(format!("{}: stored bytes do not match descriptor", path.display())));
        }
        MappedSurfaceLexicon::from_bytes(mapped.bytes())
            .map_err(|e| internal(format!("{}: invalid lexicon image: {e}", path.display())))?;
        let arc = Arc::new(mapped);
        self.lexicons.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    fn spellchecker(
        &self,
        store: &super::Store,
        id: &str,
        max_edit_distance: u32,
    ) -> Result<Arc<QuerySpellchecker>, Status> {
        let dist = match max_edit_distance {
            0 => 2,
            1..=2 => max_edit_distance,
            other => return Err(invalid(format!("max_edit_distance {other} must be 0, 1, or 2"))),
        };
        let Some(Stored::Vocabulary(desc, bytes)) = store.read(id).map_err(internal)? else {
            return Err(not_found(id));
        };
        let key = (desc.artifact_id, desc.sha256, dist);
        if let Some(hit) = self.spellcheckers.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
            return Ok(Arc::clone(hit));
        }
        let cfg = SpellcheckConfig { max_edit_distance: dist as usize, ..SpellcheckConfig::default() };
        let checker = QuerySpellchecker::from_vocabulary_tsv(&bytes, cfg).map_err(invalid)?;
        let arc = Arc::new(checker);
        self.spellcheckers.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&arc));
        Ok(arc)
    }
}

fn stemmer_for_language(tag: &str) -> Result<Algorithm, Status> {
    let primary = tag.trim().split(['-', '_']).next().unwrap_or_default().to_ascii_lowercase();
    let snow = |alg| Ok(Algorithm::Snowball(SnowballStemmer::new(alg)));
    match primary.as_str() {
        "" | "en" | "eng" => snow(SnowballAlgorithm::English),
        "porter" => Ok(Algorithm::Porter),
        "ar" | "ara" => snow(SnowballAlgorithm::Arabic),
        "ca" | "cat" => snow(SnowballAlgorithm::Catalan),
        "da" | "dan" => snow(SnowballAlgorithm::Danish),
        "nl" | "nld" => snow(SnowballAlgorithm::Dutch),
        "fi" | "fin" => snow(SnowballAlgorithm::Finnish),
        "fr" | "fra" => snow(SnowballAlgorithm::French),
        "de" | "deu" => snow(SnowballAlgorithm::German),
        "el" | "ell" => snow(SnowballAlgorithm::Greek),
        "hu" | "hun" => snow(SnowballAlgorithm::Hungarian),
        "id" | "ind" => snow(SnowballAlgorithm::Indonesian),
        "ga" | "gle" => snow(SnowballAlgorithm::Irish),
        "it" | "ita" => snow(SnowballAlgorithm::Italian),
        "no" | "nob" | "nno" | "nb" | "nn" => snow(SnowballAlgorithm::Norwegian),
        "pt" | "por" => snow(SnowballAlgorithm::Portuguese),
        "ro" | "ron" => snow(SnowballAlgorithm::Romanian),
        "ru" | "rus" => snow(SnowballAlgorithm::Russian),
        "es" | "spa" => snow(SnowballAlgorithm::Spanish),
        "sv" | "swe" => snow(SnowballAlgorithm::Swedish),
        "tr" | "tur" => snow(SnowballAlgorithm::Turkish),
        _ => Err(invalid(format!("unsupported stemmer language tag {tag:?}"))),
    }
}

fn wire_offset_mode(raw: i32) -> Result<(OffsetMode, SpellcheckOffsetMode), Status> {
    match SpellcheckOffsetMode::try_from(raw) {
        Ok(SpellcheckOffsetMode::Utf8) => Ok((OffsetMode::Utf8, SpellcheckOffsetMode::Utf8)),
        Ok(SpellcheckOffsetMode::Utf16) => Ok((OffsetMode::Utf16, SpellcheckOffsetMode::Utf16)),
        Ok(SpellcheckOffsetMode::CodePoint) => Ok((OffsetMode::CodePoint, SpellcheckOffsetMode::CodePoint)),
        Err(_) => Err(invalid(format!("unknown SpellcheckOffsetMode {raw}"))),
    }
}

enum Feed {
    Observation(SuggestObservation),
    Document(String),
}

pub(super) async fn build_suggest_lexicon(
    svc: &Vocabulary,
    req: Request<Streaming<BuildSuggestLexiconRequest>>,
) -> Reply<SuggestLexiconArtifactDescriptor> {
    let store = svc.writer("BuildSuggestLexicon")?;
    let mut s = req.into_inner();
    let h = first(&mut s, |m: BuildSuggestLexiconRequest| match m.frame {
        Some(build_suggest_lexicon_request::Frame::Header(h)) => Some(h),
        _ => None,
    })
    .await?;
    check_text("language", &h.language, MAX_LANGUAGE)?;
    let top_p = if h.top_k_prefix == 0 { DEFAULT_TOP_K } else { h.top_k_prefix.min(MAX_SUGGEST_LIMIT as u32) as usize };
    let top_s = if h.top_k_stem == 0 { DEFAULT_TOP_K } else { h.top_k_stem.min(MAX_SUGGEST_LIMIT as u32) as usize };
    let count_df = h.count_doc_freq;
    let lang = h.language.clone();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Feed>(32);
    let worker = tokio::task::spawn_blocking(move || -> Result<SurfaceLexiconBuilder, Status> {
        let mut builder = SurfaceLexiconBuilder::new().with_top_k_prefix(top_p).with_top_k_stem(top_s);
        let mut stemmer: Option<Algorithm> = None;
        while let Some(item) = rx.blocking_recv() {
            match item {
                Feed::Observation(obs) => builder.observe(&obs.surface, &obs.stem, obs.count),
                Feed::Document(text) => {
                    let alg = match &stemmer {
                        Some(a) => *a,
                        None => *stemmer.insert(stemmer_for_language(&lang)?),
                    };
                    builder.observe_document(&text, &alg, count_df);
                }
            }
        }
        Ok(builder)
    });

    let mut read_bytes = 0u64;
    while let Some(m) = s.message().await? {
        let feed = match m.frame {
            Some(build_suggest_lexicon_request::Frame::Observation(obs)) => {
                read_bytes += (obs.surface.len() + obs.stem.len()) as u64;
                Feed::Observation(obs)
            }
            Some(build_suggest_lexicon_request::Frame::Document(doc)) => {
                read_bytes += doc.text.len() as u64;
                Feed::Document(doc.text)
            }
            _ => return Err(invalid("every frame after the header must be an observation or a document")),
        };
        if read_bytes > MAX_LEARN_BYTES {
            return Err(fail(TURBO_E_CAPACITY, format!("past max_learn_bytes {MAX_LEARN_BYTES}")));
        }
        if tx.send(feed).await.is_err() {
            break;
        }
    }
    drop(tx);
    let builder = worker.await.map_err(|e| internal(format!("suggest builder task failed: {e}")))??;
    if builder.is_empty() {
        return Err(invalid("no non-empty surface observations to build a suggest lexicon from"));
    }

    let descriptor = blocking(move || {
        let bytes = builder.build_bytes().map_err(|e| fail(TURBO_E_CAPACITY, e.to_string()))?;
        let image = turbo_suggest::ImageBuf::copy_from(&bytes);
        let lex = MappedSurfaceLexicon::from_bytes(image.bytes()).map_err(|e| internal(e.to_string()))?;
        let (entry_count, stem_count) = (lex.entry_count() as u64, lex.stem_count() as u64);
        let sha256 = turbo_maxent::sha256(&bytes).to_hex();
        let desc = SuggestLexiconArtifactDescriptor {
            artifact_id: artifact_id("suggest", &[&h.language, &sha256]),
            language: h.language,
            entry_count,
            stem_count,
            byte_size: bytes.len() as u64,
            sha256,
        };
        store.put_suggest(desc, &bytes).map_err(internal)
    })
    .await?;
    Ok(Response::new(descriptor))
}

pub(super) async fn suggest_prefix(
    svc: &Vocabulary,
    req: Request<SuggestPrefixRequest>,
) -> Reply<SuggestPrefixResponse> {
    let r = req.into_inner();
    let store = svc.store.clone().ok_or_else(|| not_found(&r.artifact_id))?;
    let limit = if r.limit == 0 { DEFAULT_SUGGEST_LIMIT } else { (r.limit as usize).min(MAX_SUGGEST_LIMIT) };
    let cache = Arc::clone(&svc.suggest);
    let completions = blocking(move || {
        let mapped = cache.mapped(&store, &r.artifact_id)?;
        let lex = MappedSurfaceLexicon::from_bytes(mapped.bytes()).map_err(|e| internal(e.to_string()))?;
        Ok(lex
            .suggest_prefix(&r.prefix, limit)
            .into_iter()
            .map(|c| SurfaceCompletionItem { surface: c.surface.to_owned(), stem: c.stem.to_owned(), count: c.count })
            .collect())
    })
    .await?;
    Ok(Response::new(SuggestPrefixResponse { completions }))
}

pub(super) async fn lookup_stem_surfaces(
    svc: &Vocabulary,
    req: Request<LookupStemSurfacesRequest>,
) -> Reply<LookupStemSurfacesResponse> {
    let r = req.into_inner();
    let store = svc.store.clone().ok_or_else(|| not_found(&r.artifact_id))?;
    let limit = if r.limit == 0 { DEFAULT_SUGGEST_LIMIT } else { (r.limit as usize).min(MAX_SUGGEST_LIMIT) };
    let cache = Arc::clone(&svc.suggest);
    let surfaces = blocking(move || {
        let mapped = cache.mapped(&store, &r.artifact_id)?;
        let lex = MappedSurfaceLexicon::from_bytes(mapped.bytes()).map_err(|e| internal(e.to_string()))?;
        Ok(lex
            .stem_surfaces(&r.stem, limit)
            .into_iter()
            .map(|s| StemSurfaceItem { surface: s.surface.to_owned(), count: s.count })
            .collect())
    })
    .await?;
    Ok(Response::new(LookupStemSurfacesResponse { surfaces }))
}

pub(super) async fn spellcheck_query(
    svc: &Vocabulary,
    req: Request<SpellcheckQueryRequest>,
) -> Reply<SpellcheckQueryResponse> {
    let r = req.into_inner();
    if r.query.len() > MAX_QUERY_BYTES {
        return Err(fail(TURBO_E_CAPACITY, format!("query exceeds {MAX_QUERY_BYTES} bytes")));
    }
    let (mode, proto_mode) = wire_offset_mode(r.offsets)?;
    let offsetter = Offsetter::new(&r.query, mode);
    let mut protected = Vec::with_capacity(r.protected_spans.len());
    for sp in &r.protected_spans {
        let u8_span = offsetter
            .from_wire(&r.query, "protected_spans", sp.start, sp.end)
            .map_err(|f| crate::status::to_status(&f, &[]))?;
        protected.push(u8_span);
    }
    let store = svc.store.clone().ok_or_else(|| not_found(&r.vocabulary_artifact_id))?;
    let cache = Arc::clone(&svc.suggest);
    let out = blocking(move || {
        let checker = cache.spellchecker(&store, &r.vocabulary_artifact_id, r.max_edit_distance)?;
        let res = checker.check_query(&r.query, &protected);
        let edits = res
            .edits
            .into_iter()
            .map(|e| {
                let (start, end) = match mode {
                    OffsetMode::Utf8 => (e.utf8_span.start, e.utf8_span.end),
                    OffsetMode::Utf16 => (e.utf16_span.start.0, e.utf16_span.end.0),
                    OffsetMode::CodePoint => (e.code_point_span.start.0, e.code_point_span.end.0),
                };
                SpellcheckEditItem {
                    span: Some(SpellcheckSpan { start, end }),
                    original: e.original,
                    replacement: e.replacement,
                    edit_distance: e.edit_distance,
                }
            })
            .collect();
        Ok(SpellcheckQueryResponse {
            corrected_query: res.corrected_query,
            did_change: res.did_change,
            confidence: res.confidence,
            offsets: proto_mode as i32,
            edits,
        })
    })
    .await?;
    Ok(Response::new(out))
}
