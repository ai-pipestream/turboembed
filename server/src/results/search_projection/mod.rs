//! Schema-driven Search Index Envelope Projection (`#2345`,
//! `proto/turbo_search_projection.proto`).
//!
//! Projects a `turbo_analysis::Document` (or imported binary `DocumentGraph`)
//! in a single pass into ready-to-index chunks, aligned `Sentences` spans,
//! position-length lexical token lattices (`#2340`), first-class NLP typed
//! columns, syntactic negation and modality entity polarity buckets,
//! near-duplicate sketches, and WAL replay provenance digests.
//!
//! Composes with `#1219` (`TypedDocumentResult`) and enforces `#1904`/`#1905`
//! PII disclosure and layer assessment status invariants without duplicating
//! `#1901`..`#1904` PII wire structures.

pub mod columns;
pub mod entities;
pub mod validate;

use std::any::TypeId;
use std::collections::HashMap;

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Execution, Ix, Key, Layer, Provenance, Span, View, ViewId};
use turbo_analysis_layers::{Boilerplate, Chunks, Lemmas, Sentences, Stems, TokenGraph, Tokens};
use turbo_group::Sketches;

use entities::{boilerplate_kind_str, cut_reason_str, token_arc_kind_str};

use super::disclosure::{Disclosure, PiiOutputPolicy};
use super::document::codecs;
use super::model_to_proto;
use super::typed::identity::{check_binding, input_identity, result_identity};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{
    ExecutionInfo, InputBinding, ModelProvenance, ProjectSearchDocumentRequest, SearchChunkRow,
    SearchDocumentProjection, SearchLexicalToken, SearchReplayProvenance, SearchSentenceRow, SearchSketchRow,
    ViewAlignmentSummary,
};
use crate::results::{MAX_GRAPH_BYTES, Wire, over};

pub use validate::validate_search_projection;

pub(crate) fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProvKey {
    annotator: String,
    version: String,
    config_hash: u64,
    model_id: String,
    model_rev: String,
    model_sha: String,
}

pub(crate) struct SearchProjCx<'a> {
    pub(crate) doc: &'a Document,
    pub(crate) target_view: ViewId,
    pub(crate) target_text: &'a str,
    pub(crate) offsetter: Offsetter,
    pub(crate) models: Vec<ModelProvenance>,
    model_map: HashMap<ProvKey, String>,
}

impl<'a> SearchProjCx<'a> {
    fn new(doc: &'a Document, target_view: ViewId, target_text: &'a str, offsets: OffsetMode) -> Self {
        Self {
            doc,
            target_view,
            target_text,
            offsetter: Offsetter::new(target_text, offsets),
            models: Vec::new(),
            model_map: HashMap::new(),
        }
    }

    pub(crate) fn register_provenance(&mut self, prov: &Provenance) -> String {
        let (model_id, model_rev, model_sha) = match &prov.model {
            Some(m) => (m.id.clone(), m.revision.clone(), m.sha256.clone()),
            None => (String::new(), String::new(), String::new()),
        };
        let key = ProvKey {
            annotator: prov.annotator.to_owned(),
            version: prov.version.to_owned(),
            config_hash: prov.config,
            model_id,
            model_rev,
            model_sha,
        };
        if let Some(existing) = self.model_map.get(&key) {
            return existing.clone();
        }
        let model_ref = format!("m{}", self.models.len());
        let execution = prov.execution.as_ref().map(|exec: &Execution| ExecutionInfo {
            backend: exec.backend.clone(),
            device: exec.device.clone(),
            precision: exec.precision.clone(),
        });
        self.models.push(ModelProvenance {
            model_ref: model_ref.clone(),
            annotator: prov.annotator.to_owned(),
            version: prov.version.to_owned(),
            config_hash: prov.config,
            bundle: prov.model.as_ref().map(model_to_proto),
            execution,
        });
        self.model_map.insert(key, model_ref.clone());
        model_ref
    }

    pub(crate) fn map_to_wire(&self, layer_view: ViewId, span: Span) -> Result<(u32, u32), Failure> {
        let mapped = if layer_view == self.target_view {
            span
        } else {
            self.doc.map_span(layer_view, self.target_view, span).map_err(|e| {
                bad(format!(
                    "cannot map span [{}..{}) from view {} to target view {}: {e}",
                    span.start, span.end, layer_view.0, self.target_view.0
                ))
            })?
        };
        Ok(self.offsetter.to_wire(self.target_text, mapped))
    }
}

fn sketch_row<L: turbo_analysis::Layer>(sketches: &Sketches<L>, idx: usize) -> Option<SearchSketchRow> {
    if idx >= sketches.len() {
        return None;
    }
    let exact = sketches.exact[idx];
    let minhash = sketches.row(Ix::new(idx as u32)).to_vec();
    Some(SearchSketchRow {
        shingle: u32::from(sketches.shingle),
        words: sketches.words[idx],
        exact_lo: exact as u64,
        exact_hi: (exact >> 64) as u64,
        shape_hash: sketches.shape[idx],
        simhash: sketches.simhash[idx],
        minhash,
    })
}

/// Projects `doc` into a [`SearchDocumentProjection`] on `target_view` in
/// `offsets` under [`Disclosure::FULL`] (for in-process callers holding their
/// own text).
pub fn document_to_search_projection(
    doc_id: &str,
    doc: &Document,
    target_view: ViewId,
    offsets: OffsetMode,
) -> Result<SearchDocumentProjection, Failure> {
    document_to_search_projection_bound(doc_id, doc, target_view, offsets, None, &[], &Disclosure::FULL)
}

/// Projects `doc` under explicit `disclosure`, `binding`, and optional
/// 32-byte `analyzer_config_digest` (`#2345`).
pub fn document_to_search_projection_bound(
    doc_id: &str,
    doc: &Document,
    target_view: ViewId,
    offsets: OffsetMode,
    binding: Option<&InputBinding>,
    analyzer_config_digest: &[u8],
    disclosure: &Disclosure,
) -> Result<SearchDocumentProjection, Failure> {
    check_binding(binding, "binding")?;
    if !analyzer_config_digest.is_empty() && analyzer_config_digest.len() != 32 {
        return Err(bad(format!(
            "analyzer_config_digest must be empty or 32 SHA-256 bytes, got {}",
            analyzer_config_digest.len()
        )));
    }
    let view_ref = doc
        .layers()
        .filter(|k| k.ty == TypeId::of::<View>())
        .find_map(|k| doc.get_key(&Key::<View>::named(k.name)).filter(|v| v.id() == target_view))
        .ok_or_else(|| bad(format!("document has no view {}", target_view.0)))?;
    disclosure.require_view_text(doc, target_view, "ProjectSearchDocument")?;

    let target_text = view_ref.text();
    let mut cx = SearchProjCx::new(doc, target_view, target_text, offsets);
    for k in doc.layers() {
        if let Some(p) = doc.provenance(k) {
            cx.register_provenance(p);
        }
    }

    let exported = export(doc, codecs()).map_err(|e| bad(format!("document graph export: {e}")))?;
    let graph_bytes = exported.to_bytes().map_err(|e| bad(format!("document graph encode: {e}")))?;
    let mut views = Vec::new();
    for k in doc.layers().filter(|k| k.ty == TypeId::of::<View>()) {
        if let Some(v) = doc.get_key(&Key::<View>::named(k.name)) {
            views.push(ViewAlignmentSummary {
                view_id: u32::from(v.id().0),
                name: k.name.to_owned(),
                parent_view: v.parent().map(|p| u32::from(p.0)),
                utf8_bytes: v.text().len() as u32,
                text_sha256: v.text_digest().as_bytes().to_vec(),
                lossy: v.alignment().edits().iter().any(|e| !e.parent.is_empty()),
            });
        }
    }

    let sentences = project_sentences(&cx)?;
    let chunks = project_chunks(&cx)?;
    let tokens = project_lexical_tokens(&cx, &sentences)?;
    let numeric_temporal = columns::project_numeric_temporal(&cx)?;
    let geo_locations = columns::project_geo(&cx)?;
    let categories_sentiment = columns::project_categories_sentiment(&cx)?;
    let quality = columns::project_quality(doc)?;
    let entities = entities::project_entities(&cx)?;
    let document_sketch = doc.get::<Sketches<View>>().and_then(|s| sketch_row(s, 0));

    let (_, target_text_len) = cx.offsetter.to_wire(target_text, Span::new(0, target_text.len() as u32));
    let provenance = SearchReplayProvenance {
        input: Some(input_identity(doc, binding)),
        result: Some(result_identity(&graph_bytes)),
        views,
        models: cx.models,
        analyzer_config_digest: analyzer_config_digest.to_vec(),
    };

    let out = SearchDocumentProjection {
        doc_id: doc_id.to_owned(),
        target_view: u32::from(target_view.0),
        offset_mode: offsets.to_proto(),
        target_text_len,
        provenance: Some(provenance),
        sentences,
        chunks,
        tokens,
        numeric_temporal: Some(numeric_temporal),
        geo_locations,
        categories_sentiment: Some(categories_sentiment),
        quality: Some(quality),
        entities: Some(entities),
        document_sketch,
    };
    debug_assert!(validate_search_projection(&out).is_ok());
    Ok(out)
}

fn project_sentences(cx: &SearchProjCx<'_>) -> Result<Vec<SearchSentenceRow>, Failure> {
    let Some(sents) = cx.doc.get::<Sentences>() else {
        return Ok(Vec::new());
    };
    let bp = cx.doc.get::<Boilerplate<Sentences>>();
    let mut rows = Vec::with_capacity(sents.len());
    let mut prev_end = 0u32;
    for i in 0..sents.len() {
        let (mut start, end) = cx.map_to_wire(sents.spans.view(), sents.spans.get(Ix::new(i as u32)))?;
        if i > 0 && start < prev_end {
            start = prev_end.min(end);
        }
        prev_end = prev_end.max(end);
        let (boilerplate_kind, boilerplate_score) = match bp.filter(|b| i < b.len()) {
            Some(b) => (b.kind[i].map(|k| boilerplate_kind_str(k).to_owned()), Some(b.score[i])),
            None => (None, None),
        };
        rows.push(SearchSentenceRow {
            index: i as u32,
            span_start: start,
            span_end: end,
            boilerplate_kind,
            boilerplate_score,
        });
    }
    Ok(rows)
}

fn project_chunks(cx: &SearchProjCx<'_>) -> Result<Vec<SearchChunkRow>, Failure> {
    let Some(chunks) = cx.doc.get::<Chunks>() else {
        return Ok(Vec::new());
    };
    let bp = cx.doc.get::<Boilerplate<Chunks>>();
    let sketches = cx.doc.get::<Sketches<Chunks>>();
    let mut rows = Vec::with_capacity(chunks.len());
    for i in 0..chunks.len() {
        let (span_start, span_end) = cx.map_to_wire(chunks.spans.view(), chunks.spans.get(Ix::new(i as u32)))?;
        let sr = chunks.sentences[i];
        let (boilerplate_kind, boilerplate_score) = match bp.filter(|b| i < b.len()) {
            Some(b) => (b.kind[i].map(|k| boilerplate_kind_str(k).to_owned()), Some(b.score[i])),
            None => (None, None),
        };
        rows.push(SearchChunkRow {
            index: i as u32,
            span_start,
            span_end,
            level: u32::from(chunks.level[i]),
            parent_index: chunks.parent[i].map(|p| p.get() as u32),
            sentence_start: sr.start.get() as u32,
            sentence_end: sr.end.get() as u32,
            cut_reason: cut_reason_str(chunks.cut[i]),
            cut_margin: chunks.cost[i],
            model_tokens: chunks.model_tokens[i],
            over_budget: chunks.over_budget[i],
            boilerplate_kind,
            boilerplate_score,
            sketch: sketches.and_then(|s| sketch_row(s, i)),
        });
    }
    Ok(rows)
}

fn find_sentence_index(sentences: &[SearchSentenceRow], span_start: u32) -> Option<u32> {
    if sentences.is_empty() {
        return None;
    }
    let idx = sentences.partition_point(|s| s.span_end <= span_start);
    sentences.get(idx).filter(|s| span_start >= s.span_start && span_start < s.span_end).map(|s| s.index)
}

fn project_lexical_tokens(
    cx: &SearchProjCx<'_>,
    sentences: &[SearchSentenceRow],
) -> Result<Vec<SearchLexicalToken>, Failure> {
    let mut out = Vec::new();
    let mut had_graph = false;
    for k in cx.doc.layers().filter(|k| k.ty == TypeId::of::<TokenGraph>()) {
        let Some(tg) = cx.doc.get_key(&Key::<TokenGraph>::named(k.name)) else { continue };
        had_graph = true;
        let view = tg.spans.view();
        for (_, arc) in tg.iter() {
            if arc.term.is_empty() {
                continue;
            }
            let (span_start, span_end) = cx.map_to_wire(view, arc.span)?;
            out.push(SearchLexicalToken {
                instance: k.name.to_owned(),
                term: arc.term.to_owned(),
                pos: arc.position,
                pos_len: arc.position_length,
                span_start,
                span_end,
                sentence_index: find_sentence_index(sentences, span_start),
                kind: token_arc_kind_str(arc.kind).to_owned(),
            });
        }
    }
    if had_graph {
        return Ok(out);
    }
    if let Some(tokens) = cx.doc.get::<Tokens>() {
        let stems = cx.doc.get::<Stems>();
        let lemmas = cx.doc.get::<Lemmas>();
        let view = tokens.spans.view();
        let view_text = cx.doc.text(view);
        for i in 0..tokens.len() {
            let span = tokens.spans.get(Ix::new(i as u32));
            let term = if let Some(s) = stems.filter(|s| i < s.len()) {
                s.stem.get(i)
            } else if let Some(l) = lemmas.filter(|l| i < l.len()) {
                l.lemma.get(i)
            } else {
                view_text.get(span.range()).unwrap_or_default()
            };
            if term.is_empty() {
                continue;
            }
            let (span_start, span_end) = cx.map_to_wire(view, span)?;
            let sent_idx = tokens
                .sentence
                .as_ref()
                .and_then(|s| s.get(i).map(|ix| ix.get() as u32))
                .or_else(|| find_sentence_index(sentences, span_start));
            out.push(SearchLexicalToken {
                instance: String::new(),
                term: term.to_owned(),
                pos: i as u32,
                pos_len: 1,
                span_start,
                span_end,
                sentence_index: sent_idx,
                kind: "primary".to_owned(),
            });
        }
    }
    Ok(out)
}

/// Imports the binary `DocumentGraph` in `req` and projects it under
/// [`PiiOutputPolicy::ALL`].
pub fn project_search_document(req: &ProjectSearchDocumentRequest) -> Result<SearchDocumentProjection, Failure> {
    project_search_document_under(req, PiiOutputPolicy::ALL)
}

/// [`project_search_document`] under the server's `policy`.
pub fn project_search_document_under(
    req: &ProjectSearchDocumentRequest,
    policy: PiiOutputPolicy,
) -> Result<SearchDocumentProjection, Failure> {
    let disclosure = Disclosure::decide(req.pii_output.as_ref(), policy)?;
    check_binding(req.binding.as_ref(), "binding")?;
    if req.graph.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "document graph bytes", req.graph.len() as u64));
    }
    let offsets = OffsetMode::from_proto(req.offsets)?;
    let view_u16 = u16::try_from(req.view).map_err(|_| bad(format!("view {} exceeds u16::MAX", req.view)))?;
    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, codecs()).map_err(|e| bad(format!("document graph import: {e}")))?;
    document_to_search_projection_bound(
        &req.doc_id,
        &doc,
        ViewId(view_u16),
        offsets,
        req.binding.as_ref(),
        &req.analyzer_config_digest,
        &disclosure,
    )
}

/// Validates `msg` and returns an unchanged clone (`RoundTripSearchProjection`).
pub fn round_trip_search_projection(msg: &SearchDocumentProjection) -> Result<SearchDocumentProjection, Failure> {
    validate_search_projection(msg)?;
    Ok(msg.clone())
}

impl Wire for SearchDocumentProjection {
    type Msg = SearchDocumentProjection;

    fn to_msg(&self) -> Self::Msg {
        self.clone()
    }

    fn from_msg(msg: &Self::Msg) -> Result<Self, Failure> {
        round_trip_search_projection(msg)
    }
}
