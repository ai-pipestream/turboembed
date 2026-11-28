//! Document and chunk ingestion into `HybridIndex` (`#1221`, `#2609`).

use std::collections::{BTreeMap, BTreeSet};

use turbo::TURBO_PROMPT_DOCUMENT;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, Layer, Span, ViewId};
use turbo_analysis_layers::LayoutBoxes;
use turbo_vector_index::{IndexBuilder, Normalization, Space, TokenIndexBuilder};

use crate::api::Result;
use crate::deadline::Deadline;
use crate::priority::Priority;
use crate::proto::search::{ChunkInput, IndexDocumentRequest, IndexSpec, LayoutBoxInput, VectorStorageKind};
use crate::results::document::codecs;
use crate::results::{MAX_GRAPH_BYTES, over};
use crate::search::registry::{
    CompiledDense, HybridIndex, RetainedDocument, StoredChunk, bad_arg, bad_state, verify_embed_bundle_matches_space,
};
use crate::search::segment::{apply_layout_boxes_to_chunks, extract_chunks_from_doc, segment_raw_text};
use crate::state::{Loaded, Slot, State};
use crate::tasks::embed::exec::{embed_f32_batches, prompt_embed_options};

pub(crate) struct PreparedDocument {
    pub doc_id: String,
    pub doc_text: String,
    pub graph: Vec<u8>,
    pub raw_chunks: Vec<ChunkInput>,
    pub prepared_chunks: Vec<ChunkInput>,
    pub doc_attrs: BTreeMap<String, String>,
}

pub(crate) struct StagedDocument {
    pub retained: RetainedDocument,
    pub chunks: Vec<StoredChunk>,
    pub chunk_terms: Vec<Vec<String>>,
}

pub(crate) fn prepare_document(spec: &IndexSpec, sealed: bool, req: IndexDocumentRequest) -> Result<PreparedDocument> {
    if sealed {
        return Err(bad_state(format!("search index `{}` is sealed and accepts no writes", spec.index_name)));
    }
    if req.doc_id.is_empty() {
        return Err(bad_arg("doc_id must not be empty"));
    }
    if req.graph.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "document graph bytes", req.graph.len() as u64));
    }

    let imported_doc = if !req.graph.is_empty() {
        let g = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad_arg(format!("invalid DocumentGraph: {e}")))?;
        Some(import(&g, codecs()).map_err(|e| bad_arg(format!("DocumentGraph import failed: {e}")))?)
    } else {
        None
    };

    let doc_text = if !req.text.is_empty() {
        req.text.clone()
    } else if let Some(d) = &imported_doc {
        d.source().to_owned()
    } else {
        return Err(bad_arg("either text or graph must be provided on IndexDocumentRequest"));
    };

    let raw_chunks = req.chunks.clone();
    let doc_attrs: BTreeMap<String, String> = req.attributes.clone().into_iter().collect();
    let mut prepared_chunks: Vec<ChunkInput> = if !req.chunks.is_empty() {
        req.chunks
    } else if let Some(d) = &imported_doc {
        extract_chunks_from_doc(d, &req.doc_id, spec.chunk_budget_tokens)?
    } else {
        segment_raw_text(&doc_text, &req.doc_id, spec.chunk_budget_tokens)
    };

    for (ci, c) in prepared_chunks.iter().enumerate() {
        validate_utf8_span(&doc_text, c.span_start, c.span_end, &format!("chunk {ci} span"))?;
        if c.has_parent {
            validate_utf8_span(&doc_text, c.parent_start, c.parent_end, &format!("chunk {ci} parent_span"))?;
        }
    }

    if !req.layout_boxes.is_empty() {
        let layout = build_request_layout_boxes(&doc_text, &req.layout_boxes)?;
        apply_layout_boxes_to_chunks(imported_doc.as_ref(), ViewId::SOURCE, &layout, &mut prepared_chunks);
    } else if let Some(d) = &imported_doc
        && let Some(layout) = d.get::<LayoutBoxes>()
    {
        apply_layout_boxes_to_chunks(Some(d), ViewId::SOURCE, layout, &mut prepared_chunks);
    }

    Ok(PreparedDocument { doc_id: req.doc_id, doc_text, graph: req.graph, raw_chunks, prepared_chunks, doc_attrs })
}

pub(crate) fn embed_prepared_documents(
    state: &State,
    loaded: &Loaded,
    slot: &Slot,
    space: &Space,
    docs: &mut [PreparedDocument],
    deadline: &Deadline,
) -> Result<()> {
    let mut targets: Vec<(usize, usize)> = Vec::new();
    let mut slices: Vec<String> = Vec::new();
    for (di, d) in docs.iter().enumerate() {
        for (ci, c) in d.prepared_chunks.iter().enumerate() {
            if c.dense_vector.is_empty() {
                let slice = d
                    .doc_text
                    .get(c.span_start as usize..c.span_end as usize)
                    .ok_or_else(|| bad_arg(format!("chunk span {}..{} is out of bounds", c.span_start, c.span_end)))?;
                targets.push((di, ci));
                slices.push(slice.to_owned());
            }
        }
    }
    if slices.is_empty() {
        return Ok(());
    }
    let unit = matches!(space.normalization, Normalization::Unit);
    let opts = prompt_embed_options(TURBO_PROMPT_DOCUMENT, unit, loaded.model_info.dim, space.dim);
    let _docs = loaded.metrics.track_docs(slices.len(), Some(loaded.device));
    let vectors = embed_f32_batches(state, loaded, slot, &slices, &opts, deadline)?;
    for ((di, ci), vec) in targets.into_iter().zip(vectors) {
        docs[di].prepared_chunks[ci].dense_vector = vec;
    }
    Ok(())
}

pub(crate) fn finalize_prepared_document(space: &Space, doc: PreparedDocument) -> Result<StagedDocument> {
    let dim = space.dim as usize;
    let mut staged_chunks = Vec::with_capacity(doc.prepared_chunks.len());
    let mut chunk_terms = Vec::with_capacity(doc.prepared_chunks.len());

    for (ci, c) in doc.prepared_chunks.iter().enumerate() {
        let chunk_slice = doc
            .doc_text
            .get(c.span_start as usize..c.span_end as usize)
            .ok_or_else(|| bad_arg(format!("chunk {ci} span {}..{} is invalid", c.span_start, c.span_end)))?;
        let (parent_span, parent_text) = if c.has_parent {
            let ps_slice = doc.doc_text.get(c.parent_start as usize..c.parent_end as usize).ok_or_else(|| {
                bad_arg(format!("chunk {ci} parent_span {}..{} is invalid", c.parent_start, c.parent_end))
            })?;
            (Some(Span::new(c.parent_start, c.parent_end)), ps_slice.to_owned())
        } else {
            (None, String::new())
        };

        let dense_vector = if c.dense_vector.is_empty() {
            None
        } else {
            if c.dense_vector.len() != dim {
                return Err(bad_arg(format!(
                    "chunk {ci} dense_vector has dimension {}, index holds {dim}",
                    c.dense_vector.len()
                )));
            }
            if c.dense_vector.iter().any(|v| !v.is_finite()) {
                return Err(bad_arg(format!("chunk {ci} dense_vector contains non-finite value")));
            }
            Some(c.dense_vector.clone())
        };

        for (ti, tv) in c.token_vectors.iter().enumerate() {
            if tv.values.len() != dim {
                return Err(bad_arg(format!(
                    "chunk {ci} token_vector {ti} has dimension {}, index holds {dim}",
                    tv.values.len()
                )));
            }
            if tv.values.iter().any(|v| !v.is_finite()) {
                return Err(bad_arg(format!("chunk {ci} token_vector {ti} contains non-finite value")));
            }
        }

        let terms = analyze_chunk_terms(chunk_slice, c.span_start);
        chunk_terms.push(terms.iter().map(|(t, _)| t.clone()).collect());
        let mut attrs = doc.doc_attrs.clone();
        for (k, v) in &c.attributes {
            attrs.insert(k.clone(), v.clone());
        }
        let chunk_id = if c.chunk_id.is_empty() { format!("{}#{}", doc.doc_id, ci) } else { c.chunk_id.clone() };
        let level_kind = if c.level_kind.is_empty() { "chunk".to_owned() } else { c.level_kind.clone() };
        let page_span = validate_chunk_layout(ci, c)?;
        staged_chunks.push(StoredChunk {
            chunk_id,
            doc_id: doc.doc_id.clone(),
            level_kind,
            span: Span::new(c.span_start, c.span_end),
            parent_span,
            text: chunk_slice.to_owned(),
            parent_text,
            terms,
            dense_vector,
            sparse_terms: c.sparse_terms.clone(),
            token_vectors: c.token_vectors.clone(),
            attributes: attrs,
            page_span,
            page_boxes: c.page_boxes.clone(),
        });
    }

    Ok(StagedDocument {
        retained: RetainedDocument {
            doc_id: doc.doc_id,
            text: doc.doc_text,
            graph: doc.graph,
            raw_chunks: doc.raw_chunks,
            chunks: doc.prepared_chunks,
            attributes: doc.doc_attrs,
        },
        chunks: staged_chunks,
        chunk_terms,
    })
}

pub(crate) fn stage_document(ix: &HybridIndex, state: &State, req: IndexDocumentRequest) -> Result<StagedDocument> {
    let mut docs = [prepare_document(&ix.spec, ix.sealed, req)?];
    if !ix.spec.embed_bundle.is_empty() && !docs[0].prepared_chunks.is_empty() {
        let loaded = verify_embed_bundle_matches_space(state, &ix.spec.embed_bundle, &ix.space)?;
        if docs[0].prepared_chunks.iter().any(|c| c.dense_vector.is_empty()) {
            let deadline = Deadline::none();
            let held = tokio::runtime::Handle::current().block_on(loaded.acquire_async(Priority::Bulk, &deadline))?;
            embed_prepared_documents(state, &loaded, held.slot(), &ix.space, &mut docs, &deadline)?;
        }
    }
    let [doc] = docs;
    finalize_prepared_document(&ix.space, doc)
}

pub(crate) fn commit_staged_documents(
    ix: &mut HybridIndex,
    staged: Vec<StagedDocument>,
    batch_chunk_terms: &mut Vec<Vec<String>>,
) -> Result<(u32, u32)> {
    let mut indexed_docs = 0u32;
    let mut indexed_chunks = 0u32;
    for sd in staged {
        for terms in &sd.chunk_terms {
            let mut unique = BTreeSet::new();
            for t in terms {
                ix.seen_terms.insert(t.clone());
                unique.insert(t.clone());
            }
            for t in unique {
                *ix.term_df.entry(t).or_insert(0) += 1;
            }
            batch_chunk_terms.push(terms.clone());
        }
        indexed_docs += 1;
        indexed_chunks += sd.chunks.len() as u32;
        ix.chunks.extend(sd.chunks);
        ix.docs.push(sd.retained);
    }
    rebuild_compiled_indexes(ix)?;
    Ok((indexed_docs, indexed_chunks))
}

fn validate_utf8_span(doc_text: &str, start: u32, end: u32, label: &str) -> Result<()> {
    let (s, e) = (start as usize, end as usize);
    if s > e || e > doc_text.len() {
        return Err(bad_arg(format!("{label} {start}..{end} is out of bounds for document length {}", doc_text.len())));
    }
    if !doc_text.is_char_boundary(s) || !doc_text.is_char_boundary(e) {
        return Err(bad_arg(format!("{label} {start}..{end} is not on UTF-8 char boundaries")));
    }
    Ok(())
}

fn build_request_layout_boxes(doc_text: &str, boxes: &[LayoutBoxInput]) -> Result<LayoutBoxes> {
    let mut layout = LayoutBoxes::with_capacity(ViewId::SOURCE, boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        let sp = b.span.as_ref().ok_or_else(|| bad_arg(format!("layout_box {i} span is missing")))?;
        validate_utf8_span(doc_text, sp.start, sp.end, &format!("layout_box {i} span"))?;
        layout.push(Span::new(sp.start, sp.end), b.page, b.x0, b.y0, b.x1, b.y1);
    }
    layout.check().map_err(bad_arg)?;
    Ok(layout)
}

fn validate_chunk_layout(ci: usize, c: &ChunkInput) -> Result<Option<(u32, u32)>> {
    for (bi, b) in c.page_boxes.iter().enumerate() {
        if b.page == 0 || ![b.x0, b.y0, b.x1, b.y1].into_iter().all(f32::is_finite) || b.x0 > b.x1 || b.y0 > b.y1 {
            return Err(bad_arg(format!("chunk {ci} page_box {bi} has invalid page or bounds")));
        }
    }
    if !c.page_boxes.windows(2).all(|w| w[0].page < w[1].page) {
        return Err(bad_arg(format!("chunk {ci} page_boxes are not in strictly ascending page order")));
    }
    if c.has_page_span {
        if c.page_start == 0 || c.page_start > c.page_end {
            return Err(bad_arg(format!("chunk {ci} has invalid page_span {}..={}", c.page_start, c.page_end)));
        }
        if let (Some(first), Some(last)) = (c.page_boxes.first(), c.page_boxes.last())
            && (first.page != c.page_start || last.page != c.page_end)
        {
            return Err(bad_arg(format!("chunk {ci} page_span does not match page_boxes")));
        }
        Ok(Some((c.page_start, c.page_end)))
    } else {
        if c.page_start != 0 || c.page_end != 0 || !c.page_boxes.is_empty() {
            return Err(bad_arg(format!("chunk {ci} without has_page_span must have zero pages and no page_boxes")));
        }
        Ok(None)
    }
}

pub fn rebuild_compiled_indexes(ix: &mut HybridIndex) -> Result<()> {
    let mut b = turbo_chunker::terms::Bm25PostingIndexBuilder::new();
    for c in &ix.chunks {
        let refs: Vec<&str> = c.terms.iter().map(|(t, _)| t.as_str()).collect();
        b.add_row_tokens(&refs).map_err(bad_arg)?;
    }
    ix.bm25 = b.build();

    let has_dense = ix.chunks.iter().any(|c| c.dense_vector.is_some());
    if has_dense {
        let mut vb = IndexBuilder::new(ix.space.clone(), ix.metric).map_err(|e| bad_arg(e.to_string()))?;
        for (row_idx, c) in ix.chunks.iter().enumerate() {
            if let Some(vec) = &c.dense_vector {
                vb.push(vec, row_idx as u32).map_err(|e| bad_arg(e.to_string()))?;
            }
        }
        ix.dense = match ix.storage {
            VectorStorageKind::FlatF32 => CompiledDense::Flat(vb.build()),
            VectorStorageKind::TurboQuant => {
                let q = ix.quant.expect("validated TurboQuant config");
                let q_ix = vb.build_quantized(q).map_err(|e| bad_arg(e.to_string()))?;
                CompiledDense::Quantized(Box::new(q_ix))
            }
        };
    } else {
        ix.dense = CompiledDense::Empty;
    }

    let has_tokens = ix.chunks.iter().any(|c| !c.token_vectors.is_empty());
    if has_tokens {
        let mut tb = TokenIndexBuilder::new(ix.space.clone()).map_err(|e| bad_arg(e.to_string()))?;
        let mut candidates = Vec::new();
        for (row_idx, c) in ix.chunks.iter().enumerate() {
            if c.token_vectors.is_empty() {
                continue;
            }
            let flat: Vec<f32> = c.token_vectors.iter().flat_map(|t| t.values.iter().copied()).collect();
            tb.push(&flat, row_idx as u32).map_err(|e| bad_arg(e.to_string()))?;
            candidates.push(row_idx as u32);
        }
        ix.token_index = Some(tb.build());
        ix.token_candidates = candidates;
    } else {
        ix.token_index = None;
        ix.token_candidates.clear();
    }
    Ok(())
}

pub fn analyze_chunk_terms(chunk_text: &str, chunk_offset: u32) -> Vec<(String, Span)> {
    let tok = turbo_tokenize::WordTokenizer::new();
    let doc = Document::new(chunk_text);
    let view = doc.get::<turbo_analysis::View>().expect("source view");
    let Ok((tokens, _)) = turbo_analysis::Annotator::annotate(&tok, (view, None)) else {
        return Vec::new();
    };
    let stemmer = turbo_stem::snowball::SnowballStemmer::new(turbo_stem::snowball::SnowballAlgorithm::English);
    let (starts, ends) = tokens.spans.columns();
    let mut out = Vec::new();
    for (&s, &e) in starts.iter().zip(ends) {
        let Some(slice) = chunk_text.get(s as usize..e as usize) else {
            continue;
        };
        if !slice.chars().any(char::is_alphanumeric) {
            continue;
        }
        let lower = slice.to_lowercase();
        let stemmed = turbo_stem::stemmer::Stemmer::stem(&stemmer, &lower);
        out.push((stemmed, Span::new(chunk_offset + s, chunk_offset + e)));
    }
    out
}
