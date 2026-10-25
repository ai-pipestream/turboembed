//! Document and chunk ingestion into `HybridIndex` (`#1221`).

use std::collections::{BTreeMap, BTreeSet};

use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, Span};
use turbo_vector_index::{IndexBuilder, Normalization, Space, TokenIndexBuilder};

use crate::api::Result;
use crate::proto::search::{ChunkInput, IndexDocumentRequest, VectorStorageKind};
use crate::results::document::codecs;
use crate::results::{MAX_GRAPH_BYTES, over};
use crate::search::registry::{
    CompiledDense, HybridIndex, RetainedDocument, StoredChunk, bad_arg, bad_state, io_err,
    verify_embed_bundle_matches_space,
};
use crate::search::segment::{extract_chunks_from_doc, segment_raw_text};
use crate::state::State;

pub(crate) struct StagedDocument {
    pub retained: RetainedDocument,
    pub chunks: Vec<StoredChunk>,
    pub chunk_terms: Vec<Vec<String>>,
}

pub(crate) fn stage_document(ix: &HybridIndex, state: &State, req: IndexDocumentRequest) -> Result<StagedDocument> {
    if ix.sealed {
        return Err(bad_state(format!("search index `{}` is sealed and accepts no writes", ix.spec.index_name)));
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
        req.chunks.clone()
    } else if let Some(d) = &imported_doc {
        extract_chunks_from_doc(d, &req.doc_id, ix.spec.chunk_budget_tokens)?
    } else {
        segment_raw_text(&doc_text, &req.doc_id, ix.spec.chunk_budget_tokens)
    };

    for (ci, c) in prepared_chunks.iter().enumerate() {
        validate_utf8_span(&doc_text, c.span_start, c.span_end, &format!("chunk {ci} span"))?;
        if c.has_parent {
            validate_utf8_span(&doc_text, c.parent_start, c.parent_end, &format!("chunk {ci} parent_span"))?;
        }
    }

    if !ix.spec.embed_bundle.is_empty() {
        populate_embeddings_from_bundle(state, &ix.spec.embed_bundle, &ix.space, &doc_text, &mut prepared_chunks)?;
    }

    let dim = ix.space.dim as usize;
    let mut staged_chunks = Vec::with_capacity(prepared_chunks.len());
    let mut chunk_terms = Vec::with_capacity(prepared_chunks.len());

    for (ci, c) in prepared_chunks.iter().enumerate() {
        let chunk_slice = doc_text
            .get(c.span_start as usize..c.span_end as usize)
            .ok_or_else(|| bad_arg(format!("chunk {ci} span {}..{} is invalid", c.span_start, c.span_end)))?;
        let (parent_span, parent_text) = if c.has_parent {
            let ps_slice = doc_text.get(c.parent_start as usize..c.parent_end as usize).ok_or_else(|| {
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
        let mut attrs = doc_attrs.clone();
        for (k, v) in &c.attributes {
            attrs.insert(k.clone(), v.clone());
        }
        let chunk_id = if c.chunk_id.is_empty() { format!("{}#{}", req.doc_id, ci) } else { c.chunk_id.clone() };
        let level_kind = if c.level_kind.is_empty() { "chunk".to_owned() } else { c.level_kind.clone() };
        staged_chunks.push(StoredChunk {
            chunk_id,
            doc_id: req.doc_id.clone(),
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
        });
    }

    Ok(StagedDocument {
        retained: RetainedDocument {
            doc_id: req.doc_id,
            text: doc_text,
            graph: req.graph,
            raw_chunks,
            chunks: prepared_chunks,
            attributes: doc_attrs,
        },
        chunks: staged_chunks,
        chunk_terms,
    })
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

pub(crate) fn populate_embeddings_from_bundle(
    state: &State,
    bundle_name: &str,
    space: &Space,
    doc_text: &str,
    chunks: &mut [ChunkInput],
) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let loaded = verify_embed_bundle_matches_space(state, bundle_name, space)?;
    let held = loaded.take()?;
    let slot = held.slot();
    let max_batch = slot.info.max_batch as usize;
    let output_dim = if space.dim != loaded.model_info.dim { space.dim } else { 0 };
    let normalize = match space.normalization {
        Normalization::Unit => turbo::TURBO_NORMALIZE_L2,
        Normalization::None => turbo::TURBO_NORMALIZE_NONE,
    };
    let opts = turbo::turbo_embed_options {
        struct_size: size_of::<turbo::turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: turbo::TURBO_PROMPT_DOCUMENT,
        normalize,
        pooling: 0,
        output_dim,
    };
    for batch in chunks.chunks_mut(max_batch.max(1)) {
        let mut need_indices = Vec::new();
        let mut views = Vec::new();
        for (i, c) in batch.iter().enumerate() {
            if c.dense_vector.is_empty() {
                let slice = doc_text
                    .get(c.span_start as usize..c.span_end as usize)
                    .ok_or_else(|| bad_arg(format!("chunk span {}..{} is out of bounds", c.span_start, c.span_end)))?;
                views.push(turbo::turbo_text { ptr: slice.as_ptr().cast(), len: slice.len() as u64 });
                need_indices.push(i);
            }
        }
        if views.is_empty() {
            continue;
        }
        slot.session.write_text(&views, &opts)?;
        let res = slot.session.run()?;
        let info = res.info()?;
        let bytes = res.read(info.bytes)?;
        let dim = info.dim as usize;
        let expected_bytes = need_indices.len().saturating_mul(dim).saturating_mul(4);
        if bytes.len() < expected_bytes {
            return Err(io_err(format!(
                "embedding session returned {} bytes, expected at least {expected_bytes}",
                bytes.len()
            )));
        }
        for (bi, &ci) in need_indices.iter().enumerate() {
            let offset = bi * dim * 4;
            let floats: Vec<f32> =
                bytes[offset..offset + dim * 4].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            batch[ci].dense_vector = floats;
        }
    }
    Ok(())
}
