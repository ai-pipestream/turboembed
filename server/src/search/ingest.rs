//! Document and chunk ingestion into `HybridIndex` (`#1221`).

use std::collections::{BTreeMap, BTreeSet};

use turbo::TURBO_TASK_EMBED;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, Ix, Layer, Span};
use turbo_analysis_layers::{Chunks, Level, Levels, Sentences, SparseTerms, Subwords, Vectors};
use turbo_vector_index::{IndexBuilder, Source, TokenIndexBuilder};

use crate::api::Result;
use crate::proto::search::{ChunkInput, IndexDocumentRequest, SparseEntryProto, TokenVectorInput, VectorStorageKind};
use crate::results::document::codecs;
use crate::results::{MAX_GRAPH_BYTES, over};
use crate::search::registry::{CompiledDense, HybridIndex, RetainedDocument, StoredChunk, bad_arg, bad_state};
use crate::state::State;

pub(crate) fn ingest_document(
    ix: &mut HybridIndex,
    state: &State,
    req: IndexDocumentRequest,
    batch_chunk_terms: &mut Vec<Vec<String>>,
) -> Result<u32> {
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

    let doc_attrs: BTreeMap<String, String> = req.attributes.clone().into_iter().collect();
    let mut prepared_chunks: Vec<ChunkInput> = if !req.chunks.is_empty() {
        req.chunks.clone()
    } else if let Some(d) = &imported_doc {
        extract_chunks_from_doc(d, &req.doc_id, ix.spec.chunk_budget_tokens)?
    } else {
        segment_raw_text(&doc_text, &req.doc_id, ix.spec.chunk_budget_tokens)
    };

    if !ix.spec.embed_bundle.is_empty() {
        populate_embeddings_from_bundle(state, &ix.spec.embed_bundle, &doc_text, &mut prepared_chunks)?;
    }

    let dim = ix.space.dim as usize;
    let mut staged_chunks = Vec::with_capacity(prepared_chunks.len());
    for (ci, c) in prepared_chunks.iter().enumerate() {
        if c.span_start > c.span_end || (c.span_end as usize) > doc_text.len() {
            return Err(bad_arg(format!(
                "chunk {ci} span {}..{} is out of bounds for document length {}",
                c.span_start,
                c.span_end,
                doc_text.len()
            )));
        }
        if !doc_text.is_char_boundary(c.span_start as usize) || !doc_text.is_char_boundary(c.span_end as usize) {
            return Err(bad_arg(format!(
                "chunk {ci} span {}..{} is not on UTF-8 char boundaries",
                c.span_start, c.span_end
            )));
        }
        let chunk_slice = &doc_text[c.span_start as usize..c.span_end as usize];
        let parent_span = if c.has_parent {
            if c.parent_start > c.parent_end || (c.parent_end as usize) > doc_text.len() {
                return Err(bad_arg(format!(
                    "chunk {ci} parent_span {}..{} out of bounds",
                    c.parent_start, c.parent_end
                )));
            }
            Some(Span::new(c.parent_start, c.parent_end))
        } else {
            None
        };
        let parent_text =
            parent_span.map(|ps| doc_text[ps.start as usize..ps.end as usize].to_owned()).unwrap_or_default();

        let dense_vector = if c.dense_vector.is_empty() {
            None
        } else {
            if c.dense_vector.len() != dim {
                return Err(bad_arg(format!(
                    "chunk {ci} dense_vector has dimension {}, index holds {dim}",
                    c.dense_vector.len()
                )));
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
        }

        let terms = analyze_chunk_terms(chunk_slice, c.span_start);
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

    for sc in &staged_chunks {
        let term_refs: Vec<&str> = sc.terms.iter().map(|(t, _)| t.as_str()).collect();
        ix.bm25_builder.add_row_tokens(&term_refs).map_err(bad_arg)?;
        let term_strings: Vec<String> = sc.terms.iter().map(|(t, _)| t.clone()).collect();
        let mut unique = BTreeSet::new();
        for t in &term_strings {
            ix.seen_terms.insert(t.clone());
            unique.insert(t.clone());
        }
        for t in unique {
            *ix.term_df.entry(t).or_insert(0) += 1;
        }
        batch_chunk_terms.push(term_strings);
    }

    let added = staged_chunks.len() as u32;
    ix.chunks.extend(staged_chunks);
    ix.docs.push(RetainedDocument {
        doc_id: req.doc_id,
        text: doc_text,
        graph: req.graph,
        chunks: prepared_chunks,
        attributes: doc_attrs,
    });
    Ok(added)
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
        let _ = Source { piece: 0, span: None };
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
        let slice = &chunk_text[s as usize..e as usize];
        if !slice.chars().any(char::is_alphanumeric) {
            continue;
        }
        let lower = slice.to_lowercase();
        let stemmed = turbo_stem::stemmer::Stemmer::stem(&stemmer, &lower);
        out.push((stemmed, Span::new(chunk_offset + s, chunk_offset + e)));
    }
    out
}

fn extract_chunks_from_doc(doc: &Document, doc_id: &str, budget_tokens: u32) -> Result<Vec<ChunkInput>> {
    let text = doc.source();
    let dense_chunks = doc.get::<Vectors<Chunks>>();
    let sparse_chunks = doc.get::<SparseTerms<Chunks>>();
    let subwords = doc.get::<Subwords>();
    let subword_vecs = doc.get::<Vectors<Subwords>>();

    if let Some(levels) = doc.get::<Levels>()
        && !levels.is_empty()
    {
        let mut out = Vec::new();
        for i in 0..levels.len() {
            let ix = Ix::<Levels>::new(i as u32);
            let span = levels.spans.get(ix);
            let kind_str = match levels.level(ix) {
                Level::Chunk => "chunk",
                Level::Section => "section",
                Level::Document => "document",
            };
            let (parent_start, parent_end, has_parent) = match levels.parent[i] {
                Some(p) => {
                    let ps = levels.spans.get(p);
                    (ps.start, ps.end, true)
                }
                None => (0, 0, false),
            };
            out.push(ChunkInput {
                chunk_id: format!("{doc_id}#L{i}"),
                level_kind: kind_str.to_owned(),
                span_start: span.start,
                span_end: span.end,
                parent_start,
                parent_end,
                has_parent,
                dense_vector: Vec::new(),
                sparse_terms: Vec::new(),
                token_vectors: extract_token_vecs_in_span(subwords, subword_vecs, span),
                attributes: Default::default(),
            });
        }
        return Ok(out);
    }

    if let Some(chunks) = doc.get::<Chunks>()
        && !chunks.is_empty()
    {
        let mut out = Vec::with_capacity(chunks.len());
        for i in 0..chunks.len() {
            let ix = Ix::<Chunks>::new(i as u32);
            let span = chunks.spans.get(ix);
            let dense_vector =
                dense_chunks.filter(|v| i < v.len() && !v.is_empty(ix)).map(|v| v.row(ix).to_vec()).unwrap_or_default();
            let sparse_terms =
                sparse_chunks.filter(|s| i < s.len()).map(|s| extract_sparse_unit(s, i)).unwrap_or_default();
            out.push(ChunkInput {
                chunk_id: format!("{doc_id}#C{i}"),
                level_kind: "chunk".to_owned(),
                span_start: span.start,
                span_end: span.end,
                parent_start: 0,
                parent_end: text.len() as u32,
                has_parent: true,
                dense_vector,
                sparse_terms,
                token_vectors: extract_token_vecs_in_span(subwords, subword_vecs, span),
                attributes: Default::default(),
            });
        }
        return Ok(out);
    }

    if let Some(sents) = doc.get::<Sentences>()
        && !sents.is_empty()
    {
        let mut out = Vec::with_capacity(sents.len());
        for i in 0..sents.len() {
            let ix = Ix::<Sentences>::new(i as u32);
            let span = sents.spans.get(ix);
            out.push(ChunkInput {
                chunk_id: format!("{doc_id}#S{i}"),
                level_kind: "sentence".to_owned(),
                span_start: span.start,
                span_end: span.end,
                parent_start: 0,
                parent_end: text.len() as u32,
                has_parent: true,
                dense_vector: Vec::new(),
                sparse_terms: Vec::new(),
                token_vectors: extract_token_vecs_in_span(subwords, subword_vecs, span),
                attributes: Default::default(),
            });
        }
        return Ok(out);
    }

    Ok(segment_raw_text(text, doc_id, budget_tokens))
}

fn extract_sparse_unit(sparse: &SparseTerms<Chunks>, unit: usize) -> Vec<SparseEntryProto> {
    let start = sparse.row[unit] as usize;
    let end = sparse.row[unit + 1] as usize;
    (start..end)
        .map(|e| {
            let sp = sparse.span.get(e).copied().flatten();
            SparseEntryProto {
                term_id: sparse.term[e],
                term_text: String::new(),
                weight: sparse.weight[e],
                span_start: sp.map_or(0, |s| s.start),
                span_end: sp.map_or(0, |s| s.end),
                has_span: sp.is_some(),
            }
        })
        .collect()
}

fn extract_token_vecs_in_span(
    subwords: Option<&Subwords>,
    vecs: Option<&Vectors<Subwords>>,
    span: Span,
) -> Vec<TokenVectorInput> {
    let (Some(sub), Some(v)) = (subwords, vecs) else { return Vec::new() };
    let r = sub.spans.within(span);
    let mut out = Vec::new();
    for i in r.start.get()..r.end.get() {
        let ix = Ix::<Subwords>::new(i as u32);
        if !v.is_empty(ix) {
            let sp = sub.spans.get(ix);
            out.push(TokenVectorInput {
                values: v.row(ix).to_vec(),
                span_start: sp.start,
                span_end: sp.end,
                has_span: true,
            });
        }
    }
    out
}

pub fn segment_raw_text(text: &str, doc_id: &str, budget_tokens: u32) -> Vec<ChunkInput> {
    if text.is_empty() {
        return Vec::new();
    }
    let budget = if budget_tokens == 0 { 64usize } else { budget_tokens as usize };
    let det = turbo_sentdetect::RuleSentenceDetector::new("en");
    let doc = Document::new(text);
    let view = doc.get::<turbo_analysis::View>().expect("source view");
    let sents = turbo_analysis::Annotator::annotate(&det, (view,)).map(|(s,)| s).unwrap_or_else(|_| {
        let mut s = Sentences { spans: turbo_analysis::Spans::new(view.id()) };
        s.spans.push(Span::new(0, text.len() as u32));
        s
    });
    let mut chunks = Vec::new();
    let mut cur_start = sents.spans.get(Ix::new(0)).start;
    let mut cur_end = sents.spans.get(Ix::new(0)).end;
    let mut cur_words = text[cur_start as usize..cur_end as usize].split_whitespace().count();
    let make_chunk = |idx: usize, start: u32, end: u32| ChunkInput {
        chunk_id: format!("{doc_id}#{idx}"),
        level_kind: "chunk".to_owned(),
        span_start: start,
        span_end: end,
        parent_start: 0,
        parent_end: text.len() as u32,
        has_parent: true,
        dense_vector: Vec::new(),
        sparse_terms: Vec::new(),
        token_vectors: Vec::new(),
        attributes: Default::default(),
    };
    if sents.is_empty() {
        return vec![make_chunk(0, 0, text.len() as u32)];
    }
    for i in 1..sents.len() {
        let sp = sents.spans.get(Ix::new(i as u32));
        let words = text[sp.start as usize..sp.end as usize].split_whitespace().count();
        if cur_words + words > budget && cur_words > 0 {
            let idx = chunks.len();
            chunks.push(make_chunk(idx, cur_start, cur_end));
            cur_start = sp.start;
            cur_end = sp.end;
            cur_words = words;
        } else {
            cur_end = sp.end;
            cur_words += words;
        }
    }
    let idx = chunks.len();
    chunks.push(make_chunk(idx, cur_start, cur_end));
    chunks
}

pub(crate) fn populate_embeddings_from_bundle(
    state: &State,
    bundle_name: &str,
    doc_text: &str,
    chunks: &mut [ChunkInput],
) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let (_, loaded) = state.ready(bundle_name, "")?;
    if loaded.task != TURBO_TASK_EMBED {
        return Err(bad_arg(format!("bundle `{bundle_name}` is not a TASK_EMBED model")));
    }
    let held = loaded.take()?;
    let slot = held.slot();
    let max_batch = slot.info.max_batch as usize;
    let opts = turbo::turbo_embed_options {
        struct_size: size_of::<turbo::turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: turbo::TURBO_PROMPT_DOCUMENT,
        normalize: 0,
        pooling: 0,
        output_dim: 0,
    };
    for batch in chunks.chunks_mut(max_batch.max(1)) {
        let mut need_indices = Vec::new();
        let mut views = Vec::new();
        for (i, c) in batch.iter().enumerate() {
            if c.dense_vector.is_empty() {
                let slice = &doc_text[c.span_start as usize..c.span_end as usize];
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
        for (bi, &ci) in need_indices.iter().enumerate() {
            let offset = bi * dim * 4;
            let floats: Vec<f32> =
                bytes[offset..offset + dim * 4].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            batch[ci].dense_vector = floats;
        }
    }
    Ok(())
}
