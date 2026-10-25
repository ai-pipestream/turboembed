//! Document graph chunk extraction and sentence-budget raw text segmentation
//! for `TurboSearchService` (`#1221`).

use turbo_analysis::{Document, Ix, Layer, Span};
use turbo_analysis_layers::{Chunks, Level, Levels, Sentences, SparseTerms, Subwords, Vectors};

use crate::api::Result;
use crate::proto::search::{ChunkInput, SparseEntryProto, TokenVectorInput};

pub(crate) fn extract_chunks_from_doc(doc: &Document, doc_id: &str, budget_tokens: u32) -> Result<Vec<ChunkInput>> {
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
    let budget = if budget_tokens == 0 { 64usize } else { budget_tokens as usize };
    let det = turbo_sentdetect::RuleSentenceDetector::new("en");
    let doc = Document::new(text);
    let view = doc.get::<turbo_analysis::View>().expect("source view");
    let sents = turbo_analysis::Annotator::annotate(&det, (view,)).map(|(s,)| s).unwrap_or_else(|_| {
        let mut s = Sentences { spans: turbo_analysis::Spans::new(view.id()) };
        s.spans.push(Span::new(0, text.len() as u32));
        s
    });
    if sents.is_empty() {
        return vec![make_chunk(0, 0, text.len() as u32)];
    }
    let mut chunks = Vec::new();
    let first_sp = sents.spans.get(Ix::new(0));
    let mut cur_start = first_sp.start;
    let mut cur_end = first_sp.end;
    let mut cur_words = text.get(cur_start as usize..cur_end as usize).map_or(0, |s| s.split_whitespace().count());
    for i in 1..sents.len() {
        let sp = sents.spans.get(Ix::new(i as u32));
        let words = text.get(sp.start as usize..sp.end as usize).map_or(0, |s| s.split_whitespace().count());
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
