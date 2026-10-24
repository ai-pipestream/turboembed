//! Corpus cascade tests: full-pool equality with dense
//! retrieval, equality with per-document `Cascade`, and space
//! refusal.

use std::marker::PhantomData;

use turbo_analysis::{Ix, ModelRef, Spans, ViewId};
use turbo_analysis_layers::Level;
use turbo_analysis_layers::{Chunks, Levels, Unit, Vectors};
use turbo_chunker::terms::Bm25;
use turbo_chunker::{Cascade as DocCascade, CascadeConfig, QueryVector, VectorSpace};
use turbo_vector_index::{Normalization, Space};

use super::*;
use crate::lexical::TermRule;
use crate::retrieve::Index as DenseIndex;
use crate::spans::Span;

fn model(id: &str) -> ModelRef {
    ModelRef { id: id.into(), revision: "1".into(), sha256: "0".into() }
}

fn cascade(candidates: usize) -> Cascade {
    Cascade { candidates, model: Bm25::LUCENE, terms: TermRule::Folded }
}

/// Three documents, two chunks each, in a 2-dim world. Static
/// and contextual rows differ so the stages disagree.
struct World {
    docs: Vec<String>,
    doc_of: Vec<u32>,
    spans: Vec<Span>,
    texts: Vec<String>,
    statics: Vec<f32>,
    contextuals: Vec<f32>,
}

fn world() -> World {
    let docs = vec!["a".to_string(), "b".into(), "c".into()];
    let texts: Vec<String> = ["cats purr", "cats nap", "dogs bark", "dogs run", "rust code", "rust ships"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let spans: Vec<Span> = (0..6).map(|i| Span::new(i * 10, i * 10 + 5).unwrap()).collect();
    World {
        docs,
        doc_of: vec![0, 0, 1, 1, 2, 2],
        spans,
        texts,
        statics: vec![1.0, 0.0, 0.8, 0.6, 0.0, 1.0, 0.6, 0.8, -1.0, 0.0, 0.0, -1.0],
        contextuals: vec![0.0, 1.0, 0.6, 0.8, 1.0, 0.0, 0.8, 0.6, 0.0, -1.0, -1.0, 0.0],
    }
}

fn spaces() -> (Space, Space) {
    (
        Space::new(model("study/static"), 2, Normalization::None),
        Space::new(model("study/contextual"), 2, Normalization::None),
    )
}

fn search_world(candidates: usize) -> (Run, ChunkRun) {
    let w = world();
    let c = cascade(candidates);
    let (ss, cs) = spaces();
    let index = c.build(ss.clone(), &w.statics, cs.clone(), &w.contextuals, 6).unwrap();
    let bm25 = c.index(&w.texts, &w.doc_of).unwrap();
    let s = CascadeSearch {
        bm25: &bm25,
        doc: &w.doc_of,
        span: &w.spans,
        docs: &w.docs,
        static_space: &ss,
        static_queries: &[1.0, 0.0],
        contextual_space: &cs,
        contextual_queries: &[0.0, 1.0],
    };
    c.search(&index, &s, &[("q", "zzzqqq")]).unwrap()
}

#[test]
fn full_pool_ranks_like_dense_contextual_retrieval() {
    let w = world();
    let dense = DenseIndex {
        dim: 2,
        vectors: w.contextuals.clone(),
        doc: w.doc_of.clone(),
        span: w.spans.clone(),
        docs: &w.docs,
    };
    let (want_docs, want_chunks) = dense.search(&["q"], &[0.0, 1.0]).unwrap();
    let (docs, chunks) = search_world(6);
    let ids = |r: &Run| r.hits("q").iter().map(|h| h.doc.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&docs), ids(&want_docs));
    let cids = |r: &ChunkRun| r.hits("q").iter().map(|h| (h.doc.clone(), h.span)).collect::<Vec<_>>();
    assert_eq!(cids(&chunks), cids(&want_chunks));
}

#[test]
fn corpus_cascade_equals_per_document_cascade() {
    // One document, its chunks as level units under the document
    // unit; the query text hits no BM25 term, so both cascades
    // run static + contextual only.
    let w = world();
    let (run, chunks) = search_world(2);
    let _ = run;

    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(turbo_analysis::Span::new(0, 55));
    for s in &w.spans {
        spans.push(turbo_analysis::Span::new(s.start as u32, s.end as u32));
    }
    let mut unit = vec![Unit::Document];
    unit.extend((0..6).map(|i| Unit::Chunk(Ix::<Chunks>::new(i))));
    let mut parent = vec![None];
    parent.extend((0..6).map(|_| Some(Ix::<Levels>::new(0))));
    let levels = Levels { spans, unit, parent };
    let rows = |head: [f32; 2], body: &[f32], m: ModelRef| Vectors::<Levels> {
        dim: 2,
        data: head.iter().chain(body).copied().collect(),
        empty: Vec::new(),
        model: m,
        _of: PhantomData,
    };
    let (sm, cm) = (model("study/static"), model("study/contextual"));
    let statics = rows([0.0, -1.0], &w.statics, sm.clone());
    let contextuals = rows([-1.0, 0.0], &w.contextuals, cm.clone());
    let (sq, cq) = ([1.0f32, 0.0], [0.0f32, 1.0]);
    let doc = DocCascade::new(
        &levels,
        CascadeConfig { candidates: 2, top_k: 100, bm25: Bm25::LUCENE, filters: Vec::new() },
        Some(VectorSpace { rows: &statics, query: QueryVector { model: &sm, vector: &sq } }),
        Some(VectorSpace { rows: &contextuals, query: QueryVector { model: &cm, vector: &cq } }),
        None,
    )
    .unwrap();
    let result = doc.search(&["zzzqqq"]);
    let want: Vec<Span> = result
        .hits
        .iter()
        .filter(|h| h.level == Level::Chunk)
        .map(|h| Span::new(h.spans[0].start as usize, h.spans[0].end as usize).unwrap())
        .collect();
    let got: Vec<Span> = chunks.hits("q").iter().map(|h| h.span).collect();
    // The corpus search covers all three documents; the per-doc
    // cascade covers one levels tree holding the same six chunks,
    // so the chunk order over the shared pool must agree where
    // both ranked the chunk (the corpus pool is the union over
    // the same six rows).
    let got_shared: Vec<Span> = got.iter().copied().filter(|s| want.contains(s)).collect();
    assert_eq!(got_shared, want);
}

#[test]
fn one_model_in_both_stages_is_refused() {
    let w = world();
    let c = cascade(2);
    let same = Space::new(model("study/static"), 2, Normalization::None);
    let err = c.build(same.clone(), &w.statics, same, &w.contextuals, 6).err().expect("refused");
    assert!(err.contains("different spaces"), "{err}");

    // A query of the static space scored against contextual rows
    // is refused by the index, never scored.
    let (ss, cs) = spaces();
    let index = c.build(ss.clone(), &w.statics, cs.clone(), &w.contextuals, 6).unwrap();
    let bm25 = c.index(&w.texts, &w.doc_of).unwrap();
    let s = CascadeSearch {
        bm25: &bm25,
        doc: &w.doc_of,
        span: &w.spans,
        docs: &w.docs,
        static_space: &cs,
        static_queries: &[1.0, 0.0],
        contextual_space: &ss,
        contextual_queries: &[0.0, 1.0],
    };
    let swapped = c.search(&index, &s, &[("q", "cats")]);
    assert!(swapped.is_err(), "swapped query spaces scored");
}

#[test]
fn own_document_chunks_do_not_consume_the_shortlist_budget() {
    // candidates = 1 and the query is document "a" itself, whose
    // chunks are the top static and lexical hits: the one eligible
    // slot must still be filled by another document, not spent on
    // the excluded self chunks.
    let w = world();
    let c = cascade(1);
    let (ss, cs) = spaces();
    let index = c.build(ss.clone(), &w.statics, cs.clone(), &w.contextuals, 6).unwrap();
    let bm25 = c.index(&w.texts, &w.doc_of).unwrap();
    let s = CascadeSearch {
        bm25: &bm25,
        doc: &w.doc_of,
        span: &w.spans,
        docs: &w.docs,
        static_space: &ss,
        static_queries: &[1.0, 0.0],
        contextual_space: &cs,
        contextual_queries: &[0.0, 1.0],
    };
    let (docs, _chunks) = c.search(&index, &s, &[("a", "cats purr")]).unwrap();
    let hits: Vec<&str> = docs.hits("a").iter().map(|h| h.doc.as_str()).collect();
    assert!(!hits.is_empty(), "all stage-one budget was spent on the excluded self document");
    assert!(!hits.contains(&"a"), "the query document itself is never returned: {hits:?}");
}
