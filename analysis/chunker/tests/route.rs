//! Query-time level routing over a real pipeline (Markdown
//! structure, the chunker, the levels and their term counts):
//! factoid queries land on chunks with parent context, sibling
//! co-activation promotes their section, broad queries route to
//! the document, dense scores fuse with BM25, and ranking is
//! deterministic.

use std::marker::PhantomData;

use turbo_analysis::*;
use turbo_analysis_layers::Strs;
use turbo_analysis_layers::{Level, Levels, Sentences, Subwords, Tokens, Vectors};
use turbo_chunker::levels::LevelBuilder;
use turbo_chunker::route::RouteQuery;
use turbo_chunker::terms::{Bm25, LocalCollection, TermCounter, TermIndexer, TermStats, Vocabulary};
use turbo_chunker::{Budget, Chunker, Costs, LevelRouter, QueryKind, StructureBoundaries};
use turbo_structure::{StructureAnnotator, Syntax};

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

fn words(text: &str) -> Spans<Tokens> {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = None;
    for (i, c) in text.char_indices().chain(std::iter::once((text.len(), ' '))) {
        let i = i as u32;
        match (at, c.is_whitespace() || c == '.') {
            (Some(a), true) => {
                spans.push(Span::new(a, i));
                at = None;
            }
            (None, false) => at = Some(i),
            _ => {}
        }
        if c == '.' {
            spans.push(Span::new(i, i + 1));
        }
    }
    spans
}

fn document(text: &str, sentences: &[&str]) -> Document {
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut from = 0;
    for s in sentences {
        let a = from + text[from..].find(s).unwrap();
        spans.push(Span::new(a as u32, (a + s.len()) as u32));
        from = a + s.len();
    }
    let w = words(text);
    let n = w.len();
    let mut pieces = Spans::new(ViewId::SOURCE);
    for (_, s) in w.iter() {
        pieces.push(s);
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("t")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords { spans: pieces, ids: vec![0; n], model: toy_model(), token: None },
        Provenance::by("t"),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: w, sentence: None }, Provenance::by("t")).unwrap();
    doc
}

const TEXT: &str = "# Cats\n\nCats purr. Cats nap. Cats hunt mice.\n\n# Dogs\n\nDogs bark. Dogs run.\n";
const SENTENCES: [&str; 7] =
    ["# Cats", "Cats purr.", "Cats nap.", "Cats hunt mice.", "# Dogs", "Dogs bark.", "Dogs run."];

fn run(text: &str, sentences: &[&str]) -> Document {
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(StructureAnnotator::new(Syntax::Markdown))
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max: 5, target: 5, size_weight: 1000, overhead: 0 } })
        .add(LevelBuilder)
        .add(TermIndexer)
        .add(TermCounter::<Levels>::new())
        .build()
        .unwrap();
    let mut docs = vec![document(text, sentences)];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    docs.pop().unwrap()
}

fn routed<'a>(
    router: &LevelRouter,
    doc: &'a Document,
    query: &[&str],
    vectors: Option<(&'a Vectors<Levels>, &[f32])>,
    top_k: usize,
) -> turbo_chunker::Routed {
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    router
        .route(
            doc.get::<Levels>().unwrap(),
            stats,
            &LocalCollection::new(stats, vocab),
            vocab,
            &RouteQuery { terms: query, vectors },
            top_k,
        )
        .unwrap()
}

fn hit_text<'a>(doc: &'a Document, hit: &turbo_chunker::Hit) -> &'a str {
    &doc.source()[hit.span.range()]
}

#[test]
fn factoid_queries_route_to_chunks_with_parent_context() {
    let doc = run(TEXT, &SENTENCES);
    let out = routed(&LevelRouter::default(), &doc, &["mice"], None, 5);
    assert_eq!(out.profile.kind, QueryKind::Factoid);
    assert_eq!(out.profile.present, 1);
    let top = &out.hits[0];
    assert_eq!(top.level, Level::Chunk);
    assert!(hit_text(&doc, top).contains("mice"), "top hit: {:?}", hit_text(&doc, top));
    let context = top.context.as_ref().expect("a chunk hit carries parent context");
    let levels = doc.get::<Levels>().unwrap();
    assert_eq!(levels.level(context.parent), Level::Section);
    assert!(&doc.source()[context.parent_span.range()].contains("# Cats"));
    assert!(context.parent_span.contains(top.span), "parent span holds the chunk span");
}

#[test]
fn sibling_coactivation_promotes_the_section() {
    // Two short Cats chunks inside a section diluted by long
    // filler chunks: the section's own BM25 signal trails its
    // children's, so promotion is what lifts it.
    const DILUTED: &str = "# Cats\n\nCats purr. Cats nap. Filler words wander onward today. More filler text rambles further along.\n\n# Dogs\n\nDogs bark. Dogs run.\n";
    const DILUTED_SENTENCES: [&str; 8] = [
        "# Cats",
        "Cats purr.",
        "Cats nap.",
        "Filler words wander onward today.",
        "More filler text rambles further along.",
        "# Dogs",
        "Dogs bark.",
        "Dogs run.",
    ];
    let doc = run(DILUTED, &DILUTED_SENTENCES);
    let out = routed(&LevelRouter::default(), &doc, &["Cats"], None, 10);
    let section = out
        .hits
        .iter()
        .find(|h| h.level == Level::Section && hit_text(&doc, h).contains("# Cats"))
        .expect("the Cats section is a hit");
    assert!(section.active_children >= 2, "its chunks co-activated: {section:?}");
    // The same query with co-activation disabled leaves the
    // section's raw signal strictly below the promoted one.
    let plain = LevelRouter { min_siblings: usize::MAX, ..LevelRouter::default() };
    let unpromoted = routed(&plain, &doc, &["Cats"], None, 10);
    let plain_section = unpromoted.hits.iter().find(|h| h.unit == section.unit).unwrap();
    assert!(section.score > plain_section.score, "{} promoted vs {} plain", section.score, plain_section.score);
    // One hot chunk alone (the only chunk holding "mice") promotes
    // nothing: its section reports fewer active children.
    let doc = run(TEXT, &SENTENCES);
    let section = routed(&LevelRouter::default(), &doc, &["Cats"], None, 10)
        .hits
        .into_iter()
        .find(|h| h.level == Level::Section && hit_text(&doc, h).contains("# Cats"))
        .unwrap();
    let factoid = routed(&LevelRouter::default(), &doc, &["mice"], None, 10);
    if let Some(s) = factoid.hits.iter().find(|h| h.unit == section.unit) {
        assert!(s.active_children < 2);
    }
}

#[test]
fn broad_common_queries_route_to_the_document() {
    let doc = run(TEXT, &SENTENCES);
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let router = LevelRouter::default();
    // Frequent terms only, several of them: a summary profile.
    let levels = doc.get::<Levels>().unwrap();
    let profile =
        router.profile(levels, stats, &collection, vocab, &["Cats", "Dogs", "Cats", "Dogs", "Cats", "Dogs"]).unwrap();
    assert_eq!(profile.kind, QueryKind::Summary, "profile: {profile:?}");
    let out = routed(&router, &doc, &["Cats", "Dogs", "Cats", "Dogs", "Cats", "Dogs"], None, 3);
    assert_eq!(out.hits[0].level, Level::Document, "hits: {:?}", out.hits);
    // A mid-specificity query is topical.
    let topical = router.profile(levels, stats, &collection, vocab, &["Cats", "purr"]).unwrap();
    assert!(matches!(topical.kind, QueryKind::Topical | QueryKind::Factoid));
    // Nothing present: an empty result, not an error.
    let none = routed(&router, &doc, &["zebra"], None, 3);
    assert!(none.hits.is_empty() && none.profile.present == 0);
}

#[test]
fn dense_scores_fuse_and_ranking_is_deterministic() {
    let doc = run(TEXT, &SENTENCES);
    let levels = doc.get::<Levels>().unwrap();
    // Every unit points at [0, 1] except the Dogs chunks at [1, 0];
    // the document row is an empty (unusable) vector.
    let mut data = vec![0.0f32; levels.len() * 2];
    for i in 0..levels.len() {
        data[i * 2 + 1] = 1.0;
    }
    let mut dogs = Vec::new();
    for (i, s) in levels.spans.iter() {
        if levels.level(i) == Level::Chunk && doc.source()[s.range()].contains("Dogs") {
            data[i.get() * 2] = 1.0;
            data[i.get() * 2 + 1] = 0.0;
            dogs.push(i);
        }
    }
    assert!(!dogs.is_empty());
    data[0] = 0.0;
    data[1] = 0.0;
    let vectors = Vectors::<Levels> { dim: 2, data, empty: vec![Ix::new(0)], model: toy_model(), _of: PhantomData };
    vectors.check().unwrap();

    let router = LevelRouter::default();
    let a = routed(&router, &doc, &["Cats"], Some((&vectors, &[1.0, 0.0])), 10);
    let b = routed(&router, &doc, &["Cats"], Some((&vectors, &[1.0, 0.0])), 10);
    assert_eq!(a, b, "routing is deterministic");
    // A Dogs chunk scores on dense alone (no lexical hit) and the
    // document, whose row is empty, carries no dense signal.
    let dense_hit = a.hits.iter().find(|h| dogs.contains(&h.unit)).expect("a dense-only hit");
    assert_eq!(dense_hit.lexical, 0.0);
    assert_eq!(dense_hit.dense, Some(1.0));
    if let Some(doc_hit) = a.hits.iter().find(|h| h.level == Level::Document) {
        assert_eq!(doc_hit.dense, None);
    }
    // Scores are sorted, ties in span then unit order.
    for w in a.hits.windows(2) {
        assert!(
            w[0].score > w[1].score
                || (w[0].score == w[1].score
                    && (w[0].span.start, w[0].span.end, w[0].unit) <= (w[1].span.start, w[1].span.end, w[1].unit))
        );
    }
}

#[test]
fn invalid_inputs_are_refused() {
    let doc = run(TEXT, &SENTENCES);
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let levels = doc.get::<Levels>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let router = LevelRouter::default();
    let q = RouteQuery { terms: &["Cats"], vectors: None };
    assert!(router.route(levels, stats, &collection, vocab, &q, 0).is_err(), "top_k of zero");
    let bad = LevelRouter { lexical_weight: f64::NAN, ..LevelRouter::default() };
    assert!(bad.route(levels, stats, &collection, vocab, &q, 3).is_err(), "NaN signal weight");
    let other = run("# Solo\n\nAlone here.\n", &["# Solo", "Alone here."]);
    let other_stats = other.get::<TermStats<Levels>>().unwrap();
    if other_stats.len() != stats.len() {
        assert!(router.route(levels, other_stats, &collection, vocab, &q, 3).is_err(), "stats of another document");
    }
    let zero = Vectors::<Levels> {
        dim: 2,
        data: vec![0.0; levels.len() * 2],
        empty: vec![],
        model: toy_model(),
        _of: PhantomData,
    };
    let zero_q = RouteQuery { terms: &["Cats"], vectors: Some((&zero, &[0.0, 0.0])) };
    assert!(router.route(levels, stats, &collection, vocab, &zero_q, 3).is_err(), "zero query vector");
    let short_q = RouteQuery { terms: &["Cats"], vectors: Some((&zero, &[1.0])) };
    assert!(router.route(levels, stats, &collection, vocab, &short_q, 3).is_err(), "dimension mismatch");
}

#[test]
fn extreme_router_values_neither_overflow_nor_change_ranking() {
    let doc = run(TEXT, &SENTENCES);
    let levels = doc.get::<Levels>().unwrap();
    // Valid vectors for every unit, all pointing one way.
    let mut data = vec![0.0f32; levels.len() * 2];
    for i in 0..levels.len() {
        data[i * 2] = 1.0;
    }
    let vectors = Vectors::<Levels> { dim: 2, data, empty: vec![], model: toy_model(), _of: PhantomData };
    vectors.check().unwrap();
    let plain = routed(&LevelRouter::default(), &doc, &["mice"], Some((&vectors, &[1.0, 0.0])), 10);
    assert!(!plain.hits.is_empty());
    // f64::MAX signal weights fuse exactly as 1.0 weights do.
    let huge = LevelRouter { lexical_weight: f64::MAX, dense_weight: f64::MAX, ..LevelRouter::default() };
    assert_eq!(routed(&huge, &doc, &["mice"], Some((&vectors, &[1.0, 0.0])), 10), plain);
    // A usize::MAX context window saturates to all siblings on
    // chunk hits and to every child on section hits, never panics.
    let wide = LevelRouter { context_siblings: usize::MAX, ..LevelRouter::default() };
    let out = routed(&wide, &doc, &["Cats"], Some((&vectors, &[1.0, 0.0])), 10);
    let chunk = out.hits.iter().find(|h| h.level == Level::Chunk).unwrap();
    let siblings = levels
        .of(Level::Chunk)
        .filter(|&i| levels.parent[i.get()] == levels.parent[chunk.unit.get()] && i != chunk.unit)
        .count();
    assert_eq!(chunk.context.as_ref().unwrap().window.len(), siblings);
    let section = out.hits.iter().find(|h| h.level == Level::Section).unwrap();
    let children = levels.of(Level::Chunk).filter(|&i| levels.parent[i.get()] == Some(section.unit)).count();
    assert!(children > 0);
    assert_eq!(section.context.as_ref().unwrap().window.len(), children);
}

#[test]
fn profile_and_route_refuse_mismatched_inputs_and_bad_bm25() {
    let doc = run(TEXT, &SENTENCES);
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let router = LevelRouter::default();
    let q = RouteQuery { terms: &["Cats"], vectors: None };

    // Stats from a smaller document: both entrypoints refuse,
    // where profile used to index past the shorter columns.
    let other = run("# Solo\n\nAlone here.\n", &["# Solo", "Alone here."]);
    let other_stats = other.get::<TermStats<Levels>>().unwrap();
    assert_ne!(other_stats.len(), levels.len());
    assert!(router.profile(levels, other_stats, &collection, vocab, &["Cats"]).is_err());
    assert!(router.route(levels, other_stats, &collection, vocab, &q, 3).is_err());

    // A vocabulary larger than the stats' df column.
    let mut text = Strs::default();
    for i in 0..stats.df.len() + 1 {
        text.push(&format!("t{i:04}"));
    }
    let wide_vocab = Vocabulary { text };
    assert!(router.profile(levels, stats, &collection, &wide_vocab, &["Cats"]).is_err());
    assert!(router.route(levels, stats, &collection, &wide_vocab, &q, 3).is_err());

    // Invalid BM25 parameters, through public fields.
    for bm25 in [
        Bm25 { k1: -1.0, b: 0.75 },
        Bm25 { k1: f64::NAN, b: 0.75 },
        Bm25 { k1: 1.2, b: 1.5 },
        Bm25 { k1: 1.2, b: f64::INFINITY },
    ] {
        let bad = LevelRouter { bm25, ..LevelRouter::default() };
        assert!(bad.profile(levels, stats, &collection, vocab, &["Cats"]).is_err(), "profile accepted {bm25:?}");
        assert!(bad.route(levels, stats, &collection, vocab, &q, 3).is_err(), "route accepted {bm25:?}");
    }
}
