//! The static-to-contextual cascade over a real pipeline
//! (Markdown structure, the chunker, levels, term counts):
//! build-time space isolation, pool-only contextual
//! reranking, structured filters, determinism, and a
//! recall sweep over candidate counts.

use std::collections::BTreeSet;
use std::marker::PhantomData;

use turbo_analysis::*;
use turbo_analysis_layers::{Levels, Sentences, Subwords, Tokens, Vectors};
use turbo_chunker::levels::LevelBuilder;
use turbo_chunker::terms::{LocalCollection, TermCounter, TermIndexer, TermStats, Vocabulary};
use turbo_chunker::{
    Budget, Cascade, CascadeConfig, Chunker, Costs, FilterMode, LexicalSpace, QueryVector, Stage, StructureBoundaries,
    UnitFilter, VectorSpace, recall_at_k,
};
use turbo_structure::{StructureAnnotator, Syntax};

fn model(id: &str) -> ModelRef {
    ModelRef { id: id.into(), revision: "1".into(), sha256: "0".into() }
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

const TEXT: &str = "# Cats\n\nCats purr. Cats nap. Cats hunt mice.\n\n# Dogs\n\nDogs bark. Dogs run.\n";
const SENTENCES: [&str; 7] =
    ["# Cats", "Cats purr.", "Cats nap.", "Cats hunt mice.", "# Dogs", "Dogs bark.", "Dogs run."];

fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut from = 0;
    for s in SENTENCES {
        let a = from + TEXT[from..].find(s).unwrap();
        spans.push(Span::new(a as u32, (a + s.len()) as u32));
        from = a + s.len();
    }
    let w = words(TEXT);
    let n = w.len();
    let mut pieces = Spans::new(ViewId::SOURCE);
    for (_, s) in w.iter() {
        pieces.push(s);
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("t")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords { spans: pieces, ids: vec![0; n], model: model("toy"), token: None },
        Provenance::by("t"),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: w, sentence: None }, Provenance::by("t")).unwrap();
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
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    docs.pop().unwrap()
}

/// Topic rows from the unit text itself: one axis per probe
/// word, so rankings are derivable by hand.
fn rows(doc: &Document, probes: &[&str], id: &str) -> Vectors<Levels> {
    let levels = doc.get::<Levels>().unwrap();
    let mut data = Vec::new();
    for i in 0..levels.len() as u32 {
        let unit = Ix::new(i);
        let text: String = levels
            .spans
            .fragments(unit)
            .map(|f| &doc.source()[f.start as usize..f.end as usize])
            .collect::<Vec<_>>()
            .join(" ");
        for probe in probes {
            data.push(if text.contains(probe) { 1.0 } else { 0.0 });
        }
    }
    Vectors { dim: probes.len() as u32, data, empty: Vec::new(), model: model(id), _of: PhantomData }
}

fn lexical<'a>(doc: &'a Document, collection: &'a LocalCollection<'a>) -> LexicalSpace<'a> {
    LexicalSpace {
        stats: doc.get::<TermStats<Levels>>().unwrap(),
        vocabulary: doc.get::<Vocabulary>().unwrap(),
        collection,
    }
}

#[test]
fn space_mismatches_are_refused_at_build_time() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let contextual_rows = rows(&doc, &["Cats", "Dogs", "mice"], "contextual");
    let (sm, cm) = (model("static"), model("contextual"));
    let cfg = CascadeConfig::default();

    // A query from another space than the rows'.
    let bad = VectorSpace { rows: &contextual_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0, 0.0] } };
    let err = Cascade::new(levels, cfg.clone(), None, Some(bad), None).err().unwrap_or_default();
    assert!(err.contains("space mismatch"), "{err}");

    // The same space in both stages is not a cascade.
    let same = VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } };
    let same2 = VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } };
    assert!(Cascade::new(levels, cfg.clone(), Some(same), Some(same2), None).is_err());

    // Dimension and direction are build-time checks too.
    let dim = VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0] } };
    assert!(Cascade::new(levels, cfg.clone(), Some(dim), None, None).is_err());
    let zero = VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[0.0, 0.0] } };
    assert!(Cascade::new(levels, cfg.clone(), Some(zero), None, None).is_err());
    let ok = VectorSpace { rows: &contextual_rows, query: QueryVector { model: &cm, vector: &[0.0, 0.0, 1.0] } };
    assert!(Cascade::new(levels, cfg, None, Some(ok), None).is_err(), "contextual alone has no stage 1");
}

#[test]
fn the_cascade_reranks_only_the_pool_and_names_its_stages() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let contextual_rows = rows(&doc, &["Cats", "Dogs", "mice"], "contextual");
    let (sm, cm) = (model("static"), model("contextual"));
    let cascade = Cascade::new(
        levels,
        CascadeConfig { candidates: 3, top_k: 3, ..CascadeConfig::default() },
        Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
        Some(VectorSpace { rows: &contextual_rows, query: QueryVector { model: &cm, vector: &[0.0, 0.0, 1.0] } }),
        Some(lexical(&doc, &collection)),
    )
    .unwrap();
    let out = cascade.search(&["cats"]);
    assert_eq!(out.contextual_scored, out.pool, "stage 2 scores the pool, not the document");
    assert!(out.pool < levels.len(), "pool {} of {}", out.pool, levels.len());
    let top = &out.hits[0];
    let text: String = top.spans.iter().map(|s| &TEXT[s.range()]).collect::<Vec<_>>().join(" ");
    assert!(text.contains("mice"), "contextual rerank puts the mice unit first: {text}");
    assert!(top.contextual_score.is_some());
    assert!(!top.admitted_by.is_empty());
    assert!(out.hits.iter().all(|h| h.admitted_by.contains(&Stage::Static) || h.admitted_by.contains(&Stage::Lexical)));
    // Deterministic: the same search, byte for byte.
    assert_eq!(cascade.search(&["cats"]), out);
}

#[test]
fn filters_require_and_boost_after_reranking() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let sm = model("static");
    let mice = {
        let tokens = doc.get::<Tokens>().unwrap();
        let mut spans = Spans::<Tokens>::new(ViewId::SOURCE);
        for (_, s) in tokens.spans.iter() {
            if &TEXT[s.range()] == "mice" {
                spans.push(s);
            }
        }
        spans
    };
    let require = UnitFilter::overlapping("mice-entity", FilterMode::Require, levels, &mice).unwrap();
    let cascade = Cascade::new(
        levels,
        CascadeConfig { candidates: 64, top_k: 10, filters: vec![require], ..CascadeConfig::default() },
        Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
        None,
        Some(lexical(&doc, &collection)),
    )
    .unwrap();
    let out = cascade.search(&["cats"]);
    assert!(!out.hits.is_empty());
    for hit in &out.hits {
        let text: String = hit.spans.iter().map(|s| &TEXT[s.range()]).collect::<Vec<_>>().join(" ");
        assert!(text.contains("mice"), "required filter keeps only mice units: {text}");
    }

    let boost = UnitFilter::overlapping("mice-entity", FilterMode::Boost(4.0), levels, &mice).unwrap();
    let cascade = Cascade::new(
        levels,
        CascadeConfig { candidates: 64, top_k: 1, filters: vec![boost], ..CascadeConfig::default() },
        Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
        None,
        Some(lexical(&doc, &collection)),
    )
    .unwrap();
    let out = cascade.search(&["cats"]);
    assert_eq!(out.hits[0].boosted_by, ["mice-entity"]);
    let text: String = out.hits[0].spans.iter().map(|s| &TEXT[s.range()]).collect::<Vec<_>>().join(" ");
    assert!(text.contains("mice"), "{text}");
}

#[test]
fn recall_grows_with_the_candidate_count() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let contextual_rows = rows(&doc, &["Cats", "Dogs", "mice"], "contextual");
    let (sm, cm) = (model("static"), model("contextual"));
    let search = |candidates: usize| {
        Cascade::new(
            levels,
            CascadeConfig { candidates, top_k: levels.len(), ..CascadeConfig::default() },
            Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
            Some(VectorSpace { rows: &contextual_rows, query: QueryVector { model: &cm, vector: &[0.0, 0.0, 1.0] } }),
            Some(lexical(&doc, &collection)),
        )
        .unwrap()
        .search(&["cats"])
    };
    // Gold: the contextual stage's own top 2 over everything.
    let full = search(levels.len());
    let gold: BTreeSet<Ix<Levels>> = full.hits.iter().take(2).map(|h| h.unit).collect();
    let mut curve = Vec::new();
    for candidates in [1usize, 2, 3, 5, 64] {
        let out = search(candidates);
        let ranking: Vec<Ix<Levels>> = out.hits.iter().map(|h| h.unit).collect();
        curve.push(recall_at_k(&ranking, &gold, 2));
        assert!(out.contextual_scored <= 2 * candidates, "stage 2 work follows the pool");
    }
    assert!(curve.windows(2).all(|w| w[0] <= w[1]), "recall is monotone in candidates: {curve:?}");
    assert_eq!(*curve.last().unwrap(), 1.0, "a full pool recovers the contextual ranking");
}
