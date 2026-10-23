//! The static-to-contextual cascade over a real pipeline
//! (Markdown structure, the chunker, levels, term counts):
//! build-time space isolation, pool-only contextual
//! reranking, structured filters, determinism, and a
//! recall sweep over candidate counts.

use std::collections::BTreeSet;
use std::marker::PhantomData;

use turbo_analysis::*;
use turbo_analysis_layers::{Entities, Levels, Sentences, Subwords, Tokens, Vectors};
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
    document_with(TEXT, &SENTENCES)
}

fn document_with(text: &str, sentences: &[&str]) -> Document {
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
fn entity_layer_filter_drops_and_boosts_with_measured_recall() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    // A real Entities layer over the "mice" mention, as a name
    // finder would write it; the filter is derived from its spans.
    let entities = {
        let tokens = doc.get::<Tokens>().unwrap();
        let mut spans = Spans::<Entities>::new(ViewId::SOURCE);
        let mut ix = Vec::new();
        for (i, s) in tokens.spans.iter() {
            if &TEXT[s.range()] == "mice" {
                spans.push(s);
                ix.push(IxRange::new(i.get() as u32, i.get() as u32 + 1));
            }
        }
        let kinds = TagSet::new("animals", &["animal"]);
        let n = ix.len();
        Entities { spans, tokens: ix, kind: vec![kinds.tag("animal").unwrap(); n], kinds, score: vec![0.9; n] }
    };
    entities.check().unwrap();
    let (sm, cm) = (model("static"), model("contextual"));
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let contextual_rows = rows(&doc, &["Cats", "Dogs", "mice"], "contextual");
    let run = |filters: Vec<UnitFilter>| {
        Cascade::new(
            levels,
            CascadeConfig { candidates: 64, top_k: 10, filters, ..CascadeConfig::default() },
            Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
            Some(VectorSpace { rows: &contextual_rows, query: QueryVector { model: &cm, vector: &[0.0, 0.0, 1.0] } }),
            Some(lexical(&doc, &collection)),
        )
        .unwrap()
        .search(&["cats"])
    };
    let relevant: BTreeSet<Ix<Levels>> =
        UnitFilter::overlapping("animal", FilterMode::Require, levels, &entities.spans).unwrap().units;
    let plain = run(Vec::new());
    let required = run(vec![UnitFilter { name: "animal".into(), mode: FilterMode::Require, units: relevant.clone() }]);
    assert!(!required.hits.is_empty() && required.hits.iter().all(|h| relevant.contains(&h.unit)));
    // Stage 3 drops after reranking, so the scored count is the
    // pool either way; the work saving is the pool, not the filter.
    assert_eq!(required.contextual_scored, plain.contextual_scored);
    let plain_ids: Vec<Ix<Levels>> = plain.hits.iter().map(|h| h.unit).collect();
    let required_ids: Vec<Ix<Levels>> = required.hits.iter().map(|h| h.unit).collect();
    assert_eq!(recall_at_k(&required_ids, &relevant, 10), 1.0);
    assert!(recall_at_k(&plain_ids, &relevant, 1) <= recall_at_k(&required_ids, &relevant, 1));
    let boosted = run(vec![UnitFilter { name: "animal".into(), mode: FilterMode::Boost(4.0), units: relevant }]);
    assert_eq!(boosted.hits[0].boosted_by, ["animal"]);
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

#[test]
fn overflowing_boosts_are_refused_and_scores_stay_finite() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let sm = model("static");
    let all: BTreeSet<Ix<Levels>> = (0..levels.len() as u32).map(Ix::new).collect();
    let boost = |name: &str, m: f64| UnitFilter { name: name.into(), mode: FilterMode::Boost(m), units: all.clone() };
    let space = || VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } };

    // Two maximal boosts over every unit: refused at build.
    let err = Cascade::new(
        levels,
        CascadeConfig { filters: vec![boost("a", f64::MAX), boost("b", f64::MAX)], ..CascadeConfig::default() },
        Some(space()),
        None,
        Some(lexical(&doc, &collection)),
    )
    .err()
    .unwrap_or_default();
    assert!(err.contains("overflows"), "{err}");
    // A later shrinking boost does not rescue an overflowing prefix.
    assert!(
        Cascade::new(
            levels,
            CascadeConfig {
                filters: vec![boost("a", 1e308), boost("b", 1e308), boost("c", 1e-308)],
                ..CascadeConfig::default()
            },
            Some(space()),
            None,
            Some(lexical(&doc, &collection)),
        )
        .is_err()
    );
    // One maximal boost is finite, and so is every score it produces.
    let cascade = Cascade::new(
        levels,
        CascadeConfig { filters: vec![boost("a", f64::MAX)], ..CascadeConfig::default() },
        Some(space()),
        None,
        Some(lexical(&doc, &collection)),
    )
    .unwrap();
    let out = cascade.search(&["cats"]);
    assert!(out.hits.iter().all(|h| h.score.is_finite()));
}

#[test]
fn recall_at_k_counts_each_relevant_unit_once() {
    let u = Ix::<Levels>::new(7);
    assert_eq!(recall_at_k(&[u, u], &BTreeSet::from([u]), 2), 1.0);
    let (a, b) = (Ix::<Levels>::new(1), Ix::<Levels>::new(2));
    assert_eq!(recall_at_k(&[a, a, a], &BTreeSet::from([a, b]), 3), 0.5);
    assert_eq!(recall_at_k(&[a, b, b], &BTreeSet::from([a, b]), 2), 1.0);
    assert_eq!(recall_at_k(&[a, a], &BTreeSet::from([a, b]), 0), 0.0);
}

#[test]
fn disjoint_reducing_filter_cannot_hide_boost_overflow() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let sm = model("static");
    let all: BTreeSet<Ix<Levels>> = (0..levels.len() as u32).map(Ix::new).collect();
    let space = || VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } };
    // The shrinking boost names no unit, so it never applies;
    // the two amplifying boosts meet on every unit. A single
    // sequential product (1e-310 * MAX * MAX, finite) would
    // accept this and search would emit +inf scores.
    let filters = vec![
        UnitFilter { name: "shrink-nowhere".into(), mode: FilterMode::Boost(1e-310), units: BTreeSet::new() },
        UnitFilter { name: "big-a".into(), mode: FilterMode::Boost(f64::MAX), units: all.clone() },
        UnitFilter { name: "big-b".into(), mode: FilterMode::Boost(f64::MAX), units: all.clone() },
    ];
    let err = Cascade::new(
        levels,
        CascadeConfig { filters, ..CascadeConfig::default() },
        Some(space()),
        None,
        Some(lexical(&doc, &collection)),
    )
    .err()
    .unwrap_or_default();
    assert!(err.contains("amplifying"), "{err}");
    // The same shrinking boost beside boosts whose amplifying
    // product is finite is accepted, and scores stay finite.
    let filters = vec![
        UnitFilter { name: "shrink-nowhere".into(), mode: FilterMode::Boost(1e-310), units: BTreeSet::new() },
        UnitFilter { name: "big-a".into(), mode: FilterMode::Boost(1e300), units: all.clone() },
        UnitFilter { name: "big-b".into(), mode: FilterMode::Boost(4.0), units: all },
    ];
    let cascade = Cascade::new(
        levels,
        CascadeConfig { filters, ..CascadeConfig::default() },
        Some(space()),
        None,
        Some(lexical(&doc, &collection)),
    )
    .unwrap();
    assert!(cascade.search(&["cats"]).hits.iter().all(|h| h.score.is_finite()));
}

#[test]
fn unembedded_units_never_outrank_contextually_scored_hits() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let mut contextual_rows = rows(&doc, &["Cats", "Dogs", "mice"], "contextual");
    // The mice chunk loses its contextual row entirely.
    let mice_chunk = levels
        .of(turbo_analysis_layers::Level::Chunk)
        .find(|&u| levels.spans.fragments(u).any(|f| TEXT[f.range()].contains("mice")))
        .unwrap();
    let d = contextual_rows.dim as usize;
    for v in &mut contextual_rows.data[mice_chunk.get() * d..(mice_chunk.get() + 1) * d] {
        *v = 0.0;
    }
    contextual_rows.empty = vec![mice_chunk];
    let (sm, cm) = (model("static"), model("contextual"));
    let cascade = Cascade::new(
        levels,
        CascadeConfig { candidates: 64, top_k: levels.len(), ..CascadeConfig::default() },
        Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
        Some(VectorSpace { rows: &contextual_rows, query: QueryVector { model: &cm, vector: &[0.0, 0.0, 1.0] } }),
        Some(lexical(&doc, &collection)),
    )
    .unwrap();
    let out = cascade.search(&["mice"]);
    let fallback = out.hits.iter().position(|h| h.unit == mice_chunk).expect("the mice chunk is pooled lexically");
    let hit = &out.hits[fallback];
    assert_eq!(hit.contextual_score, None);
    assert_eq!(hit.lexical_score, Some(1.0), "it is the top lexical hit, stage-1 score 1.0 on its own scale");
    assert!(hit.stage1_score > 0.0);
    assert!(
        out.hits[..fallback].iter().all(|h| h.contextual_score.is_some()),
        "every hit above the fallback was contextually scored"
    );
    assert!(
        out.hits[fallback..].iter().all(|h| h.contextual_score.is_none() || h.unit == mice_chunk),
        "no contextually scored hit sits below the fallback tier"
    );
}

mod facets;
mod latency;
