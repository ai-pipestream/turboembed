//! Term statistics and BM25 over hand-made tokens, structure and chunks,
//! with the terms read from a case folded view.

use std::collections::HashMap;

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Chunks, Cut, Sentences, Structure, Tokens};
use turbo_chunker::terms::{Bm25, Collection, LocalCollection, Query, TermCounter, TermIndexer, TermStats, Vocabulary};
use turbo_normalize::{CaseFold, NormalizeStep};

const TEXT: &str = "Cats purr. Cats nap. Dogs nap too. Rust compiles. RUST ships fast.";

/// The source span from the start of `first` to the end of `last`.
fn span(text: &str, first: &str, last: &str) -> Span {
    let a = text.find(first).unwrap();
    let b = text.find(last).unwrap() + last.len();
    Span::new(a as u32, b as u32)
}

/// Words split at spaces, with a sentence's closing period a token of its
/// own; sentences end at a period.
fn tokens_and_sentences(text: &str) -> (Tokens, Sentences) {
    let (mut tokens, mut sentences) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    let (mut at, mut start) = (0u32, 0u32);
    for word in text.split(' ') {
        let w = word.trim_end_matches('.');
        tokens.push(Span::new(at, at + w.len() as u32));
        if w.len() < word.len() {
            tokens.push(Span::new(at + w.len() as u32, at + word.len() as u32));
            sentences.push(Span::new(start, at + word.len() as u32));
            start = at + word.len() as u32 + 1;
        }
        at += word.len() as u32 + 1;
    }
    (Tokens { spans: tokens, sentence: None }, Sentences { spans: sentences })
}

/// Two top-level sections, the second with a section nested in it, and a
/// paragraph in the first; chunks one level up per section and two cut
/// from the first.
fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let (tokens, sentences) = tokens_and_sentences(TEXT);
    let mut st =
        Structure { spans: Spans::new(ViewId::SOURCE), kind: Vec::new(), parent: Vec::new(), depth: Vec::new() };
    for (first, last, kind, parent) in [
        ("Cats purr", "too.", Block::Section, None),
        ("Cats purr", "nap.", Block::Paragraph, Some(0)),
        ("Rust", "fast.", Block::Section, None),
        ("RUST", "fast.", Block::Section, Some(2)),
    ] {
        st.spans.push(span(TEXT, first, last));
        st.kind.push(kind);
        st.parent.push(parent.map(Ix::new));
        st.depth.push(u8::from(parent.is_some()));
    }
    let mut ch = Chunks {
        spans: Spans::new(ViewId::SOURCE),
        level: Vec::new(),
        parent: Vec::new(),
        sentences: Vec::new(),
        cut: Vec::new(),
        cost: Vec::new(),
        model_tokens: Vec::new(),
    };
    for (first, last, level, parent, s) in [
        ("Cats purr", "too.", 1, None, (0, 3)),
        ("Cats purr", "nap.", 0, Some(0), (0, 2)),
        ("Dogs", "too.", 0, Some(0), (2, 3)),
        ("Rust", "fast.", 1, None, (3, 5)),
        ("Rust", "fast.", 0, Some(3), (3, 5)),
    ] {
        ch.spans.push(span(TEXT, first, last));
        ch.level.push(level);
        ch.parent.push(parent.map(Ix::new));
        ch.sentences.push(IxRange::new(s.0, s.1));
        ch.cut.push(Cut::Sentence);
        ch.cost.push(0.0);
        ch.model_tokens.push(0);
    }
    doc.add(&Key::DEFAULT, sentences, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, st, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, ch, Provenance::by("t")).unwrap();
    doc
}

/// Case folds the source into "folded", reads the terms from it, and
/// counts them per chunk, per structure block and for the document.
fn pipeline() -> Analyzer {
    Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(CaseFold::full())).write(0, "folded"))
        .step(Bound::new(TermIndexer).read(0, "folded"))
        .add(TermCounter::<Chunks>::new())
        .add(TermCounter::<Structure>::new())
        .add(TermCounter::<View>::new())
        .build()
        .unwrap()
}

fn run() -> Document {
    let mut docs = vec![document()];
    let status = pipeline().run(&mut docs);
    assert_eq!(status[0], Ok(()));
    docs.pop().unwrap()
}

/// A unit's terms and counts, as text.
fn row<L: Layer>(stats: &TermStats<L>, v: &Vocabulary, i: u32) -> Vec<(String, u32)> {
    let (terms, tf) = stats.unit(Ix::new(i));
    terms.iter().zip(tf).map(|(&t, &c)| (v.get(t).to_string(), c)).collect()
}

fn pairs(xs: &[(&str, u32)]) -> Vec<(String, u32)> {
    xs.iter().map(|&(t, c)| (t.to_string(), c)).collect()
}

#[test]
fn terms_come_from_the_folded_view_and_punctuation_is_no_term() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();
    let words: Vec<&str> = v.text.iter().collect();
    assert_eq!(words, ["cats", "compiles", "dogs", "fast", "nap", "purr", "rust", "ships", "too"]);
    let terms = doc.get::<turbo_chunker::terms::Terms>().unwrap();
    let (tokens, _) = tokens_and_sentences(TEXT);
    assert_eq!(terms.term.len(), tokens.spans.len());
    // "Cats" "purr" "." ...: the period is a token with no term.
    assert_eq!(terms.term[2], None);
    assert_eq!(terms.term[0].map(|t| v.get(t)), Some("cats"));
    // "RUST" and "Rust" are one term.
    let rust: Vec<usize> = tokens
        .spans
        .iter()
        .filter(|(_, s)| TEXT[s.range()].eq_ignore_ascii_case("rust"))
        .map(|(i, _)| i.get())
        .collect();
    assert_eq!(rust.len(), 2);
    assert_eq!(terms.term[rust[0]], terms.term[rust[1]]);
}

#[test]
fn every_level_gets_its_counts_lengths_and_df() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();

    let chunks = doc.get::<TermStats<Chunks>>().unwrap();
    assert_eq!(chunks.length, [7, 4, 3, 5, 5]);
    assert_eq!(row(chunks, v, 1), pairs(&[("cats", 2), ("nap", 1), ("purr", 1)]));
    assert_eq!(row(chunks, v, 2), pairs(&[("dogs", 1), ("nap", 1), ("too", 1)]));
    assert_eq!(row(chunks, v, 0), pairs(&[("cats", 2), ("dogs", 1), ("nap", 2), ("purr", 1), ("too", 1)]));
    assert_eq!(row(chunks, v, 4), pairs(&[("compiles", 1), ("fast", 1), ("rust", 2), ("ships", 1)]));
    // "nap" is in the first section chunk and in both of its children.
    assert_eq!(chunks.df[v.lookup("nap").unwrap().get()], 3);

    let blocks = doc.get::<TermStats<Structure>>().unwrap();
    assert_eq!(blocks.length, [7, 4, 5, 3]);
    assert_eq!(row(blocks, v, 3), pairs(&[("fast", 1), ("rust", 1), ("ships", 1)]));
    // The nested section's terms count in it and in the section around it.
    assert_eq!(blocks.tf(Ix::new(2), v.lookup("rust").unwrap()), 2);

    let whole = doc.get::<TermStats<View>>().unwrap();
    assert_eq!(whole.length, [12]);
    assert_eq!(whole.tf(Ix::new(0), v.lookup("cats").unwrap()), 2);
    assert!(whole.df.iter().all(|&d| d == 1));

    let p = doc.provenance(Key::<TermStats<Chunks>>::DEFAULT.id()).unwrap();
    assert_eq!(p.annotator, "turbo:term-counter");
    assert_eq!(Key::<TermStats<Chunks>>::DEFAULT.id().to_string(), "turbo:term-stats<turbo:chunks>");
}

/// BM25 as textbooks write it, from lists of words: the classic form, with
/// the (k1 + 1) factor, and Lucene's idf. Units with no words are not in
/// the collection.
fn reference(units: &[Vec<&str>], query: &[&str], k1: f64, b: f64) -> Vec<f64> {
    let n = units.iter().filter(|u| !u.is_empty()).count() as f64;
    let avgdl = units.iter().map(|u| u.len()).sum::<usize>() as f64 / n;
    let mut scores = Vec::new();
    for u in units {
        let mut score = 0.0;
        for q in query {
            let df = units.iter().filter(|v| v.contains(q)).count() as f64;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            let tf = u.iter().filter(|w| *w == q).count() as f64;
            let dl = u.len() as f64;
            score += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl));
        }
        scores.push(score);
    }
    scores
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn scores_match_a_plain_reference_bm25() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();
    let stats = doc.get::<TermStats<Chunks>>().unwrap();
    let chunks = doc.get::<Chunks>().unwrap();
    let leaves: Vec<Ix<Chunks>> = stats.units().filter(|i| chunks.level[i.get()] == 0).collect();
    let words = [
        vec!["cats", "purr", "cats", "nap"],
        vec!["dogs", "nap", "too"],
        vec!["rust", "compiles", "rust", "ships", "fast"],
    ];
    let local = LocalCollection::only(stats, v, |i| chunks.level[i.get()] == 0);
    assert_eq!((local.units(), local.mean_length()), (3, 4.0));
    for model in [Bm25::LUCENE, Bm25 { k1: 2.0, b: 0.3 }, Bm25 { k1: 0.5, b: 1.0 }, Bm25 { k1: 1.2, b: 0.0 }] {
        for query in [&["cats"][..], &["nap", "rust"], &["cats", "nap", "missing", "cats"], &["missing"]] {
            let q = Query::new(model, &local, v, query);
            let expected = reference(&words, query, model.k1, model.b);
            for (k, &i) in leaves.iter().enumerate() {
                // Lucene leaves out (k1 + 1): the same ranking, scaled.
                let got = q.score(stats, i) * (model.k1 + 1.0);
                assert!(close(got, expected[k]), "{model:?} {query:?} chunk {}: {got} != {}", i.get(), expected[k]);
            }
        }
    }
}

#[test]
fn idf_is_lucenes() {
    // ln(1 + (10 - 1 + 0.5) / (1 + 0.5)) = ln(22 / 3).
    assert!(close(Bm25::idf(10, 1), (22.0f64 / 3.0).ln()));
    // A term in every unit still weighs a little: ln(1 + 0.5 / 4.5).
    assert!(close(Bm25::idf(4, 4), (1.0f64 + 0.5 / 4.5).ln()));
    assert!(Bm25::idf(4, 4) > 0.0);
    assert!(Bm25::idf(1000, 1) > Bm25::idf(1000, 10));
}

/// A corpus's statistics, as an index over many documents would hand them.
struct Corpus {
    units: u64,
    mean_length: f64,
    df: HashMap<&'static str, u64>,
}

impl Collection for Corpus {
    fn units(&self) -> u64 {
        self.units
    }
    fn mean_length(&self) -> f64 {
        self.mean_length
    }
    fn df(&self, term: &str) -> u64 {
        self.df.get(term).copied().unwrap_or(0)
    }
}

#[test]
fn the_idf_comes_from_the_collection_the_caller_gives() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();
    let stats = doc.get::<TermStats<Chunks>>().unwrap();
    let corpus = Corpus { units: 1000, mean_length: 20.0, df: HashMap::from([("cats", 900), ("dogs", 3)]) };
    let q = Query::new(Bm25::LUCENE, &corpus, v, &["cats", "dogs"]);
    // Chunk 1 holds "cats" twice, chunk 2 "dogs" once. Across the corpus
    // "cats" is in nearly every unit and "dogs" is rare, so chunk 2 wins.
    let (one, two) = (Ix::new(1), Ix::new(2));
    let by_hand = |tf: f64, dl: f64, df: f64| {
        let idf = (1.0 + (1000.0 - df + 0.5) / (df + 0.5)).ln();
        idf * tf / (tf + 1.2 * (1.0 - 0.75 + 0.75 * dl / 20.0))
    };
    assert!(close(q.score(stats, one), by_hand(2.0, 4.0, 900.0)));
    assert!(close(q.score(stats, two), by_hand(1.0, 3.0, 3.0)));
    assert!(q.score(stats, two) > q.score(stats, one));
    let chunks = doc.get::<Chunks>().unwrap();
    let local = LocalCollection::only(stats, v, |i| chunks.level[i.get()] == 0);
    let q = Query::new(Bm25::LUCENE, &local, v, &["cats", "dogs"]);
    // Within the document both terms are in one leaf of three: equal idf,
    // and the repeated "cats" wins.
    assert!(q.score(stats, one) > q.score(stats, two));
}

#[test]
fn ranking_is_best_first_and_ties_go_to_the_earlier_unit() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();
    let stats = doc.get::<TermStats<Chunks>>().unwrap();
    let q = Query::new(Bm25::LUCENE, &LocalCollection::new(stats, v), v, &["rust"]);
    // Chunks 3 and 4 are the same text: the same score, in unit order.
    let ranked: Vec<usize> = q.rank(stats, stats.units()).iter().map(|(i, _)| i.get()).collect();
    assert_eq!(ranked, [3, 4]);
    let q = Query::new(Bm25::LUCENE, &LocalCollection::new(stats, v), v, &["cats", "too"]);
    let ranked = q.rank(stats, stats.units());
    assert_eq!(ranked.iter().map(|(i, _)| i.get()).collect::<Vec<_>>(), [0, 1, 2]);
    assert!(ranked.windows(2).all(|w| w[0].1 >= w[1].1));
    assert!(
        Query::new(Bm25::LUCENE, &LocalCollection::new(stats, v), v, &["missing"])
            .rank(stats, stats.units())
            .is_empty()
    );
}

#[test]
fn local_df_counts_only_the_units_kept() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();
    let blocks = doc.get::<TermStats<Structure>>().unwrap();
    let st = doc.get::<Structure>().unwrap();
    let all = LocalCollection::new(blocks, v);
    let sections = LocalCollection::only(blocks, v, |i| st.kind[i.get()] == Block::Section);
    assert_eq!((all.units(), all.df("cats"), all.mean_length()), (4, 2, 19.0 / 4.0));
    assert_eq!((sections.units(), sections.df("cats"), sections.df("rust")), (3, 1, 2));
    assert_eq!(sections.mean_length(), 5.0);
    assert_eq!(sections.df("missing"), 0);
}

#[test]
fn full_case_folding_maps_through_the_view() {
    // "Straße" folds to "strasse", "ẞ" (U+1E9E) to "ss" as well; the term
    // view's bytes differ from the source's, and each token's span is
    // mapped to it.
    let text = "Straße STRASSE STRAẞE.";
    let mut doc = Document::new(text);
    let (tokens, sentences) = tokens_and_sentences(text);
    doc.add(&Key::DEFAULT, sentences, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(CaseFold::full())).write(0, "folded"))
        .step(Bound::new(TermIndexer).read(0, "folded"))
        .add(TermCounter::<View>::new())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let v = docs[0].get::<Vocabulary>().unwrap();
    assert_eq!(v.text.iter().collect::<Vec<_>>(), ["strasse"]);
    let whole = docs[0].get::<TermStats<View>>().unwrap();
    assert_eq!((whole.length[0], whole.tf(Ix::new(0), Ix::new(0))), (3, 3));
}

#[test]
fn terms_of_the_source_view_keep_their_case() {
    let mut doc = document();
    let pipeline =
        Analyzer::builder().given(Key::<View>::DEFAULT).given(Key::<Tokens>::DEFAULT).add(TermIndexer).build().unwrap();
    let mut docs = vec![doc.clone()];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let v = docs[0].get::<Vocabulary>().unwrap();
    assert!(v.lookup("RUST").is_some() && v.lookup("Rust").is_some() && v.lookup("rust").is_none());
    // Units in a view other than the tokens' are refused.
    let (text, alignment) = turbo_normalize::normalize(&CaseFold::full(), TEXT);
    let folded = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("folded"), folded, Provenance::by("t")).unwrap();
    let folded_id = doc.get_key(&Key::<View>::named("folded")).unwrap().id();
    let mut other =
        Structure { spans: Spans::new(folded_id), kind: vec![Block::Section], parent: vec![None], depth: vec![0] };
    other.spans.push(Span::new(0, 4));
    doc.add(&Key::named("other"), other, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::named("other"))
        .add(TermIndexer)
        .step(Bound::new(TermCounter::<Structure>::new()).read(3, "other").write(0, "other"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let err = pipeline.run(&mut docs).pop().unwrap().unwrap_err();
    assert!(err.to_string().contains("the units are in view 1, the tokens in view 0"), "{err}");
}

#[test]
fn tokens_of_an_unrelated_view_are_refused() {
    // Terms read from a view two steps from the tokens' view: the step has
    // no document to follow the chain through, so it refuses.
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(CaseFold::full())).write(0, "folded"))
        .step(Bound::new(NormalizeStep::new(CaseFold::lowercase())).read(0, "folded").write(0, "twice"))
        .step(Bound::new(TermIndexer).read(0, "twice"))
        .build()
        .unwrap();
    let mut docs = vec![document()];
    let err = pipeline.run(&mut docs)[0].clone().unwrap_err();
    assert!(err.to_string().contains("neither the terms' view 2 nor its parent"), "{err}");
    // And the document says the terms failed rather than were empty.
    assert!(matches!(docs[0].status(Key::<Vocabulary>::DEFAULT.id()), Some(Status::Failed(_))));
    assert!(docs[0].get::<Vocabulary>().is_none());
}

#[test]
fn a_token_a_unit_end_cuts_counts_once_in_the_unit_it_starts_in() {
    // Units "ab c" and "d ef" over tokens "ab", "cd", "ef": "cd" starts in
    // the first.
    let text = "ab cd ef";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 2), (3, 5), (6, 8)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let mut st = Structure {
        spans: Spans::new(ViewId::SOURCE),
        kind: vec![Block::Section; 2],
        parent: vec![None; 2],
        depth: vec![0; 2],
    };
    st.spans.push(Span::new(0, 4));
    st.spans.push(Span::new(4, 8));
    doc.add(&Key::DEFAULT, st, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(TermIndexer)
        .add(TermCounter::<Structure>::new())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let blocks = docs[0].get::<TermStats<Structure>>().unwrap();
    assert_eq!(blocks.length, [2, 1]);
}

#[test]
fn a_document_with_no_terms_has_empty_rows_and_scores_nothing() {
    let text = ". , ;";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 1), (2, 3), (4, 5)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(TermIndexer)
        .add(TermCounter::<View>::new())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let v = docs[0].get::<Vocabulary>().unwrap();
    let whole = docs[0].get::<TermStats<View>>().unwrap();
    assert!(v.is_empty());
    assert_eq!((whole.length.as_slice(), whole.row.as_slice()), (&[0][..], &[0, 0][..]));
    let local = LocalCollection::new(whole, v);
    assert_eq!((local.units(), local.mean_length()), (0, 0.0));
    let q = Query::new(Bm25::LUCENE, &local, v, &["anything"]);
    assert!(q.terms.is_empty());
    assert_eq!(q.score(whole, Ix::new(0)), 0.0);
}

#[test]
fn scores_match_lucenes_formula_worked_by_hand() {
    // Lucene 9's BM25Similarity, k1 = 1.2, b = 0.75:
    //   idf = ln(1 + (N - n + 0.5) / (n + 0.5))
    //   score = idf * tf / (tf + k1 * (1 - b + b * dl / avgdl))
    // The collection is the three leaf chunks, of 4, 3 and 5 terms: N = 3,
    // avgdl = 12 / 3 = 4.
    //
    // "cats" in chunk 1 ("cats purr cats nap"): tf = 2, dl = 4, n = 1.
    //   idf = ln(1 + 2.5 / 1.5) = ln(8 / 3) = 0.9808292530117262
    //   tf part = 2 / (2 + 1.2 * (0.25 + 0.75 * 4 / 4)) = 2 / 3.2 = 0.625
    //   score = 0.6130182831323289
    // "nap" in chunk 2 ("dogs nap too"): tf = 1, dl = 3, n = 2.
    //   idf = ln(1 + 1.5 / 2.5) = ln(1.6) = 0.4700036292457356
    //   tf part = 1 / (1 + 1.2 * (0.25 + 0.75 * 3 / 4)) = 1 / 1.975
    //   score = 0.2379765211370813
    // Lucene computes in f32, so its own scores agree to about 1e-7.
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();
    let stats = doc.get::<TermStats<Chunks>>().unwrap();
    let chunks = doc.get::<Chunks>().unwrap();
    let local = LocalCollection::only(stats, v, |i| chunks.level[i.get()] == 0);
    let cats = Query::new(Bm25::LUCENE, &local, v, &["cats"]).score(stats, Ix::new(1));
    let nap = Query::new(Bm25::LUCENE, &local, v, &["nap"]).score(stats, Ix::new(2));
    assert!(close(cats, 0.6130182831323289), "{cats}");
    assert!(close(nap, 0.2379765211370813), "{nap}");
}

#[test]
fn a_query_term_given_twice_counts_twice() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();
    let stats = doc.get::<TermStats<Chunks>>().unwrap();
    let local = LocalCollection::new(stats, v);
    let once = Query::new(Bm25::LUCENE, &local, v, &["cats"]).score(stats, Ix::new(1));
    let twice = Query::new(Bm25::LUCENE, &local, v, &["cats", "cats"]).score(stats, Ix::new(1));
    assert!(once > 0.0);
    assert!(close(twice, 2.0 * once));
}

#[test]
fn units_with_no_terms_are_not_in_the_local_collection() {
    // Four units over the tokens "a . b a" (the period is no term): all of
    // them, the period alone, "b", and none, so lengths 3, 0, 1 and 0. N
    // counts the two with terms, and the mean is 4 / 2, not 4 / 4.
    let term = [Some(Ix::new(0)), None, Some(Ix::new(1)), Some(Ix::new(0))];
    let units = [0..4, 1..2, 2..3, 4..4];
    let stats: TermStats<Chunks> = turbo_chunker::terms::count::count(&term, 2, &units).unwrap();
    assert_eq!(stats.length, [3, 0, 1, 0]);
    let (v, _) = turbo_chunker::terms::index::intern(&[Some("a"), Some("b")]);
    let local = LocalCollection::new(&stats, &v);
    assert_eq!((local.units(), local.mean_length(), local.df("a"), local.df("b")), (2, 2.0, 1, 2));
    let kept = LocalCollection::only(&stats, &v, |i| i.get() != 0);
    assert_eq!((kept.units(), kept.mean_length(), kept.df("a")), (1, 1.0, 0));
}

#[test]
fn a_mean_length_of_zero_counts_the_length_as_the_mean() {
    let m = Bm25::LUCENE;
    assert!(close(m.weight(2, 5, 0.0), 2.0 / (2.0 + 1.2)));
    assert!(close(m.weight(2, 5, 0.0), m.weight(2, 5, 5.0)));
    assert_eq!(m.weight(0, 5, 0.0), 0.0);
    assert!(m.weight(1, 0, 0.0).is_finite());
}

#[test]
fn a_token_edge_inside_an_edit_of_the_terms_view_is_refused() {
    // One edit turns "don't" into "do not" over the tokens "do" and "n't":
    // mapped, each would read the whole "do not".
    let text = "don't stop";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 2), (2, 5), (6, 10)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let mut out = ViewBuilder::new(text);
    out.replace(5, "do not");
    let (expanded, alignment) = out.finish();
    let view = View::derived(doc.get::<View>().unwrap(), expanded, alignment);
    doc.add(&Key::named("expanded"), view, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::named("expanded"))
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(TermIndexer).read(0, "expanded"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let err = pipeline.run(&mut docs)[0].clone().unwrap_err();
    assert!(err.to_string().contains("token 0 at 0..2 has an edge inside the edit of 0..5"), "{err}");
    assert!(err.to_string().contains("must not cross a token edge"), "{err}");
    assert!(docs[0].get::<Vocabulary>().is_none());
}

#[test]
fn an_edit_that_is_a_whole_token_or_an_insertion_is_fine() {
    // "can't" is one token and one edit; an insertion before "go" lies at
    // a token edge, not inside a token.
    let text = "can't go";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 5), (6, 8)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let mut out = ViewBuilder::new(text);
    out.replace(5, "cannot");
    out.keep(6);
    out.replace(6, "now ");
    let (changed, alignment) = out.finish();
    assert_eq!(changed, "cannot now go");
    let view = View::derived(doc.get::<View>().unwrap(), changed, alignment);
    doc.add(&Key::named("changed"), view, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::named("changed"))
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(TermIndexer).read(0, "changed"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let v = docs[0].get::<Vocabulary>().unwrap();
    assert_eq!(v.text.iter().collect::<Vec<_>>(), ["cannot", "go"]);
}

/// Pages with no spans: a layer `TermCounter` cannot place tokens in when
/// it has more than one item.
struct Pages(usize);

impl Layer for Pages {
    const ID: &'static str = "test:pages";
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        self.0
    }
}

#[test]
fn the_counter_refuses_tokens_out_of_order_and_many_units_with_no_spans() {
    let text = "ab cd";
    let counter = |tokens: &[(u32, u32)], pages: usize| {
        let mut doc = Document::new(text);
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(a, b) in tokens {
            spans.push(Span::new(a, b));
        }
        doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
        doc.add(&Key::DEFAULT, Pages(pages), Provenance::by("t")).unwrap();
        let pipeline = Analyzer::builder()
            .given(Key::<View>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Pages>::DEFAULT)
            .add(TermIndexer)
            .add(TermCounter::<Pages>::new())
            .build()
            .unwrap();
        let mut docs = vec![doc];
        let status = pipeline.run(&mut docs).pop().unwrap();
        (status, docs.pop().unwrap())
    };
    let (status, doc) = counter(&[(0, 2), (3, 5)], 1);
    assert_eq!(status, Ok(()));
    assert_eq!(doc.get::<TermStats<Pages>>().unwrap().length, [2]);
    let err = counter(&[(3, 5), (0, 2)], 1).0.unwrap_err();
    assert!(err.to_string().contains("the tokens are not in text order"), "{err}");
    let err = counter(&[(0, 2), (3, 5)], 2).0.unwrap_err();
    assert!(err.to_string().contains("2 units with no spans; only one can hold every token"), "{err}");
}
