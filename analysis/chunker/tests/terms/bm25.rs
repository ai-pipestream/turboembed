//! BM25 scores against a plain reference and Lucene's formula, the
//! collection the idf comes from, and ranking.

use super::*;

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
