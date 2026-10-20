//! Corpus statistics, the posting-list BM25 index and BM25 term weights,
//! checked against the full scan `Query` gives over every unit.

use std::marker::PhantomData;

use turbo_analysis::Ix;
use turbo_analysis_layers::{Chunks, Strs};
use turbo_chunker::terms::{
    Bm25, Bm25PostingIndexBuilder, Collection, CollectionStatistics, CollectionStatisticsBuilder, LocalCollection,
    Query, TermStats, Vocabulary, WeightedTerms,
};

/// One document: its vocabulary and the term counts of its units.
struct Doc {
    vocabulary: Vocabulary,
    stats: TermStats<Chunks>,
}

fn doc(units: &[Vec<&str>]) -> Doc {
    let mut words: Vec<&str> = units.iter().flatten().copied().collect();
    words.sort_unstable();
    words.dedup();
    let mut text = Strs::default();
    for w in &words {
        text.push(w);
    }
    let mut stats = TermStats {
        row: vec![0],
        term: Vec::new(),
        tf: Vec::new(),
        length: Vec::new(),
        df: vec![0; words.len()],
        _of: PhantomData,
    };
    for unit in units {
        let mut ids: Vec<usize> = unit.iter().map(|w| words.binary_search(w).unwrap()).collect();
        ids.sort_unstable();
        for (i, &t) in ids.iter().enumerate() {
            if i > 0 && ids[i - 1] == t {
                *stats.tf.last_mut().unwrap() += 1;
            } else {
                stats.term.push(Ix::new(t as u32));
                stats.tf.push(1);
                stats.df[t] += 1;
            }
        }
        stats.length.push(unit.len() as u32);
        stats.row.push(stats.term.len() as u32);
    }
    Doc { vocabulary: Vocabulary { text }, stats }
}

const WORDS: [&str; 24] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
    "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
];

/// A fixed corpus: skewed word choice so some terms are common and some
/// rare, units of 0 to 15 words, some repeated whole.
fn corpus(seed: u64, docs: usize) -> Vec<Doc> {
    let mut s = seed | 1;
    let mut next = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as usize
    };
    (0..docs)
        .map(|_| {
            let n = 1 + next() % 6;
            let mut units: Vec<Vec<&str>> = Vec::new();
            for _ in 0..n {
                if !units.is_empty() && next() % 7 == 0 {
                    units.push(units[0].clone());
                    continue;
                }
                let len = next() % 16;
                units.push((0..len).map(|_| WORDS[(next() % 24) * (next() % 24) / 24]).collect());
            }
            doc(&units)
        })
        .collect()
}

fn statistics(docs: &[Doc]) -> CollectionStatistics {
    let mut b = CollectionStatisticsBuilder::new();
    for d in docs {
        b.add(&d.stats, &d.vocabulary).unwrap();
    }
    b.build()
}

/// Every unit of every document scored by `Query`, best first, equal scores
/// to the earlier row, the first `k` kept.
fn full_scan(docs: &[Doc], model: Bm25, c: &impl Collection, query: &[&str], k: usize) -> Vec<(u32, f64)> {
    let mut all = Vec::new();
    let mut first = 0u32;
    for d in docs {
        let q = Query::new(model, c, &d.vocabulary, query);
        for (i, s) in q.rank(&d.stats, d.stats.units()) {
            all.push((first + i.get() as u32, s));
        }
        first += d.stats.length.len() as u32;
    }
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

const MODELS: [Bm25; 4] = [Bm25::LUCENE, Bm25 { k1: 0.9, b: 0.4 }, Bm25 { k1: 2.0, b: 1.0 }, Bm25 { k1: 1.2, b: 0.0 }];

fn queries() -> Vec<Vec<&'static str>> {
    vec![
        vec!["alpha"],
        vec!["omega"],
        vec!["beta", "gamma"],
        vec!["alpha", "alpha", "delta"],
        vec!["unknown", "eta", "theta", "iota"],
        vec!["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
        vec!["unknown"],
    ]
}

#[test]
fn one_document_gives_its_local_collection() {
    let docs = corpus(7, 1);
    let local = LocalCollection::new(&docs[0].stats, &docs[0].vocabulary);
    let stats = statistics(&docs);
    assert_eq!(stats.units(), local.units());
    assert_eq!(Collection::mean_length(&stats), local.mean_length());
    for w in WORDS.iter().chain(&["unknown"]) {
        assert_eq!(Collection::df(&stats, w), local.df(w), "{w}");
    }
}

#[test]
fn adding_and_merging_in_any_order_gives_the_same_bytes() {
    let docs = corpus(11, 12);
    let whole = statistics(&docs);
    let (mut a, mut b) = (CollectionStatisticsBuilder::new(), CollectionStatisticsBuilder::new());
    for (i, d) in docs.iter().enumerate().rev() {
        if i % 2 == 0 { a.add(&d.stats, &d.vocabulary).unwrap() } else { b.add(&d.stats, &d.vocabulary).unwrap() }
    }
    let (mut ab, mut ba) = (a.clone(), b.clone());
    ab.merge(b).unwrap();
    ba.merge(a).unwrap();
    assert_eq!(ab.build().to_bytes(), whole.to_bytes());
    assert_eq!(ba.build(), whole);
    assert_eq!(whole.units(), docs.iter().map(|d| d.stats.length.iter().filter(|&&l| l > 0).count() as u64).sum());
    assert!(whole.idf("alpha") > 0.0 && whole.idf("unknown") > whole.idf("alpha"));
}

#[test]
fn bytes_round_trip_and_anything_else_is_refused() {
    let stats = statistics(&corpus(3, 9));
    let bytes = stats.to_bytes();
    assert_eq!(CollectionStatistics::from_bytes(&bytes).unwrap(), stats);
    assert_eq!(CollectionStatistics::from_bytes(&statistics(&[]).to_bytes()).unwrap().units(), 0);
    for cut in 0..bytes.len() {
        assert!(CollectionStatistics::from_bytes(&bytes[..cut]).is_err(), "cut at {cut}");
    }
    let mut longer = bytes.clone();
    longer.push(0);
    let mut magic = bytes.clone();
    magic[7] = 2;
    // The first term's document frequency set to 0, then past the units.
    let first_df = 8 + 24 + 4 + stats.iter().next().unwrap().0.len();
    let mut zero = bytes.clone();
    zero[first_df..first_df + 8].copy_from_slice(&0u64.to_le_bytes());
    let mut past = bytes.clone();
    past[first_df..first_df + 8].copy_from_slice(&(stats.units() + 1).to_le_bytes());
    // A huge term count, and two terms swapped out of byte order.
    let mut count = bytes.clone();
    count[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut order = bytes.clone();
    let t = 8 + 24 + 4;
    order.swap(t, first_df + 8 + 4);
    let mut empty = Vec::from(&bytes[..8]);
    empty.extend_from_slice(&0u64.to_le_bytes());
    empty.extend_from_slice(&5u64.to_le_bytes());
    empty.extend_from_slice(&0u64.to_le_bytes());
    // Every unit holds a term, so the total length is at least the units.
    let mut short = bytes.clone();
    short[16..24].copy_from_slice(&(stats.units() - 1).to_le_bytes());
    for (what, b) in [
        ("trailing", longer),
        ("version", magic),
        ("zero df", zero),
        ("df", past),
        ("count", count),
        ("length", empty),
        ("short", short),
    ] {
        assert!(CollectionStatistics::from_bytes(&b).is_err(), "{what}");
    }
    let mut unsorted = statistics(&[doc(&[vec!["b", "a"]])]).to_bytes();
    let n = unsorted.len();
    unsorted.swap(8 + 24 + 4, n - 8 - 1);
    assert!(CollectionStatistics::from_bytes(&unsorted).is_err());
}

#[test]
fn top_k_is_the_full_scan_bit_for_bit() {
    let docs = corpus(42, 40);
    let mut builder = Bm25PostingIndexBuilder::new();
    let mut first = 0;
    for d in &docs {
        assert_eq!(builder.add(&d.stats, &d.vocabulary).unwrap(), first);
        first += d.stats.length.len() as u32;
    }
    let index = builder.build();
    assert_eq!(index.rows(), first as usize);
    assert_eq!(index.statistics(), &statistics(&docs));
    // A larger corpus the rows are part of, for top_k_with.
    let mut wide = corpus(42, 40);
    wide.extend(corpus(99, 25));
    let wide = statistics(&wide);
    for model in MODELS {
        for query in queries() {
            for k in [1, 3, 10, 1000] {
                let own = full_scan(&docs, model, index.statistics(), &query, k);
                assert_eq!(index.top_k(model, &query, k), own, "{model:?} {query:?} {k}");
                let corpus = full_scan(&docs, model, &wide, &query, k);
                assert_eq!(index.top_k_with(model, &wide, &query, k), corpus, "{model:?} {query:?} {k}");
            }
        }
    }
    assert!(index.top_k(Bm25::LUCENE, &["alpha"], 0).is_empty());
}

#[test]
fn equal_scores_go_to_the_earlier_row() {
    let docs = [doc(&[vec!["a", "b"], vec!["a", "b"]]), doc(&[vec!["c"], vec!["b", "a"]])];
    let mut builder = Bm25PostingIndexBuilder::new();
    for d in &docs {
        builder.add(&d.stats, &d.vocabulary).unwrap();
    }
    let index = builder.build();
    let rows: Vec<u32> = index.top_k(Bm25::LUCENE, &["a", "b"], 2).iter().map(|h| h.0).collect();
    assert_eq!(rows, [0, 1]);
    let all = index.top_k(Bm25::LUCENE, &["a"], 10);
    assert_eq!(all.iter().map(|h| h.0).collect::<Vec<_>>(), [0, 1, 3]);
    assert_eq!(index.postings("a"), &[(0, 1), (1, 1), (3, 1)]);
}

/// A collection that says every term is in more units than it holds, so
/// every idf is negative and the bounds cannot prune.
struct Odd;

impl Collection for Odd {
    fn units(&self) -> u64 {
        3
    }
    fn mean_length(&self) -> f64 {
        4.0
    }
    fn df(&self, _: &str) -> u64 {
        10
    }
}

#[test]
fn negative_idf_and_odd_models_still_match_the_full_scan() {
    let docs = corpus(5, 15);
    let mut builder = Bm25PostingIndexBuilder::new();
    for d in &docs {
        builder.add(&d.stats, &d.vocabulary).unwrap();
    }
    let index = builder.build();
    for model in [Bm25::LUCENE, Bm25 { k1: 1.2, b: 1.5 }, Bm25 { k1: 1.2, b: -0.5 }] {
        for query in queries() {
            assert_eq!(index.top_k_with(model, &Odd, &query, 4), full_scan(&docs, model, &Odd, &query, 4));
            let own = full_scan(&docs, model, index.statistics(), &query, 4);
            assert_eq!(index.top_k(model, &query, 4), own, "{model:?} {query:?}");
        }
    }
}

#[test]
fn weighted_terms_are_each_terms_share_of_the_score() {
    let docs = corpus(17, 6);
    let stats = statistics(&docs);
    for d in &docs {
        let w = WeightedTerms::bm25(&d.stats, &d.vocabulary, &stats, Bm25::LUCENE).unwrap();
        assert_eq!(turbo_analysis::Layer::check(&w), Ok(()));
        for i in d.stats.units() {
            let (terms, _) = d.stats.unit(i);
            for (t, &weight) in terms.iter().zip(w.unit(i.get())) {
                let q = Query::new(Bm25::LUCENE, &stats, &d.vocabulary, &[d.vocabulary.get(*t)]);
                assert_eq!(weight, q.score(&d.stats, i));
            }
        }
    }
}

#[test]
fn counts_that_would_pass_u64_are_refused_and_change_nothing() {
    let one = doc(&[vec!["alpha"]]);
    let mut b = CollectionStatisticsBuilder::new();
    b.add(&one.stats, &one.vocabulary).unwrap();
    // Doubling 63 times reaches 2^63 units; the 64th would wrap to zero.
    for _ in 0..63 {
        b.merge(b.clone()).unwrap();
    }
    let before = b.clone().build();
    assert_eq!(before.units(), 1 << 63);
    assert!(b.merge(b.clone()).is_err());
    assert!(b.add(&one.stats, &one.vocabulary).is_ok());
    let mut full = b.clone();
    for _ in 0..1000 {
        if full.add(&one.stats, &one.vocabulary).is_err() {
            break;
        }
    }
    assert!(b.clone().merge(full.clone()).is_err());
    let after = b.build();
    assert_eq!(after.units(), before.units() + 1);
    assert_eq!(CollectionStatistics::from_bytes(&after.to_bytes()).unwrap(), after);
}

#[test]
fn a_nan_model_ranks_as_the_full_scan_does() {
    let docs = corpus(23, 10);
    let mut builder = Bm25PostingIndexBuilder::new();
    for d in &docs {
        builder.add(&d.stats, &d.vocabulary).unwrap();
    }
    let index = builder.build();
    let model = Bm25 { k1: f64::NAN, b: 0.75 };
    for query in queries() {
        let got: Vec<(u32, u64)> = index.top_k(model, &query, 5).iter().map(|&(r, s)| (r, s.to_bits())).collect();
        let want: Vec<(u32, u64)> =
            full_scan(&docs, model, index.statistics(), &query, 5).iter().map(|&(r, s)| (r, s.to_bits())).collect();
        assert_eq!(got, want, "{query:?}");
    }
}

/// Counts no `TermIndexer` writes, each refused by
/// `TermStats::validate_with`: a df longer than the vocabulary, a df past
/// the units, a term past the vocabulary, a vocabulary out of order, and
/// an empty term that a unit counts.
fn broken() -> Vec<(&'static str, Doc)> {
    let base = || doc(&[vec!["alpha", "beta"], vec!["beta"]]);
    let mut long = base();
    long.stats.df.push(0);
    let mut over = base();
    over.stats.df[1] = 5;
    let mut past = base();
    past.stats.term[1] = Ix::new(7);
    let mut order = base();
    let mut text = Strs::default();
    text.push("beta");
    text.push("alpha");
    order.vocabulary = Vocabulary { text };
    // An empty term, counted: it sorts first, so the vocabulary is in
    // order, but statistics holding it would not decode.
    let empty = doc(&[vec!["", "alpha"]]);
    vec![
        ("df past the vocabulary", long),
        ("df past the units", over),
        ("term past the vocabulary", past),
        ("vocabulary out of order", order),
        ("an empty term", empty),
    ]
}

#[test]
fn invalid_term_stats_are_refused_before_anything_changes() {
    let good = doc(&[vec!["gamma"]]);
    let stats = statistics(&corpus(3, 4));
    for (what, d) in broken() {
        let mut b = CollectionStatisticsBuilder::new();
        b.add(&good.stats, &good.vocabulary).unwrap();
        let before = b.clone().build();
        assert!(b.add(&d.stats, &d.vocabulary).is_err(), "{what}: collection statistics");
        assert_eq!(b.build(), before, "{what}: the statistics changed");

        let mut p = Bm25PostingIndexBuilder::new();
        p.add(&good.stats, &good.vocabulary).unwrap();
        assert!(p.add(&d.stats, &d.vocabulary).is_err(), "{what}: posting index");
        // Nothing was added: the next rows start where they did.
        assert_eq!(p.add(&good.stats, &good.vocabulary), Ok(1), "{what}: the rows moved");
        assert_eq!(p.build().statistics().units(), 2, "{what}: the posting statistics changed");

        assert!(WeightedTerms::bm25(&d.stats, &d.vocabulary, &stats, Bm25::LUCENE).is_err(), "{what}: weights");
    }
}
