//! Times BM25 top-k from posting lists against the full scan `Query` gives.
//!
//! `cargo run --release -p turbo-chunker --example bm25_posting_bench --
//! DOCS UNITS VOCABULARY K`: DOCS documents of UNITS units each, 40 words
//! a unit drawn from VOCABULARY words with a Zipf-like skew, from a fixed
//! LCG, and 200 queries of 1 to 4 words. Both answers are compared.

use std::marker::PhantomData;
use std::time::Instant;

use turbo_analysis::Ix;
use turbo_analysis_layers::{Chunks, Strs};
use turbo_chunker::terms::{Bm25, Bm25PostingIndexBuilder, Collection, Query, TermStats, Vocabulary};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    /// A word index with a Zipf-like skew: small indexes are common.
    fn word(&mut self, vocabulary: u64) -> u64 {
        let u = (self.next() % 1_000_000) as f64 / 1_000_000.0;
        ((vocabulary as f64).powf(u) as u64).saturating_sub(1).min(vocabulary - 1)
    }
}

fn document(rng: &mut Lcg, units: usize, vocabulary: u64) -> (Vocabulary, TermStats<Chunks>) {
    let rows: Vec<Vec<u64>> = (0..units).map(|_| (0..40).map(|_| rng.word(vocabulary)).collect()).collect();
    let mut words: Vec<u64> = rows.iter().flatten().copied().collect();
    let mut names: Vec<String> = Vec::new();
    words.sort_unstable();
    words.dedup();
    for w in &words {
        names.push(format!("w{w}"));
    }
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by(|&a, &b| names[a].cmp(&names[b]));
    let mut text = Strs::default();
    let mut rank = vec![0; names.len()];
    for (r, &i) in order.iter().enumerate() {
        text.push(&names[i]);
        rank[i] = r;
    }
    let mut stats = TermStats {
        row: vec![0],
        term: vec![],
        tf: vec![],
        length: vec![],
        df: vec![0; names.len()],
        _of: PhantomData,
    };
    for unit in &rows {
        let mut ids: Vec<usize> = unit.iter().map(|w| rank[words.binary_search(w).unwrap()]).collect();
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
    (Vocabulary { text }, stats)
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    if args.len() != 4 {
        eprintln!("usage: bm25_posting_bench DOCS UNITS VOCABULARY K, for example 2000 20 50000 10");
        std::process::exit(2);
    }
    let (docs, units, vocabulary, k) = (args[0], args[1], args[2] as u64, args[3]);
    let mut rng = Lcg(7);
    let corpus: Vec<_> = (0..docs).map(|_| document(&mut rng, units, vocabulary)).collect();
    let t = Instant::now();
    let mut builder = Bm25PostingIndexBuilder::new();
    for (v, s) in &corpus {
        builder.add(s, v).unwrap();
    }
    let index = builder.build();
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    let names: Vec<Vec<String>> =
        (0..200).map(|_| (0..1 + rng.next() % 4).map(|_| format!("w{}", rng.word(vocabulary))).collect()).collect();
    let queries: Vec<Vec<&str>> = names.iter().map(|q| q.iter().map(String::as_str).collect()).collect();
    let stats = index.statistics();
    let t = Instant::now();
    let mut scan = Vec::new();
    for q in &queries {
        let mut all = Vec::new();
        let mut first = 0u32;
        for (v, s) in &corpus {
            for (i, score) in Query::new(Bm25::LUCENE, stats, v, q).rank(s, s.units()) {
                all.push((first + i.get() as u32, score));
            }
            first += s.length.len() as u32;
        }
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        scan.push(all);
    }
    let scan_ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
    let t = Instant::now();
    let posting: Vec<_> = queries.iter().map(|q| index.top_k(Bm25::LUCENE, q, k)).collect();
    let posting_ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
    assert_eq!(scan, posting, "the posting index and the full scan disagree");
    println!(
        "docs={docs} units={} terms={} mean_length={:.1} k={k} build_ms={build_ms:.1} \
         scan_ms_per_query={scan_ms:.3} posting_ms_per_query={posting_ms:.4}",
        index.rows(),
        stats.terms(),
        stats.mean_length()
    );
}
