//! BM25 retrieval by hand on the toy set the dense tests use.

use super::*;
use crate::analysis::{Analysis, Sentencer};
use crate::chunkers::SentenceWindows;
use crate::embedder::stub::Stub;
use crate::pipeline::tests::set;
use crate::pipeline::{Pipeline, Titles};
use crate::set::Query;

fn terms(rule: TermRule, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    rule.terms(text, &mut out);
    out
}

#[test]
fn folded_terms_are_lowercased_word_tokens_with_a_letter_or_digit() {
    let text = "The cat's 2 CATS, running ... over the Café's hills!";
    assert_eq!(terms(TermRule::Folded, text), ["the", "cat's", "2", "cats", "running", "over", "the", "café's", "hills"]);
    // Stop words out, the possessive dropped, Porter's stems.
    assert_eq!(terms(TermRule::English, text), ["cat", "2", "cat", "run", "over", "café", "hill"]);
    assert_eq!(terms(TermRule::English, "it\u{2019}s a test"), ["test"]);
    assert!(terms(TermRule::Folded, " ... !! ").is_empty());
}

fn pipeline(lexical: Lexical) -> Pipeline {
    Pipeline {
        analysis: Analysis { sentences: Sentencer::Newline, structure: None },
        chunker: Box::new(SentenceWindows::new(1).unwrap()),
        embedder: Box::new(Stub::new(&["cat", "dog", "rust", "tea"])),
        titles: Titles::Ignore,
        lexical: Some(lexical),
    }
}

const LUCENE: Lexical = Lexical { model: Bm25::LUCENE, terms: TermRule::Folded, dump: None };

/// Lucene's BM25, written out: idf * tf / (tf + k1 * (1 - b + b * len / mean)).
fn bm25(units: f64, df: f64, tf: f64, len: f64, mean: f64) -> f64 {
    let idf = (1.0 + (units - df + 0.5) / (df + 0.5)).ln();
    idf * tf / (tf + 1.2 * (1.0 - 0.75 + 0.75 * len / mean))
}

#[test]
fn documents_rank_by_their_best_chunk_s_bm25() {
    // Chunks, a sentence each: d1 "cat purrs", "rust compiles"; d2 "dog
    // barks", "cat sleeps with dog"; d3 "tea brews". Five rows of 2, 2, 2,
    // 4 and 2 terms, a mean of 2.4.
    let mut p = pipeline(LUCENE);
    let o = p.evaluate(&set(None)).unwrap();
    let ranked = |q: &str| o.run.hits(q).iter().map(|h| (h.doc.clone(), h.score)).collect::<Vec<_>>();
    // q1 "cat": in 2 of 5 rows; d1's short chunk beats d2's long one, and d3
    // holds no "cat", so it is not ranked.
    let q1 = ranked("q1");
    assert_eq!(q1.iter().map(|h| h.0.as_str()).collect::<Vec<_>>(), ["d1", "d2"]);
    assert!((q1[0].1 - bm25(5.0, 2.0, 1.0, 2.0, 2.4)).abs() < 1e-12, "{}", q1[0].1);
    assert!((q1[1].1 - bm25(5.0, 2.0, 1.0, 4.0, 2.4)).abs() < 1e-12, "{}", q1[1].1);
    // q2 "dog": d2's "dog barks" (2 terms) beats its other chunk.
    let q2 = ranked("q2");
    assert_eq!(q2.len(), 1);
    assert!((q2[0].1 - bm25(5.0, 2.0, 1.0, 2.0, 2.4)).abs() < 1e-12);
    // q3 "rust tea": d1 and d3 score the same; the tie goes to the higher id.
    assert_eq!(ranked("q3").iter().map(|h| h.0.as_str()).collect::<Vec<_>>(), ["d3", "d1"]);
    // The chunks of q1 follow: d1 [0, 9), then d2 [10, 29).
    let spans: Vec<_> = o.chunks.hits("q1").iter().map(|h| (h.doc.clone(), h.span)).collect();
    let s = |a, b| Span::new(a, b).unwrap();
    assert_eq!(spans, [("d1".to_string(), s(0, 9)), ("d2".to_string(), s(10, 29))]);
    // q1 finds d2 second, q2 first, q3 d1 second: nDCG@10 as the dense
    // test's, with nothing embedded.
    assert!((o.metrics.ndcg_at_10 - (2.0 * 0.630_929_753_571_457_4 + 1.0) / 3.0).abs() < 1e-9);
    assert_eq!(o.cost.tokens_embedded, 0);
    assert_eq!(o.cost.wall.embedding_ms, 0.0);
}

#[test]
fn the_query_s_own_document_is_left_out() {
    let mut s = set(None);
    s.queries.push(Query { id: "d2".into(), text: "dog cat".into() });
    let o = pipeline(LUCENE).evaluate(&s).unwrap();
    // d2 matches best but is the query itself; d1 is left.
    assert_eq!(o.run.hits("d2").iter().map(|h| h.doc.as_str()).collect::<Vec<_>>(), ["d1"]);
    assert!(o.chunks.hits("d2").iter().all(|h| h.doc != "d2"));
}

#[test]
fn the_record_names_the_retrieval_and_a_dense_one_names_none() {
    let lexical = Lexical { model: Bm25 { k1: 0.9, b: 0.4 }, terms: TermRule::English, dump: None };
    let d = pipeline(lexical).description();
    let r = d.retrieval.unwrap();
    assert_eq!(r.name, "bm25");
    let config: Vec<(String, String)> = r.config.into_iter().map(|s| (s.name, s.value)).collect();
    assert_eq!(config, [("k1".into(), "0.9".into()), ("b".into(), "0.4".into()), ("terms".into(), "english".into())]);
    let mut dense = pipeline(LUCENE);
    dense.lexical = None;
    let d = dense.description();
    assert!(d.retrieval.is_none());
    // A dense pipeline's record has no retrieval key, as before.
    assert!(!serde_json::to_string(&d).unwrap().contains("retrieval"));
}
