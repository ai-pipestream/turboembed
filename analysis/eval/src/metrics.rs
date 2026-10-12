//! Document level measures: nDCG@10 and recall@k.
//!
//! The rules, which follow `trec_eval` (and so `pytrec_eval`, which BEIR
//! reports with) except where noted:
//!
//! - A document is relevant to a query when its grade is above 0. The
//!   gain of a document is its grade, linear, and a grade of 0 or below
//!   gains nothing (`trec_eval`'s `ndcg_cut`, not the `2^grade - 1` gain
//!   some papers use).
//! - The discount at rank `r` (1 based) is `log2(r + 1)`. The ideal DCG
//!   is the DCG of the query's judged documents sorted by grade, best
//!   first, cut at the same depth.
//! - Recall@k is the relevant documents in the top k over all relevant
//!   documents of the query, not over `min(k, relevant)`.
//! - Ties: hits are ordered by score, highest first, and hits with the
//!   same score by document id in descending byte order. This is the
//!   order `trec_eval` uses, so a run scored here and there agrees when
//!   scores tie. The order the hits were given in does not matter.
//! - A query whose judgements hold no relevant document is left out of
//!   the mean, as `trec_eval` leaves them out, since nDCG and recall are not
//!   defined for it. A query in the run but not in the judgements is left
//!   out too.
//! - A query with relevant documents that the run has no ranking for
//!   scores 0 on every measure and counts in the mean. Here this differs
//!   from `trec_eval`'s default, which skips it (its `-c` flag counts
//!   it): a pipeline that returns nothing for a query has failed on it.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{RECALL_KS, Result};

/// The depth nDCG is cut at.
pub const NDCG_DEPTH: usize = 10;

/// One retrieved document and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub doc: String,
    pub score: f64,
}

/// What a pipeline retrieved for one query, in any order.
#[derive(Debug, Clone, PartialEq)]
pub struct Ranking {
    pub query: String,
    pub hits: Vec<Hit>,
}

/// One judgement: how relevant `doc` is to `query`.
#[derive(Debug, Clone, PartialEq)]
pub struct Qrel {
    pub query: String,
    pub doc: String,
    pub grade: i32,
}

/// The judgements of a set, by query and then by document.
#[derive(Debug, Clone, PartialEq)]
pub struct Qrels {
    by_query: BTreeMap<String, BTreeMap<String, i32>>,
}

impl Qrels {
    /// Collects judgements. A (query, document) pair judged twice is an
    /// error, even with the same grade, since it means the set was read
    /// wrong.
    pub fn new(rows: impl IntoIterator<Item = Qrel>) -> Result<Qrels> {
        let mut by_query: BTreeMap<String, BTreeMap<String, i32>> = BTreeMap::new();
        for q in rows {
            let docs = by_query.entry(q.query.clone()).or_default();
            if docs.insert(q.doc.clone(), q.grade).is_some() {
                return Err(format!("query {} judges document {} twice", q.query, q.doc));
            }
        }
        Ok(Qrels { by_query })
    }

    /// The grade of `doc` for `query`; 0 when it was not judged.
    pub fn grade(&self, query: &str, doc: &str) -> i32 {
        self.by_query.get(query).and_then(|d| d.get(doc)).copied().unwrap_or(0)
    }

    /// The queries with at least one relevant document, in byte order:
    /// the queries a mean is taken over.
    pub fn scored_queries(&self) -> impl Iterator<Item = &str> {
        self.by_query.iter().filter(|(_, d)| d.values().any(|&g| g > 0)).map(|(q, _)| q.as_str())
    }

    fn relevant_count(&self, query: &str) -> usize {
        self.by_query.get(query).map_or(0, |d| d.values().filter(|&&g| g > 0).count())
    }

    /// DCG of the best possible ranking of the query's judged documents.
    fn ideal_dcg(&self, query: &str, depth: usize) -> f64 {
        let mut grades: Vec<i32> =
            self.by_query.get(query).map_or_else(Vec::new, |d| d.values().copied().filter(|&g| g > 0).collect());
        grades.sort_unstable_by(|a, b| b.cmp(a));
        grades.iter().take(depth).enumerate().map(|(i, &g)| f64::from(g) / discount(i)).sum()
    }
}

/// A run: one ranking per query, each sorted by the tie rule above.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    by_query: BTreeMap<String, Vec<Hit>>,
}

impl Run {
    /// Checks and sorts the rankings. Refused: a query ranked twice, a
    /// document twice in one ranking, and a score that is not finite.
    pub fn new(rankings: impl IntoIterator<Item = Ranking>) -> Result<Run> {
        let mut by_query = BTreeMap::new();
        for r in rankings {
            let mut seen = BTreeSet::new();
            for h in &r.hits {
                if !h.score.is_finite() {
                    return Err(format!("query {}: document {} has score {}", r.query, h.doc, h.score));
                }
                if !seen.insert(h.doc.as_str()) {
                    return Err(format!("query {}: document {} is ranked twice", r.query, h.doc));
                }
            }
            let mut hits = r.hits;
            hits.sort_by(rank_order);
            if by_query.insert(r.query.clone(), hits).is_some() {
                return Err(format!("query {} is ranked twice", r.query));
            }
        }
        Ok(Run { by_query })
    }

    /// The query's hits in rank order; empty when the run has none.
    pub fn hits(&self, query: &str) -> &[Hit] {
        self.by_query.get(query).map_or(&[], Vec::as_slice)
    }
}

/// Score descending, then document id descending.
fn rank_order(a: &Hit, b: &Hit) -> Ordering {
    b.score.total_cmp(&a.score).then_with(|| b.doc.cmp(&a.doc))
}

/// The discount at 0 based position `i`: `log2(i + 2)`.
fn discount(i: usize) -> f64 {
    ((i + 2) as f64).log2()
}

/// Recall at each of [`RECALL_KS`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallAtK {
    pub at_1: f64,
    pub at_5: f64,
    pub at_10: f64,
    pub at_100: f64,
}

impl RecallAtK {
    fn from_fn(mut f: impl FnMut(usize) -> f64) -> RecallAtK {
        let [a, b, c, d] = RECALL_KS;
        RecallAtK { at_1: f(a), at_5: f(b), at_10: f(c), at_100: f(d) }
    }

    /// The mean of each field, summed in the order given.
    pub fn mean<'a>(items: impl ExactSizeIterator<Item = &'a RecallAtK>) -> RecallAtK {
        let n = items.len() as f64;
        let mut sum = RecallAtK { at_1: 0.0, at_5: 0.0, at_10: 0.0, at_100: 0.0 };
        for r in items {
            sum.at_1 += r.at_1;
            sum.at_5 += r.at_5;
            sum.at_10 += r.at_10;
            sum.at_100 += r.at_100;
        }
        RecallAtK { at_1: sum.at_1 / n, at_5: sum.at_5 / n, at_10: sum.at_10 / n, at_100: sum.at_100 / n }
    }
}

/// The document level measures of one query.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryScores {
    pub query: String,
    pub ndcg_at_10: f64,
    pub recall: RecallAtK,
}

/// The document level measures of a run: per query, in query byte order,
/// and their means.
#[derive(Debug, Clone, PartialEq)]
pub struct DocScores {
    pub per_query: Vec<QueryScores>,
    pub ndcg_at_10: f64,
    pub recall: RecallAtK,
}

/// nDCG of the hits at `depth`, against the query's judgements.
pub fn ndcg(hits: &[Hit], qrels: &Qrels, query: &str, depth: usize) -> f64 {
    let ideal = qrels.ideal_dcg(query, depth);
    if ideal == 0.0 {
        return 0.0;
    }
    let dcg: f64 = hits
        .iter()
        .take(depth)
        .enumerate()
        .map(|(i, h)| f64::from(qrels.grade(query, &h.doc).max(0)) / discount(i))
        .sum();
    dcg / ideal
}

/// Recall of the hits at `k`.
pub fn recall(hits: &[Hit], qrels: &Qrels, query: &str, k: usize) -> f64 {
    let relevant = qrels.relevant_count(query);
    if relevant == 0 {
        return 0.0;
    }
    let found = hits.iter().take(k).filter(|h| qrels.grade(query, &h.doc) > 0).count();
    found as f64 / relevant as f64
}

/// Scores every query that has a relevant document. An error when no
/// query has one, since there is then nothing to take a mean of.
pub fn evaluate(run: &Run, qrels: &Qrels) -> Result<DocScores> {
    let per_query: Vec<QueryScores> = qrels
        .scored_queries()
        .map(|q| {
            let hits = run.hits(q);
            QueryScores {
                query: q.to_owned(),
                ndcg_at_10: ndcg(hits, qrels, q, NDCG_DEPTH),
                recall: RecallAtK::from_fn(|k| recall(hits, qrels, q, k)),
            }
        })
        .collect();
    if per_query.is_empty() {
        return Err("no query has a relevant document".into());
    }
    let ndcg_at_10 = mean(per_query.iter().map(|s| s.ndcg_at_10));
    let recall = RecallAtK::mean(per_query.iter().map(|s| &s.recall));
    Ok(DocScores { per_query, ndcg_at_10, recall })
}

/// The mean of the values, summed in the order given. Every mean in this
/// crate is taken this way, so a mean recomputed from a record's per
/// query values gives the same bits.
pub fn mean(values: impl ExactSizeIterator<Item = f64>) -> f64 {
    let n = values.len() as f64;
    values.sum::<f64>() / n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(doc: &str, score: f64) -> Hit {
        Hit { doc: doc.into(), score }
    }

    fn qrel(query: &str, doc: &str, grade: i32) -> Qrel {
        Qrel { query: query.into(), doc: doc.into(), grade }
    }

    fn close(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
    }

    #[test]
    fn graded_ndcg_and_recall_by_hand() {
        // Judgements for q1: d1 = 3, d2 = 2, d3 = 1, d4 = 0.
        // Ranking: d3 (0.9), d1 (0.8), dx (0.7), d2 (0.6), d4 (0.5).
        //
        // DCG@10 = 1/log2(2) + 3/log2(3) + 0/log2(4) + 2/log2(5) + 0/log2(6)
        //        = 1 + 1.892789261 + 0 + 0.861353116
        //        = 3.754142377
        // Ideal: d1, d2, d3 = 3/log2(2) + 2/log2(3) + 1/log2(4)
        //        = 3 + 1.261859507 + 0.5 = 4.761859507
        // nDCG@10 = 3.754142377 / 4.761859507 = 0.788377391
        //
        // Three relevant documents (d4 has grade 0):
        // recall@1 = 1/3 (d3), recall@5 = recall@10 = recall@100 = 3/3.
        let qrels =
            Qrels::new([qrel("q1", "d1", 3), qrel("q1", "d2", 2), qrel("q1", "d3", 1), qrel("q1", "d4", 0)]).unwrap();
        let run = Run::new([Ranking {
            query: "q1".into(),
            hits: vec![hit("d4", 0.5), hit("d2", 0.6), hit("dx", 0.7), hit("d1", 0.8), hit("d3", 0.9)],
        }])
        .unwrap();
        let s = evaluate(&run, &qrels).unwrap();
        close(s.ndcg_at_10, 0.788377391);
        close(s.recall.at_1, 1.0 / 3.0);
        close(s.recall.at_5, 1.0);
        close(s.recall.at_10, 1.0);
        close(s.recall.at_100, 1.0);
    }

    #[test]
    fn ndcg_cuts_at_ten() {
        // One relevant document of grade 1, at rank 11: past the cut, so
        // DCG@10 = 0 and nDCG@10 = 0, while recall@100 = 1/1 and
        // recall@10 = 0.
        let qrels = Qrels::new([qrel("q", "rel", 1)]).unwrap();
        let mut hits: Vec<Hit> = (0..10).map(|i| hit(&format!("n{i}"), 10.0 - i as f64)).collect();
        hits.push(hit("rel", -1.0));
        let run = Run::new([Ranking { query: "q".into(), hits }]).unwrap();
        let s = evaluate(&run, &qrels).unwrap();
        assert_eq!(s.ndcg_at_10, 0.0);
        assert_eq!(s.recall.at_10, 0.0);
        assert_eq!(s.recall.at_100, 1.0);
    }

    #[test]
    fn mean_over_two_queries_by_hand() {
        // qa: one relevant document (grade 2) at rank 2:
        //   DCG = 2/log2(3) = 1.261859507, ideal = 2/log2(2) = 2,
        //   nDCG = 0.630929754; recall@1 = 0, recall@5 = 1.
        // qb: one relevant document (grade 1) at rank 1: nDCG = 1,
        //   recall@1 = 1, recall@5 = 1.
        // Mean nDCG = (0.630929754 + 1) / 2 = 0.815464877.
        // Mean recall@1 = (0 + 1) / 2 = 0.5.
        let qrels = Qrels::new([qrel("qa", "a", 2), qrel("qb", "b", 1)]).unwrap();
        let run = Run::new([
            Ranking { query: "qa".into(), hits: vec![hit("x", 2.0), hit("a", 1.0)] },
            Ranking { query: "qb".into(), hits: vec![hit("b", 2.0), hit("y", 1.0)] },
        ])
        .unwrap();
        let s = evaluate(&run, &qrels).unwrap();
        close(s.per_query[0].ndcg_at_10, 0.630929754);
        close(s.per_query[1].ndcg_at_10, 1.0);
        close(s.ndcg_at_10, 0.815464877);
        close(s.recall.at_1, 0.5);
        close(s.recall.at_5, 1.0);
    }

    #[test]
    fn ties_break_by_document_id_descending() {
        // a and b score the same. The tie rule ranks b first (descending
        // id), so the relevant a is at rank 2: nDCG = (1/log2(3)) / 1 =
        // 0.630929754 and recall@1 = 0, whichever order they came in.
        let qrels = Qrels::new([qrel("q", "a", 1)]).unwrap();
        for hits in [vec![hit("a", 0.5), hit("b", 0.5)], vec![hit("b", 0.5), hit("a", 0.5)]] {
            let run = Run::new([Ranking { query: "q".into(), hits }]).unwrap();
            assert_eq!(run.hits("q")[0].doc, "b");
            let s = evaluate(&run, &qrels).unwrap();
            close(s.ndcg_at_10, 0.630929754);
            assert_eq!(s.recall.at_1, 0.0);
        }
    }

    #[test]
    fn queries_without_relevant_documents_are_left_out() {
        // q0 has only a grade 0 judgement and qr is not judged at all:
        // both are left out. qm has a relevant document and no ranking:
        // it scores 0 and counts. qok scores 1. Mean = (0 + 1) / 2.
        let qrels = Qrels::new([qrel("q0", "a", 0), qrel("qm", "m", 1), qrel("qok", "k", 1)]).unwrap();
        let run = Run::new([
            Ranking { query: "q0".into(), hits: vec![hit("a", 1.0)] },
            Ranking { query: "qr".into(), hits: vec![hit("a", 1.0)] },
            Ranking { query: "qok".into(), hits: vec![hit("k", 1.0)] },
        ])
        .unwrap();
        let s = evaluate(&run, &qrels).unwrap();
        let queries: Vec<&str> = s.per_query.iter().map(|q| q.query.as_str()).collect();
        assert_eq!(queries, ["qm", "qok"]);
        assert_eq!(s.per_query[0].ndcg_at_10, 0.0);
        assert_eq!(s.per_query[0].recall.at_100, 0.0);
        assert_eq!(s.ndcg_at_10, 0.5);
    }

    #[test]
    fn no_scored_query_is_an_error() {
        let qrels = Qrels::new([qrel("q", "a", 0)]).unwrap();
        let run = Run::new([]).unwrap();
        assert!(evaluate(&run, &qrels).is_err());
    }

    #[test]
    fn bad_input_is_refused() {
        assert!(Qrels::new([qrel("q", "a", 1), qrel("q", "a", 1)]).is_err());
        let twice = Ranking { query: "q".into(), hits: vec![hit("a", 1.0), hit("a", 0.5)] };
        assert!(Run::new([twice]).is_err());
        let nan = Ranking { query: "q".into(), hits: vec![hit("a", f64::NAN)] };
        assert!(Run::new([nan]).is_err());
        let q = Ranking { query: "q".into(), hits: vec![] };
        assert!(Run::new([q.clone(), q]).is_err());
    }

    #[test]
    fn negative_grades_gain_nothing() {
        // a = -1 (judged harmful), b = 1. Ranking a, b:
        // DCG = 0 + 1/log2(3), ideal = 1, nDCG = 0.630929754.
        let qrels = Qrels::new([qrel("q", "a", -1), qrel("q", "b", 1)]).unwrap();
        let run = Run::new([Ranking { query: "q".into(), hits: vec![hit("a", 2.0), hit("b", 1.0)] }]).unwrap();
        close(evaluate(&run, &qrels).unwrap().ndcg_at_10, 0.630929754);
    }
}
