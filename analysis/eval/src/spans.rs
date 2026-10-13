//! Span level measures: how well the top k retrieved chunks cover the
//! byte spans judged relevant.
//!
//! A chunk is a byte range `[start, end)` of one document's source text,
//! in UTF-8 bytes (the offsets the analysis layers use). For one query and
//! a cutoff k, with R the union of the relevant spans and C the union of
//! the top k chunks' spans, both per document:
//!
//! - precision = |R ∩ C| / |C|, the share of retrieved bytes that are
//!   relevant; 0 when nothing was retrieved.
//! - recall = |R ∩ C| / |R|, the share of relevant bytes retrieved.
//! - contained = the relevant spans that lie wholly inside a single top k
//!   chunk, over all relevant spans. A span split across two chunks does
//!   not count, even when both are retrieved.
//!
//! Overlapping chunks or spans are counted once in the byte measures,
//! since they are unions. A span is relevant when its grade is above 0;
//! grades do not weight the bytes. Queries are handled as in
//! [`crate::metrics`]: a query with no relevant span is left out of the
//! mean, and a query with relevant spans and no chunks scores 0.
//!
//! Ties: chunks are ordered by score, highest first, then by document id
//! descending (as for documents), then by start and end ascending.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{RECALL_KS, Result};

/// A byte range `[start, end)` of a document's UTF-8 source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// A span; an error when it is empty or backwards.
    pub fn new(start: usize, end: usize) -> Result<Span> {
        if start >= end {
            return Err(format!("span [{start}, {end}) is empty"));
        }
        Ok(Span { start, end })
    }

    fn len(self) -> usize {
        self.end - self.start
    }

    fn contains(self, other: Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }
}

/// One span judgement: how relevant a span of `doc` is to `query`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanRel {
    pub query: String,
    pub doc: String,
    pub span: Span,
    pub grade: i32,
}

/// The relevant spans of a set, by query and document. Spans graded 0 or
/// below are dropped when read.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanRels {
    by_query: BTreeMap<String, BTreeMap<String, Vec<Span>>>,
}

impl SpanRels {
    /// Collects judgements. The same span judged twice for a query and
    /// document is an error.
    pub fn new(rows: impl IntoIterator<Item = SpanRel>) -> Result<SpanRels> {
        let mut seen = BTreeSet::new();
        let mut by_query: BTreeMap<String, BTreeMap<String, Vec<Span>>> = BTreeMap::new();
        for r in rows {
            Span::new(r.span.start, r.span.end).map_err(|e| format!("query {} document {}: {e}", r.query, r.doc))?;
            if !seen.insert((r.query.clone(), r.doc.clone(), r.span)) {
                return Err(format!(
                    "query {} judges span [{}, {}) of document {} twice",
                    r.query, r.span.start, r.span.end, r.doc
                ));
            }
            if r.grade > 0 {
                by_query.entry(r.query).or_default().entry(r.doc).or_default().push(r.span);
            }
        }
        Ok(SpanRels { by_query })
    }

    /// The relevant spans of `doc` for `query`, in the order they were
    /// given; empty when there are none.
    pub fn spans(&self, query: &str, doc: &str) -> &[Span] {
        self.by_query.get(query).and_then(|d| d.get(doc)).map_or(&[], Vec::as_slice)
    }
}

/// One retrieved chunk: a span of a document, and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkHit {
    pub doc: String,
    pub span: Span,
    pub score: f64,
}

/// The chunks a pipeline retrieved for one query, in any order.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRanking {
    pub query: String,
    pub hits: Vec<ChunkHit>,
}

/// A run of chunks: one ranking per query, each sorted by the tie rule.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRun {
    by_query: BTreeMap<String, Vec<ChunkHit>>,
}

impl ChunkRun {
    /// Checks and sorts the rankings. Refused: a query ranked twice, the
    /// same chunk twice in a ranking, an empty span, a score that is not
    /// finite.
    pub fn new(rankings: impl IntoIterator<Item = ChunkRanking>) -> Result<ChunkRun> {
        let mut by_query = BTreeMap::new();
        for r in rankings {
            let mut seen = BTreeSet::new();
            for h in &r.hits {
                let at = format!("query {}: chunk [{}, {}) of {}", r.query, h.span.start, h.span.end, h.doc);
                Span::new(h.span.start, h.span.end).map_err(|e| format!("{at}: {e}"))?;
                if !h.score.is_finite() {
                    return Err(format!("{at} has score {}", h.score));
                }
                if !seen.insert((h.doc.as_str(), h.span)) {
                    return Err(format!("{at} is ranked twice"));
                }
            }
            let mut hits = r.hits;
            hits.sort_by(chunk_order);
            if by_query.insert(r.query.clone(), hits).is_some() {
                return Err(format!("query {} is ranked twice", r.query));
            }
        }
        Ok(ChunkRun { by_query })
    }

    /// The query's chunks in rank order; empty when the run has none.
    pub fn hits(&self, query: &str) -> &[ChunkHit] {
        self.by_query.get(query).map_or(&[], Vec::as_slice)
    }
}

fn chunk_order(a: &ChunkHit, b: &ChunkHit) -> Ordering {
    b.score.total_cmp(&a.score).then_with(|| b.doc.cmp(&a.doc)).then_with(|| a.span.cmp(&b.span))
}

/// The three span measures at one cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanScores {
    pub precision: f64,
    pub recall: f64,
    pub contained: f64,
}

/// The span measures at each of [`RECALL_KS`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanAtK {
    pub at_1: SpanScores,
    pub at_5: SpanScores,
    pub at_10: SpanScores,
    pub at_100: SpanScores,
}

/// The span measures of one query.
#[derive(Debug, Clone, PartialEq)]
pub struct QuerySpanScores {
    pub query: String,
    pub scores: SpanAtK,
}

/// The span measures of a run: per query, in query byte order, and their
/// means.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanEval {
    pub per_query: Vec<QuerySpanScores>,
    pub mean: SpanAtK,
}

/// Sorts and merges spans into disjoint, ordered ranges.
fn union(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_unstable();
    let mut out: Vec<Span> = Vec::with_capacity(spans.len());
    for s in spans {
        match out.last_mut() {
            Some(last) if s.start <= last.end => last.end = last.end.max(s.end),
            _ => out.push(s),
        }
    }
    out
}

/// The bytes two merged unions share.
fn overlap(a: &[Span], b: &[Span]) -> usize {
    let (mut i, mut j, mut total) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let start = a[i].start.max(b[j].start);
        let end = a[i].end.min(b[j].end);
        if start < end {
            total += end - start;
        }
        if a[i].end < b[j].end {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

/// The span measures of `hits` (in rank order) at cutoff `k`, against the
/// query's relevant spans by document.
pub fn span_scores(hits: &[ChunkHit], relevant: &BTreeMap<String, Vec<Span>>, k: usize) -> SpanScores {
    let mut retrieved: BTreeMap<&str, Vec<Span>> = BTreeMap::new();
    for h in hits.iter().take(k) {
        retrieved.entry(h.doc.as_str()).or_default().push(h.span);
    }
    let retrieved: BTreeMap<&str, Vec<Span>> = retrieved.into_iter().map(|(d, s)| (d, union(s))).collect();
    let retrieved_bytes: usize = retrieved.values().flatten().map(|s| s.len()).sum();

    let (mut relevant_bytes, mut shared, mut spans, mut contained) = (0, 0, 0, 0);
    for (doc, rel) in relevant {
        let chunks = retrieved.get(doc.as_str()).map_or(&[][..], Vec::as_slice);
        let merged = union(rel.clone());
        relevant_bytes += merged.iter().map(|s| s.len()).sum::<usize>();
        shared += overlap(&merged, chunks);
        spans += rel.len();
        contained += rel.iter().filter(|&&s| hits.iter().take(k).any(|h| h.doc == *doc && h.span.contains(s))).count();
    }
    let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    SpanScores {
        precision: ratio(shared, retrieved_bytes),
        recall: ratio(shared, relevant_bytes),
        contained: ratio(contained, spans),
    }
}

fn at_k(mut f: impl FnMut(usize) -> SpanScores) -> SpanAtK {
    let [a, b, c, d] = RECALL_KS;
    SpanAtK { at_1: f(a), at_5: f(b), at_10: f(c), at_100: f(d) }
}

/// Scores every query with a relevant span. An error when none has one.
pub fn evaluate_spans(run: &ChunkRun, rels: &SpanRels) -> Result<SpanEval> {
    let per_query: Vec<QuerySpanScores> = rels
        .by_query
        .iter()
        .map(|(q, relevant)| QuerySpanScores {
            query: q.clone(),
            scores: at_k(|k| span_scores(run.hits(q), relevant, k)),
        })
        .collect();
    if per_query.is_empty() {
        return Err("no query has a relevant span".into());
    }
    let n = per_query.len() as f64;
    let mean_of = |pick: fn(&SpanAtK) -> SpanScores| {
        let mut sum = SpanScores { precision: 0.0, recall: 0.0, contained: 0.0 };
        for q in &per_query {
            let s = pick(&q.scores);
            sum.precision += s.precision;
            sum.recall += s.recall;
            sum.contained += s.contained;
        }
        SpanScores { precision: sum.precision / n, recall: sum.recall / n, contained: sum.contained / n }
    };
    let mean = SpanAtK {
        at_1: mean_of(|s| s.at_1),
        at_5: mean_of(|s| s.at_5),
        at_10: mean_of(|s| s.at_10),
        at_100: mean_of(|s| s.at_100),
    };
    Ok(SpanEval { per_query, mean })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(start: usize, end: usize) -> Span {
        Span::new(start, end).unwrap()
    }

    fn rel(query: &str, doc: &str, start: usize, end: usize, grade: i32) -> SpanRel {
        SpanRel { query: query.into(), doc: doc.into(), span: span(start, end), grade }
    }

    fn chunk(doc: &str, start: usize, end: usize, score: f64) -> ChunkHit {
        ChunkHit { doc: doc.into(), span: span(start, end), score }
    }

    #[test]
    fn coverage_by_hand() {
        // Relevant for q: d [10, 20) and d [40, 50), 20 bytes in all.
        // Chunks: d [0, 15) 0.9, d [35, 60) 0.8, e [0, 100) 0.7.
        //
        // k = 1: C = d [0, 15), 15 bytes. Shared = [10, 15) = 5 bytes.
        //   precision = 5/15, recall = 5/20, contained = 0/2.
        // k = 5 (all three): C = 15 + 25 + 100 = 140 bytes.
        //   Shared = [10, 15) + [40, 50) = 5 + 10 = 15 bytes.
        //   precision = 15/140, recall = 15/20,
        //   contained = 1/2 ([40, 50) is inside [35, 60); [10, 20) is
        //   inside no chunk).
        let rels = SpanRels::new([rel("q", "d", 10, 20, 1), rel("q", "d", 40, 50, 2)]).unwrap();
        let run = ChunkRun::new([ChunkRanking {
            query: "q".into(),
            hits: vec![chunk("e", 0, 100, 0.7), chunk("d", 35, 60, 0.8), chunk("d", 0, 15, 0.9)],
        }])
        .unwrap();
        let s = evaluate_spans(&run, &rels).unwrap().mean;
        assert_eq!(s.at_1, SpanScores { precision: 5.0 / 15.0, recall: 5.0 / 20.0, contained: 0.0 });
        assert_eq!(s.at_5, SpanScores { precision: 15.0 / 140.0, recall: 15.0 / 20.0, contained: 0.5 });
        assert_eq!(s.at_100, s.at_5);
    }

    #[test]
    fn overlapping_chunks_count_bytes_once() {
        // Relevant: d [0, 10). Chunks d [0, 6) and d [4, 10): the union
        // is [0, 10), 10 bytes, all relevant. precision = 10/10,
        // recall = 10/10, contained = 0/1 (no single chunk holds it).
        let rels = SpanRels::new([rel("q", "d", 0, 10, 1)]).unwrap();
        let run = ChunkRun::new([ChunkRanking {
            query: "q".into(),
            hits: vec![chunk("d", 0, 6, 0.9), chunk("d", 4, 10, 0.8)],
        }])
        .unwrap();
        let s = evaluate_spans(&run, &rels).unwrap().mean.at_5;
        assert_eq!(s, SpanScores { precision: 1.0, recall: 1.0, contained: 0.0 });
    }

    #[test]
    fn overlapping_relevant_spans_count_bytes_once() {
        // Relevant: d [0, 10) and d [5, 15): union [0, 15), 15 bytes.
        // Chunk d [0, 10): shared 10. precision = 10/10, recall = 10/15,
        // contained = 1/2 (only [0, 10)).
        let rels = SpanRels::new([rel("q", "d", 0, 10, 1), rel("q", "d", 5, 15, 1)]).unwrap();
        let run = ChunkRun::new([ChunkRanking { query: "q".into(), hits: vec![chunk("d", 0, 10, 1.0)] }]).unwrap();
        let s = evaluate_spans(&run, &rels).unwrap().mean.at_1;
        assert_eq!(s, SpanScores { precision: 1.0, recall: 10.0 / 15.0, contained: 0.5 });
    }

    #[test]
    fn ties_break_by_document_then_start() {
        // Same score: e before d (descending id), and within d, [0, 5)
        // before [5, 10). At k = 1 only e [0, 5) is retrieved, so the
        // relevant d [0, 5) is missed: recall 0.
        let rels = SpanRels::new([rel("q", "d", 0, 5, 1)]).unwrap();
        let run = ChunkRun::new([ChunkRanking {
            query: "q".into(),
            hits: vec![chunk("d", 5, 10, 0.5), chunk("d", 0, 5, 0.5), chunk("e", 0, 5, 0.5)],
        }])
        .unwrap();
        let order: Vec<(&str, usize)> = run.hits("q").iter().map(|h| (h.doc.as_str(), h.span.start)).collect();
        assert_eq!(order, [("e", 0), ("d", 0), ("d", 5)]);
        assert_eq!(evaluate_spans(&run, &rels).unwrap().mean.at_1.recall, 0.0);
    }

    #[test]
    fn empty_queries() {
        // q0 has only a grade 0 span: left out. qm has a relevant span
        // and no chunks: every measure 0. The mean is over qm alone.
        let rels = SpanRels::new([rel("q0", "d", 0, 5, 0), rel("qm", "d", 0, 5, 1)]).unwrap();
        let run = ChunkRun::new([]).unwrap();
        let e = evaluate_spans(&run, &rels).unwrap();
        assert_eq!(e.per_query.len(), 1);
        assert_eq!(e.mean.at_10, SpanScores { precision: 0.0, recall: 0.0, contained: 0.0 });

        let none = SpanRels::new([rel("q0", "d", 0, 5, 0)]).unwrap();
        assert!(evaluate_spans(&run, &none).is_err());
    }

    #[test]
    fn bad_input_is_refused() {
        assert!(Span::new(5, 5).is_err());
        assert!(Span::new(6, 5).is_err());
        assert!(SpanRels::new([rel("q", "d", 0, 5, 1), rel("q", "d", 0, 5, 2)]).is_err());
        let twice = ChunkRanking { query: "q".into(), hits: vec![chunk("d", 0, 5, 1.0), chunk("d", 0, 5, 0.5)] };
        assert!(ChunkRun::new([twice]).is_err());
        let bad = ChunkRanking {
            query: "q".into(),
            hits: vec![ChunkHit { doc: "d".into(), span: Span { start: 3, end: 1 }, score: 1.0 }],
        };
        assert!(ChunkRun::new([bad]).is_err());
    }
}
