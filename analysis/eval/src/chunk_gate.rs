//! The chunker gate (#172): does a chunker keep the evidence whole, at no
//! real cost to document ranking?
//!
//! The document gate (gate.rs) asks a chunker for +2 points of nDCG@10,
//! which measures which document ranks first. On sets whose documents are
//! one or two chunks under any chunker, boundaries barely move that, and
//! the ceiling (the better of two chunkers per query) is under +2. A
//! chunker is about what a reader gets back, so this gate asks:
//!
//! - **Evidence**, on every set with span judgements where chunking has
//!   something to decide (under `SHORT_SHARE` of its judged documents are
//!   one or two chunks under the candidate or under the reference,
//!   whichever splits more, so a candidate cannot excuse a set by keeping
//!   its documents whole): the candidate's span recall@10 and contained@10
//!   each beat the reference baseline (sentence windows) with the 95
//!   percent paired bootstrap interval above `EVIDENCE_LOW`, and it cuts
//!   through fewer judged spans than the reference. At least one such set
//!   is needed. Sets with mostly short documents are reported, not gated.
//! - **Floor**, on every set: against the baseline with the highest mean
//!   nDCG@10 on that set, the interval's high end of the candidate's
//!   difference is at least `FLOOR_POINTS`: never significantly worse by
//!   more than half a point.

use std::fmt;

use crate::Result;
use crate::record::{EvalRecord, SpanMetrics};
use crate::significance::{Bootstrap, RESAMPLES, SEED, bootstrap};

/// An evidence measure's interval must lie above this (fraction, 0 to 1).
pub const EVIDENCE_LOW: f64 = 0.0;
/// A set is gated on evidence when under this share of its judged
/// documents are one or two chunks under the candidate.
pub const SHORT_SHARE: f64 = 0.5;
/// The nDCG@10 floor, in points: the high end of the candidate's
/// difference from the best baseline may not be below it.
pub const FLOOR_POINTS: f64 = -0.5;

/// One evidence set's comparison with the reference baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    pub set: String,
    /// Whether the set is gated (else reported only).
    pub gated: bool,
    /// The smaller of the candidate's and the reference's shares of
    /// judged documents in one or two chunks.
    pub short_documents: f64,
    pub recall: Bootstrap,
    pub contained: Bootstrap,
    /// Shares of judged spans cut through, candidate and reference.
    pub split: (f64, f64),
    pub pass: bool,
}

/// One set's nDCG@10 floor against its best baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct Floor {
    pub set: String,
    pub baseline: String,
    /// In points.
    pub diff: Bootstrap,
    pub pass: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkGateReport {
    pub evidence: Vec<Evidence>,
    pub floors: Vec<Floor>,
    pub pass: bool,
}

/// The records of `set` in `records`.
fn of<'a>(records: &'a [EvalRecord], set: &str) -> Option<&'a EvalRecord> {
    records.iter().find(|r| r.set.name == set)
}

/// The paired differences of two per-query lists, which must name the
/// same queries in the same order.
fn paired<T>(a: &[T], b: &[T], key: impl Fn(&T) -> &str, value: impl Fn(&T) -> f64) -> Result<Vec<f64>> {
    if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| key(x) != key(y)) {
        return Err("the records score different queries".into());
    }
    Ok(a.iter().zip(b).map(|(x, y)| value(x) - value(y)).collect())
}

fn same_set(a: &EvalRecord, b: &EvalRecord) -> Result<()> {
    if a.set != b.set {
        return Err(format!(
            "set {}: the records hold different bytes ({} and {})",
            a.set.name, a.set.sha256, b.set.sha256
        ));
    }
    Ok(())
}

/// The gate of `candidate`'s records against the reference `windows`
/// and the other baselines; every baseline must hold every set the
/// candidate does.
pub fn chunk_gate(
    candidate: &[EvalRecord],
    windows: &[EvalRecord],
    others: &[&[EvalRecord]],
) -> Result<ChunkGateReport> {
    // The candidate is judged on every set the reference holds, once.
    let sets = |records: &[EvalRecord], who: &str| -> Result<Vec<String>> {
        let mut names: Vec<String> = records.iter().map(|r| r.set.name.clone()).collect();
        names.sort();
        if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
            return Err(format!("{who} holds set {} twice", w[0]));
        }
        Ok(names)
    };
    let (cs, ws) = (sets(candidate, "the candidate")?, sets(windows, "the reference")?);
    if cs != ws {
        return Err(format!("the candidate holds sets {cs:?} and the reference {ws:?}"));
    }
    for (k, o) in others.iter().enumerate() {
        sets(o, &format!("baseline {}", k + 1))?;
    }
    let mut report = ChunkGateReport { evidence: Vec::new(), floors: Vec::new(), pass: true };
    for c in candidate {
        let set = c.set.name.as_str();
        let w = of(windows, set).ok_or_else(|| format!("set {set}: no reference record"))?;
        same_set(c, w)?;
        if c.metrics.span.is_some() != w.metrics.span.is_some() {
            return Err(format!("set {set}: one record scores evidence and the other does not"));
        }
        if let (Some(cs), Some(ws)) = (&c.metrics.span, &w.metrics.span) {
            if cs.per_query.is_empty() || ws.per_query.is_empty() {
                return Err(format!(
                    "set {set}: a record holds no per-query evidence scores (made before #172's gate)"
                ));
            }
            let (Some(ci), Some(wi)) = (cs.integrity, ws.integrity) else {
                return Err(format!("set {set}: a record holds no evidence integrity (made before #172's gate)"));
            };
            let pq = |f: fn(&crate::record::QuerySpanAt10) -> f64| {
                paired(&cs.per_query, &ws.per_query, |q| q.query.as_str(), f)
                    .and_then(|d| bootstrap(&d, RESAMPLES, SEED))
                    .map_err(|e| format!("set {set}: {e}"))
            };
            let (recall, contained) = (pq(|q| q.recall)?, pq(|q| q.contained)?);
            // Gated unless both chunkers leave most judged documents in one
            // or two chunks: the candidate alone cannot demote a set by
            // keeping its documents whole.
            let short = ci.short_documents.min(wi.short_documents);
            let gated = short < SHORT_SHARE;
            let pass = recall.low > EVIDENCE_LOW && contained.low > EVIDENCE_LOW && ci.split < wi.split;
            report.evidence.push(Evidence {
                set: set.into(),
                gated,
                short_documents: short,
                recall,
                contained,
                split: (ci.split, wi.split),
                pass,
            });
        }
        let mut b = w;
        for (k, o) in others.iter().enumerate() {
            let o = of(o, set).ok_or_else(|| format!("set {set}: baseline {} has no record", k + 1))?;
            same_set(c, o)?;
            if o.metrics.ndcg_at_10 > b.metrics.ndcg_at_10 {
                b = o;
            }
        }
        let diffs = paired(&c.metrics.per_query, &b.metrics.per_query, |q| q.query.as_str(), |q| 100.0 * q.ndcg_at_10)
            .map_err(|e| format!("set {set}: {e}"))?;
        let diff = bootstrap(&diffs, RESAMPLES, SEED)?;
        report.floors.push(Floor {
            set: set.into(),
            baseline: b.pipeline.chunker.name.clone(),
            diff,
            pass: diff.high >= FLOOR_POINTS,
        });
    }
    let gated: Vec<&Evidence> = report.evidence.iter().filter(|e| e.gated).collect();
    report.pass = !gated.is_empty() && gated.iter().all(|e| e.pass) && report.floors.iter().all(|f| f.pass);
    Ok(report)
}

impl fmt::Display for ChunkGateReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pts = |b: &Bootstrap| format!("{:+.2} ({:+.2}..{:+.2})", 100.0 * b.mean, 100.0 * b.low, 100.0 * b.high);
        for e in &self.evidence {
            writeln!(
                f,
                "evidence {}{}: recall@10 {}, contained@10 {}, split {:.1}% vs {:.1}%, short documents {:.0}%{}",
                e.set,
                if e.gated { "" } else { " (reported, mostly short documents)" },
                pts(&e.recall),
                pts(&e.contained),
                100.0 * e.split.0,
                100.0 * e.split.1,
                100.0 * e.short_documents,
                if e.gated { if e.pass { "  pass" } else { "  FAIL" } } else { "" }
            )?;
        }
        for x in &self.floors {
            writeln!(
                f,
                "floor {} against {}: nDCG@10 {:+.2} ({:+.2}..{:+.2}){}",
                x.set,
                x.baseline,
                x.diff.mean,
                x.diff.low,
                x.diff.high,
                if x.pass { "" } else { "  below the floor" }
            )?;
        }
        let gated = self.evidence.iter().filter(|e| e.gated).count();
        write!(
            f,
            "{}: {} of {gated} gated evidence sets pass; {} of {} sets above the nDCG@10 floor {FLOOR_POINTS}",
            if self.pass { "PASS" } else { "FAIL" },
            self.evidence.iter().filter(|e| e.gated && e.pass).count(),
            self.floors.iter().filter(|x| x.pass).count(),
            self.floors.len()
        )
    }
}

fn in_unit(what: &str, v: f64) -> Result<()> {
    if (0.0..=1.0).contains(&v) { Ok(()) } else { Err(format!("{what} is {v}, outside [0, 1]")) }
}

/// The evidence fields the chunk gate reads: integrity shares in [0, 1],
/// and per-query scores (when present) one per span-scored query, in
/// byte order, in [0, 1], whose means are the record's at 10.
pub(crate) fn check_evidence(s: &SpanMetrics) -> Result<()> {
    if let Some(i) = s.integrity {
        in_unit("metrics.span.integrity.split", i.split)?;
        in_unit("metrics.span.integrity.short_documents", i.short_documents)?;
    }
    if s.per_query.is_empty() {
        return Ok(());
    }
    if s.per_query.len() != s.queries as usize {
        return Err(format!("metrics.span.queries is {} and per_query has {}", s.queries, s.per_query.len()));
    }
    for w in s.per_query.windows(2) {
        if w[0].query >= w[1].query {
            return Err(format!("metrics.span.per_query: {} is not before {}", w[0].query, w[1].query));
        }
    }
    for q in &s.per_query {
        in_unit(&format!("metrics.span.per_query {} recall", q.query), q.recall)?;
        in_unit(&format!("metrics.span.per_query {} contained", q.query), q.contained)?;
    }
    for (what, from, mean_at_10) in [
        ("recall", crate::metrics::mean(s.per_query.iter().map(|q| q.recall)), s.mean.at_10.recall),
        ("contained", crate::metrics::mean(s.per_query.iter().map(|q| q.contained)), s.mean.at_10.contained),
    ] {
        if (from - mean_at_10).abs() > 1e-9 {
            return Err(format!("metrics.span.mean.at_10.{what} is {mean_at_10} and the per query mean is {from}"));
        }
    }
    Ok(())
}

/// `turbo-eval chunk-gate <candidate> <windows> [<baseline> ...]`: prints
/// the report; exit status 1 when the gate fails.
pub fn command(args: &[String]) -> Result<std::process::ExitCode> {
    if args.len() < 2 {
        return Err("chunk-gate <candidate-dir> <windows-dir> [<baseline-dir> ...]".into());
    }
    let read = |p: &String| crate::run::read_records(std::path::Path::new(p));
    let (cand, windows) = (read(&args[0])?, read(&args[1])?);
    let others = args[2..].iter().map(read).collect::<Result<Vec<_>>>()?;
    let others: Vec<&[EvalRecord]> = others.iter().map(Vec::as_slice).collect();
    let report = chunk_gate(&cand, &windows, &others)?;
    println!("{report}");
    Ok(if report.pass { std::process::ExitCode::SUCCESS } else { std::process::ExitCode::FAILURE })
}

#[cfg(test)]
#[path = "chunk_gate_tests.rs"]
mod tests;
