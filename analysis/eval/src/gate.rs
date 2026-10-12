//! The gate a chunker change passes: candidate records against baseline
//! records, one of each per set.
//!
//! The candidate passes when its nDCG@10 is at least [`GAIN_POINTS`]
//! points above the baseline's on at least [`GAINS_NEEDED`] sets, and no
//! set is more than [`FLOOR_POINTS`] below. A point is 0.01 of nDCG@10.
//! Both rules look at the observed mean difference; the bootstrap
//! interval and the permutation p value are reported beside each
//! difference, not used to decide.
//!
//! The two sides must cover the same sets, each with the same SHA-256 and
//! the same scored queries, so each difference is paired query by query.

use std::collections::BTreeMap;
use std::fmt;

use crate::Result;
use crate::record::EvalRecord;
use crate::significance::{Bootstrap, RESAMPLES, SEED, bootstrap, permutation_p};

/// The gain, in nDCG@10 points, a set must show to count.
pub const GAIN_POINTS: f64 = 2.0;
/// The sets that must show it.
pub const GAINS_NEEDED: usize = 2;
/// The lowest difference, in points, any set may show.
pub const FLOOR_POINTS: f64 = -0.5;

/// One set's comparison. Every figure is in nDCG@10 points.
#[derive(Debug, Clone, PartialEq)]
pub struct SetComparison {
    pub set: String,
    pub queries: usize,
    pub baseline: f64,
    pub candidate: f64,
    /// The mean per query difference, candidate minus baseline, with its
    /// 95 percent bootstrap interval.
    pub difference: Bootstrap,
    /// The paired permutation test's two sided p value.
    pub p_value: f64,
    /// The difference is at least GAIN_POINTS.
    pub gained: bool,
    /// The difference is below FLOOR_POINTS.
    pub below_floor: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateReport {
    /// One comparison per set, in set name order.
    pub sets: Vec<SetComparison>,
    /// Sets that gained at least GAIN_POINTS.
    pub gains: usize,
    pub pass: bool,
}

fn by_set<'a>(side: &str, records: &'a [EvalRecord]) -> Result<BTreeMap<&'a str, &'a EvalRecord>> {
    let mut out = BTreeMap::new();
    for r in records {
        if out.insert(r.set.name.as_str(), r).is_some() {
            return Err(format!("the {side} has two records for set {}", r.set.name));
        }
    }
    Ok(out)
}

/// One set's per query differences in points, paired by query id.
fn paired_points(set: &str, candidate: &EvalRecord, baseline: &EvalRecord) -> Result<Vec<f64>> {
    let (c, b) = (&candidate.metrics.per_query, &baseline.metrics.per_query);
    if c.len() != b.len() || c.iter().zip(b).any(|(x, y)| x.query != y.query) {
        return Err(format!(
            "set {set}: the candidate scored {} queries and the baseline {}, not the same ones",
            c.len(),
            b.len()
        ));
    }
    Ok(c.iter().zip(b).map(|(x, y)| 100.0 * (x.ndcg_at_10 - y.ndcg_at_10)).collect())
}

/// Compares the candidate's records with the baseline's, set by set.
/// Each record is checked first.
pub fn gate(candidate: &[EvalRecord], baseline: &[EvalRecord]) -> Result<GateReport> {
    for r in candidate.iter().chain(baseline) {
        r.check().map_err(|e| format!("set {}: {e}", r.set.name))?;
    }
    let cand = by_set("candidate", candidate)?;
    let base = by_set("baseline", baseline)?;
    if cand.keys().ne(base.keys()) {
        return Err(format!(
            "the candidate has sets {:?} and the baseline {:?}",
            cand.keys().collect::<Vec<_>>(),
            base.keys().collect::<Vec<_>>()
        ));
    }
    if cand.is_empty() {
        return Err("no records to compare".into());
    }
    let mut sets = Vec::with_capacity(cand.len());
    for (name, c) in &cand {
        let b = base[name];
        if c.set.sha256 != b.set.sha256 {
            return Err(format!("set {name}: the candidate read {} and the baseline {}", c.set.sha256, b.set.sha256));
        }
        let diffs = paired_points(name, c, b)?;
        let difference = bootstrap(&diffs, RESAMPLES, SEED)?;
        let p_value = permutation_p(&diffs, RESAMPLES, SEED)?;
        sets.push(SetComparison {
            set: (*name).to_owned(),
            queries: diffs.len(),
            baseline: 100.0 * b.metrics.ndcg_at_10,
            candidate: 100.0 * c.metrics.ndcg_at_10,
            difference,
            p_value,
            gained: difference.mean >= GAIN_POINTS,
            below_floor: difference.mean < FLOOR_POINTS,
        });
    }
    let gains = sets.iter().filter(|s| s.gained).count();
    let pass = gains >= GAINS_NEEDED && !sets.iter().any(|s| s.below_floor);
    Ok(GateReport { sets, gains, pass })
}

impl fmt::Display for GateReport {
    /// One line per set, the difference with its interval beside it, and
    /// the verdict.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for s in &self.sets {
            let mark = if s.below_floor {
                "  below the floor"
            } else if s.gained {
                "  gain"
            } else {
                ""
            };
            writeln!(
                f,
                "{}: {:.2} -> {:.2}, {:+.2} points (95% interval {:+.2} to {:+.2}, p = {:.4}, {} queries){mark}",
                s.set,
                s.baseline,
                s.candidate,
                s.difference.mean,
                s.difference.low,
                s.difference.high,
                s.p_value,
                s.queries,
            )?;
        }
        let verdict = if self.pass { "PASS" } else { "FAIL" };
        write!(
            f,
            "{verdict}: {} of {} sets gained at least {GAIN_POINTS} points ({GAINS_NEEDED} needed), floor {FLOOR_POINTS}",
            self.gains,
            self.sets.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::tests::sample;

    /// Baseline scores for 20 queries, and the same shifted by `points`
    /// on every query.
    fn pair(set: &str, points: f64) -> (EvalRecord, EvalRecord) {
        let base: Vec<f64> = (0..20).map(|i| 0.3 + 0.01 * i as f64).collect();
        let cand: Vec<f64> = base.iter().map(|b| b + points / 100.0).collect();
        (sample(set, &cand), sample(set, &base))
    }

    fn run(shifts: &[(&str, f64)]) -> GateReport {
        let (c, b): (Vec<_>, Vec<_>) = shifts.iter().map(|&(s, p)| pair(s, p)).unzip();
        gate(&c, &b).unwrap()
    }

    #[test]
    fn two_gains_and_no_loss_pass() {
        // +3 and +2.5 on two sets, +0 and -0.4 on the other two: two
        // gains, nothing below -0.5.
        let r = run(&[("a", 3.0), ("b", 2.5), ("c", 0.0), ("d", -0.4)]);
        assert_eq!(r.gains, 2);
        assert!(r.pass, "{r}");
        // Every query moved by the same amount, so each interval is that
        // amount at both ends (up to rounding of the scores).
        let a = &r.sets[0];
        assert!((a.difference.mean - 3.0).abs() < 1e-9);
        assert!((a.difference.low - 3.0).abs() < 1e-9 && (a.difference.high - 3.0).abs() < 1e-9);
    }

    #[test]
    fn one_gain_fails() {
        let r = run(&[("a", 3.0), ("b", 1.9), ("c", 0.0), ("d", 0.0)]);
        assert_eq!(r.gains, 1);
        assert!(!r.pass);
    }

    #[test]
    fn a_loss_below_the_floor_fails() {
        let r = run(&[("a", 3.0), ("b", 3.0), ("c", 3.0), ("d", -0.6)]);
        assert_eq!(r.gains, 3);
        assert!(r.sets[3].below_floor);
        assert!(!r.pass);
        assert!(r.to_string().contains("below the floor"));
        assert!(r.to_string().ends_with("floor -0.5"));
    }

    #[test]
    fn mismatched_records_are_refused() {
        let (c, b) = pair("a", 1.0);
        let (c2, _) = pair("b", 1.0);
        assert!(gate(&[c.clone(), c2], std::slice::from_ref(&b)).is_err(), "sets differ");
        assert!(gate(&[c.clone(), c.clone()], &[b.clone(), b.clone()]).is_err(), "a set twice");
        let mut other = b.clone();
        other.set.sha256 = "e".repeat(64);
        assert!(gate(std::slice::from_ref(&c), &[other]).is_err(), "different set bytes");
        let fewer = sample("a", &[0.5]);
        assert!(gate(&[c], &[fewer]).is_err(), "different queries");
        assert!(gate(&[], &[]).is_err());
    }

    #[test]
    fn mixed_differences_give_an_interval_around_the_mean() {
        // Differences alternate +6 and -2 points: mean +2, a gain, with
        // an interval that is not a single point.
        let base: Vec<f64> = (0..20).map(|i| 0.3 + 0.01 * i as f64).collect();
        let cand: Vec<f64> =
            base.iter().enumerate().map(|(i, b)| b + if i.is_multiple_of(2) { 0.06 } else { -0.02 }).collect();
        let r = gate(&[sample("a", &cand)], &[sample("a", &base)]).unwrap();
        let d = r.sets[0].difference;
        assert!((d.mean - 2.0).abs() < 1e-9);
        assert!(d.low < d.mean && d.mean < d.high, "{d:?}");
        assert!(r.sets[0].p_value < 0.1, "{}", r.sets[0].p_value);
    }
}
