//! Equal-width top-label ECE, multiclass Brier (sum over classes), and
//! empirical set coverage. Abstention never counts as delivered coverage.
use crate::{Prediction, Result, check_distribution};
#[derive(Clone, Debug, PartialEq)]
pub struct Reliability {
    pub ece: f64,
    pub brier: f64,
    pub accuracy: f64,
    pub samples: usize,
}
pub fn reliability(rows: &[Vec<f64>], gold: &[usize], bins: usize) -> Result<Reliability> {
    let selected: Vec<_> = rows
        .iter()
        .map(|row| (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a))).unwrap_or(0))
        .collect();
    reliability_selected(rows, gold, &selected, bins)
}
/// Reliability of caller-selected labels, which need not be row maxima
/// when a structured decoder uses surrounding text. Brier still scores
/// the complete distribution; accuracy and ECE score the selected label.
pub fn reliability_selected(rows: &[Vec<f64>], gold: &[usize], selected: &[usize], bins: usize) -> Result<Reliability> {
    if rows.is_empty() || rows.len() != gold.len() || rows.len() != selected.len() || bins == 0 || bins > 65536 {
        return Err("invalid reliability inputs or bin count".into());
    }
    let k = rows[0].len();
    let mut counts = vec![0usize; bins];
    let mut conf = vec![0.0; bins];
    let mut correct = vec![0usize; bins];
    let mut brier = 0.0;
    let n = rows.len() as f64;
    for ((row, &y), &best) in rows.iter().zip(gold).zip(selected) {
        check_distribution(row)?;
        if row.len() != k || y >= k || best >= k {
            return Err("reliability label dimension mismatch".into());
        }
        let bin = ((row[best] * bins as f64) as usize).min(bins - 1);
        counts[bin] += 1;
        conf[bin] += row[best];
        correct[bin] += usize::from(best == y);
        for (j, &p) in row.iter().enumerate() {
            brier += (p - f64::from(j == y)).powi(2) / n;
        }
    }
    let ece = (0..bins).map(|i| (conf[i] - correct[i] as f64).abs() / n).sum();
    Ok(Reliability { ece, brier, accuracy: correct.iter().sum::<usize>() as f64 / n, samples: rows.len() })
}
#[derive(Clone, Debug, PartialEq)]
pub struct Coverage {
    pub set_coverage: f64,
    pub delivered_coverage: f64,
    pub abstention_rate: f64,
    pub mean_set_size: f64,
    pub samples: usize,
}
pub fn coverage(rows: &[Prediction], gold: &[usize]) -> Result<Coverage> {
    if rows.is_empty() || rows.len() != gold.len() {
        return Err("invalid coverage input lengths".into());
    }
    let (mut covered, mut delivered, mut abstained, mut size) = (0, 0, 0, 0.0);
    let k = rows[0].raw.len();
    for (r, &y) in rows.iter().zip(gold) {
        r.check()?;
        if y >= k || r.raw.len() != k {
            return Err("coverage label dimension mismatch".into());
        }
        let found = r.set.contains(&y);
        covered += usize::from(found);
        delivered += usize::from(found && r.abstention.is_none());
        abstained += usize::from(r.abstention.is_some());
        size += r.set.len() as f64 / rows.len() as f64;
    }
    let n = rows.len() as f64;
    Ok(Coverage {
        set_coverage: covered as f64 / n,
        delivered_coverage: delivered as f64 / n,
        abstention_rate: abstained as f64 / n,
        mean_set_size: size,
        samples: rows.len(),
    })
}
