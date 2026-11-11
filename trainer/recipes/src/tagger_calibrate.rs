//! Scalar temperature calibration for `TokenTagger` heads on held-out `dev` (#1888).
//!
//! Fits a positive temperature `T` minimizing token multiclass negative
//! log-likelihood on the corpus's `dev` split (never `test`), and reports
//! NLL, 15-bin top-label ECE, multiclass Brier score, accuracy, and
//! entity-token ECE before and after scaling.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use turbo::head::TokenTagger;
use turbo::manifest::HeadKind;

use crate::cache::{CacheRecord, same_corpus_release};
use crate::cli::{Args, commit};
use crate::corpus::FileRecord;
use crate::fit_cache::examples_at;
use crate::prior;
use crate::receipt::unix_time;
use crate::{Error, Result, read_corpus};

/// Schema identifier for serialized tagger calibration reports.
pub const SCHEMA: &str = "turboembed.trainer.tagger-calibration/1";

/// Number of equal-width reliability bins used for ECE.
pub const ECE_BINS: usize = 15;

const MIN_BETA: f64 = 0.05;
const MAX_BETA: f64 = 20.0;
const GOLDEN_STEPS: usize = 80;

/// Reliability and proper-scoring metrics at one temperature on held-out `dev`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationMetrics {
    pub nll: f64,
    pub ece: f64,
    pub entity_ece: f64,
    pub brier: f64,
    pub accuracy: f64,
}

/// Recorded held-out `dev` temperature calibration report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaggerCalibrationReport {
    pub schema: String,
    pub command: Vec<String>,
    pub commit: String,
    pub unix_time: u64,
    pub corpus_id: String,
    pub split: String,
    pub layer: u32,
    pub head_files: Vec<FileRecord>,
    pub dev_sentences: usize,
    pub dev_tokens: usize,
    pub dev_entity_tokens: usize,
    pub temperature: f64,
    pub uncalibrated: CalibrationMetrics,
    pub calibrated: CalibrationMetrics,
}

/// Mean multiclass negative log-likelihood on `[tokens, labels]` logits at
/// inverse temperature `beta = 1 / T`.
pub fn mean_nll(logits: &[f32], gold: &[u32], labels: usize, beta: f64) -> f64 {
    if gold.is_empty() || labels == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for (i, &y) in gold.iter().enumerate() {
        let row = &logits[i * labels..][..labels];
        let max = row.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64 * beta));
        let z: f64 = row.iter().map(|&v| (v as f64 * beta - max).exp()).sum();
        let log_p = (row[y as usize] as f64 * beta - max) - z.ln();
        sum -= log_p;
    }
    sum / gold.len() as f64
}

/// Fits positive scalar temperature `T > 0` minimizing [`mean_nll`] over
/// `beta = 1 / T` in `[MIN_BETA, MAX_BETA]`.
///
/// Multiclass cross-entropy is strictly convex in `beta`, so golden-section
/// search converges globally.
pub fn fit_temperature(logits: &[f32], gold: &[u32], labels: usize) -> Result<f64> {
    if gold.is_empty() || labels < 2 || logits.len() != gold.len() * labels {
        return Err(Error("fit_temperature requires non-empty [tokens, labels] logits with >= 2 labels".into()));
    }
    let phi = (5.0f64.sqrt() - 1.0) * 0.5;
    let (mut lo, mut hi) = (MIN_BETA, MAX_BETA);
    let mut c = hi - phi * (hi - lo);
    let mut d = lo + phi * (hi - lo);
    let mut fc = mean_nll(logits, gold, labels, c);
    let mut fd = mean_nll(logits, gold, labels, d);
    for _ in 0..GOLDEN_STEPS {
        if fc <= fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - phi * (hi - lo);
            fc = mean_nll(logits, gold, labels, c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + phi * (hi - lo);
            fd = mean_nll(logits, gold, labels, d);
        }
    }
    let beta = 0.5 * (lo + hi);
    Ok(1.0 / beta)
}

/// Evaluates NLL, 15-bin top-label ECE, entity-token ECE, Brier score, and
/// accuracy at `temperature`.
pub fn evaluate_metrics(
    logits: &[f32],
    gold: &[u32],
    labels: usize,
    outside: Option<u32>,
    temperature: f64,
) -> Result<(CalibrationMetrics, usize)> {
    if gold.is_empty()
        || labels < 2
        || logits.len() != gold.len() * labels
        || !temperature.is_finite()
        || temperature <= 0.0
    {
        return Err(Error("invalid calibration evaluation inputs".into()));
    }
    let beta = 1.0 / temperature;
    let n = gold.len() as f64;
    let mut nll = 0.0f64;
    let mut brier = 0.0f64;
    let mut correct_total = 0usize;
    let mut conf = [0.0f64; ECE_BINS];
    let mut corr = [0usize; ECE_BINS];
    let mut ent_conf = [0.0f64; ECE_BINS];
    let mut ent_corr = [0usize; ECE_BINS];
    let mut ent_count = 0usize;

    let mut probs = vec![0.0f64; labels];
    for (i, &y) in gold.iter().enumerate() {
        let row = &logits[i * labels..][..labels];
        let mut best = 0usize;
        for (j, &v) in row.iter().enumerate().skip(1) {
            if v > row[best] {
                best = j;
            }
        }
        let max = row.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64 * beta));
        let mut z = 0.0f64;
        for (p, &v) in probs.iter_mut().zip(row) {
            *p = (v as f64 * beta - max).exp();
            z += *p;
        }
        for p in &mut probs {
            *p /= z;
        }
        let py = probs[y as usize].max(f64::MIN_POSITIVE);
        nll -= py.ln();
        let p_best = probs[best];
        let hit = usize::from(best == y as usize);
        correct_total += hit;
        let bin = ((p_best * ECE_BINS as f64) as usize).min(ECE_BINS - 1);
        conf[bin] += p_best;
        corr[bin] += hit;
        if outside.is_none_or(|o| y != o || best as u32 != o) {
            ent_conf[bin] += p_best;
            ent_corr[bin] += hit;
            ent_count += 1;
        }
        for (j, &p) in probs.iter().enumerate() {
            let diff = p - f64::from(j == y as usize);
            brier += diff * diff;
        }
    }
    let ece = (0..ECE_BINS).map(|b| (conf[b] - corr[b] as f64).abs() / n).sum();
    let entity_ece = if ent_count == 0 {
        0.0
    } else {
        (0..ECE_BINS).map(|b| (ent_conf[b] - ent_corr[b] as f64).abs() / ent_count as f64).sum()
    };
    Ok((
        CalibrationMetrics { nll: nll / n, ece, entity_ece, brier: brier / n, accuracy: correct_total as f64 / n },
        ent_count,
    ))
}

/// Runs `turbo-train-head calibrate` on held-out `dev` features from `--cache`.
pub fn command(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let cache_dir = PathBuf::from(a.required("--cache")?);
    let run_dir = PathBuf::from(a.required("--run")?);
    let out_file = PathBuf::from(a.required("--out")?);
    let layer_arg: Option<u32> =
        a.value("--layer")?.map(|s| s.parse().map_err(|e| Error(format!("--layer: {e}")))).transpose()?;
    let commit = commit(a.value("--commit")?);
    a.done()?;

    let cache_bytes = std::fs::read(cache_dir.join("cache.json"))
        .map_err(|e| Error(format!("{}/cache.json: {e}", cache_dir.display())))?;
    let cache: CacheRecord =
        serde_json::from_slice(&cache_bytes).map_err(|e| Error(format!("{}/cache.json: {e}", cache_dir.display())))?;
    let corpus = read_corpus(corpus_id, dir, unrecorded)?;
    if !same_corpus_release(&corpus.record, &cache.corpus) {
        return Err(Error(format!("{}: cache is from another corpus or release", cache_dir.display())));
    }
    let layer = match layer_arg {
        Some(l) => l,
        None => *cache.layers.last().ok_or_else(|| Error("cache lists no layers".into()))?,
    };
    let prior = prior::read(&run_dir, &cache.encoder)?;
    if prior.spec.kind != HeadKind::TokenTagger {
        return Err(Error(format!("{}: expected HEAD_TOKEN_TAGGER, got {:?}", run_dir.display(), prior.spec.kind)));
    }
    let tagger = prior.head(|get| TokenTagger::from_tensors(&prior.spec, cache.encoder.hidden, get))?;
    let (dev, left_out) = examples_at(&cache_dir, &cache, "dev", &[layer], &corpus.splits.dev, &prior.spec.labels)?;
    if left_out != 0 {
        return Err(Error(format!("dev has {left_out} sentences with labels unknown to head")));
    }

    let labels = prior.spec.labels.len();
    let outside = prior.spec.labels.iter().position(|l| l == "O").map(|i| i as u32);
    let total_tokens: usize = dev.iter().map(|e| e.labels.len()).sum();
    let mut logits = Vec::with_capacity(total_tokens * labels);
    let mut gold = Vec::with_capacity(total_tokens);
    let (mut scores, mut work) = (Vec::new(), turbo::head::tagger::Work::default());
    for ex in &dev {
        let n = ex.labels.len();
        let word_ids: Vec<u32> = (0..n as u32).collect();
        let forms: Vec<&str> = ex.forms.iter().map(String::as_str).collect();
        tagger
            .scores_language(&ex.layers[0], &word_ids, Some(&forms), None, &mut scores, &mut work)
            .map_err(|e| Error(e.message))?;
        logits.extend_from_slice(&scores);
        gold.extend_from_slice(&ex.labels);
    }

    let temperature = fit_temperature(&logits, &gold, labels)?;
    let (uncalibrated, dev_entity_tokens) = evaluate_metrics(&logits, &gold, labels, outside, 1.0)?;
    let (calibrated, _) = evaluate_metrics(&logits, &gold, labels, outside, temperature)?;
    let report = TaggerCalibrationReport {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        corpus_id: corpus_id.to_owned(),
        split: "dev".into(),
        layer,
        head_files: prior.files,
        dev_sentences: dev.len(),
        dev_tokens: total_tokens,
        dev_entity_tokens,
        temperature,
        uncalibrated,
        calibrated,
    };
    let json = serde_json::to_string_pretty(&report).expect("calibration report serializes");
    if let Some(parent) = out_file.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| Error(format!("{}: {e}", parent.display())))?;
    }
    std::fs::write(&out_file, format!("{json}\n")).map_err(|e| Error(format!("{}: {e}", out_file.display())))?;
    println!(
        "{corpus_id}: dev ({total_tokens} tokens, {dev_entity_tokens} entity) T={temperature:.4} | NLL {:.5} -> {:.5} | ECE {:.5} -> {:.5} | entity ECE {:.5} -> {:.5}",
        uncalibrated.nll,
        calibrated.nll,
        uncalibrated.ece,
        calibrated.ece,
        uncalibrated.entity_ece,
        calibrated.entity_ece,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_scaling_recovers_overconfident_scale_and_reduces_nll_and_ece() {
        // True probabilities come from logits z, while the uncalibrated model
        // emits 2.5 * z (overconfident by factor 2.5, so optimal T ~ 2.5).
        let base_logits: [[f32; 3]; 8] = [
            [1.2, 0.2, -0.4],
            [1.2, 0.2, -0.4],
            [1.2, 0.2, -0.4],
            [1.2, 0.2, -0.4],
            [0.9, 0.6, -0.2],
            [0.9, 0.6, -0.2],
            [0.1, 1.1, -0.3],
            [-0.5, 0.2, 1.0],
        ];
        // Include imperfect outcomes matching moderate true confidence.
        let gold: [u32; 8] = [0, 0, 0, 1, 0, 1, 1, 2];
        let logits: Vec<f32> = base_logits.iter().flatten().map(|&z| z * 4.0).collect();

        let t = fit_temperature(&logits, &gold, 3).unwrap();
        assert!(t > 1.5 && t < 5.0, "expected T > 1.5 for overconfident logits, got {t}");
        let (raw, ent) = evaluate_metrics(&logits, &gold, 3, Some(0), 1.0).unwrap();
        let (cal, _) = evaluate_metrics(&logits, &gold, 3, Some(0), t).unwrap();
        assert_eq!(ent, 4);
        assert!(cal.nll < raw.nll, "calibrated NLL {} should beat raw {}", cal.nll, raw.nll);
        assert!(cal.ece < raw.ece, "calibrated ECE {} should beat raw {}", cal.ece, raw.ece);
        assert_eq!(cal.accuracy, raw.accuracy, "positive temperature preserves argmax accuracy");
    }
}
