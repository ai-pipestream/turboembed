//! Precision, recall and identity-term bias of the incivility annotator
//! on real data (#323, #889).
//!
//! Embeds the Civil Comments splits and the English bias madlibs with one
//! encoder bundle, fits an `IncivilityVectorHead` to the train split's
//! labels (0 and 1 from the rater fractions by default), chooses each
//! stage's threshold by micro F1 on the val split, scores the test split
//! and the madlibs with the lexicon, learned and fused stages at 0.5 and
//! at that threshold, and writes `evaluate_incivility_corpus` (precision,
//! recall, F1 and evidence coverage) and `evaluate_incivility_bias` for
//! each to a JSON file.
//!
//! ```text
//! cargo run --release -p turbo-analysis-embed --example incivility_bias -- \
//!     --bundle <bundle dir> --device cuda --civil <dir> --madlibs <dir> \
//!     --cache <dir> --out <file.json>
//! ```
//!
//! `--civil` holds `all_data_with_identities.csv` (the WILDS release of
//! Civil Comments with identity columns) and `--madlibs` the
//! `en_sentence_templates.csv` and `words.csv` of
//! conversationai/unintended-ml-bias-analysis; `docs/sentiment.md` names
//! the revisions and digests.

mod data;
mod embed;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};
use turbo_analysis::Ix;
use turbo_analysis_embed::{Device, EncoderOptions};
use turbo_analysis_layers::{Incivility, IncivilityItem, IncivilityLabel, IncivilityScores, IncivilityTarget};
use turbo_doccat::{
    GoldBiasUnit, GoldIncivilityUnit, IncivilityAnnotator, IncivilityBiasMetrics, IncivilityFitOptions,
    IncivilityMetrics, IncivilityStage, IncivilityVectorHead, IncivilityVectorScorer, evaluate_incivility_bias,
    evaluate_incivility_corpus,
};

use data::Unit;

/// The thresholds `tune` tries on the validation split.
const THRESHOLDS: [f64; 10] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];

const STAGES: [(&str, IncivilityStage); 3] =
    [("lexicon", IncivilityStage::Lexicon), ("learned", IncivilityStage::Learned), ("fused", IncivilityStage::Fused)];

struct Args {
    bundle: PathBuf,
    device: Device,
    civil: PathBuf,
    madlibs: PathBuf,
    cache: PathBuf,
    out: PathBuf,
    fit: IncivilityFitOptions,
    /// Fit to 0 and 1 (a rater fraction at 0.5 and above); with
    /// `--fraction-targets`, to the fractions themselves.
    hard: bool,
}

fn args() -> Result<Args, String> {
    let mut a = std::env::args().skip(1);
    let (mut bundle, mut device, mut civil, mut madlibs, mut cache, mut out) =
        (None, Device::Cpu, None, None, None, None);
    let mut fit = IncivilityFitOptions { learning_rate: 16.0, ..Default::default() };
    let mut hard = true;
    while let Some(flag) = a.next() {
        let mut value = || a.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(value()?)),
            "--device" => {
                device = match value()?.as_str() {
                    "cpu" => Device::Cpu,
                    d => Device::Backend(d.into()),
                }
            }
            "--civil" => civil = Some(PathBuf::from(value()?)),
            "--madlibs" => madlibs = Some(PathBuf::from(value()?)),
            "--cache" => cache = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--epochs" => fit.epochs = value()?.parse().map_err(|e| format!("--epochs: {e}"))?,
            "--learning-rate" => fit.learning_rate = value()?.parse().map_err(|e| format!("--learning-rate: {e}"))?,
            "--fraction-targets" => hard = false,
            "--l2" => fit.l2 = value()?.parse().map_err(|e| format!("--l2: {e}"))?,
            f => return Err(format!("unknown flag {f}")),
        }
    }
    let need = |v: Option<PathBuf>, f: &str| v.ok_or_else(|| format!("{f} is required"));
    Ok(Args {
        bundle: need(bundle, "--bundle")?,
        device,
        civil: need(civil, "--civil")?,
        madlibs: need(madlibs, "--madlibs")?,
        cache: need(cache, "--cache")?,
        out: need(out, "--out")?,
        fit,
        hard,
    })
}

/// The annotator's own token form: trimmed of all but letters, digits and
/// the apostrophe, then lowercased.
fn token_lower(text: &str) -> Vec<String> {
    data::tokens(text)
        .into_iter()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_ascii_lowercase())
        .collect()
}

/// Each unit's tokens in the annotator's form, and its gold labels and
/// subgroups, a label positive at 0.5 and above.
struct Prepared {
    tokens: Vec<Vec<String>>,
    gold: Vec<GoldBiasUnit>,
    units: Vec<GoldIncivilityUnit>,
}

fn prepare(units: &[Unit]) -> Prepared {
    let gold: Vec<GoldBiasUnit> = units
        .iter()
        .map(|u| GoldBiasUnit {
            positive_labels: IncivilityLabel::ALL.into_iter().filter(|&l| u.targets.get(l) >= 0.5).collect(),
            subgroups: u.subgroups.clone(),
        })
        .collect();
    Prepared {
        tokens: units.iter().map(|u| token_lower(&u.text)).collect(),
        units: gold.iter().map(|g| GoldIncivilityUnit { positive_labels: g.positive_labels.clone() }).collect(),
        gold,
    }
}

/// The layer `annotator` writes for the units, one sentence each.
fn score(annotator: &IncivilityAnnotator, p: &Prepared, vectors: &[f32], dim: usize) -> Result<Incivility, String> {
    let mut layer = Incivility::new();
    for (i, tokens) in p.tokens.iter().enumerate() {
        let v = &vectors[i * dim..(i + 1) * dim];
        let (scores, evidence) = annotator.score_unit_in_context(tokens, 0, tokens.len(), None, Some(v), None)?;
        layer.push(IncivilityItem {
            target: IncivilityTarget::Sentence(Ix::new(i as u32)),
            scores,
            evidence: &evidence,
        });
    }
    Ok(layer)
}

/// Precision, recall and bias of the units at `threshold`.
fn measure(annotator: &IncivilityAnnotator, p: &Prepared, vectors: &[f32], dim: usize) -> Result<Value, String> {
    let layer = score(annotator, p, vectors, dim)?;
    let t = annotator.threshold;
    Ok(json!({
        "threshold": t,
        "precision_recall": corpus_json(&evaluate_incivility_corpus(&p.units, &layer, t)),
        "bias": to_json(&evaluate_incivility_bias(&p.gold, &layer, t)?),
    }))
}

/// The threshold of `THRESHOLDS` with the best micro F1 on `p`, and that F1.
fn tune(annotator: &IncivilityAnnotator, p: &Prepared, vectors: &[f32], dim: usize) -> Result<(f64, f64), String> {
    let mut best = (0.5, f64::NEG_INFINITY);
    for t in THRESHOLDS {
        let a = annotator.clone().with_threshold(t);
        let f1 = evaluate_incivility_corpus(&p.units, &score(&a, p, vectors, dim)?, t).micro_f1;
        if f1 > best.1 {
            best = (t, f1);
        }
    }
    Ok(best)
}

fn corpus_json(m: &IncivilityMetrics) -> Value {
    let labels: Vec<Value> = m
        .per_label
        .iter()
        .map(|(l, x)| {
            json!({
                "label": l.name(), "tp": x.tp, "fp": x.fp, "fn": x.fn_count, "tn": x.tn,
                "precision": x.precision, "recall": x.recall, "f1": x.f1,
            })
        })
        .collect();
    json!({
        "per_label": labels,
        "macro": { "precision": m.macro_precision, "recall": m.macro_recall, "f1": m.macro_f1 },
        "micro": { "precision": m.micro_precision, "recall": m.micro_recall, "f1": m.micro_f1 },
        "positive_predictions": m.total_positive_predictions,
        "positive_predictions_with_evidence": m.positive_with_evidence,
    })
}

fn to_json(m: &IncivilityBiasMetrics) -> Value {
    let labels: Vec<Value> = m
        .per_label
        .iter()
        .map(|l| {
            let subgroups: Vec<Value> = l
                .subgroups
                .iter()
                .map(|s| {
                    json!({
                        "subgroup": s.subgroup, "units": s.units, "positives": s.positives,
                        "subgroup_auc": s.subgroup_auc, "bpsn_auc": s.bpsn_auc, "bnsp_auc": s.bnsp_auc,
                        "neutral_mentions": s.neutral_mentions, "neutral_false_positives": s.neutral_false_positives,
                        "neutral_fpr": s.neutral_fpr,
                    })
                })
                .collect();
            json!({ "label": l.label.name(), "overall_auc": l.overall_auc, "subgroups": subgroups })
        })
        .collect();
    json!(labels)
}

fn run() -> Result<(), String> {
    let a = args()?;
    eprintln!("reading the sets");
    let (madlibs, _) = data::madlibs(&a.madlibs)?;
    let data::Civil { train, val, test } = data::civil(&a.civil)?;
    eprintln!("{} madlibs, {} train, {} val and {} test comments", madlibs.len(), train.len(), val.len(), test.len());

    let mut embedder =
        embed::Embedder::open(&a.bundle, &EncoderOptions { device: a.device.clone(), ..Default::default() })?;
    let dim = embedder.dim;
    let mut vectors = |units: &[Unit], name: &str| {
        eprintln!("embedding {name}");
        let texts: Vec<&str> = units.iter().map(|u| u.text.as_str()).collect();
        embedder.embed(&texts, &a.cache)
    };
    let (train_v, val_v, test_v, madlibs_v) =
        (vectors(&train, "train")?, vectors(&val, "val")?, vectors(&test, "test")?, vectors(&madlibs, "madlibs")?);

    eprintln!("fitting the head on {} comments, {:?}", train.len(), a.fit);
    let started = Instant::now();
    let rows: Vec<&[f32]> = train_v.chunks_exact(dim).collect();
    let targets: Vec<IncivilityScores> = train
        .iter()
        .map(|u| {
            let mut t = u.targets;
            if a.hard {
                for l in IncivilityLabel::ALL {
                    t.set(l, if t.get(l) >= 0.5 { 1.0 } else { 0.0 });
                }
            }
            t
        })
        .collect();
    let options = a.fit;
    let head = IncivilityVectorHead::fit(&rows, None, &targets, &options)
        .and_then(|h| h.with_model(embedder.model.clone()))
        .map_err(|e| e.to_string())?;
    let fit_seconds = started.elapsed().as_secs_f64();
    let scorer: Arc<dyn IncivilityVectorScorer> = Arc::new(head);

    let val_p = prepare(&val);
    let sets_p = [("civil_comments_test", prepare(&test), &test_v), ("madlibs", prepare(&madlibs), &madlibs_v)];
    let mut sets = serde_json::Map::new();
    let mut tuned = serde_json::Map::new();
    for (stage_name, stage) in STAGES {
        let annotator = IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&scorer)).with_stage(stage);
        eprintln!("tuning the {stage_name} stage's threshold on the val split");
        let (t, f1) = tune(&annotator, &val_p, &val_v, dim)?;
        tuned.insert(stage_name.into(), json!({ "threshold": t, "val_micro_f1": f1 }));
        for (name, p, v) in &sets_p {
            eprintln!("scoring {name} with the {stage_name} stage");
            let at_default = measure(&annotator, p, v, dim)?;
            let at_tuned = measure(&annotator.clone().with_threshold(t), p, v, dim)?;
            let stages = sets.entry(name.to_string()).or_insert_with(|| json!({}));
            stages[stage_name] = json!({ "default_threshold": at_default, "tuned_threshold": at_tuned });
        }
    }
    let record = json!({
        "encoder": { "id": embedder.model.id, "revision": embedder.model.revision, "manifest_sha256": embedder.model.sha256 },
        "head": {
            "train_comments": train.len(), "epochs": options.epochs, "learning_rate": options.learning_rate,
            "l2": options.l2, "targets": if a.hard { "hard" } else { "fractions" }, "fit_seconds": fit_seconds,
        },
        "tuned_on_val": { "val_comments": val.len(), "stages": tuned },
        "sets": sets,
    });
    let text = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
    std::fs::write(&a.out, text + "\n").map_err(|e| format!("{}: {e}", a.out.display()))
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("incivility_bias: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
