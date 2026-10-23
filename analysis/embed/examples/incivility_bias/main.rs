//! Identity-term bias of the incivility annotator on real data (#889).
//!
//! Embeds the Civil Comments train and test splits and the English bias
//! madlibs with one encoder bundle, fits an `IncivilityVectorHead` to the
//! train split's rater fractions, scores the test split and the madlibs
//! with the lexicon, learned and fused stages, and writes
//! `evaluate_incivility_bias` for each to a JSON file.
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
use turbo_analysis_layers::{Incivility, IncivilityItem, IncivilityLabel, IncivilityTarget};
use turbo_doccat::{
    GoldBiasUnit, IncivilityAnnotator, IncivilityBiasMetrics, IncivilityFitOptions, IncivilityStage,
    IncivilityVectorHead, IncivilityVectorScorer, evaluate_incivility_bias,
};

use data::Unit;

const STAGES: [(&str, IncivilityStage); 3] =
    [("lexicon", IncivilityStage::Lexicon), ("learned", IncivilityStage::Learned), ("fused", IncivilityStage::Fused)];

struct Args {
    bundle: PathBuf,
    device: Device,
    civil: PathBuf,
    madlibs: PathBuf,
    cache: PathBuf,
    out: PathBuf,
    epochs: u32,
}

fn args() -> Result<Args, String> {
    let mut a = std::env::args().skip(1);
    let (mut bundle, mut device, mut civil, mut madlibs, mut cache, mut out) =
        (None, Device::Cpu, None, None, None, None);
    let mut epochs = IncivilityFitOptions::default().epochs;
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
            "--epochs" => epochs = value()?.parse().map_err(|e| format!("--epochs: {e}"))?,
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
        epochs,
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

/// Scores `units` (vectors in `vectors`) with `annotator` and measures the
/// bias against their targets, a label positive at 0.5 and above.
fn measure(annotator: &IncivilityAnnotator, units: &[Unit], vectors: &[f32], dim: usize) -> Result<Value, String> {
    let mut layer = Incivility::new();
    let mut gold = Vec::with_capacity(units.len());
    for (i, u) in units.iter().enumerate() {
        let tokens = token_lower(&u.text);
        let v = &vectors[i * dim..(i + 1) * dim];
        let (scores, _) = annotator.score_unit_in_context(&tokens, 0, tokens.len(), None, Some(v), None)?;
        layer.push(IncivilityItem { target: IncivilityTarget::Sentence(Ix::new(i as u32)), scores, evidence: &[] });
        let positive_labels = IncivilityLabel::ALL.into_iter().filter(|&l| u.targets.get(l) >= 0.5).collect();
        gold.push(GoldBiasUnit { positive_labels, subgroups: u.subgroups.clone() });
    }
    Ok(to_json(&evaluate_incivility_bias(&gold, &layer, 0.5)?))
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
    let (train, test) = data::civil(&a.civil)?;
    eprintln!("{} madlibs, {} train and {} test comments", madlibs.len(), train.len(), test.len());

    let mut embedder =
        embed::Embedder::open(&a.bundle, &EncoderOptions { device: a.device.clone(), ..Default::default() })?;
    let dim = embedder.dim;
    let mut vectors = |units: &[Unit], name: &str| {
        eprintln!("embedding {name}");
        let texts: Vec<&str> = units.iter().map(|u| u.text.as_str()).collect();
        embedder.embed(&texts, &a.cache)
    };
    let (train_v, test_v, madlibs_v) =
        (vectors(&train, "train")?, vectors(&test, "test")?, vectors(&madlibs, "madlibs")?);

    eprintln!("fitting the head on {} comments, {} epochs", train.len(), a.epochs);
    let started = Instant::now();
    let rows: Vec<&[f32]> = train_v.chunks_exact(dim).collect();
    let targets: Vec<_> = train.iter().map(|u| u.targets).collect();
    let options = IncivilityFitOptions { epochs: a.epochs, ..Default::default() };
    let head = IncivilityVectorHead::fit(&rows, None, &targets, &options)
        .and_then(|h| h.with_model(embedder.model.clone()))
        .map_err(|e| e.to_string())?;
    let fit_seconds = started.elapsed().as_secs_f64();
    let scorer: Arc<dyn IncivilityVectorScorer> = Arc::new(head);

    let mut sets = serde_json::Map::new();
    for (name, units, v) in [("civil_comments_test", &test, &test_v), ("madlibs", &madlibs, &madlibs_v)] {
        let mut stages = serde_json::Map::new();
        for (stage_name, stage) in STAGES {
            eprintln!("scoring {name} with the {stage_name} stage");
            let annotator = IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&scorer)).with_stage(stage);
            stages.insert(stage_name.into(), measure(&annotator, units, v, dim)?);
        }
        sets.insert(name.into(), Value::Object(stages));
    }
    let record = json!({
        "encoder": { "id": embedder.model.id, "revision": embedder.model.revision, "manifest_sha256": embedder.model.sha256 },
        "head": {
            "train_comments": train.len(), "epochs": options.epochs, "learning_rate": options.learning_rate,
            "l2": options.l2, "fit_seconds": fit_seconds,
        },
        "threshold": 0.5,
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
