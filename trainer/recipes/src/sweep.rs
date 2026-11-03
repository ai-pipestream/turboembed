//! `turbo-train-head sweep` (`#1822`): runs every cached encoder layer from
//! `--from-layer` to the top layer plus `--mix` for one corpus and task,
//! writes `receipt.json` and `fit.json` in each run subdirectory, and
//! writes `sweep.json` and `sweep.md` with paired bootstrap against the
//! last layer and an optional peer reference receipt.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use turbo::manifest::Biaffine;

use crate::cached_ud::CachedParams;
use crate::cli::{Args, HeadArgs, OutDir, commit, head_args};
use crate::eval::{Comparison, bootstrap};
use crate::fit_cache::{
    CacheTask, DEFAULT_ARC, DEFAULT_MIN_TREE_FREQ, DEFAULT_REL, Mix, read_cache_record, run_cached,
};
use crate::receipt::{host, unix_time};
use crate::run::write;
use crate::{Error, Result};

#[path = "sweep_peer.rs"]
mod peer;

pub const SCHEMA: &str = "turboembed.layer-sweep.v1";
const RESAMPLES: usize = 1000;

/// Peer reference summary extracted from `--peer-receipt <receipt.json>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepPeer {
    pub name: String,
    pub test: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uas: Option<f64>,
}

/// One row in a layer + scalar-mix sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepRow {
    pub name: String,
    pub dir: String,
    pub layers: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mix: Option<Mix>,
    pub best_epoch: usize,
    pub dev: f64,
    pub test: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uas: Option<f64>,
    pub vs_last_layer: Comparison,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vs_peer: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uas_vs_peer: Option<f64>,
}

/// Full sweep summary persisted to `<out>/sweep.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepRecord {
    pub schema: String,
    pub command: Vec<String>,
    pub commit: String,
    pub host: String,
    pub unix_time: u64,
    pub corpus_id: String,
    pub task: CacheTask,
    pub layers: Vec<u32>,
    pub reference_layer: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer: Option<SweepPeer>,
    pub runs: Vec<SweepRow>,
    pub best_dev_run: String,
    pub best_test_run: String,
}

/// CLI entry point for `turbo-train-head sweep`.
///
/// # Errors
///
/// Invalid flags, a cache without `"test"`, or any training failure.
pub fn command(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let cache_dir = PathBuf::from(a.required("--cache")?);
    let mut out = OutDir::claim(&a.required("--out")?)?;
    let task = match a.value("--task")? {
        Some(s) => CacheTask::parse(&s)?,
        None => CacheTask::infer(corpus_id),
    };
    let from: Option<u32> =
        a.value("--from-layer")?.map(|v| v.parse()).transpose().map_err(|_| Error("--from-layer N".into()))?;
    let peer_path = a.value("--peer-receipt")?.map(PathBuf::from);
    let min_tree_freq: usize = a.parsed("--min-tree-freq", DEFAULT_MIN_TREE_FREQ)?;
    let seen_lookup: u32 = a.parsed("--seen-lookup", 0)?;
    let biaffine = Biaffine { arc: a.parsed("--arc", DEFAULT_ARC)?, rel: a.parsed("--rel", DEFAULT_REL)? };
    let max_train: usize = a.parsed("--max-train", 0)?;
    let HeadArgs { crf, bilstm, hash, config } = head_args(&mut a)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;

    let cache = read_cache_record(&cache_dir)?;
    if !cache.splits.iter().any(|s| s.name == "test") {
        return Err(Error(format!("{}: sweep requires the test split in cache.json", cache_dir.display())));
    }
    let layers: Vec<u32> = cache.layers.iter().copied().filter(|&l| from.is_none_or(|f| l >= f)).collect();
    if layers.is_empty() {
        return Err(Error(format!("--from-layer {:?}: the cache holds {:?}", from, cache.layers)));
    }
    let base = CachedParams {
        argv: argv.to_vec(),
        commit: commit.clone(),
        corpus_id: corpus_id.to_owned(),
        corpus_dir: dir.to_path_buf(),
        cache_dir,
        out_dir: out.path.clone(),
        unrecorded,
        layers: Vec::new(),
        mixed: false,
        crf,
        bilstm,
        hash,
        min_tree_freq,
        seen_lookup,
        biaffine,
        config,
        max_train,
    };

    let peer = peer_path.as_ref().map(|path| peer::read(path, &base, task)).transpose()?;

    let mut specs: Vec<(String, String, Vec<u32>, bool)> =
        layers.iter().map(|&k| (format!("layer-{k}"), format!("layer-{k:02}"), vec![k], false)).collect();
    specs.push(("mix".into(), "mix".into(), layers.clone(), true));

    let mut outcomes = Vec::with_capacity(specs.len());
    for (name, subdir, run_layers, mixed) in specs {
        let run_dir = out.path.join(&subdir);
        std::fs::create_dir_all(&run_dir).map_err(|e| Error(format!("{}: {e}", run_dir.display())))?;
        let mut p = base.clone();
        p.out_dir = run_dir;
        p.layers = run_layers.clone();
        p.mixed = mixed;
        let outcome = run_cached(&p, task, cache.clone())?;
        outcomes.push((name, subdir, run_layers, outcome));
    }

    let reference_layer = *layers.last().expect("non-empty layers");
    let ref_idx = outcomes.iter().position(|(_, _, l, m)| l == &[reference_layer] && m.fit.mix.is_none()).unwrap_or(0);
    let ref_gold = outcomes[ref_idx].3.test_gold.clone();
    let ref_pred = outcomes[ref_idx].3.test_pred.clone();
    let eval_task = outcomes[ref_idx]
        .3
        .receipt
        .as_ref()
        .map(|r| r.corpus.task)
        .ok_or_else(|| Error("missing test receipt in sweep run".into()))?;

    let mut runs = Vec::with_capacity(outcomes.len());
    for (name, subdir, run_layers, outcome) in outcomes {
        let receipt = outcome.receipt.as_ref().ok_or_else(|| Error("missing test receipt in sweep run".into()))?;
        let test = receipt.eval.model.as_ref().map_or(0.0, |m| m.score());
        let uas = receipt.eval.uas.as_ref().map(|u| u.model);
        let vs_last_layer = bootstrap(eval_task, &ref_gold, &outcome.test_pred, &ref_pred, RESAMPLES, base.config.seed);
        let vs_peer = peer.as_ref().map(|p| test - p.test);
        let uas_vs_peer = peer.as_ref().and_then(|p| uas.zip(p.uas).map(|(u, pu)| u - pu));
        runs.push(SweepRow {
            name,
            dir: subdir,
            layers: run_layers,
            mix: outcome.fit.mix,
            best_epoch: outcome.fit.best_epoch,
            dev: outcome.fit.best_dev,
            test,
            uas,
            vs_last_layer,
            vs_peer,
            uas_vs_peer,
        });
    }

    let best_dev_run = runs.iter().max_by(|a, b| a.dev.total_cmp(&b.dev)).map(|r| r.name.clone()).unwrap_or_default();
    let best_test_run =
        runs.iter().max_by(|a, b| a.test.total_cmp(&b.test)).map(|r| r.name.clone()).unwrap_or_default();

    let record = SweepRecord {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus_id: corpus_id.to_owned(),
        task,
        layers,
        reference_layer,
        peer,
        runs,
        best_dev_run,
        best_test_run,
    };
    let md = format_markdown(&record);
    write(&out.path, "sweep.json", serde_json::to_string_pretty(&record).expect("serializes").as_bytes())?;
    write(&out.path, "sweep.md", md.as_bytes())?;
    out.keep = true;
    print!("{md}");
    Ok(())
}

/// Formats a [`SweepRecord`] as a Markdown comparison table.
pub fn format_markdown(record: &SweepRecord) -> String {
    let has_uas = record.runs.iter().any(|r| r.uas.is_some());
    let mut out = String::new();
    let _ = writeln!(
        out,
        "### `{}` (`{}`): layer and scalar-mix sweep (ref `layer-{}`)\n",
        record.corpus_id,
        record.task.as_str(),
        record.reference_layer
    );
    match (&record.peer, has_uas) {
        (Some(p), true) => {
            let _ = writeln!(
                out,
                "| Run | Layers | Dev | Test (LAS) | UAS | Delta vs layer-{} | 95% CI | Delta LAS vs {} | Delta UAS vs {} |",
                record.reference_layer, p.name, p.name
            );
            let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|");
        }
        (Some(p), false) => {
            let _ = writeln!(
                out,
                "| Run | Layers | Dev | Test | Delta vs layer-{} | 95% CI | Delta vs {} |",
                record.reference_layer, p.name
            );
            let _ = writeln!(out, "|---|---|---|---|---|---|---|");
        }
        (None, true) => {
            let _ = writeln!(
                out,
                "| Run | Layers | Dev | Test (LAS) | UAS | Delta vs layer-{} | 95% CI |",
                record.reference_layer
            );
            let _ = writeln!(out, "|---|---|---|---|---|---|---|");
        }
        (None, false) => {
            let _ = writeln!(out, "| Run | Layers | Dev | Test | Delta vs layer-{} | 95% CI |", record.reference_layer);
            let _ = writeln!(out, "|---|---|---|---|---|---|");
        }
    }
    for r in &record.runs {
        let layers_str = if r.layers.len() == 1 {
            format!("{}", r.layers[0])
        } else {
            format!("{}..={}", r.layers.first().unwrap_or(&0), r.layers.last().unwrap_or(&0))
        };
        let c = &r.vs_last_layer;
        match (&record.peer, r.uas) {
            (Some(_), Some(u)) => {
                let d_las = r.vs_peer.map_or("-".into(), |d| format!("{d:+.4}"));
                let d_uas = r.uas_vs_peer.map_or("-".into(), |d| format!("{d:+.4}"));
                let _ = writeln!(
                    out,
                    "| `{}` | `{}` | {:.4} | {:.4} | {:.4} | {:+.4} | [{:+.4}, {:+.4}] | {} | {} |",
                    r.name, layers_str, r.dev, r.test, u, c.delta, c.low, c.high, d_las, d_uas
                );
            }
            (Some(_), None) => {
                let d_peer = r.vs_peer.map_or("-".into(), |d| format!("{d:+.4}"));
                let _ = writeln!(
                    out,
                    "| `{}` | `{}` | {:.4} | {:.4} | {:+.4} | [{:+.4}, {:+.4}] | {} |",
                    r.name, layers_str, r.dev, r.test, c.delta, c.low, c.high, d_peer
                );
            }
            (None, Some(u)) => {
                let _ = writeln!(
                    out,
                    "| `{}` | `{}` | {:.4} | {:.4} | {:.4} | {:+.4} | [{:+.4}, {:+.4}] |",
                    r.name, layers_str, r.dev, r.test, u, c.delta, c.low, c.high
                );
            }
            (None, None) => {
                let _ = writeln!(
                    out,
                    "| `{}` | `{}` | {:.4} | {:.4} | {:+.4} | [{:+.4}, {:+.4}] |",
                    r.name, layers_str, r.dev, r.test, c.delta, c.low, c.high
                );
            }
        }
    }
    if let Some(m) = record.runs.iter().find(|r| r.name == "mix").and_then(|r| r.mix.as_ref()) {
        let w: Vec<String> = m.weights.iter().map(|v| format!("{v:.3}")).collect();
        let _ = writeln!(out, "\n- `mix` weights: `[{}]`, gain: `{:.3}`", w.join(", "), m.gain);
    }
    let _ = writeln!(out, "- Best dev: `{}` | Best test: `{}`", record.best_dev_run, record.best_test_run);
    out
}
