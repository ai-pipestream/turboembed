//! `turbo-train-head summarize`: one table from a set of `receipt.json`
//! files, with the peer rows (#1830).
//!
//! ```text
//! turbo-train-head summarize <dir>... [--peer <dir>...] [--format md|json] [--out <dir>] [--check]
//! ```
//!
//! Every `<dir>` holds a run's `receipt.json` beside its outputs. The
//! receipts group by corpus and task; each group prints as one table
//! whose rows are the head variants, then the baseline, then the peers.
//! `--out` writes the tables as `summary.md` with the numbers as
//! `summary.json`, with any `host` field scrubbed (#1804): the summary
//! never carries one. `--check` refuses receipts whose `commit`
//! disagrees with the others, and digests in a receipt or a summary
//! that no longer match the file on disk. A summary records each row's
//! `dir` and `receipt` relative to the directory it is written to, and
//! `--check` reads them relative to the summary's own directory, so a
//! copied run directory is checked as copied (summarize/check.rs).

mod check;

use std::path::{Path, PathBuf};

use check::{relative, verify_summary};
use serde::Serialize;
use turbo::bundle::sha256_hex;

use turbo_trainer_recipes::cli::Args;
use turbo_trainer_recipes::corpus::{FileRecord, Task};
use turbo_trainer_recipes::eval::Comparison;
use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::{Error, Result};

pub const SCHEMA: &str = "turboembed.trainer.summary/1";

/// A run's receipt as read, with the digest of the file it came from.
struct Loaded {
    dir: String,
    /// `dir` relative to where the summary is written, as recorded.
    shown: String,
    receipt: String,
    receipt_sha256: String,
    receipt_data: Receipt,
}

/// A table row: a head variant, the baseline, or a peer.
struct Row {
    label: String,
    kind: &'static str,
    dir: String,
    receipt: String,
    receipt_sha256: String,
    commit: String,
    model: Option<f64>,
    baseline: f64,
    comparison: Option<Comparison>,
    epochs: Option<(usize, usize)>,
    wall_s: f64,
    outputs: Vec<FileRecord>,
}

/// One table: the rows of one corpus and task.
struct Group {
    corpus: String,
    task: String,
    metric: String,
    rows: Vec<Row>,
}

/// A row of `summary.json`. It carries no `host` field (#1804).
#[derive(Serialize)]
struct JsonRow {
    label: String,
    kind: &'static str,
    dir: String,
    receipt: String,
    receipt_sha256: String,
    commit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<f64>,
    baseline: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<Comparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    epochs_run: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    epochs_kept: Option<usize>,
    wall_s: f64,
    outputs: Vec<FileRecord>,
}

/// One corpus table of `summary.json`.
#[derive(Serialize)]
struct JsonCorpus {
    corpus: String,
    task: String,
    metric: String,
    rows: Vec<JsonRow>,
}

/// `summary.json`: the tables, without any `host` field (#1804).
#[derive(Serialize)]
struct Summary {
    schema: &'static str,
    command: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    commit: Option<String>,
    corpora: Vec<JsonCorpus>,
}

pub fn run(argv: &[String]) -> Result<()> {
    let mut a = Args { rest: argv.to_vec() };
    let peers = a.values("--peer")?;
    let format = match a.value("--format")?.as_deref() {
        None | Some("md") => "md",
        Some("json") => "json",
        Some(f) => return Err(Error(format!("--format {f:?}: md or json"))),
    };
    let out = a.value("--out")?.map(PathBuf::from);
    let check = a.flag("--check");
    let mut dirs = Vec::new();
    for arg in std::mem::take(&mut a.rest) {
        if arg.starts_with("--") {
            return Err(Error(format!("unknown argument: {arg}")));
        }
        dirs.push(arg);
    }
    a.done()?;
    if dirs.is_empty() && peers.is_empty() {
        return Err(Error("a run <dir> is required".into()));
    }
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    }
    let base = out.clone().unwrap_or_else(|| PathBuf::from("."));
    let mut loaded: Vec<Loaded> = Vec::new();
    for dir in dirs.iter().chain(peers.iter()) {
        loaded.push(load(dir, &base)?);
    }
    if check {
        for l in &loaded {
            let summary = Path::new(&l.dir).join("summary.json");
            if summary.is_file() {
                verify_summary(&summary)?;
            }
        }
        check_commits(&loaded)?;
        for l in &loaded {
            check_receipt(l)?;
        }
    }
    let groups = group(&loaded)?;
    let md = render(&groups);
    let summary = summarize(command(&loaded, dirs.len(), format, check), &groups);
    let json = serde_json::to_string_pretty(&summary).expect("a summary serializes");
    if let Some(dir) = &out {
        let (md_path, json_path) = (dir.join("summary.md"), dir.join("summary.json"));
        std::fs::write(&md_path, md.as_bytes()).map_err(|e| Error(format!("{}: {e}", md_path.display())))?;
        std::fs::write(&json_path, format!("{json}\n").as_bytes())
            .map_err(|e| Error(format!("{}: {e}", json_path.display())))?;
        if check {
            verify_summary(&json_path)?;
        }
    }
    if format == "json" {
        println!("{json}");
    } else {
        print!("{md}");
    }
    Ok(())
}

/// Reads `dir/receipt.json` with the digest of the file as read; the
/// recorded paths are relative to `base`, where the summary goes.
fn load(dir: &str, base: &Path) -> Result<Loaded> {
    let path = Path::new(dir).join("receipt.json");
    let bytes = std::fs::read(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let receipt: Receipt = serde_json::from_slice(&bytes).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let shown = relative(Path::new(dir), base)?;
    Ok(Loaded {
        dir: dir.into(),
        receipt: if shown == "." { "receipt.json".into() } else { format!("{shown}/receipt.json") },
        shown,
        receipt_sha256: sha256_hex(&bytes),
        receipt_data: receipt,
    })
}

/// Every receipt names one commit, or the copy to the models folder may
/// mix runs.
fn check_commits(loaded: &[Loaded]) -> Result<()> {
    let first = &loaded[0];
    for l in &loaded[1..] {
        if l.receipt_data.commit != first.receipt_data.commit {
            return Err(Error(format!(
                "{}: commit {} differs from {} commit {}",
                l.dir, l.receipt_data.commit, first.dir, first.receipt_data.commit
            )));
        }
    }
    Ok(())
}

/// Every digest a receipt records still matches the file beside it.
fn check_receipt(l: &Loaded) -> Result<()> {
    let dir = Path::new(&l.dir);
    for o in &l.receipt_data.outputs {
        let path = dir.join(&o.path);
        let bytes = std::fs::read(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let digest = sha256_hex(&bytes);
        if digest != o.sha256 {
            return Err(Error(format!("{}: sha256 {digest} on disk, receipt says {}", path.display(), o.sha256)));
        }
    }
    Ok(())
}

/// The command as recorded: `summarize`, the run and peer directories
/// as the summary records them, and the flags that shape the output.
/// `--out` is where the summary sits, so it is not repeated.
fn command(loaded: &[Loaded], heads: usize, format: &str, check: bool) -> Vec<String> {
    let mut c = vec!["summarize".to_owned()];
    for (i, l) in loaded.iter().enumerate() {
        if i >= heads {
            c.push("--peer".into());
        }
        c.push(l.shown.clone());
    }
    if format == "json" {
        c.extend(["--format".into(), "json".into()]);
    }
    if check {
        c.push("--check".into());
    }
    c
}

/// Groups the receipts by corpus and task, heads first, then the
/// baseline, then the peers.
fn group(loaded: &[Loaded]) -> Result<Vec<Group>> {
    let mut groups: Vec<Group> = Vec::new();
    for l in loaded {
        let r = &l.receipt_data;
        let task = match r.corpus.task {
            Task::Tags => "tags",
            Task::Bio => "bio",
            Task::Coref => "coref",
            Task::Relations => "relations",
        };
        let metric = r.eval.model.as_ref().map_or_else(|| r.eval.baseline.metric_name(), |m| m.metric_name());
        let at = groups.iter().position(|g| g.corpus == r.corpus.id && g.task == task);
        let g = match at {
            Some(i) if groups[i].metric != metric => {
                return Err(Error(format!(
                    "{}: metric {metric} differs from {} in the {} {task} table",
                    l.dir, groups[i].metric, r.corpus.id
                )));
            }
            Some(i) => &mut groups[i],
            None => {
                groups.push(Group {
                    corpus: r.corpus.id.clone(),
                    task: task.into(),
                    metric: metric.into(),
                    rows: Vec::new(),
                });
                groups.last_mut().expect("a group was pushed")
            }
        };
        let (kind, label) = if let Some(p) = &r.peer {
            ("peer", if p.map.is_empty() { p.name.clone() } else { format!("{} (mapped)", p.name) })
        } else if r.eval.model.is_none() {
            ("baseline", "majority baseline".into())
        } else {
            ("head", dir_name(&l.dir))
        };
        let timing = &r.timing_s;
        g.rows.push(Row {
            label,
            kind,
            dir: l.shown.clone(),
            receipt: l.receipt.clone(),
            receipt_sha256: l.receipt_sha256.clone(),
            commit: r.commit.clone(),
            model: r.eval.model.as_ref().map(|m| m.score()),
            baseline: r.eval.baseline.score(),
            comparison: r.eval.comparison.clone(),
            epochs: r.training.as_ref().map(|t| (t.dev.len(), t.best_epoch)),
            wall_s: timing.read + timing.features + timing.train + timing.eval,
            outputs: r.outputs.clone(),
        });
        let head_at = |kind: &str| {
            if kind == "head" {
                0
            } else if kind == "baseline" {
                1
            } else {
                2
            }
        };
        let last = g.rows.len() - 1;
        let row = g.rows.remove(last);
        let at = g.rows.iter().position(|r| head_at(r.kind) > head_at(row.kind)).unwrap_or(g.rows.len());
        g.rows.insert(at, row);
    }
    Ok(groups)
}

/// The run directory's name, for a head row.
fn dir_name(dir: &str) -> String {
    let real = Path::new(dir).canonicalize().unwrap_or_else(|_| PathBuf::from(dir));
    real.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| dir.into())
}

fn score(x: f64) -> String {
    format!("{x:.4}")
}

fn wall(s: f64) -> String {
    format!("{s:.1} s")
}

fn epochs(epochs: Option<(usize, usize)>) -> String {
    epochs.map_or_else(|| "-".into(), |(run, kept)| format!("{run} run / best {kept}"))
}

fn outputs(outputs: &[FileRecord]) -> String {
    if outputs.is_empty() {
        return "-".into();
    }
    outputs
        .iter()
        .map(|o| format!("{} {}", o.path, o.sha256.get(..12).unwrap_or(o.sha256.as_str())))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One markdown table per corpus.
fn render(groups: &[Group]) -> String {
    let mut md = String::new();
    for g in groups {
        md += &format!("# {} ({}, {})\n\n", g.corpus, g.task, g.metric);
        md += "| row | dir | score | vs baseline (paired bootstrap, 95%) | epochs | wall | outputs |\n";
        md += "|---|---|---|---|---|---|---|\n";
        for r in &g.rows {
            let vs = r
                .comparison
                .as_ref()
                .map_or_else(String::new, |c| format!("{:+.4} [{:+.4}, {:+.4}]", c.delta, c.low, c.high));
            md += &format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                r.label,
                r.dir,
                score(r.model.unwrap_or(r.baseline)),
                vs,
                epochs(r.epochs),
                wall(r.wall_s),
                outputs(&r.outputs)
            );
        }
        md += "\n";
    }
    md
}

/// The `summary.json` value: the tables without any `host` field.
fn summarize(command: Vec<String>, groups: &[Group]) -> Summary {
    let unanimous = groups
        .iter()
        .flat_map(|g| &g.rows)
        .map(|r| r.commit.clone())
        .reduce(|a, b| if a == b { a } else { String::new() });
    Summary {
        schema: SCHEMA,
        command,
        commit: unanimous.filter(|c| !c.is_empty()),
        corpora: groups
            .iter()
            .map(|g| JsonCorpus {
                corpus: g.corpus.clone(),
                task: g.task.clone(),
                metric: g.metric.clone(),
                rows: g
                    .rows
                    .iter()
                    .map(|r| JsonRow {
                        label: r.label.clone(),
                        kind: r.kind,
                        dir: r.dir.clone(),
                        receipt: r.receipt.clone(),
                        receipt_sha256: r.receipt_sha256.clone(),
                        commit: r.commit.clone(),
                        model: r.model,
                        baseline: r.baseline,
                        comparison: r.comparison.clone(),
                        epochs_run: r.epochs.map(|(run, _)| run),
                        epochs_kept: r.epochs.map(|(_, kept)| kept),
                        wall_s: r.wall_s,
                        outputs: r.outputs.clone(),
                    })
                    .collect(),
            })
            .collect(),
    }
}
