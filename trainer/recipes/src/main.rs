//! `turbo-train-head`: checks a corpus, scores its majority baseline, or
//! trains a head on a frozen encoder, writing a receipt each time.
//!
//! ```text
//! turbo-train-head check    --corpus <id> --dir <corpus dir> [--unrecorded-corpus]
//! turbo-train-head baseline --corpus <id> --dir <corpus dir> --out <dir> [--unrecorded-corpus]
//! turbo-train-head train    --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <dir>
//!     [--device cpu|cuda|levelzero|metal] [--precision model|exact] [--pooling first|mean] [--crf]
//!     [--bilstm <hidden>x<layers>] [--seed N] [--epochs N] [--batch N] [--lr X] [--patience N]
//!     [--max-train N] [--commit <sha>] [--unrecorded-corpus]
//! ```
//!
//! Exit status 2 on any error. `--out` must not exist yet. A run claims
//! it before anything else, by making it, which fails when it exists, so
//! of two runs given one `--out` exactly one goes on; a run that fails
//! removes it, so a refused run leaves none.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use turbo::manifest::WordPooling;
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_tagger::WordEncoder;
use turbo_trainer::Config;
use turbo_trainer_recipes::corpus::{Corpus, Task};
use turbo_trainer_recipes::eval::{Majority, Seen, bootstrap, evaluate};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{HeadOptions, examples, named, predict, spec, write};
use turbo_trainer_recipes::{Error, Result, coarse, read_corpus, safetensors};

const RESAMPLES: usize = 1000;

struct Args {
    rest: Vec<String>,
}

impl Args {
    fn flag(&mut self, name: &str) -> bool {
        let at = self.rest.iter().position(|a| a == name);
        at.map(|i| self.rest.remove(i)).is_some()
    }

    fn value(&mut self, name: &str) -> Result<Option<String>> {
        let Some(i) = self.rest.iter().position(|a| a == name) else { return Ok(None) };
        if i + 1 >= self.rest.len() {
            return Err(Error(format!("{name} needs a value")));
        }
        let v = self.rest.remove(i + 1);
        self.rest.remove(i);
        Ok(Some(v))
    }

    fn required(&mut self, name: &str) -> Result<String> {
        self.value(name)?.ok_or_else(|| Error(format!("{name} is required")))
    }

    fn parsed<T: std::str::FromStr>(&mut self, name: &str, default: T) -> Result<T> {
        match self.value(name)? {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| Error(format!("{name} {v:?} is not a valid value"))),
        }
    }

    fn done(self) -> Result<()> {
        if self.rest.is_empty() { Ok(()) } else { Err(Error(format!("unknown arguments: {}", self.rest.join(" ")))) }
    }
}

fn main() -> ExitCode {
    match run(std::env::args().collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("turbo-train-head: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(argv: Vec<String>) -> Result<()> {
    let command = argv.get(1).cloned().unwrap_or_default();
    let mut a = Args { rest: argv[2.min(argv.len())..].to_vec() };
    let corpus_id = a.required("--corpus")?;
    let dir = PathBuf::from(a.required("--dir")?);
    let unrecorded = a.flag("--unrecorded-corpus");
    match command.as_str() {
        "check" => {
            a.done()?;
            let c = read_corpus(&corpus_id, &dir, unrecorded)?;
            println!("{}", serde_json::to_string_pretty(&c.record).expect("the record serializes"));
            println!("labels: {}", c.labels().len());
            Ok(())
        }
        "baseline" => {
            let out = OutDir::claim(&a.required("--out")?)?;
            let commit = commit(a.value("--commit")?);
            a.done()?;
            let t = Instant::now();
            let c = read_corpus(&corpus_id, &dir, unrecorded)?;
            let read = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let base = baseline(&c);
            let receipt = Receipt {
                schema: SCHEMA.into(),
                command: argv.clone(),
                commit,
                host: host(),
                unix_time: unix_time(),
                corpus: c.record.clone(),
                encoder: None,
                head: None,
                config: None,
                training: None,
                eval: EvalRecord {
                    split: "test".into(),
                    model: None,
                    baseline: evaluate(
                        c.record.task,
                        &c.splits.test,
                        &base,
                        &Seen::of(&c.splits.train),
                        coarse(&corpus_id),
                    ),
                    comparison: None,
                },
                timing_s: Timing { read, eval: t.elapsed().as_secs_f64(), ..Timing::default() },
                outputs: Vec::new(),
            };
            finish(out, receipt)
        }
        "train" => train(&argv, &corpus_id, &dir, unrecorded, a),
        _ => Err(Error(format!("unknown command {command:?}: check, baseline or train"))),
    }
}

fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let bundle = PathBuf::from(a.required("--bundle")?);
    let out = OutDir::claim(&a.required("--out")?)?;
    let device = match a.value("--device")?.as_deref() {
        None | Some("cpu") => Device::Cpu,
        Some(d) => Device::Backend(d.to_owned()),
    };
    let precision = match a.value("--precision")?.as_deref() {
        None | Some("model") => Precision::Model,
        Some("exact") => Precision::Exact,
        Some(p) => return Err(Error(format!("--precision {p:?}: model or exact"))),
    };
    let pooling = match a.value("--pooling")?.as_deref() {
        None | Some("first") => WordPooling::First,
        Some("mean") => WordPooling::Mean,
        Some(p) => return Err(Error(format!("--pooling {p:?}: first or mean"))),
    };
    let crf = a.flag("--crf");
    let bilstm = match a.value("--bilstm")? {
        None => None,
        Some(v) => {
            let parse = |s: &str| s.parse::<u32>().map_err(|_| Error(format!("--bilstm {v:?}: <hidden>x<layers>")));
            let (h, l) = v.split_once('x').ok_or_else(|| Error(format!("--bilstm {v:?}: <hidden>x<layers>")))?;
            Some((parse(h)?, parse(l)?))
        }
    };
    let d = Config::default();
    let c = Config {
        seed: a.parsed("--seed", d.seed)?,
        epochs: a.parsed("--epochs", d.epochs)?,
        batch: a.parsed("--batch", d.batch)?,
        lr: a.parsed("--lr", d.lr)?,
        patience: a.parsed("--patience", d.patience)?,
        ..d
    };
    let max_train: usize = a.parsed("--max-train", 0)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;

    let t = Instant::now();
    let corpus = read_corpus(corpus_id, dir, unrecorded)?;
    let read = t.elapsed().as_secs_f64();
    let head = spec(&corpus, &HeadOptions { pooling, crf, bilstm })?;
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;

    let t = Instant::now();
    let train_set = match max_train {
        0 => &corpus.splits.train[..],
        n => &corpus.splits.train[..n.min(corpus.splits.train.len())],
    };
    let (train_ex, _) = examples(&mut enc, train_set, &head.labels, pooling)?;
    let (dev_ex, dev_left_out) = examples(&mut enc, &corpus.splits.dev, &head.labels, pooling)?;
    let features = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (model, report) =
        turbo_trainer::fit(&head, enc.hidden(), &train_ex, &dev_ex, &c).map_err(|e| Error(e.message))?;
    drop((train_ex, dev_ex));
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let tagger = model.to_token_tagger().map_err(|e| Error(e.message))?;
    let test = &corpus.splits.test;
    let predicted = predict(&mut enc, &tagger, test)?;
    let base = baseline(&corpus);
    let seen = Seen::of(train_set);
    let task: Task = corpus.record.task;
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(task, test, &predicted, &seen, coarse(corpus_id))),
        baseline: evaluate(task, test, &base, &seen, coarse(corpus_id)),
        comparison: Some(bootstrap(task, test, &predicted, &base, RESAMPLES, c.seed)),
    };
    let eval_s = t.elapsed().as_secs_f64();

    let weights = safetensors::write(&named(&head, enc.hidden(), model.tensors()));
    let spec_json = serde_json::to_vec_pretty(&head).expect("a head spec serializes");
    let outputs = vec![write(&out.path, "head.safetensors", &weights)?, write(&out.path, "head.json", &spec_json)?];
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus: corpus.record.clone(),
        encoder: Some(encoder),
        head: Some(head),
        config: Some(ConfigRecord {
            seed: c.seed,
            epochs: c.epochs,
            batch: c.batch,
            lr: c.lr,
            weight_decay: c.weight_decay,
            clip: c.clip,
            warmup: c.warmup,
            patience: c.patience,
            max_train,
            dev_left_out,
        }),
        training: Some(TrainingRecord {
            train_loss: report.train_loss,
            dev: report.dev,
            best_epoch: report.best_epoch,
        }),
        eval,
        timing_s: Timing { read, features, train: trained, eval: eval_s },
        outputs,
    };
    finish(out, receipt)
}

/// The majority baseline's labels for the test split.
fn baseline(c: &Corpus) -> Vec<Vec<String>> {
    let m = Majority::train(&c.splits.train);
    c.splits.test.iter().map(|s| m.predict(&s.words)).collect()
}

fn encoder_record(bundle: &Path, enc: &WordEncoder) -> Result<EncoderRecord> {
    let manifest =
        std::fs::read(bundle.join("manifest.json")).map_err(|e| Error(format!("{}: {e}", bundle.display())))?;
    let m: turbo::manifest::Manifest =
        serde_json::from_slice(&manifest).map_err(|e| Error(format!("{}: {e}", bundle.display())))?;
    let x = enc.execution();
    Ok(EncoderRecord {
        model: m.model.id,
        revision: m.model.revision,
        manifest_sha256: turbo::bundle::sha256_hex(&manifest),
        hidden: enc.hidden(),
        width: enc.width(),
        backend: x.backend.clone(),
        device: x.device.clone(),
        precision: x.precision.clone(),
    })
}

/// A run's output directory, claimed before the run reads or writes
/// anything: `create_dir` fails when the directory exists, so two runs
/// given one `--out` cannot both have it. Dropped unkept, as when the run
/// fails, it is removed with what the run wrote in it.
struct OutDir {
    path: PathBuf,
    keep: bool,
}

impl OutDir {
    fn claim(p: &str) -> Result<OutDir> {
        let path = PathBuf::from(p);
        if let Some(parent) = path.parent().filter(|q| !q.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| Error(format!("{}: {e}", parent.display())))?;
        }
        match std::fs::create_dir(&path) {
            Ok(()) => Ok(OutDir { path, keep: false }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(Error(format!("{}: already there; a run writes a new directory", path.display())))
            }
            Err(e) => Err(Error(format!("{}: {e}", path.display()))),
        }
    }
}

impl Drop for OutDir {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn commit(given: Option<String>) -> String {
    given
        .or_else(|| {
            let o = std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
            o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn finish(mut out: OutDir, receipt: Receipt) -> Result<()> {
    let text = serde_json::to_string_pretty(&receipt).expect("the receipt serializes");
    write(&out.path, "receipt.json", text.as_bytes())?;
    out.keep = true;
    let model = receipt.eval.model.as_ref().map_or("-".into(), |m| format!("{:.4}", m.score()));
    println!(
        "{}: {} model {model} baseline {:.4}; {}",
        receipt.corpus.id,
        receipt.eval.split,
        receipt.eval.baseline.score(),
        out.path.join("receipt.json").display()
    );
    Ok(())
}
