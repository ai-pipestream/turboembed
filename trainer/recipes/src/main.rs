//! `turbo-train-head`: checks a corpus, scores its majority baseline, or
//! trains a head on a frozen encoder, writing a receipt each time.
//!
//! ```text
//! turbo-train-head check    --corpus <id> --dir <corpus dir> [--unrecorded-corpus]
//! turbo-train-head baseline --corpus <id> --dir <corpus dir> --out <dir> [--unrecorded-corpus]
//! turbo-train-head train    --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <dir>
//!     [--device cpu|cuda|levelzero|metal] [--precision model|exact] [--pooling first|mean] [--crf]
//!     [--bilstm <hidden>x<layers>] [--hash-embed <dim>] [--hash-rows <norm>,<prefix>,<suffix>,<shape>]
//!     [--seed N] [--epochs N] [--batch N] [--lr X] [--patience N] [--tolerance X]
//!     [--threads N] [--max-train N] [--commit <sha>] [--unrecorded-corpus]
//! turbo-train-head cache    --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <dir> --layers N
//!     [--device ..] [--precision ..] [--pooling ..] [--max-batch N] [--splits train,dev,test]
//!     [--commit <sha>] [--unrecorded-corpus]
//! ```
//!
//! `cache` writes every split's word features at the encoder's last N
//! layers, for heads on one layer or a mix of them (`cache` module).
//!
//! Exit status 2 on any error. `--out` must not exist yet. A run claims
//! it before anything else, by making it, which fails when it exists, so
//! of two runs given one `--out` exactly one goes on; a run that fails
//! removes it, so a refused run leaves none.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use turbo::manifest::{HashEmbed, WordPooling};
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_tagger::WordEncoder;
use turbo_trainer::Config;
use turbo_trainer_recipes::cache::{self, CacheRecord};
use turbo_trainer_recipes::corpus::{Sentence, Task};
use turbo_trainer_recipes::eval::{Majority, Seen, bootstrap, evaluate};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{
    CHECKPOINT_DEV, CHECKPOINT_EPOCH, HASH_ROWS, HeadOptions, examples, named, predict, replace, spec, write,
};
use turbo_trainer_recipes::{Error, Result, coarse, read_corpus, safetensors};

const RESAMPLES: usize = 1000;

/// The default patience of a head with a BiLSTM adapter: its dev score
/// keeps creeping up by less than the tolerance an epoch for many epochs
/// (MASC POS, hashed words and a 256x1 BiLSTM: 0.9733 at epoch 32, 0.9755
/// at 57), and each better epoch is saved as it comes, so waiting costs
/// only time.
const BILSTM_PATIENCE: u32 = 15;

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
            let base = baseline(&c.splits.train, &c.splits.test);
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
        "cache" => cache(&argv, &corpus_id, &dir, unrecorded, a),
        _ => Err(Error(format!("unknown command {command:?}: check, baseline, train or cache"))),
    }
}

/// --device, --precision and --pooling.
fn encoder_args(a: &mut Args) -> Result<(Device, Precision, WordPooling)> {
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
    Ok((device, precision, pooling))
}

/// Writes the word features of every split at the encoder's last
/// `--layers` layers, and `cache.json` last.
fn cache(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let bundle = PathBuf::from(a.required("--bundle")?);
    let mut out = OutDir::claim(&a.required("--out")?)?;
    let (device, precision, pooling) = encoder_args(&mut a)?;
    let taps: u32 = a.parsed("--layers", 0)?;
    let max_batch: u32 = a.parsed("--max-batch", 32)?;
    let wanted = a.value("--splits")?.unwrap_or_else(|| "train,dev,test".into());
    let commit = commit(a.value("--commit")?);
    a.done()?;
    let wanted: Vec<&str> = wanted.split(',').collect();
    if wanted.is_empty() || wanted.iter().any(|w| !["train", "dev", "test"].contains(w)) {
        return Err(Error(format!("--splits {wanted:?}: train, dev and test, comma separated")));
    }
    let corpus = read_corpus(corpus_id, dir, unrecorded)?;
    let o = EncoderOptions { device, precision, max_batch: max_batch.max(taps), ..EncoderOptions::default() };
    let mut enc = WordEncoder::open(&bundle, &o).map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let model_layers = encoder.layers.ok_or_else(|| Error(format!("{}: no architecture.layers", bundle.display())))?;
    let t = Instant::now();
    let s = &corpus.splits;
    let splits: Vec<(&str, &[_])> = [("train", &s.train[..]), ("dev", &s.dev[..]), ("test", &s.test[..])]
        .into_iter()
        .filter(|(n, _)| wanted.contains(n))
        .collect();
    let (layers, splits) = cache::write(&mut enc, &splits, pooling, taps, model_layers, &out.path)?;
    let record = CacheRecord {
        schema: cache::SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus: corpus.record.clone(),
        encoder,
        pooling,
        layers,
        splits,
        features_s: t.elapsed().as_secs_f64(),
    };
    write(&out.path, "cache.json", serde_json::to_string_pretty(&record).expect("serializes").as_bytes())?;
    out.keep = true;
    let words: u64 = record.splits.iter().map(|s| s.words).sum();
    println!(
        "{}: layers {:?} of {words} words in {:.1} s, {:.0} words/s",
        out.path.display(),
        record.layers,
        record.features_s,
        words as f64 / record.features_s.max(1e-9)
    );
    Ok(())
}

fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let bundle = PathBuf::from(a.required("--bundle")?);
    let out = OutDir::claim(&a.required("--out")?)?;
    let (device, precision, pooling) = encoder_args(&mut a)?;
    let crf = a.flag("--crf");
    let bilstm = match a.value("--bilstm")? {
        None => None,
        Some(v) => {
            let parse = |s: &str| s.parse::<u32>().map_err(|_| Error(format!("--bilstm {v:?}: <hidden>x<layers>")));
            let (h, l) = v.split_once('x').ok_or_else(|| Error(format!("--bilstm {v:?}: <hidden>x<layers>")))?;
            Some((parse(h)?, parse(l)?))
        }
    };
    let hash_dim: Option<u32> =
        a.value("--hash-embed")?.map(|v| v.parse()).transpose().map_err(|_| Error("--hash-embed <dim>".into()))?;
    let rows = match a.value("--hash-rows")? {
        None => HASH_ROWS,
        Some(v) => {
            let r: Vec<u32> = v.split(',').map(str::parse).collect::<std::result::Result<_, _>>().unwrap_or_default();
            r.try_into().map_err(|_| Error(format!("--hash-rows {v:?}: four counts, comma separated")))?
        }
    };
    let hash = hash_dim.map(|dim| HashEmbed { dim, rows });
    let d = Config::default();
    let c = Config {
        seed: a.parsed("--seed", d.seed)?,
        epochs: a.parsed("--epochs", d.epochs)?,
        batch: a.parsed("--batch", d.batch)?,
        lr: a.parsed("--lr", d.lr)?,
        patience: a.parsed("--patience", if bilstm.is_some() { BILSTM_PATIENCE } else { d.patience })?,
        tolerance: a.parsed("--tolerance", d.tolerance)?,
        threads: a.parsed("--threads", d.threads)?,
        verbose: true,
        ..d
    };
    let max_train: usize = a.parsed("--max-train", 0)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;

    let t = Instant::now();
    let corpus = read_corpus(corpus_id, dir, unrecorded)?;
    let read = t.elapsed().as_secs_f64();
    let head = spec(&corpus, &HeadOptions { pooling, crf, bilstm, hash })?;
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
    let feature_words = train_ex.iter().chain(&dev_ex).map(|e| e.labels.len() as u64).sum();

    // The best epoch's head is on disk whenever it improves, the epoch and
    // its dev score in the same file's metadata, so a run stopped midway
    // keeps a head and the epoch it is from, replaced together by one
    // rename; from the first one on, the directory stays if the run fails.
    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&head).expect("a head spec serializes");
    let hidden = enc.hidden();
    let mut out = out;
    let (model, report) = {
        let dir = out.path.clone();
        let mut saved = false;
        let mut save = |m: &turbo_trainer::Tagger<f32>, epoch: usize, dev: f64| -> turbo_trainer_recipes::Result<()> {
            if !saved {
                replace(&dir, "head.json", &spec_json)?;
                saved = true;
            }
            let at = [(CHECKPOINT_EPOCH, epoch.to_string()), (CHECKPOINT_DEV, dev.to_string())];
            replace(&dir, "head.safetensors", &safetensors::write_with(&named(&head, hidden, m.tensors()), &at))?;
            Ok(())
        };
        let r = turbo_trainer::fit_with(&head, hidden, &train_ex, &dev_ex, &c, &mut |m, e, d| {
            save(m, e, d).map_err(|x| turbo::status::invalid(x.0))
        });
        out.keep = out.keep || saved_any(&dir);
        r.map_err(|e| Error(e.message))?
    };
    drop((train_ex, dev_ex));
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let tagger = model.to_token_tagger().map_err(|e| Error(e.message))?;
    let test = &corpus.splits.test;
    let predicted = predict(&mut enc, &tagger, test)?;
    let base = baseline(train_set, test);
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
    let outputs = vec![replace(&out.path, "head.safetensors", &weights)?, replace(&out.path, "head.json", &spec_json)?];
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
            tolerance: c.tolerance,
            threads: c.threads,
            max_train,
            dev_left_out,
        }),
        training: Some(TrainingRecord {
            train_loss: report.train_loss,
            dev: report.dev,
            best_epoch: report.best_epoch,
            epoch_seconds: report.epoch_seconds,
        }),
        eval,
        timing_s: Timing { read, features, train: trained, eval: eval_s, feature_words },
        outputs,
    };
    finish(out, receipt)
}

/// The labels for `test` of the majority baseline learned from `train`,
/// the sentences the model is given, so the two compare on equal data.
fn baseline(train: &[Sentence], test: &[Sentence]) -> Vec<Vec<String>> {
    let m = Majority::train(train);
    test.iter().map(|s| m.predict(&s.words)).collect()
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
        layers: m.architecture.as_ref().map(|a| a.layers),
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

/// Whether a checkpoint was written into `dir`.
fn saved_any(dir: &Path) -> bool {
    dir.join("head.safetensors").is_file()
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
    if let Some(w) = receipt.timing_s.words_per_s() {
        println!(
            "features: {} train and dev words in {:.1} s, {w:.0} words/s",
            receipt.timing_s.feature_words, receipt.timing_s.features
        );
    }
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
