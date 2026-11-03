//! `turbo-train-head fit-cache` (and alias `train-cached`, `#1616`, `#1822`):
//! a head trained on a cache of per-layer word features (`cache`), on one
//! layer (`--layer N`) or a learned scalar mix (`--mix [--from-layer N]`),
//! across `tag`, `lemma`, `parse`, and `split` tasks. When the cache holds
//! the `"test"` split, the trained head is also scored on `"test"` and
//! writes `receipt.json` alongside `fit.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use turbo::manifest::{Biaffine, HeadSpec};
use turbo_trainer::mix::{MixExample, fit_mix};
use turbo_trainer::{Example, Report};

use crate::cache::{CacheRecord, layer_mix_for, read_layer, same_corpus_release};
use crate::cached_ud::{CachedParams, eval_cached_tagger, run_split, run_ud};
use crate::cli::{Args, HeadArgs, OutDir, commit, head_args};
use crate::corpus::{CorpusRecord, Sentence};
use crate::receipt::{ConfigRecord, EvalRecord, Receipt, SCHEMA as RECEIPT_SCHEMA, Timing, TrainingRecord, unix_time};
use crate::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, HeadOptions, named as named_tag, spec, write};
use crate::ud_heads::{UdHead, named as named_ud};
use crate::{Error, Result, read_corpus, safetensors};

pub type TestEval = Option<(EvalRecord, f64, Vec<Sentence>, Vec<Vec<String>>)>;

pub const SCHEMA: &str = "turboembed.fit-cache.v1";

/// Default minimum frequency for edit-tree mining (`lemma` and `split`).
pub const DEFAULT_MIN_TREE_FREQ: usize = 3;
/// Default biaffine arc projection width (`parse`).
pub const DEFAULT_ARC: u32 = 256;
/// Default biaffine relation projection width (`parse`).
pub const DEFAULT_REL: u32 = 64;

/// Which head family a cached training or sweep run fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTask {
    Tag,
    Lemma,
    Parse,
    Split,
}

impl CacheTask {
    /// Parses `--task tag|lemma|parse|deps|split`.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "tag" | "tags" | "upos" | "xpos" | "feats" => Ok(Self::Tag),
            "lemma" => Ok(Self::Lemma),
            "parse" | "deps" => Ok(Self::Parse),
            "split" => Ok(Self::Split),
            _ => Err(Error(format!("--task {s:?}: tag, lemma, parse or split"))),
        }
    }

    /// Infers a default task from `corpus_id` when `--task` is omitted.
    pub fn infer(corpus_id: &str) -> Self {
        if corpus_id.ends_with("-lemma") {
            Self::Lemma
        } else if corpus_id.ends_with("-deps") || corpus_id.ends_with("-parse") {
            Self::Parse
        } else if corpus_id.ends_with("-split") {
            Self::Split
        } else {
            Self::Tag
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tag => "tag",
            Self::Lemma => "lemma",
            Self::Parse => "parse",
            Self::Split => "split",
        }
    }
}

/// What a run on a cache did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitRecord {
    pub schema: String,
    pub command: Vec<String>,
    pub commit: String,
    pub unix_time: u64,
    /// The cache's record, as read.
    pub cache: CacheRecord,
    /// The layers the head read, counted from 1: one, or those mixed.
    pub layers: Vec<u32>,
    /// The mix's weights over `layers` and its gain, for a mix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mix: Option<Mix>,
    pub head: HeadSpec,
    pub seed: u64,
    pub epochs: u32,
    pub batch: usize,
    pub lr: f64,
    pub patience: u32,
    pub tolerance: f64,
    /// Sentences left out for a label the head does not have.
    pub train_left_out: usize,
    pub dev_left_out: usize,
    pub train_loss: Vec<f64>,
    pub dev: Vec<f64>,
    pub best_epoch: usize,
    pub best_dev: f64,
    pub read_s: f64,
    pub train_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mix {
    pub weights: Vec<f64>,
    pub gain: f64,
}

/// Output of [`run_cached`], including optional test-split [`Receipt`] and
/// predictions for [`crate::sweep`].
#[derive(Debug, Clone)]
pub struct CachedOutcome {
    pub fit: FitRecord,
    pub receipt: Option<Receipt>,
    pub test_gold: Vec<Sentence>,
    pub test_pred: Vec<Vec<String>>,
}

/// Each sentence of `split` whose labels are all the head's, with its
/// features at each of `layers` from the cache, and how many were left
/// out.
///
/// # Errors
///
/// What [`read_layer`] refuses.
pub fn examples_at(
    dir: &Path,
    record: &CacheRecord,
    split: &str,
    layers: &[u32],
    sentences: &[Sentence],
    labels: &[String],
) -> Result<(Vec<MixExample>, usize)> {
    examples_prefix(dir, record, split, layers, sentences, labels, sentences.len())
}

/// Select a source prefix before label filtering, while validating and
/// reading the full cache so retained features keep their original rows.
/// `limit` counts source sentences, including those with unsupported labels.
///
/// # Errors
///
/// What [`read_layer`] refuses.
pub(crate) fn examples_prefix(
    dir: &Path,
    record: &CacheRecord,
    split: &str,
    layers: &[u32],
    sentences: &[Sentence],
    labels: &[String],
    limit: usize,
) -> Result<(Vec<MixExample>, usize)> {
    let selected = &sentences[..limit.min(sentences.len())];
    let index: BTreeMap<&str, u32> = labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let ids: Vec<Option<Vec<u32>>> = selected
        .iter()
        .map(|s| s.labels.iter().map(|l| index.get(l.as_str()).copied()).collect::<Option<Vec<u32>>>())
        .collect();
    let mut out: Vec<MixExample> = selected
        .iter()
        .zip(&ids)
        .filter_map(|(s, l)| l.clone().map(|labels| MixExample { layers: Vec::new(), labels, forms: s.words.clone() }))
        .collect();
    for &l in layers {
        let feats = read_layer(dir, record, split, l, sentences)?;
        let kept = feats.into_iter().zip(&ids).filter(|(_, l)| l.is_some()).map(|(f, _)| f);
        out.iter_mut().zip(kept).for_each(|(e, f)| e.layers.push(f));
    }
    let left = selected.len() - out.len();
    Ok((out, left))
}

/// Reads `<cache_dir>/cache.json`.
pub fn read_cache_record(cache_dir: &Path) -> Result<CacheRecord> {
    let p = cache_dir.join("cache.json");
    let text = std::fs::read(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    serde_json::from_slice(&text).map_err(|e| Error(format!("{}: {e}", p.display())))
}

/// Resolves `--layer N` or `--mix [--from-layer N]` against `cache.layers`.
pub fn resolve_layers(cache: &CacheRecord, layer: Option<u32>, mixed: bool, from: Option<u32>) -> Result<Vec<u32>> {
    let layers: Vec<u32> = match (layer, mixed) {
        (Some(l), false) if from.is_none() => vec![l],
        (None, true) => cache.layers.iter().copied().filter(|&l| from.is_none_or(|f| l >= f)).collect(),
        _ => return Err(Error("one of --layer N or --mix [--from-layer N]".into())),
    };
    if layers.is_empty() || layers.iter().any(|l| !cache.layers.contains(l)) {
        return Err(Error(format!("layers {layers:?}: the cache holds {:?}", cache.layers)));
    }
    Ok(layers)
}

/// The command: `--cache <dir>` from `cache` on this corpus, and either
/// `--layer N` or `--mix` (all cached layers, or from `--from-layer N`).
///
/// # Errors
///
/// Arguments, a cache of another corpus, and what training refuses.
pub fn command(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, a: Args) -> Result<()> {
    command_with_task(argv, corpus_id, dir, unrecorded, None, a)
}

/// Runs `fit-cache` with an optional pre-selected [`CacheTask`] (used when
/// `train`, `train-lemma`, `train-parse`, or `train-split` is invoked with
/// `--cache`).
pub fn command_with_task(
    argv: &[String],
    corpus_id: &str,
    dir: &Path,
    unrecorded: bool,
    forced_task: Option<CacheTask>,
    mut a: Args,
) -> Result<()> {
    let cache_dir = PathBuf::from(a.required("--cache")?);
    let mut out = OutDir::claim(&a.required("--out")?)?;
    let flag_task = a.value("--task")?.map(|s| CacheTask::parse(&s)).transpose()?;
    let task = match (forced_task, flag_task) {
        (Some(f), Some(g)) if f != g => {
            return Err(Error(format!("--task {} conflicts with command task {}", g.as_str(), f.as_str())));
        }
        (Some(f), _) => f,
        (None, Some(g)) => g,
        (None, None) => CacheTask::infer(corpus_id),
    };
    let layer: Option<u32> =
        a.value("--layer")?.map(|v| v.parse()).transpose().map_err(|_| Error("--layer N".into()))?;
    let mixed = a.flag("--mix");
    let from: Option<u32> =
        a.value("--from-layer")?.map(|v| v.parse()).transpose().map_err(|_| Error("--from-layer N".into()))?;
    let min_tree_freq: usize = a.parsed("--min-tree-freq", DEFAULT_MIN_TREE_FREQ)?;
    let seen_lookup: u32 = a.parsed("--seen-lookup", 0)?;
    let biaffine = Biaffine { arc: a.parsed("--arc", DEFAULT_ARC)?, rel: a.parsed("--rel", DEFAULT_REL)? };
    let max_train: usize = a.parsed("--max-train", 0)?;
    let HeadArgs { crf, bilstm, hash, config } = head_args(&mut a)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;

    let cache = read_cache_record(&cache_dir)?;
    let layers = resolve_layers(&cache, layer, mixed, from)?;
    let params = CachedParams {
        argv: argv.to_vec(),
        commit,
        corpus_id: corpus_id.to_owned(),
        corpus_dir: dir.to_path_buf(),
        cache_dir,
        out_dir: out.path.clone(),
        unrecorded,
        layers,
        mixed,
        crf,
        bilstm,
        hash,
        min_tree_freq,
        seen_lookup,
        biaffine,
        config,
        max_train,
    };
    let outcome = run_cached(&params, task, cache)?;
    out.keep = true;
    let record = &outcome.fit;
    println!(
        "{}: layers {:?}, dev {:.4} at epoch {}{}",
        out.path.display(),
        record.layers,
        record.best_dev,
        record.best_epoch,
        record.mix.as_ref().map_or(String::new(), |m| format!(", mix {:.3?} gain {:.3}", m.weights, m.gain))
    );
    Ok(())
}

/// Runs a cached training job (`tag`, `lemma`, `parse`, or `split`) into
/// `p.out_dir`.
pub fn run_cached(p: &CachedParams, task: CacheTask, cache: CacheRecord) -> Result<CachedOutcome> {
    if p.layers.is_empty() || p.layers.iter().any(|l| !cache.layers.contains(l)) {
        return Err(Error(format!("layers {:?}: the cache holds {:?}", p.layers, cache.layers)));
    }
    match task {
        CacheTask::Tag => run_tag(p, cache),
        CacheTask::Lemma => run_ud(p, UdHead::Lemma, cache),
        CacheTask::Parse => run_ud(p, UdHead::Deps, cache),
        CacheTask::Split => run_split(p, cache),
    }
}

fn run_tag(p: &CachedParams, cache: CacheRecord) -> Result<CachedOutcome> {
    let t = Instant::now();
    let mut corpus = read_corpus(&p.corpus_id, &p.corpus_dir, p.unrecorded)?;
    if !same_corpus_release(&corpus.record, &cache.corpus) {
        return Err(Error(format!("{}: a cache of another corpus or release", p.cache_dir.display())));
    }
    let full_train = corpus.splits.train.clone();
    if p.max_train > 0 {
        corpus.splits.train.truncate(p.max_train);
    }
    let mut head = spec(&corpus, &HeadOptions { pooling: cache.pooling, crf: p.crf, bilstm: p.bilstm, hash: p.hash })?;
    head.layer_mix = layer_mix_for(&p.layers, p.mixed, cache.encoder.layers)?;
    let hidden = cache.encoder.hidden;
    let s = &corpus.splits;
    let (train, train_left_out) =
        examples_prefix(&p.cache_dir, &cache, "train", &p.layers, &full_train, &head.labels, s.train.len())?;
    let (dev, dev_left_out) = examples_at(&p.cache_dir, &cache, "dev", &p.layers, &s.dev, &head.labels)?;
    let read_s = t.elapsed().as_secs_f64();
    let feature_words: u64 = train.iter().chain(&dev).map(|e| e.labels.len() as u64).sum();

    let t = Instant::now();
    let (tensors, tagger, report, mix) = if p.layers.len() == 1 && !p.mixed {
        let one = |v: Vec<MixExample>| -> Vec<Example> {
            v.into_iter()
                .map(|mut e| Example { features: e.layers.remove(0), labels: e.labels, forms: e.forms })
                .collect()
        };
        let (train, dev) = (one(train), one(dev));
        let (model, report) =
            turbo_trainer::fit(&head, hidden, &train, &dev, &p.config).map_err(|e| Error(e.message))?;
        let tagger = model.to_token_tagger().map_err(|e| Error(e.message))?;
        (model.tensors(), tagger, report, None)
    } else {
        let (model, report) = fit_mix(&head, hidden, p.layers.len(), &train, &dev, &p.config, &mut |_, _, _| Ok(()))
            .map_err(|e| Error(e.message))?;
        let (w, gain) = model.weights();
        let mix = Mix { weights: w.iter().map(|&v| f64::from(v)).collect(), gain: f64::from(gain) };
        let tagger = model.to_token_tagger().map_err(|e| Error(e.message))?;
        (model.tensors(), tagger, report, Some(mix))
    };
    let train_s = t.elapsed().as_secs_f64();
    let test_eval = eval_cached_tagger(p, &cache, &tagger, &s.train, &s.test, corpus.record.task)?;
    finish_cached(
        p,
        cache,
        corpus.record,
        head,
        tensors,
        report,
        mix,
        train_left_out,
        dev_left_out,
        read_s,
        train_s,
        feature_words,
        test_eval,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn finish_cached(
    p: &CachedParams,
    cache: CacheRecord,
    corpus: CorpusRecord,
    head: HeadSpec,
    tensors: Vec<(String, Vec<f32>)>,
    report: Report,
    mix: Option<Mix>,
    train_left_out: usize,
    dev_left_out: usize,
    read_s: f64,
    train_s: f64,
    feature_words: u64,
    test_eval: TestEval,
) -> Result<CachedOutcome> {
    let hidden = cache.encoder.hidden;
    let best = report.best_epoch;
    let at = [(CHECKPOINT_EPOCH, (best + 1).to_string()), (CHECKPOINT_DEV, report.dev[best].to_string())];
    let named = match head.kind {
        turbo::manifest::HeadKind::EditTreeLemmatizer => named_ud(UdHead::Lemma, &head, hidden, tensors),
        turbo::manifest::HeadKind::BiaffineParser => named_ud(UdHead::Deps, &head, hidden, tensors),
        _ => named_tag(&head, hidden, tensors),
    };
    let weights = safetensors::write_with(&named, &at);
    let spec_json = serde_json::to_vec_pretty(&head).expect("a head spec serializes");
    let outputs = vec![write(&p.out_dir, "head.safetensors", &weights)?, write(&p.out_dir, "head.json", &spec_json)?];
    let fit = FitRecord {
        schema: SCHEMA.into(),
        command: p.argv.clone(),
        commit: p.commit.clone(),
        unix_time: unix_time(),
        cache: cache.clone(),
        layers: p.layers.clone(),
        mix,
        head: head.clone(),
        seed: p.config.seed,
        epochs: p.config.epochs,
        batch: p.config.batch,
        lr: p.config.lr,
        patience: p.config.patience,
        tolerance: p.config.tolerance,
        train_left_out,
        dev_left_out,
        best_dev: report.dev[report.best_epoch],
        train_loss: report.train_loss.clone(),
        dev: report.dev.clone(),
        best_epoch: report.best_epoch + 1,
        read_s,
        train_s,
    };
    write(&p.out_dir, "fit.json", serde_json::to_string_pretty(&fit).expect("serializes").as_bytes())?;
    let (receipt, test_gold, test_pred) = match test_eval {
        Some((eval, eval_s, gold, pred)) => {
            let receipt = Receipt {
                schema: RECEIPT_SCHEMA.into(),
                command: p.argv.clone(),
                commit: p.commit.clone(),
                unix_time: fit.unix_time,
                corpus,
                encoder: Some(cache.encoder),
                head: Some(head),
                peer: None,
                config: Some(ConfigRecord {
                    seed: p.config.seed,
                    epochs: p.config.epochs,
                    batch: p.config.batch,
                    lr: p.config.lr,
                    weight_decay: p.config.weight_decay,
                    clip: p.config.clip,
                    warmup: p.config.warmup,
                    patience: p.config.patience,
                    tolerance: p.config.tolerance,
                    threads: p.config.threads,
                    max_train: p.max_train,
                    dev_left_out,
                }),
                training: Some(TrainingRecord {
                    train_loss: report.train_loss,
                    dev: report.dev,
                    best_epoch: report.best_epoch + 1,
                    epoch_seconds: report.epoch_seconds,
                }),
                eval,
                timing_s: Timing { read: read_s, features: 0.0, train: train_s, eval: eval_s, feature_words },
                outputs,
            };
            write(&p.out_dir, "receipt.json", serde_json::to_string_pretty(&receipt).expect("serializes").as_bytes())?;
            (Some(receipt), gold, pred)
        }
        None => (None, Vec::new(), Vec::new()),
    };
    Ok(CachedOutcome { fit, receipt, test_gold, test_pred })
}
