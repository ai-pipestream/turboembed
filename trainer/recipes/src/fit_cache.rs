//! `turbo-train-head fit-cache` (#1616): a head trained on a cache of
//! per-layer word features (`cache`), on one layer or a learned mix of
//! several, scored on dev without the encoder. The comparison of a single
//! layer with the mix is the dev scores of runs like these; the winner is
//! then scored on test through the encoder.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use turbo::manifest::{HeadSpec, LayerMix};
use turbo_trainer::Example;
use turbo_trainer::mix::{MixExample, fit_mix};

use crate::cache::{CacheRecord, read_layer};
use crate::cli::{Args, HeadArgs, OutDir, commit, head_args};
use crate::corpus::Sentence;
use crate::receipt::unix_time;
use crate::run::{HeadOptions, spec, write};
use crate::{Error, Result, read_corpus};

pub const SCHEMA: &str = "turboembed.fit-cache.v1";

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
    let index: BTreeMap<&str, u32> = labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let ids: Vec<Option<Vec<u32>>> = sentences
        .iter()
        .map(|s| s.labels.iter().map(|l| index.get(l.as_str()).copied()).collect::<Option<Vec<u32>>>())
        .collect();
    let mut out: Vec<MixExample> = sentences
        .iter()
        .zip(&ids)
        .filter_map(|(s, l)| l.clone().map(|labels| MixExample { layers: Vec::new(), labels, forms: s.words.clone() }))
        .collect();
    for &l in layers {
        let feats = read_layer(dir, record, split, l, sentences)?;
        let kept = feats.into_iter().zip(&ids).filter(|(_, l)| l.is_some()).map(|(f, _)| f);
        out.iter_mut().zip(kept).for_each(|(e, f)| e.layers.push(f));
    }
    let left = sentences.len() - out.len();
    Ok((out, left))
}

/// The command: `--cache <dir>` from `cache` on this corpus, and either
/// `--layer N` or `--mix` (all cached layers, or from `--from-layer N`).
///
/// # Errors
///
/// Arguments, a cache of another corpus, and what training refuses.
pub fn command(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let cache_dir = std::path::PathBuf::from(a.required("--cache")?);
    let mut out = OutDir::claim(&a.required("--out")?)?;
    let layer: Option<u32> =
        a.value("--layer")?.map(|v| v.parse()).transpose().map_err(|_| Error("--layer N".into()))?;
    let mixed = a.flag("--mix");
    let from: Option<u32> =
        a.value("--from-layer")?.map(|v| v.parse()).transpose().map_err(|_| Error("--from-layer N".into()))?;
    let HeadArgs { crf, bilstm, hash, config: c } = head_args(&mut a)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;
    let p = cache_dir.join("cache.json");
    let text = std::fs::read(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let cache: CacheRecord = serde_json::from_slice(&text).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let layers: Vec<u32> = match (layer, mixed) {
        (Some(l), false) if from.is_none() => vec![l],
        (None, true) => cache.layers.iter().copied().filter(|&l| from.is_none_or(|f| l >= f)).collect(),
        _ => return Err(Error("one of --layer N or --mix [--from-layer N]".into())),
    };
    if layers.is_empty() || layers.iter().any(|l| !cache.layers.contains(l)) {
        return Err(Error(format!("layers {layers:?}: the cache holds {:?}", cache.layers)));
    }
    let t = Instant::now();
    let corpus = read_corpus(corpus_id, dir, unrecorded)?;
    if corpus.record != cache.corpus {
        return Err(Error(format!("{}: a cache of another corpus or release", cache_dir.display())));
    }
    let mut head = spec(&corpus, &HeadOptions { pooling: cache.pooling, crf, bilstm, hash })?;
    let hidden = cache.encoder.hidden;
    let s = &corpus.splits;
    let (train, train_left_out) = examples_at(&cache_dir, &cache, "train", &layers, &s.train, &head.labels)?;
    let (dev, dev_left_out) = examples_at(&cache_dir, &cache, "dev", &layers, &s.dev, &head.labels)?;
    let read_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (report, mix) = if layers.len() == 1 && !mixed {
        let one = |v: Vec<MixExample>| -> Vec<Example> {
            v.into_iter()
                .map(|mut e| Example { features: e.layers.remove(0), labels: e.labels, forms: e.forms })
                .collect()
        };
        let (train, dev) = (one(train), one(dev));
        let (model, report) = turbo_trainer::fit(&head, hidden, &train, &dev, &c).map_err(|e| Error(e.message))?;
        write_head(&out.path, &head, hidden, model.tensors())?;
        (report, None)
    } else {
        head.layer_mix = Some(LayerMix { layers: layers.len() as u32 });
        let (model, report) = fit_mix(&head, hidden, layers.len(), &train, &dev, &c, &mut |_, _, _| Ok(()))
            .map_err(|e| Error(e.message))?;
        write_head(&out.path, &head, hidden, model.tensors())?;
        let (w, gain) = model.weights();
        (report, Some(Mix { weights: w.iter().map(|&v| f64::from(v)).collect(), gain: f64::from(gain) }))
    };
    let record = FitRecord {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        cache,
        layers,
        mix,
        head,
        seed: c.seed,
        epochs: c.epochs,
        batch: c.batch,
        lr: c.lr,
        patience: c.patience,
        tolerance: c.tolerance,
        train_left_out,
        dev_left_out,
        best_dev: report.dev[report.best_epoch],
        train_loss: report.train_loss,
        dev: report.dev,
        best_epoch: report.best_epoch + 1,
        read_s,
        train_s: t.elapsed().as_secs_f64(),
    };
    write(&out.path, "fit.json", serde_json::to_string_pretty(&record).expect("serializes").as_bytes())?;
    out.keep = true;
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

fn write_head(dir: &Path, head: &HeadSpec, hidden: usize, tensors: Vec<(String, Vec<f32>)>) -> Result<()> {
    let w = crate::safetensors::write(&crate::run::named(head, hidden, tensors));
    write(dir, "head.safetensors", &w)?;
    write(dir, "head.json", &serde_json::to_vec_pretty(head).expect("a head spec serializes"))?;
    Ok(())
}
