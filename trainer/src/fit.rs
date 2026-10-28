//! The training loop for a token tagger head: seeded shuffles, minibatch
//! gradients summed in a fixed order, AdamW, and early stopping on a dev
//! split scored by the core's own inference path.

use turbo::head::TokenTagger;
use turbo::head::crf::{WordSpan, spans};
use turbo::head::tagger::{NO_WORD, Work as CoreWork};
use turbo::manifest::{HeadSpec, Scheme};
use turbo::status::{Result, invalid};

use crate::adamw::{AdamW, Settings};
use crate::rng::Rng;
use crate::tagger::{Tagger, Work};

/// One training sequence: per-word features [words, hidden] and a gold
/// label index per word.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub features: Vec<f32>,
    pub labels: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub seed: u64,
    pub epochs: u32,
    /// Sequences per step.
    pub batch: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub clip: f64,
    /// The share of all steps spent warming up, in [0, 1).
    pub warmup: f64,
    /// Epochs without a better dev score before stopping; 0 never stops early.
    pub patience: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config { seed: 1, epochs: 20, batch: 16, lr: 2e-3, weight_decay: 0.01, clip: 5.0, warmup: 0.1, patience: 3 }
    }
}

/// What a run did, epoch by epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Mean loss per word over each epoch's training steps.
    pub train_loss: Vec<f64>,
    /// The dev score after each epoch: accuracy under SCHEME_TAGS, exact
    /// span F1 under BIO and BILOU.
    pub dev: Vec<f64>,
    /// The epoch (from 0) whose weights were kept.
    pub best_epoch: usize,
}

/// Accuracy (SCHEME_TAGS) or exact span F1 (BIO, BILOU) of `tagger` on
/// `examples`, decoded by the core exactly as at inference.
pub fn score(tagger: &TokenTagger, examples: &[Example]) -> Result<f64> {
    let spec = tagger.spec();
    let hidden = tagger.hidden();
    let (mut right, mut total, mut found, mut gold_n) = (0usize, 0usize, 0usize, 0usize);
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        debug_assert!(!words.contains(&NO_WORD));
        tagger.scores(&e.features, &words, &mut scores, &mut work)?;
        tagger.decode(&scores, n, &mut tags)?;
        if spec.scheme == Scheme::Tags {
            right += tags.iter().zip(&e.labels).filter(|(a, b)| a == b).count();
            total += n;
        } else {
            let want: Vec<WordSpan> = spans(spec.scheme, &spec.labels, &e.labels)?;
            let got = spans(spec.scheme, &spec.labels, &tags)?;
            right += got.iter().filter(|s| want.contains(s)).count();
            found += got.len();
            gold_n += want.len();
        }
    }
    Ok(if spec.scheme == Scheme::Tags {
        if total == 0 { 0.0 } else { right as f64 / total as f64 }
    } else if found + gold_n == 0 {
        1.0
    } else {
        2.0 * right as f64 / (found + gold_n) as f64
    })
}

/// Trains a head for `spec` over features `hidden` wide on `train`,
/// keeping the weights that scored best on `dev`.
pub fn fit(
    spec: &HeadSpec,
    hidden: usize,
    train: &[Example],
    dev: &[Example],
    c: &Config,
) -> Result<(Tagger<f32>, Report)> {
    if train.is_empty() || c.batch == 0 || c.epochs == 0 || !(0.0..1.0).contains(&c.warmup) {
        return Err(invalid(
            "trainer: an empty training set, or a zero batch, epoch count, or a warmup outside [0, 1)",
        ));
    }
    for (i, e) in train.iter().chain(dev).enumerate() {
        if e.features.len() != e.labels.len() * hidden || e.labels.iter().any(|&l| l as usize >= spec.labels.len()) {
            return Err(invalid(format!(
                "trainer: example {i} does not fit {hidden} features per word and its labels"
            )));
        }
    }
    let mut rng = Rng::new(c.seed);
    let mut model = Tagger::<f32>::new(spec, hidden, &mut rng)?;
    let steps_per_epoch = train.len().div_ceil(c.batch) as u64;
    let total = steps_per_epoch * c.epochs as u64;
    let settings = Settings {
        lr: c.lr,
        weight_decay: c.weight_decay,
        clip: c.clip,
        warmup: (total as f64 * c.warmup) as u64,
        total,
        ..Settings::default()
    };
    let shapes: Vec<usize> = model.params.iter().map(Vec::len).collect();
    let mut opt = AdamW::new(settings, &shapes);
    let decay = model.decays();
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut work = Work::default();
    let mut report = Report { train_loss: Vec::new(), dev: Vec::new(), best_epoch: 0 };
    let mut best: Option<(f64, Vec<Vec<f32>>)> = None;
    for epoch in 0..c.epochs as usize {
        rng.shuffle(&mut order);
        let (mut loss, mut words) = (0.0, 0usize);
        for batch in order.chunks(c.batch) {
            let mut grads = model.zero_grads();
            let n: usize = batch.iter().map(|&i| train[i].labels.len()).sum();
            for &i in batch {
                loss += model.loss(&train[i].features, &train[i].labels, &mut grads, &mut work)?;
            }
            words += n;
            if n == 0 {
                continue;
            }
            let scale = 1.0 / n as f32;
            grads.iter_mut().flatten().for_each(|g| *g *= scale);
            opt.step(&mut model.params, &mut grads, &decay);
        }
        report.train_loss.push(if words == 0 { 0.0 } else { loss / words as f64 });
        let s = score(&model.to_token_tagger()?, dev)?;
        report.dev.push(s);
        if best.as_ref().is_none_or(|(b, _)| s > *b) {
            best = Some((s, model.params.clone()));
            report.best_epoch = epoch;
        } else if c.patience > 0 && epoch - report.best_epoch >= c.patience as usize {
            break;
        }
    }
    if let Some((_, params)) = best {
        model.params = params;
    }
    Ok((model, report))
}
