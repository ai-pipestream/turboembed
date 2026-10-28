//! The training loop for a token tagger head: seeded shuffles, minibatch
//! gradients summed in a fixed order, AdamW, and early stopping on a dev
//! split scored by the core's own inference path.

use turbo::head::TokenTagger;
use turbo::head::crf::{WordSpan, spans};
use turbo::head::tagger::{NO_WORD, Work as CoreWork};
use turbo::manifest::{HeadSpec, Scheme};
use turbo::status::{Result, invalid};

use std::time::Instant;

use crate::adamw::{AdamW, Settings};
use crate::batch::Batch;
use crate::rng::Rng;
use crate::tagger::{Tagger, Work};

/// One training sequence: per-word features [words, hidden], a gold
/// label index per word, and each word's text, which a head with a
/// `hash_embed` reads (empty for a head that reads none).
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub features: Vec<f32>,
    pub labels: Vec<u32>,
    pub forms: Vec<String>,
}

impl Example {
    /// The forms as the heads take them: None when there are none.
    pub fn forms(&self) -> Option<Vec<&str>> {
        (!self.forms.is_empty()).then(|| self.forms.iter().map(String::as_str).collect())
    }
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
    /// Epochs after warmup without a dev score within `tolerance` of the
    /// best before stopping; 0 never stops early.
    pub patience: u32,
    /// How far under the best a dev score may fall and still count as
    /// holding level for `patience`: a score holds level when it is
    /// better than `best - tolerance`. So under the default 0.001 an equal
    /// score holds level, and a run whose dev score sits flat (often a
    /// small dev set's accuracy) runs to `epochs`, the learning rate
    /// still decaying; under 0 only a better score holds, and a flat run
    /// stops after `patience` epochs. The best weights are kept only on a
    /// strictly better score.
    pub tolerance: f64,
    /// Threads a batch's examples are spread over; 0 takes every core.
    /// The weights a run gives do not depend on it.
    pub threads: usize,
    /// Logs each epoch's loss, dev score and time, and progress within
    /// it, to standard error.
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            seed: 1,
            epochs: 60,
            batch: 16,
            lr: 2e-3,
            weight_decay: 0.01,
            clip: 5.0,
            warmup: 0.1,
            patience: 6,
            tolerance: 1e-3,
            threads: 0,
            verbose: false,
        }
    }
}

/// Early stopping on the dev score: armed only once warmup is over, so a
/// plateau while the rate is still rising never stops a run, and a score
/// within `tolerance` of the best holds the run going.
#[derive(Debug, Clone)]
pub struct EarlyStop {
    warm: usize,
    patience: usize,
    tolerance: f64,
    best: Option<f64>,
    /// The last epoch whose score held level.
    level: usize,
}

impl EarlyStop {
    /// For a run of `c`: warmup is its first `ceil(warmup * epochs)` epochs.
    pub fn new(c: &Config) -> EarlyStop {
        let warm = (c.warmup * c.epochs as f64).ceil() as usize;
        EarlyStop { warm, patience: c.patience as usize, tolerance: c.tolerance, best: None, level: 0 }
    }

    /// Takes epoch `epoch`'s dev score; returns whether it is the best yet
    /// (keep these weights) and whether to stop after it.
    pub fn update(&mut self, epoch: usize, score: f64) -> (bool, bool) {
        let better = self.best.is_none_or(|b| score > b);
        if better {
            self.best = Some(score);
        }
        if better || self.best.is_some_and(|b| score > b - self.tolerance) {
            self.level = epoch;
        }
        // Warmup epochs neither count against patience nor run it down.
        let since = epoch.saturating_sub(self.level.max(self.warm.saturating_sub(1)));
        (better, self.patience > 0 && epoch >= self.warm && since >= self.patience)
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
    /// Wall seconds each epoch took, its dev scoring included; the one
    /// field two runs of the same seed do not share.
    pub epoch_seconds: Vec<f64>,
}

/// Accuracy (SCHEME_TAGS) or exact span F1 (BIO, BILOU) of `tagger` on
/// `examples`, decoded by the core exactly as at inference. Refuses an
/// example without exactly one gold label in range per feature row.
pub fn score(tagger: &TokenTagger, examples: &[Example]) -> Result<f64> {
    let spec = tagger.spec();
    let hidden = tagger.hidden();
    check_examples(examples, hidden, spec.labels.len(), tagger.reads_forms())?;
    let (mut right, mut total, mut found, mut gold_n) = (0usize, 0usize, 0usize, 0usize);
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        debug_assert!(!words.contains(&NO_WORD));
        tagger.scores_forms(&e.features, &words, e.forms().as_deref(), &mut scores, &mut work)?;
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

/// What the training loop needs of a head in training.
pub(crate) trait Model: Sized + Sync {
    type Example: Sync;
    type Work: Default + Send;
    fn params(&self) -> &[Vec<f32>];
    /// Replaces every weight; refused unless each tensor keeps its length.
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()>;
    /// One optimizer step on the weights.
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64>;
    fn zero_grads(&self) -> Vec<Vec<f32>>;
    fn decays(&self) -> Vec<bool>;
    /// The example's loss, its gradients added to `grads`, and its words.
    fn loss(&self, e: &Self::Example, grads: &mut [Vec<f32>], w: &mut Self::Work) -> Result<(f64, usize)>;
    /// The dev score through the core's inference path; higher is better.
    fn score(&self, dev: &[Self::Example]) -> Result<f64>;
}

impl Model for Tagger<f32> {
    type Example = Example;
    type Work = Work<f32>;
    fn params(&self) -> &[Vec<f32>] {
        Tagger::params(self)
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        Tagger::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        Tagger::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        Tagger::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        Tagger::decays(self)
    }
    fn loss(&self, e: &Example, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        let forms = e.forms();
        Ok((Tagger::loss_forms(self, &e.features, forms.as_deref(), &e.labels, grads, w)?, e.labels.len()))
    }
    fn score(&self, dev: &[Example]) -> Result<f64> {
        score(&self.to_token_tagger()?, dev)
    }
}

/// The settings `c` turned into the optimizer's, for `examples` a epoch.
pub(crate) fn settings(c: &Config, examples: usize) -> Settings {
    let total = examples.div_ceil(c.batch) as u64 * c.epochs as u64;
    Settings {
        lr: c.lr,
        weight_decay: c.weight_decay,
        clip: c.clip,
        warmup: (total as f64 * c.warmup) as u64,
        total,
        ..Settings::default()
    }
}

/// Refuses settings no run can use.
pub(crate) fn check_config(c: &Config, examples: usize) -> Result<()> {
    if examples == 0
        || c.batch == 0
        || c.epochs == 0
        || !(0.0..1.0).contains(&c.warmup)
        || !(c.tolerance >= 0.0 && c.tolerance.is_finite())
    {
        return Err(invalid(
            "trainer: an empty training set, a zero batch or epoch count, a warmup outside [0, 1), or a \
             tolerance not finite and at least 0",
        ));
    }
    Ok(())
}

/// Each example has `hidden` features per gold label, every label is
/// one of `labels`, and, when the head reads them (`forms`), a form per
/// label.
fn check_examples(examples: &[Example], hidden: usize, labels: usize, forms: bool) -> Result<()> {
    for (i, e) in examples.iter().enumerate() {
        let rows = e.labels.len().checked_mul(hidden);
        if hidden == 0 || rows != Some(e.features.len()) || e.labels.iter().any(|&l| l as usize >= labels) {
            return Err(invalid(format!(
                "trainer: example {i} does not have {hidden} features per gold label, each label one of {labels}"
            )));
        }
        if forms && e.forms.len() != e.labels.len() {
            return Err(invalid(format!(
                "trainer: example {i} has {} forms for {} words; the head reads each word's text",
                e.forms.len(),
                e.labels.len()
            )));
        }
    }
    Ok(())
}

/// The loop: seeded shuffles, minibatch gradients summed in a fixed
/// order (across `c.threads` threads, see [`Batch`]) and scaled per word,
/// AdamW, dev scoring each epoch, early stopping, the best weights kept.
pub(crate) fn train<M: Model>(
    mut model: M,
    rng: &mut Rng,
    train: &[M::Example],
    dev: &[M::Example],
    c: &Config,
) -> Result<(M, Report)> {
    let shapes: Vec<usize> = model.params().iter().map(Vec::len).collect();
    let mut opt = AdamW::new(settings(c, train.len()), &shapes)?;
    let decay = model.decays();
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut batch = Batch::new(c.threads);
    let mut grads = model.zero_grads();
    let mut report = Report { train_loss: Vec::new(), dev: Vec::new(), best_epoch: 0, epoch_seconds: Vec::new() };
    let mut best: Option<Vec<Vec<f32>>> = None;
    let mut stop = EarlyStop::new(c);
    let steps = train.len().div_ceil(c.batch);
    let tick = steps.div_ceil(10).max(1);
    for epoch in 0..c.epochs as usize {
        let started = Instant::now();
        rng.shuffle(&mut order);
        let (mut loss, mut words) = (0.0, 0usize);
        for (step, ids) in order.chunks(c.batch).enumerate() {
            let examples: Vec<&M::Example> = ids.iter().map(|&i| &train[i]).collect();
            let (l, n) = batch.gradients(&model, &examples, &mut grads)?;
            loss += l;
            words += n;
            if c.verbose && (step + 1) % tick == 0 && step + 1 < steps {
                let s = started.elapsed().as_secs_f64();
                eprintln!(
                    "epoch {}: step {}/{steps}, {:.1}s, {:.0} words/s",
                    epoch + 1,
                    step + 1,
                    s,
                    words as f64 / s.max(1e-9)
                );
            }
            if n == 0 {
                continue;
            }
            let scale = 1.0 / n as f32;
            grads.iter_mut().flatten().for_each(|g| *g *= scale);
            model.step(&mut opt, &mut grads, &decay)?;
        }
        let train_s = started.elapsed().as_secs_f64();
        let mean = if words == 0 { 0.0 } else { loss / words as f64 };
        report.train_loss.push(mean);
        let s = model.score(dev)?;
        report.dev.push(s);
        let seconds = started.elapsed().as_secs_f64();
        report.epoch_seconds.push(seconds);
        if c.verbose {
            eprintln!(
                "epoch {}/{}: loss {mean:.5}, dev {s:.4}, {seconds:.1}s ({train_s:.1}s training, {:.0} words/s, {} threads)",
                epoch + 1,
                c.epochs,
                words as f64 / train_s.max(1e-9),
                batch.threads()
            );
        }
        let (better, done) = stop.update(epoch, s);
        if better {
            best = Some(model.params().to_vec());
            report.best_epoch = epoch;
        }
        if done {
            if c.verbose {
                eprintln!("stopping early after epoch {}; keeping epoch {}", epoch + 1, report.best_epoch + 1);
            }
            break;
        }
    }
    if let Some(params) = best {
        model.set_params(params)?;
    }
    Ok((model, report))
}

/// The summed gradients and loss of `examples` under `model`, computed
/// as a training step computes them over `threads` threads (0: every
/// core); the bytes do not depend on `threads`.
pub fn batch_gradients(model: &Tagger<f32>, examples: &[Example], threads: usize) -> Result<(Vec<Vec<f32>>, f64)> {
    check_examples(examples, model.hidden(), model.spec().labels.len(), model.spec().hash_embed.is_some())?;
    let mut grads = model.zero_grads();
    let refs: Vec<&Example> = examples.iter().collect();
    let (loss, _) = Batch::new(threads).gradients(model, &refs, &mut grads)?;
    Ok((grads, loss))
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
    check_config(c, train.len())?;
    let forms = spec.hash_embed.is_some();
    check_examples(train, hidden, spec.labels.len(), forms)?;
    check_examples(dev, hidden, spec.labels.len(), forms)?;
    let mut rng = Rng::new(c.seed);
    let model = Tagger::<f32>::new(spec, hidden, &mut rng)?;
    self::train(model, &mut rng, train, dev, c)
}
