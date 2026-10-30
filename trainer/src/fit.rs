//! The training loop for a token tagger head: seeded shuffles, minibatch
//! gradients summed in a fixed order, AdamW, and early stopping on a dev
//! split scored by the core's own inference path.

use turbo::head::boundary::LAST;
use turbo::head::crf::{WordSpan, spans};
use turbo::head::tagger::{NO_WORD, Work as CoreWork};
use turbo::head::{SentenceBoundary, TokenSplitter, TokenTagger};
use turbo::manifest::{HeadKind, HeadSpec, Scheme};
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
    /// Epochs after warmup without a dev score more than `tolerance`
    /// above the last one that counted before stopping; 0 never stops
    /// early.
    pub patience: u32,
    /// How much a dev score must gain over the last score that counted to
    /// count as progress for `patience` (Keras's `min_delta`): under the
    /// default 0.001 a run whose dev score creeps or sits flat stops after
    /// `patience` epochs; under 0 any better score counts. The best
    /// weights are kept on every strictly better score, whether or not it
    /// counted.
    pub tolerance: f64,
    /// Threads a batch's examples are spread over; 0 takes every core.
    /// The weights a run gives do not depend on it.
    pub threads: usize,
    /// Examples whose gradients and activations are held at once: a
    /// batch runs in rounds of this many, each round's gradients added
    /// to the step's in example order, so the bytes do not depend on it
    /// either, and memory does not grow with the batch (gradient
    /// accumulation). 0 holds the whole batch.
    pub lanes: usize,
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
            lanes: 0,
            verbose: false,
        }
    }
}

/// Early stopping on the dev score: armed only once warmup is over, so a
/// plateau while the rate is still rising never stops a run, and only a
/// gain of more than `tolerance` over the last score that counted resets
/// patience.
#[derive(Debug, Clone)]
pub struct EarlyStop {
    warm: usize,
    patience: usize,
    tolerance: f64,
    /// The best score yet, whose weights are kept.
    best: Option<f64>,
    /// The last score that counted as progress, and its epoch.
    counted: Option<f64>,
    level: usize,
}

impl EarlyStop {
    /// For a run of `c`: warmup is its first `ceil(warmup * epochs)` epochs.
    pub fn new(c: &Config) -> EarlyStop {
        let warm = (c.warmup * c.epochs as f64).ceil() as usize;
        EarlyStop { warm, patience: c.patience as usize, tolerance: c.tolerance, best: None, counted: None, level: 0 }
    }

    /// Takes epoch `epoch`'s dev score; returns whether it is the best yet
    /// (keep these weights) and whether to stop after it.
    pub fn update(&mut self, epoch: usize, score: f64) -> (bool, bool) {
        let better = self.best.is_none_or(|b| score > b);
        if better {
            self.best = Some(score);
        }
        if self.counted.is_none_or(|c| score > c + self.tolerance) {
            (self.counted, self.level) = (Some(score), epoch);
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
    let mut tally = Tally::new(spec);
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        debug_assert!(!words.contains(&NO_WORD));
        tagger.scores_forms(&e.features, &words, e.forms().as_deref(), &mut scores, &mut work)?;
        tagger.decode(&scores, n, &mut tags)?;
        tally.add(&tags, &e.labels)?;
    }
    Ok(tally.score())
}

/// The running dev score of a tagger head: right words of all under
/// SCHEME_TAGS, found and gold spans and the right ones under BIO and
/// BILOU.
pub(crate) struct Tally<'a> {
    spec: &'a HeadSpec,
    right: usize,
    total: usize,
    found: usize,
    gold: usize,
}

impl<'a> Tally<'a> {
    pub(crate) fn new(spec: &'a HeadSpec) -> Tally<'a> {
        Tally { spec, right: 0, total: 0, found: 0, gold: 0 }
    }

    /// One sentence's decoded `tags` against its `labels`.
    pub(crate) fn add(&mut self, tags: &[u32], labels: &[u32]) -> Result<()> {
        let spec = self.spec;
        if spec.scheme == Scheme::Tags {
            self.right += tags.iter().zip(labels).filter(|(a, b)| a == b).count();
            self.total += labels.len();
        } else {
            let want: Vec<WordSpan> = spans(spec.scheme, &spec.labels, labels)?;
            let got = spans(spec.scheme, &spec.labels, tags)?;
            self.right += got.iter().filter(|s| want.contains(s)).count();
            self.found += got.len();
            self.gold += want.len();
        }
        Ok(())
    }

    /// Accuracy, or exact span F1; 1 when there is nothing to find and
    /// nothing was found.
    pub(crate) fn score(&self) -> f64 {
        if self.spec.scheme == Scheme::Tags {
            if self.total == 0 { 0.0 } else { self.right as f64 / self.total as f64 }
        } else if self.found + self.gold == 0 {
            1.0
        } else {
            2.0 * self.right as f64 / (self.found + self.gold) as f64
        }
    }
}

/// Accuracy of `head` on `examples`: whether each word ends a sentence as
/// the core decodes it (the last word always does) against its gold label
/// being `last`.
pub fn score_boundary(head: &SentenceBoundary, examples: &[Example]) -> Result<f64> {
    let hidden = head.hidden();
    check_examples(examples, hidden, head.spec().labels.len(), head.reads_forms())?;
    let (mut right, mut total) = (0usize, 0usize);
    let (mut scores, mut ends, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.labels.len();
        let words: Vec<u32> = (0..n as u32).collect();
        head.scores(&e.features, &words, e.forms().as_deref(), &mut scores, &mut work)?;
        head.decode(&scores, n, &mut ends)?;
        right += ends.iter().zip(&e.labels).filter(|&(&end, &gold)| end == (gold == LAST)).count();
        total += n;
    }
    Ok(if total == 0 { 0.0 } else { right as f64 / total as f64 })
}

/// Accuracy of `head` on `examples`: each token's label as the core
/// decodes it, which skips a split that does not apply to the token's
/// form. Refuses an example without a form per token.
pub fn score_split(head: &TokenSplitter, examples: &[Example]) -> Result<f64> {
    let hidden = head.hidden();
    check_examples(examples, hidden, head.spec().labels.len(), true)?;
    let (mut right, mut total) = (0usize, 0usize);
    let (mut scores, mut labels, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.labels.len();
        let words: Vec<u32> = (0..n as u32).collect();
        let forms: Vec<&str> = e.forms.iter().map(String::as_str).collect();
        let read = head.reads_forms().then_some(forms.as_slice());
        head.scores(&e.features, &words, read, &mut scores, &mut work)?;
        head.decode(&scores, &forms, &mut labels)?;
        right += labels.iter().zip(&e.labels).filter(|(a, b)| a == b).count();
        total += n;
    }
    Ok(if total == 0 { 0.0 } else { right as f64 / total as f64 })
}

/// What the training loop needs of a head in training.
pub(crate) trait Model: Sized + Sync {
    type Example: Sync;
    type Work: Default + Send;
    /// The weights, copied: what a run keeps as its best.
    fn params(&self) -> Vec<Vec<f32>>;
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
    fn params(&self) -> Vec<Vec<f32>> {
        Tagger::params(self).to_vec()
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
        match self.spec().kind {
            HeadKind::SentenceBoundary => score_boundary(&self.to_boundary()?, dev),
            HeadKind::TokenSplit => score_split(&self.to_splitter()?, dev),
            _ => score(&self.to_token_tagger()?, dev),
        }
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

/// What a run calls each time its best improves: the model, the epoch and
/// its dev score.
pub type OnBest<'a, M> = &'a mut dyn FnMut(&M, usize, f64) -> Result<()>;

/// The loop: seeded shuffles, minibatch gradients summed in a fixed
/// order (across `c.threads` threads, `c.lanes` examples at once, see
/// [`Batch`]) and scaled per word,
/// AdamW, dev scoring each epoch, early stopping, the best weights kept;
/// `on_best` sees the model and the epoch and dev score each time the
/// best improves, so a caller can save it while the run goes on.
pub(crate) fn train<M: Model>(
    mut model: M,
    rng: &mut Rng,
    train: &[M::Example],
    dev: &[M::Example],
    c: &Config,
    on_best: OnBest<'_, M>,
) -> Result<(M, Report)> {
    let shapes: Vec<usize> = model.params().iter().map(Vec::len).collect();
    let mut opt = AdamW::new(settings(c, train.len()), &shapes)?;
    let decay = model.decays();
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut batch = Batch::new(c.threads, c.lanes);
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
            best = Some(model.params());
            report.best_epoch = epoch;
            on_best(&model, epoch, s)?;
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
    let (loss, _) = Batch::new(threads, 0).gradients(model, &refs, &mut grads)?;
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
    fit_with(spec, hidden, train, dev, c, &mut |_, _, _| Ok(()))
}

/// As [`fit`], calling `on_best` with the model, the epoch and its dev
/// score each time the best improves: the weights it sees are the ones
/// the run keeps if nothing better follows.
pub fn fit_with(
    spec: &HeadSpec,
    hidden: usize,
    train: &[Example],
    dev: &[Example],
    c: &Config,
    on_best: OnBest<'_, Tagger<f32>>,
) -> Result<(Tagger<f32>, Report)> {
    check_config(c, train.len())?;
    let forms = spec.hash_embed.is_some();
    check_examples(train, hidden, spec.labels.len(), forms)?;
    check_examples(dev, hidden, spec.labels.len(), forms)?;
    let mut rng = Rng::new(c.seed);
    let model = Tagger::<f32>::new(spec, hidden, &mut rng)?;
    self::train(model, &mut rng, train, dev, c, on_best)
}
