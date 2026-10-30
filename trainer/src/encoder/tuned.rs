//! The encoder and a token tagger head trained together (#1519): each
//! sentence's token ids run through the encoder, its subword states are
//! pooled into words as the head's `word_pooling` says, and the head's
//! loss flows back through the pooling into every trainable weight of
//! the encoder, the adapters alone when it has them. The pair trains on
//! the shared loop (`fit`): batches across threads with a fixed
//! reduction order, rounds of `lanes` examples, AdamW, early stopping on
//! a dev split scored through the core's own tagger.

use turbo::head::tagger::{Work as CoreWork, word_count};
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{self, Config, Model, OnBest, Report, Tally, check_config};
use crate::real::Real;
use crate::rng::Rng;
use crate::tagger::{Tagger, Work as TaggerWork};

use super::{Encoder, Work as EncoderWork, pool};

/// One training sentence: its token ids, each token's word (or
/// `NO_WORD`), a gold label per word, and each word's text, which a head
/// with a `hash_embed` reads (empty for a head that reads none).
#[derive(Debug, Clone, PartialEq)]
pub struct TunedExample {
    pub ids: Vec<u32>,
    pub words: Vec<u32>,
    pub labels: Vec<u32>,
    pub forms: Vec<String>,
}

impl TunedExample {
    /// The forms as the heads take them: None when there are none.
    pub fn forms(&self) -> Option<Vec<&str>> {
        (!self.forms.is_empty()).then(|| self.forms.iter().map(String::as_str).collect())
    }
}

/// An encoder and a head over its pooled states. The weights the
/// optimizer steps are the encoder's trainable slots, then the head's
/// tensors.
#[derive(Debug, Clone)]
pub struct Tuned<R: Real> {
    encoder: Encoder<R>,
    head: Tagger<R>,
    /// The encoder's trainable slots, in order.
    slots: Vec<usize>,
}

/// What one sentence's forward keeps for its backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    pub encoder: EncoderWork<R>,
    pub head: TaggerWork<R>,
    /// The pooled words [n, hidden], and each word's row count.
    pub features: Vec<R>,
    count: Vec<u32>,
    dfeatures: Vec<R>,
    dstates: Vec<R>,
    /// Gradient slots shaped as the encoder's, empty but while backward
    /// runs, when the trainable ones are borrowed from the caller's.
    all: Vec<Vec<R>>,
}

impl<R: Real> Tuned<R> {
    /// `encoder` under a seeded head for `spec`, which reads the
    /// encoder's width.
    pub fn new(encoder: Encoder<R>, spec: &HeadSpec, rng: &mut Rng) -> Result<Self> {
        let head = Tagger::new(spec, encoder.geometry().hidden, rng)?;
        Self::assemble(encoder, head)
    }

    /// `encoder` under `head`, which must read its width.
    pub fn assemble(encoder: Encoder<R>, head: Tagger<R>) -> Result<Self> {
        if head.hidden() != encoder.geometry().hidden {
            return Err(invalid(format!(
                "trainer: head {:?} reads {} features, the encoder gives {}",
                head.spec().name,
                head.hidden(),
                encoder.geometry().hidden
            )));
        }
        let slots = encoder.trainable().iter().enumerate().filter(|(_, t)| **t).map(|(s, _)| s).collect();
        Ok(Tuned { encoder, head, slots })
    }

    pub fn encoder(&self) -> &Encoder<R> {
        &self.encoder
    }

    pub fn head(&self) -> &Tagger<R> {
        &self.head
    }

    pub fn into_parts(self) -> (Encoder<R>, Tagger<R>) {
        (self.encoder, self.head)
    }

    /// The weights the optimizer steps, copied: the encoder's trainable
    /// slots, then the head's tensors.
    pub fn params(&self) -> Vec<Vec<R>> {
        let e = self.slots.iter().map(|&s| self.encoder.params()[s].clone());
        e.chain(self.head.params().iter().cloned()).collect()
    }

    /// Replaces the weights with `params`, shaped as [`params`](Self::params).
    pub fn set_params(&mut self, mut params: Vec<Vec<R>>) -> Result<()> {
        let k = self.slots.len();
        let fits = params.len() == k + self.head.params().len()
            && params.iter().zip(&self.slots).all(|(p, &s)| p.len() == self.encoder.params()[s].len());
        if !fits {
            return Err(invalid("trainer: the weights do not match the encoder and its head"));
        }
        if params[..k].iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("encoder: a weight is not finite"));
        }
        self.head.set_params(params.split_off(k))?;
        for (p, &s) in params.into_iter().zip(&self.slots) {
            self.encoder.params_mut()[s] = p;
        }
        Ok(())
    }

    /// One optimizer step on the weights, in place; returns the gradient
    /// norm before clipping.
    pub fn step(&mut self, opt: &mut AdamW<R>, grads: &mut [Vec<R>], decay: &[bool]) -> Result<f64> {
        let k = self.slots.len();
        let mut all: Vec<Vec<R>> = Vec::with_capacity(k + self.head.params().len());
        all.extend(self.slots.iter().map(|&s| std::mem::take(&mut self.encoder.params_mut()[s])));
        all.extend(self.head.params_mut().iter_mut().map(std::mem::take));
        let out = opt.step(&mut all, grads, decay);
        let mut all = all.into_iter();
        for &s in &self.slots {
            self.encoder.params_mut()[s] = all.next().expect("the encoder's slots");
        }
        for p in self.head.params_mut() {
            *p = all.next().expect("the head's tensors");
        }
        out
    }

    /// Zero gradients shaped as [`params`](Self::params).
    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        self.params().into_iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    /// Which weights take weight decay.
    pub fn decays(&self) -> Vec<bool> {
        let d = self.encoder.decay();
        self.slots.iter().map(|&s| d[s]).chain(self.head.decays()).collect()
    }

    /// Refuses an example the pair cannot run: a token id past the
    /// vocabulary or more tokens than positions (the encoder's limits), a
    /// word map of another length, labels not one per word and in range,
    /// forms missing for a head that reads them.
    pub fn check(&self, e: &TunedExample) -> Result<()> {
        let geo = self.encoder.geometry();
        if e.ids.is_empty() || e.ids.len() > geo.max_positions {
            return Err(invalid(format!("trainer: {} tokens, of 1 to {}", e.ids.len(), geo.max_positions)));
        }
        if let Some(id) = e.ids.iter().find(|&&id| id as usize >= self.encoder.vocab()) {
            return Err(invalid(format!("trainer: token id {id} is past the {} words", self.encoder.vocab())));
        }
        if e.words.len() != e.ids.len() {
            return Err(invalid(format!("trainer: {} word rows for {} tokens", e.words.len(), e.ids.len())));
        }
        let n = word_count(&e.words)?;
        let labels = self.head.spec().labels.len();
        if n == 0 || e.labels.len() != n || e.labels.iter().any(|&l| l as usize >= labels) {
            return Err(invalid(format!("trainer: {} labels for {n} words, each one of {labels}", e.labels.len())));
        }
        if self.head.spec().hash_embed.is_some() && e.forms.len() != n {
            return Err(invalid(format!(
                "trainer: {} forms for {n} words; the head reads each word's text",
                e.forms.len()
            )));
        }
        Ok(())
    }

    /// The head's scores of `e` into `w.head`, through the encoder and
    /// the pooling; returns the words.
    pub fn forward(&self, e: &TunedExample, w: &mut Work<R>) -> Result<usize> {
        self.check(e)?;
        self.encoder.forward(&e.ids, &mut w.encoder)?;
        let (h, pooling) = (self.encoder.geometry().hidden, self.head.spec().word_pooling);
        pool::forward(&w.encoder.out, &e.words, h, pooling, &mut w.features, &mut w.count)?;
        self.head.forward_forms(&w.features, e.forms().as_deref(), &mut w.head)
    }

    /// The loss of `e`, summed over its words, with its gradients added
    /// to `grads` (shaped as [`zero_grads`](Self::zero_grads)).
    pub fn loss(&self, e: &TunedExample, grads: &mut [Vec<R>], w: &mut Work<R>) -> Result<f64> {
        let k = self.slots.len();
        let shaped = grads.len() == k + self.head.params().len()
            && grads.iter().zip(&self.slots).all(|(g, &s)| g.len() == self.encoder.params()[s].len());
        if !shaped {
            return Err(invalid("trainer: the gradients do not match the encoder and its head"));
        }
        self.check(e)?;
        self.encoder.forward(&e.ids, &mut w.encoder)?;
        let (h, pooling) = (self.encoder.geometry().hidden, self.head.spec().word_pooling);
        pool::forward(&w.encoder.out, &e.words, h, pooling, &mut w.features, &mut w.count)?;
        let forms = e.forms();
        let (head, w_head, df) = (&self.head, &mut w.head, &mut w.dfeatures);
        let loss = head.loss_input(&w.features, forms.as_deref(), &e.labels, &mut grads[k..], w_head, df)?;
        w.dstates.clear();
        w.dstates.resize(e.ids.len() * h, R::ZERO);
        pool::backward(&w.dfeatures, &e.words, h, pooling, &w.count, &mut w.dstates);
        // The encoder adds into the caller's trainable slots directly.
        w.all.resize_with(self.encoder.params().len(), Vec::new);
        for (g, &s) in grads.iter_mut().zip(&self.slots) {
            std::mem::swap(g, &mut w.all[s]);
        }
        let out = self.encoder.backward(&w.encoder, &w.dstates, &mut w.all);
        for (g, &s) in grads.iter_mut().zip(&self.slots) {
            std::mem::swap(g, &mut w.all[s]);
        }
        out.map(|()| loss)
    }
}

impl Tuned<f32> {
    /// The dev score through the core's own tagger over the encoder's
    /// states: accuracy under SCHEME_TAGS, exact span F1 under BIO and
    /// BILOU.
    pub fn score(&self, dev: &[TunedExample]) -> Result<f64> {
        let tagger = self.head.to_token_tagger()?;
        let mut tally = Tally::new(tagger.spec());
        let (mut w, mut core) = (EncoderWork::default(), CoreWork::default());
        let (mut scores, mut tags) = (Vec::new(), Vec::new());
        for e in dev {
            self.check(e)?;
            self.encoder.forward(&e.ids, &mut w)?;
            let n = tagger.scores_forms(&w.out, &e.words, e.forms().as_deref(), &mut scores, &mut core)?;
            tagger.decode(&scores, n, &mut tags)?;
            tally.add(&tags, &e.labels)?;
        }
        Ok(tally.score())
    }
}

impl Model for Tuned<f32> {
    type Example = TunedExample;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        Tuned::params(self)
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        Tuned::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        Tuned::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        Tuned::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        Tuned::decays(self)
    }
    fn loss(&self, e: &TunedExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        Ok((Tuned::loss(self, e, grads, w)?, e.labels.len()))
    }
    fn score(&self, dev: &[TunedExample]) -> Result<f64> {
        Tuned::score(self, dev)
    }
}

/// The summed gradients and loss of `examples` under `model`, computed
/// as a training step computes them over `threads` threads (0: every
/// core) in rounds of `lanes` (0: all at once); the bytes depend on
/// neither.
pub fn tuned_gradients(
    model: &Tuned<f32>,
    examples: &[TunedExample],
    threads: usize,
    lanes: usize,
) -> Result<(Vec<Vec<f32>>, f64)> {
    let mut grads = model.zero_grads();
    let refs: Vec<&TunedExample> = examples.iter().collect();
    let (loss, _) = crate::batch::Batch::new(threads, lanes).gradients(model, &refs, &mut grads)?;
    Ok((grads, loss))
}

/// Fine-tunes `encoder` under a seeded head for `spec` on `train`,
/// keeping the weights that scored best on `dev`.
pub fn fit_tuned(
    encoder: Encoder<f32>,
    spec: &HeadSpec,
    train: &[TunedExample],
    dev: &[TunedExample],
    c: &Config,
) -> Result<(Tuned<f32>, Report)> {
    fit_tuned_with(encoder, spec, train, dev, c, &mut |_, _, _| Ok(()))
}

/// As [`fit_tuned`], calling `on_best` with the model, the epoch and its
/// dev score each time the best improves.
pub fn fit_tuned_with(
    encoder: Encoder<f32>,
    spec: &HeadSpec,
    train: &[TunedExample],
    dev: &[TunedExample],
    c: &Config,
    on_best: OnBest<'_, Tuned<f32>>,
) -> Result<(Tuned<f32>, Report)> {
    check_config(c, train.len())?;
    let mut rng = Rng::new(c.seed);
    let model = Tuned::new(encoder, spec, &mut rng)?;
    for (i, e) in train.iter().chain(dev).enumerate() {
        model.check(e).map_err(|err| invalid(format!("example {i}: {}", err.message)))?;
    }
    fit::train(model, &mut rng, train, dev, c, on_best)
}
