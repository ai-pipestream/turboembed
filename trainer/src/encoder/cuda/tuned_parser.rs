//! The tuned parser on a CUDA device (#1821): parser loss flows through
//! word pooling into every trainable encoder weight. The device has one
//! workspace, so a mutex serializes examples while the head and optimizer
//! keep the common training loop.

use std::sync::{Mutex, MutexGuard};

use turbo::head::tagger::word_count;
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{self, Config, Draw, Model, OnBest, Report, check_config};
use crate::fit_parse::{Attachment, ParseExample, attachment};
use crate::parser::{Parser, Work as ParserWork};
use crate::rng::Rng;

use super::super::pool;
use super::super::tuned_parser::TunedParseExample;
use super::CudaEncoder;

/// An encoder and parser on a CUDA device.
pub struct TunedCudaParser {
    encoder: Mutex<CudaEncoder>,
    head: Parser<f32>,
    slots: Vec<usize>,
}

/// What one sentence's forward keeps for backward.
#[derive(Debug, Default)]
pub struct Work {
    pub head: ParserWork<f32>,
    pub features: Vec<f32>,
    states: Vec<f32>,
    count: Vec<u32>,
    dfeatures: Vec<f32>,
    dstates: Vec<f32>,
    all: Vec<Vec<f32>>,
    scratch: Vec<Vec<f32>>,
}

impl TunedCudaParser {
    /// `encoder` under a seeded parser for `spec`.
    pub fn new(encoder: CudaEncoder, spec: &HeadSpec, rng: &mut Rng) -> Result<Self> {
        let head = Parser::new(spec, encoder.geometry().hidden, rng)?;
        Self::assemble(encoder, head)
    }

    /// `encoder` under `head`, which must read its width.
    pub fn assemble(encoder: CudaEncoder, head: Parser<f32>) -> Result<Self> {
        if head.hidden() != encoder.geometry().hidden {
            return Err(invalid(format!(
                "trainer: parser {:?} reads {} features, the encoder gives {}",
                head.spec().name,
                head.hidden(),
                encoder.geometry().hidden
            )));
        }
        let slots = encoder.trainable().iter().enumerate().filter(|(_, t)| **t).map(|(s, _)| s).collect();
        Ok(TunedCudaParser { encoder: Mutex::new(encoder), head, slots })
    }

    fn lock(&self) -> Result<MutexGuard<'_, CudaEncoder>> {
        self.encoder.lock().map_err(|_| invalid("cuda: the encoder lock is poisoned"))
    }

    pub fn encoder(&self) -> Result<MutexGuard<'_, CudaEncoder>> {
        self.lock()
    }

    pub fn head(&self) -> &Parser<f32> {
        &self.head
    }

    pub fn into_parts(self) -> Result<(CudaEncoder, Parser<f32>)> {
        let encoder = self.encoder.into_inner().map_err(|_| invalid("cuda: the encoder lock is poisoned"))?;
        Ok((encoder, self.head))
    }

    /// The optimizer's weights: trainable encoder slots, then parser tensors.
    pub fn params(&self) -> Result<Vec<Vec<f32>>> {
        let encoder = self.lock()?;
        let mut out = Vec::with_capacity(self.slots.len() + self.head.params().len());
        for &s in &self.slots {
            out.push(encoder.slot(s)?);
        }
        out.extend(self.head.params().iter().cloned());
        Ok(out)
    }

    /// Replaces the weights, shaped as [`params`](Self::params).
    pub fn set_params(&mut self, mut params: Vec<Vec<f32>>) -> Result<()> {
        let k = self.slots.len();
        let encoder = self.encoder.lock().map_err(|_| invalid("cuda: the encoder lock is poisoned"))?;
        let fits = params.len() == k + self.head.params().len()
            && params.iter().zip(&self.slots).all(|(p, &s)| p.len() == encoder.lens[s]);
        if !fits {
            return Err(invalid("trainer: the weights do not match the encoder and parser"));
        }
        if params[..k].iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("encoder: a weight is not finite"));
        }
        let head = params.split_off(k);
        drop(encoder);
        self.head.set_params(head)?;
        let mut encoder = self.encoder.lock().map_err(|_| invalid("cuda: the encoder lock is poisoned"))?;
        for (p, &s) in params.into_iter().zip(&self.slots) {
            encoder.put_slot(s, &p)?;
        }
        Ok(())
    }

    /// One optimizer step on the encoder and parser weights.
    pub fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        let k = self.slots.len();
        let mut encoder = self.encoder.lock().map_err(|_| invalid("cuda: the encoder lock is poisoned"))?;
        let mut all = Vec::with_capacity(k + self.head.params().len());
        for &s in &self.slots {
            all.push(encoder.slot(s)?);
        }
        all.extend(self.head.params_mut().iter_mut().map(std::mem::take));
        let out = opt.step(&mut all, grads, decay);
        let mut all = all.into_iter();
        let mut put = Ok(());
        for &s in &self.slots {
            put = put.and(encoder.put_slot(s, &all.next().expect("the encoder's slots")));
        }
        for p in self.head.params_mut() {
            *p = all.next().expect("the parser's tensors");
        }
        put.and(out)
    }

    /// Zero gradients shaped as [`params`](Self::params).
    pub fn zero_grads(&self) -> Vec<Vec<f32>> {
        let e = self.encoder.lock().expect("CUDA encoder lock");
        let e = self.slots.iter().map(|&s| e.lens[s]);
        e.chain(self.head.params().iter().map(Vec::len)).map(|n| vec![0.0; n]).collect()
    }

    /// Which weights take weight decay.
    pub fn decays(&self) -> Vec<bool> {
        let d = self.encoder.lock().expect("CUDA encoder lock").decay();
        self.slots.iter().map(|&s| d[s]).chain(self.head.decays()).collect()
    }

    /// Refuses an example the pair cannot run.
    pub fn check(&self, e: &TunedParseExample) -> Result<()> {
        let encoder = self.lock()?;
        let geo = encoder.geometry();
        if e.ids.is_empty() || e.ids.len() > geo.max_positions {
            return Err(invalid(format!("trainer: {} tokens, of 1 to {}", e.ids.len(), geo.max_positions)));
        }
        if e.words.len() != e.ids.len() {
            return Err(invalid(format!("trainer: {} word rows for {} tokens", e.words.len(), e.ids.len())));
        }
        let n = word_count(&e.words)?;
        let labels = self.head.spec().labels.len();
        if n == 0
            || e.heads.len() != n
            || e.rels.len() != n
            || e.heads.iter().enumerate().any(|(i, &h)| h as usize > n || h as usize == i + 1)
            || e.rels.iter().any(|&l| l as usize >= labels)
        {
            return Err(invalid("trainer: invalid parser heads or relations"));
        }
        Ok(())
    }

    fn forward_with(&self, encoder: &mut CudaEncoder, e: &TunedParseExample, w: &mut Work) -> Result<usize> {
        encoder.forward(&e.ids, &mut w.states)?;
        let (h, pooling) = (encoder.geometry().hidden, self.head.spec().word_pooling);
        pool::forward(&w.states, &e.words, h, pooling, &mut w.features, &mut w.count)?;
        self.head.forward(&w.features, &mut w.head)
    }

    /// The parser's arc scores of `e` into `w.head`.
    pub fn forward(&self, e: &TunedParseExample, w: &mut Work) -> Result<usize> {
        self.check(e)?;
        let mut encoder = self.lock()?;
        self.forward_with(&mut encoder, e, w)
    }

    /// The loss of `e`, summed over its words.
    pub fn loss(&self, e: &TunedParseExample, grads: &mut [Vec<f32>], w: &mut Work) -> Result<f64> {
        self.loss_drawn(e, None, 1.0, grads, w)
    }

    /// The loss in a training pass, with dropout and loss scaling.
    pub fn loss_at(&self, e: &TunedParseExample, draw: Draw, grads: &mut [Vec<f32>], w: &mut Work) -> Result<f64> {
        self.loss_drawn(e, Some(draw.at), draw.scale, grads, w)
    }

    fn loss_drawn(
        &self,
        e: &TunedParseExample,
        at: Option<u64>,
        scale: f32,
        grads: &mut [Vec<f32>],
        w: &mut Work,
    ) -> Result<f64> {
        self.check(e)?;
        let k = self.slots.len();
        let mut encoder = self.lock()?;
        let shaped = grads.len() == k + self.head.params().len()
            && grads.iter().zip(&self.slots).all(|(g, &s)| g.len() == encoder.lens[s]);
        if !shaped {
            return Err(invalid("trainer: the gradients do not match the encoder and parser"));
        }
        match at {
            Some(at) => encoder.forward_at(&e.ids, at, &mut w.states)?,
            None => encoder.forward(&e.ids, &mut w.states)?,
        };
        let (h, pooling) = (encoder.geometry().hidden, self.head.spec().word_pooling);
        pool::forward(&w.states, &e.words, h, pooling, &mut w.features, &mut w.count)?;
        let (head, w_head, df) = (&self.head, &mut w.head, &mut w.dfeatures);
        let loss = if scale == 1.0 {
            head.loss_input(&w.features, &e.heads, &e.rels, &mut grads[k..], w_head, df)?
        } else {
            w.scratch.resize_with(grads.len() - k, Vec::new);
            for (s, g) in w.scratch.iter_mut().zip(&grads[k..]) {
                s.clear();
                s.resize(g.len(), 0.0);
            }
            let loss = head.loss_input(&w.features, &e.heads, &e.rels, &mut w.scratch, w_head, df)?;
            for (g, s) in grads[k..].iter_mut().zip(&w.scratch) {
                for (g, &s) in g.iter_mut().zip(s) {
                    *g += s * scale;
                }
            }
            df.iter_mut().for_each(|d| *d *= scale);
            loss
        };
        w.dstates.clear();
        w.dstates.resize(e.ids.len() * h, 0.0);
        pool::backward(&w.dfeatures, &e.words, h, pooling, &w.count, &mut w.dstates);
        encoder.zero_grads()?;
        encoder.backward(&w.dstates)?;
        w.all.resize_with(encoder.trainable().len(), Vec::new);
        for (g, d) in w.all.iter_mut().zip(&encoder.grads) {
            g.clear();
            g.resize(d.len(), 0.0);
        }
        encoder.grads(&mut w.all)?;
        for (g, &s) in grads.iter_mut().zip(&self.slots) {
            for (g, &d) in g.iter_mut().zip(&w.all[s]) {
                *g += d;
            }
        }
        Ok(loss)
    }

    /// Labelled and unlabelled attachment through the core's parser.
    pub fn score(&self, dev: &[TunedParseExample]) -> Result<Attachment> {
        let parser = self.head.to_core()?;
        for e in dev {
            self.check(e)?;
        }
        let mut encoder = self.lock()?;
        let mut w = Work::default();
        let mut total = Attachment::default();
        for e in dev {
            encoder.forward(&e.ids, &mut w.states)?;
            let (h, pooling) = (encoder.geometry().hidden, self.head.spec().word_pooling);
            pool::forward(&w.states, &e.words, h, pooling, &mut w.features, &mut w.count)?;
            let example = ParseExample {
                features: w.features.clone(),
                heads: e.heads.clone(),
                rels: e.rels.clone(),
                language: None,
            };
            let score = attachment(&parser, std::slice::from_ref(&example))?;
            total.words += score.words;
            total.heads += score.heads;
            total.labelled += score.labelled;
        }
        Ok(total)
    }
}

impl Model for TunedCudaParser {
    type Example = TunedParseExample;
    type Work = Work;
    fn params(&self) -> Vec<Vec<f32>> {
        self.params().expect("CUDA weights download")
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        TunedCudaParser::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        TunedCudaParser::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        TunedCudaParser::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        TunedCudaParser::decays(self)
    }
    fn loss(&self, e: &TunedParseExample, grads: &mut [Vec<f32>], w: &mut Work) -> Result<(f64, usize)> {
        Ok((TunedCudaParser::loss(self, e, grads, w)?, e.heads.len()))
    }
    fn loss_at(&self, e: &TunedParseExample, draw: Draw, grads: &mut [Vec<f32>], w: &mut Work) -> Result<(f64, usize)> {
        Ok((TunedCudaParser::loss_at(self, e, draw, grads, w)?, e.heads.len()))
    }
    fn score(&self, dev: &[TunedParseExample]) -> Result<f64> {
        Ok(TunedCudaParser::score(self, dev)?.las())
    }
}

/// Fine-tunes `encoder` under a seeded parser for `spec`.
pub fn fit_tuned_cuda_parser(
    encoder: CudaEncoder,
    spec: &HeadSpec,
    train: &[TunedParseExample],
    dev: &[TunedParseExample],
    c: &Config,
) -> Result<(TunedCudaParser, Report)> {
    fit_tuned_cuda_parser_with(encoder, spec, train, dev, c, &mut |_, _, _| Ok(()))
}

/// As [`fit_tuned_cuda_parser`], calling `on_best` for each new best epoch.
pub fn fit_tuned_cuda_parser_with(
    encoder: CudaEncoder,
    spec: &HeadSpec,
    train: &[TunedParseExample],
    dev: &[TunedParseExample],
    c: &Config,
    on_best: OnBest<'_, TunedCudaParser>,
) -> Result<(TunedCudaParser, Report)> {
    check_config(c, train.len())?;
    let mut rng = Rng::new(c.seed);
    let model = TunedCudaParser::new(encoder, spec, &mut rng)?;
    for (i, e) in train.iter().chain(dev).enumerate() {
        model.check(e).map_err(|err| invalid(format!("example {i}: {}", err.message)))?;
    }
    fit::train(model, &mut rng, train, dev, c, on_best)
}
