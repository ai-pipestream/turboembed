//! An encoder and a biaffine parser trained together (#1821): parser
//! loss flows through word pooling into every trainable encoder weight,
//! using the same batching, optimizer and early stopping as `tuned`.

use turbo::head::tagger::word_count;
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{self, Config, Draw, Model, OnBest, Report, check_config};
use crate::fit_parse::{Attachment, ParseExample, attachment};
use crate::parser::{Parser, Work as ParserWork};
use crate::real::Real;
use crate::rng::Rng;

use super::{Encoder, Work as EncoderWork, pool};

/// One training sentence for the tuned parser.
#[derive(Debug, Clone, PartialEq)]
pub struct TunedParseExample {
    pub ids: Vec<u32>,
    pub words: Vec<u32>,
    pub heads: Vec<u32>,
    pub rels: Vec<u32>,
}

/// An encoder and parser over its pooled states.
#[derive(Debug, Clone)]
pub struct TunedParser<R: Real> {
    encoder: Encoder<R>,
    head: Parser<R>,
    slots: Vec<usize>,
}

/// What one sentence's forward keeps for backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    pub encoder: EncoderWork<R>,
    pub head: ParserWork<R>,
    pub features: Vec<R>,
    count: Vec<u32>,
    dfeatures: Vec<R>,
    dstates: Vec<R>,
    all: Vec<Vec<R>>,
    scratch: Vec<Vec<R>>,
}

impl<R: Real> TunedParser<R> {
    /// `encoder` under a seeded parser for `spec`.
    pub fn new(encoder: Encoder<R>, spec: &HeadSpec, rng: &mut Rng) -> Result<Self> {
        let head = Parser::new(spec, encoder.geometry().hidden, rng)?;
        Self::assemble(encoder, head)
    }

    /// `encoder` under `head`, which must read its width.
    pub fn assemble(encoder: Encoder<R>, head: Parser<R>) -> Result<Self> {
        if head.hidden() != encoder.geometry().hidden {
            return Err(invalid(format!(
                "trainer: parser {:?} reads {} features, the encoder gives {}",
                head.spec().name,
                head.hidden(),
                encoder.geometry().hidden
            )));
        }
        let slots = encoder.trainable().iter().enumerate().filter(|(_, t)| **t).map(|(s, _)| s).collect();
        Ok(TunedParser { encoder, head, slots })
    }

    pub fn encoder(&self) -> &Encoder<R> {
        &self.encoder
    }

    pub fn head(&self) -> &Parser<R> {
        &self.head
    }

    pub fn into_parts(self) -> (Encoder<R>, Parser<R>) {
        (self.encoder, self.head)
    }

    /// The optimizer's weights: trainable encoder slots, then parser tensors.
    pub fn params(&self) -> Vec<Vec<R>> {
        let e = self.slots.iter().map(|&s| self.encoder.params()[s].clone());
        e.chain(self.head.params().iter().cloned()).collect()
    }

    /// Replaces the weights, shaped as [`params`](Self::params).
    pub fn set_params(&mut self, mut params: Vec<Vec<R>>) -> Result<()> {
        let k = self.slots.len();
        let fits = params.len() == k + self.head.params().len()
            && params.iter().zip(&self.slots).all(|(p, &s)| p.len() == self.encoder.params()[s].len());
        if !fits {
            return Err(invalid("trainer: the weights do not match the encoder and parser"));
        }
        if params[..k].iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("encoder: a weight is not finite"));
        }
        self.head.set_params(params.split_off(k))?;
        for (p, &s) in params.into_iter().zip(&self.slots) {
            self.encoder.put_slot(s, p)?;
        }
        Ok(())
    }

    /// One optimizer step on the encoder and parser weights.
    pub fn step(&mut self, opt: &mut AdamW<R>, grads: &mut [Vec<R>], decay: &[bool]) -> Result<f64> {
        let k = self.slots.len();
        let mut all: Vec<Vec<R>> = Vec::with_capacity(k + self.head.params().len());
        all.extend(self.slots.iter().map(|&s| self.encoder.take_slot(s)));
        all.extend(self.head.params_mut().iter_mut().map(std::mem::take));
        let out = opt.step(&mut all, grads, decay);
        let mut all = all.into_iter();
        let mut put = Ok(());
        for &s in &self.slots {
            put = put.and(self.encoder.put_slot(s, all.next().expect("the encoder's slots")));
        }
        for p in self.head.params_mut() {
            *p = all.next().expect("the parser's tensors");
        }
        put.and(out)
    }

    /// Zero gradients shaped as [`params`](Self::params).
    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        let e = self.slots.iter().map(|&s| self.encoder.params()[s].len());
        e.chain(self.head.params().iter().map(Vec::len)).map(|n| vec![R::ZERO; n]).collect()
    }

    /// Which weights take weight decay.
    pub fn decays(&self) -> Vec<bool> {
        let d = self.encoder.decay();
        self.slots.iter().map(|&s| d[s]).chain(self.head.decays()).collect()
    }

    /// Refuses an example the pair cannot run.
    pub fn check(&self, e: &TunedParseExample) -> Result<()> {
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

    /// The parser's arc scores of `e` into `w.head`.
    pub fn forward(&self, e: &TunedParseExample, w: &mut Work<R>) -> Result<usize> {
        self.check(e)?;
        self.encoder.forward(&e.ids, &mut w.encoder)?;
        let (h, pooling) = (self.encoder.geometry().hidden, self.head.spec().word_pooling);
        pool::forward(&w.encoder.out, &e.words, h, pooling, &mut w.features, &mut w.count)?;
        self.head.forward(&w.features, &mut w.head)
    }

    /// The loss of `e`, summed over its words.
    pub fn loss(&self, e: &TunedParseExample, grads: &mut [Vec<R>], w: &mut Work<R>) -> Result<f64> {
        self.loss_drawn(e, None, R::ONE, grads, w)
    }

    /// The loss in a training pass, with dropout and loss scaling.
    pub fn loss_at(&self, e: &TunedParseExample, draw: Draw, grads: &mut [Vec<R>], w: &mut Work<R>) -> Result<f64> {
        self.loss_drawn(e, Some(draw.at), R::from_f64(draw.scale as f64), grads, w)
    }

    fn loss_drawn(
        &self,
        e: &TunedParseExample,
        at: Option<u64>,
        scale: R,
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<f64> {
        let k = self.slots.len();
        let shaped = grads.len() == k + self.head.params().len()
            && grads.iter().zip(&self.slots).all(|(g, &s)| g.len() == self.encoder.params()[s].len());
        if !shaped {
            return Err(invalid("trainer: the gradients do not match the encoder and parser"));
        }
        self.check(e)?;
        match at {
            Some(at) => self.encoder.forward_at(&e.ids, at, &mut w.encoder)?,
            None => self.encoder.forward(&e.ids, &mut w.encoder)?,
        }
        let (h, pooling) = (self.encoder.geometry().hidden, self.head.spec().word_pooling);
        pool::forward(&w.encoder.out, &e.words, h, pooling, &mut w.features, &mut w.count)?;
        let (head, w_head, df) = (&self.head, &mut w.head, &mut w.dfeatures);
        let loss = if scale == R::ONE {
            head.loss_input(&w.features, &e.heads, &e.rels, &mut grads[k..], w_head, df)?
        } else {
            w.scratch.resize_with(grads.len() - k, Vec::new);
            for (s, g) in w.scratch.iter_mut().zip(&grads[k..]) {
                s.clear();
                s.resize(g.len(), R::ZERO);
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
        w.dstates.resize(e.ids.len() * h, R::ZERO);
        pool::backward(&w.dfeatures, &e.words, h, pooling, &w.count, &mut w.dstates);
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

impl TunedParser<f32> {
    /// Labelled and unlabelled attachment through the core's parser.
    pub fn score(&self, dev: &[TunedParseExample]) -> Result<Attachment> {
        let parser = self.head.to_core()?;
        let mut encoder_work = EncoderWork::default();
        let (mut features, mut count) = (Vec::new(), Vec::new());
        let mut total = Attachment::default();
        for e in dev {
            self.check(e)?;
            self.encoder.forward(&e.ids, &mut encoder_work)?;
            let (h, pooling) = (self.encoder.geometry().hidden, self.head.spec().word_pooling);
            pool::forward(&encoder_work.out, &e.words, h, pooling, &mut features, &mut count)?;
            let example = ParseExample {
                features: features.clone(),
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

impl Model for TunedParser<f32> {
    type Example = TunedParseExample;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        TunedParser::params(self)
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        TunedParser::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        TunedParser::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        TunedParser::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        TunedParser::decays(self)
    }
    fn loss(&self, e: &TunedParseExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        Ok((TunedParser::loss(self, e, grads, w)?, e.heads.len()))
    }
    fn loss_at(
        &self,
        e: &TunedParseExample,
        draw: Draw,
        grads: &mut [Vec<f32>],
        w: &mut Work<f32>,
    ) -> Result<(f64, usize)> {
        Ok((TunedParser::loss_at(self, e, draw, grads, w)?, e.heads.len()))
    }
    fn score(&self, dev: &[TunedParseExample]) -> Result<f64> {
        Ok(TunedParser::score(self, dev)?.las())
    }
}

/// Fine-tunes `encoder` under a seeded parser for `spec`.
pub fn fit_tuned_parser(
    encoder: Encoder<f32>,
    spec: &HeadSpec,
    train: &[TunedParseExample],
    dev: &[TunedParseExample],
    c: &Config,
) -> Result<(TunedParser<f32>, Report)> {
    fit_tuned_parser_with(encoder, spec, train, dev, c, &mut |_, _, _| Ok(()))
}

/// As [`fit_tuned_parser`], calling `on_best` for each new best epoch.
pub fn fit_tuned_parser_with(
    encoder: Encoder<f32>,
    spec: &HeadSpec,
    train: &[TunedParseExample],
    dev: &[TunedParseExample],
    c: &Config,
    on_best: OnBest<'_, TunedParser<f32>>,
) -> Result<(TunedParser<f32>, Report)> {
    check_config(c, train.len())?;
    let mut rng = Rng::new(c.seed);
    let model = TunedParser::new(encoder, spec, &mut rng)?;
    for (i, e) in train.iter().chain(dev).enumerate() {
        model.check(e).map_err(|err| invalid(format!("example {i}: {}", err.message)))?;
    }
    fit::train(model, &mut rng, train, dev, c, on_best)
}
