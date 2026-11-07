//! Training a biaffine parser: the loop of `fit` over sentences with
//! gold heads and relations, scored on dev by labelled attachment
//! through the core's own parser (`turbo::head::BiaffineParser`).

use turbo::head::BiaffineParser;
use turbo::head::parser::Work as CoreWork;
use turbo::manifest::{HeadSpec, LayerMix};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{Config, Model, OnBest, Report, check_config, train};
use crate::mix::{check_mix_layers, mix_grad, mix_layers, weights_of};
use crate::parser::{Parser, Work};
use crate::real::Real;
use crate::rng::Rng;

/// One training sentence: per-word features [words, hidden], each word's
/// gold head (0 the root, else the word's 1-based place) and relation.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseExample {
    pub features: Vec<f32>,
    pub heads: Vec<u32>,
    pub rels: Vec<u32>,
    pub language: Option<u32>,
}

/// Unlabelled and labelled attachment over a set of sentences.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Attachment {
    pub words: usize,
    /// Words with the right head.
    pub heads: usize,
    /// Words with the right head and relation.
    pub labelled: usize,
}

impl Attachment {
    pub fn uas(&self) -> f64 {
        if self.words == 0 { 0.0 } else { self.heads as f64 / self.words as f64 }
    }

    pub fn las(&self) -> f64 {
        if self.words == 0 { 0.0 } else { self.labelled as f64 / self.words as f64 }
    }
}

/// Each sentence has `hidden` features, a head in range other than
/// itself and a relation among `labels` for every word.
fn check_parses(examples: &[ParseExample], hidden: usize, labels: usize) -> Result<()> {
    for (i, e) in examples.iter().enumerate() {
        let n = e.heads.len();
        let fits = hidden > 0
            && n.checked_mul(hidden) == Some(e.features.len())
            && e.rels.len() == n
            && e.heads.iter().enumerate().all(|(j, &h)| h as usize <= n && h as usize != j + 1)
            && e.rels.iter().all(|&l| (l as usize) < labels);
        if !fits {
            return Err(invalid(format!(
                "trainer: sentence {i} does not fit {hidden} features per word, its heads and relations"
            )));
        }
    }
    Ok(())
}

fn check_languages(examples: &[ParseExample], spec: &HeadSpec) -> Result<()> {
    for (i, example) in examples.iter().enumerate() {
        match (&spec.language, example.language) {
            (None, None) => {}
            (None, Some(_)) => {
                return Err(invalid(format!("trainer: sentence {i} has a language the parser does not read")));
            }
            (Some(language), None) if language.languages.len() == 1 => {}
            (Some(language), None) => {
                return Err(invalid(format!(
                    "trainer: sentence {i} needs a language, one of {:?}",
                    language.languages
                )));
            }
            (Some(language), Some(row)) if row as usize >= language.languages.len() => {
                return Err(invalid(format!("trainer: sentence {i} language row {row} is out of range")));
            }
            (Some(_), Some(_)) => {}
        }
    }
    Ok(())
}

/// Attachment of `parser` on `examples`, decoded by the core exactly as
/// at inference. Refuses a sentence without one gold head and relation
/// in range per word.
pub fn attachment(parser: &BiaffineParser, examples: &[ParseExample]) -> Result<Attachment> {
    let hidden = parser.hidden();
    check_parses(examples, hidden, parser.labels().len())?;
    check_languages(examples, parser.spec())?;
    let mut a = Attachment::default();
    let (mut arcs, mut heads, mut scores, mut rels, mut w) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        let language = match (&parser.spec().language, e.language) {
            (Some(language), Some(row)) => Some(language.languages[row as usize].as_str()),
            _ => None,
        };
        parser.arcs_language(&e.features, &words, language, &mut arcs, &mut w)?;
        parser.decode(&arcs, n, &mut heads)?;
        parser.relations(&heads, &mut scores, &w)?;
        parser.relation_labels(&scores, n, &mut rels)?;
        a.words += n;
        for i in 0..n {
            if heads[i] == e.heads[i] {
                a.heads += 1;
                a.labelled += usize::from(rels[i] == e.rels[i]);
            }
        }
    }
    Ok(a)
}

impl Model for Parser<f32> {
    type Example = ParseExample;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        Parser::params(self).to_vec()
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        Parser::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        Parser::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        Parser::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        Parser::decays(self)
    }
    fn loss(&self, e: &ParseExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        Ok((Parser::loss_language(self, &e.features, e.language, &e.heads, &e.rels, grads, w)?, e.heads.len()))
    }
    fn score(&self, dev: &[ParseExample]) -> Result<f64> {
        Ok(attachment(&self.to_core()?, dev)?.las())
    }
}

/// Trains a parser for `spec` over features `hidden` wide on `train`,
/// keeping the weights with the best dev LAS.
pub fn fit_parser(
    spec: &HeadSpec,
    hidden: usize,
    train_set: &[ParseExample],
    dev: &[ParseExample],
    c: &Config,
) -> Result<(Parser<f32>, Report)> {
    fit_parser_with(spec, hidden, train_set, dev, c, &mut |_, _, _| Ok(()))
}

/// [`fit_parser`], calling `on_best` with the model, its epoch and dev
/// score each time the best improves (a checkpoint to publish).
pub fn fit_parser_with(
    spec: &HeadSpec,
    hidden: usize,
    train_set: &[ParseExample],
    dev: &[ParseExample],
    c: &Config,
    on_best: OnBest<'_, Parser<f32>>,
) -> Result<(Parser<f32>, Report)> {
    check_config(c, train_set.len())?;
    check_parses(train_set, hidden, spec.labels.len())?;
    check_parses(dev, hidden, spec.labels.len())?;
    check_languages(train_set, spec)?;
    check_languages(dev, spec)?;
    let mut rng = Rng::new(c.seed);
    let model = Parser::<f32>::new(spec, hidden, &mut rng)?;
    train(model, &mut rng, train_set, dev, c, on_best)
}

/// One training sentence's features at each mixed layer ([words, hidden]
/// each), gold heads and relations.
#[derive(Debug, Clone, PartialEq)]
pub struct MixParseExample {
    pub layers: Vec<Vec<f32>>,
    pub heads: Vec<u32>,
    pub rels: Vec<u32>,
}

/// A biaffine parser head and its learned scalar mix over encoder layers.
#[derive(Debug, Clone)]
pub struct MixParser<R: Real> {
    head: Parser<R>,
    params: Vec<Vec<R>>,
}

impl<R: Real> MixParser<R> {
    /// A seeded parser over a uniform mix of `layers` layers.
    pub fn new(spec: &HeadSpec, hidden: usize, layers: usize, rng: &mut Rng) -> Result<Self> {
        check_mix_layers(spec, layers)?;
        let plain = HeadSpec { layer_mix: None, ..spec.clone() };
        let head = Parser::new(&plain, hidden, rng)?;
        let mut params = head.params().to_vec();
        params.push(vec![R::ZERO; layers]);
        params.push(vec![R::ONE]);
        Ok(MixParser { head, params })
    }

    pub fn parser(&self) -> &Parser<R> {
        &self.head
    }

    fn head_len(&self) -> usize {
        self.params.len() - 2
    }

    /// The mix's weights (the softmax of its logits) and gain.
    pub fn weights(&self) -> (Vec<R>, R) {
        let k = self.head_len();
        weights_of(&self.params[k], self.params[k + 1][0])
    }

    /// The mixed features of one sequence's layers.
    pub fn mix(&self, layers: &[Vec<R>]) -> Result<Vec<R>> {
        let k = self.head_len();
        mix_layers(layers, &self.params[k], self.params[k + 1][0])
    }

    pub fn set_params(&mut self, params: Vec<Vec<R>>) -> Result<()> {
        let lens: Vec<usize> = self.params.iter().map(Vec::len).collect();
        if params.len() != lens.len() || params.iter().zip(&lens).any(|(p, &n)| p.len() != n) {
            return Err(invalid("trainer: weights do not match the mix and its parser"));
        }
        self.head.set_params(params[..lens.len() - 2].to_vec())?;
        self.params = params;
        Ok(())
    }

    pub fn params(&self) -> &[Vec<R>] {
        &self.params
    }

    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        self.params.iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    /// The loss of one sequence (`layers` its features at each layer), its
    /// gradients added to `grads`: the head's, then the mix's.
    pub fn loss(
        &self,
        layers: &[Vec<R>],
        heads: &[u32],
        rels: &[u32],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<f64> {
        let k = self.head_len();
        if grads.len() != self.params.len() || grads.iter().zip(&self.params).any(|(g, p)| g.len() != p.len()) {
            return Err(invalid("trainer: gradients do not match the mix and its parser"));
        }
        let x = self.mix(layers)?;
        let mut dx = Vec::new();
        let (head, mix) = grads.split_at_mut(k);
        let loss = self.head.loss_input(&x, heads, rels, head, w, &mut dx)?;
        let [dlogits, dgamma] = mix else { unreachable!() };
        mix_grad(layers, &self.params[k], self.params[k + 1][0], &dx, dlogits, dgamma);
        Ok(loss)
    }
}

impl MixParser<f32> {
    /// The head's spec with `layer_mix` set to this mix's layer count.
    pub fn spec(&self) -> HeadSpec {
        let layers = self.params[self.head_len()].len() as u32;
        HeadSpec { layer_mix: Some(LayerMix { layers, layer: None }), ..self.head.spec().clone() }
    }

    /// Each tensor of the head and its mix by suffix.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        let k = self.head_len();
        let mut out = self.head.tensors();
        out.push(("mix.weight".into(), self.params[k].clone()));
        out.push(("mix.gamma".into(), self.params[k + 1].clone()));
        out
    }

    /// A [`BiaffineParser`] carrying `layer_mix` and these weights.
    pub fn to_core(&self) -> Result<BiaffineParser> {
        let spec = self.spec();
        let tensors = self.tensors();
        let get = |s: &str| tensors.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
        BiaffineParser::from_tensors(&spec, self.head.hidden(), &get)
    }
}

impl Model for MixParser<f32> {
    type Example = MixParseExample;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        self.params.clone()
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        MixParser::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        let norm = opt.step(&mut self.params, grads, decay)?;
        let k = self.head_len();
        self.head.set_params(self.params[..k].to_vec())?;
        Ok(norm)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        MixParser::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        let mut d = self.head.decays();
        d.extend([false, false]);
        d
    }
    fn loss(&self, e: &MixParseExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        Ok((MixParser::loss(self, &e.layers, &e.heads, &e.rels, grads, w)?, e.heads.len()))
    }
    fn score(&self, dev: &[MixParseExample]) -> Result<f64> {
        let mixed: Vec<ParseExample> = dev
            .iter()
            .map(|e| {
                Ok(ParseExample {
                    features: self.mix(&e.layers)?,
                    heads: e.heads.clone(),
                    rels: e.rels.clone(),
                    language: None,
                })
            })
            .collect::<Result<_>>()?;
        Ok(attachment(&self.head.to_core()?, &mixed)?.las())
    }
}

/// Trains a parser over a learned mix of `layers` layers.
pub fn fit_parser_mix(
    spec: &HeadSpec,
    hidden: usize,
    layers: usize,
    train_set: &[MixParseExample],
    dev: &[MixParseExample],
    c: &Config,
    on_best: OnBest<'_, MixParser<f32>>,
) -> Result<(MixParser<f32>, Report)> {
    check_config(c, train_set.len())?;
    let mut rng = Rng::new(c.seed);
    let model = MixParser::<f32>::new(spec, hidden, layers, &mut rng)?;
    let labels = spec.labels.len();
    for (i, e) in train_set.iter().chain(dev).enumerate() {
        let n = e.heads.len();
        let first = e.layers.first().map_or(0, Vec::len);
        let ok = hidden > 0
            && e.layers.len() == layers
            && e.layers.iter().all(|l| l.len() == first)
            && n.checked_mul(hidden) == Some(first)
            && e.rels.len() == n
            && e.heads.iter().enumerate().all(|(j, &h)| h as usize <= n && h as usize != j + 1)
            && e.rels.iter().all(|&l| (l as usize) < labels);
        if !ok {
            return Err(invalid(format!(
                "trainer: sentence {i} does not fit {layers} layers of {hidden} features per word, its heads and relations"
            )));
        }
    }
    train(model, &mut rng, train_set, dev, c, on_best)
}
