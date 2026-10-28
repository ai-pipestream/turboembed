//! Training a biaffine parser: the loop of `fit` over sentences with
//! gold heads and relations, scored on dev by labelled attachment
//! through the core's own parser (`turbo::head::BiaffineParser`).

use turbo::head::BiaffineParser;
use turbo::head::parser::Work as CoreWork;
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::fit::{Config, Model, Report, check_config, train};
use crate::parser::{Parser, Work};
use crate::rng::Rng;

/// One training sentence: per-word features [words, hidden], each word's
/// gold head (0 the root, else the word's 1-based place) and relation.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseExample {
    pub features: Vec<f32>,
    pub heads: Vec<u32>,
    pub rels: Vec<u32>,
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

/// Attachment of `parser` on `examples`, decoded by the core exactly as
/// at inference.
pub fn attachment(parser: &BiaffineParser, examples: &[ParseExample]) -> Result<Attachment> {
    let hidden = parser.hidden();
    let mut a = Attachment::default();
    let (mut arcs, mut heads, mut scores, mut rels, mut w) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        parser.arcs(&e.features, &words, &mut arcs, &mut w)?;
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
    fn params(&self) -> &Vec<Vec<f32>> {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Vec<Vec<f32>> {
        &mut self.params
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        Parser::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        Parser::decays(self)
    }
    fn loss(&self, e: &ParseExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        Ok((Parser::loss(self, &e.features, &e.heads, &e.rels, grads, w)?, e.heads.len()))
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
    check_config(c, train_set.len())?;
    for (i, e) in train_set.iter().chain(dev).enumerate() {
        let n = e.heads.len();
        let fits = e.features.len() == n * hidden
            && e.rels.len() == n
            && e.heads.iter().enumerate().all(|(j, &h)| h as usize <= n && h as usize != j + 1)
            && e.rels.iter().all(|&l| (l as usize) < spec.labels.len());
        if !fits {
            return Err(invalid(format!(
                "trainer: sentence {i} does not fit {hidden} features per word, its heads and relations"
            )));
        }
    }
    let mut rng = Rng::new(c.seed);
    let model = Parser::<f32>::new(spec, hidden, &mut rng)?;
    train(model, &mut rng, train_set, dev, c)
}
