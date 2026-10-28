//! Training a biaffine parser: the loop of `fit` over sentences with
//! gold heads and relations, scored on dev by labelled attachment
//! through the core's own parser (`turbo::head::BiaffineParser`).

use turbo::head::BiaffineParser;
use turbo::head::parser::Work as CoreWork;
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
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

/// Attachment of `parser` on `examples`, decoded by the core exactly as
/// at inference. Refuses a sentence without one gold head and relation
/// in range per word.
pub fn attachment(parser: &BiaffineParser, examples: &[ParseExample]) -> Result<Attachment> {
    let hidden = parser.hidden();
    check_parses(examples, hidden, parser.labels().len())?;
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
    fn params(&self) -> &[Vec<f32>] {
        Parser::params(self)
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
    check_parses(train_set, hidden, spec.labels.len())?;
    check_parses(dev, hidden, spec.labels.len())?;
    let mut rng = Rng::new(c.seed);
    let model = Parser::<f32>::new(spec, hidden, &mut rng)?;
    train(model, &mut rng, train_set, dev, c, &mut |_, _, _| Ok(()))
}
