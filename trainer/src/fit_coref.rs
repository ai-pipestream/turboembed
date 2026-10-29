//! Training a coreference ranker: the loop of `fit` over documents with
//! their mentions and chains, scored on dev by antecedent accuracy
//! through the core's own ranker (`turbo::head::CorefRanker`), decoded as
//! at inference.

use turbo::head::CorefRanker;
use turbo::head::tagger::Work as CoreWork;
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::coref::{Ranker, Work};
use crate::fit::{Config, Model, Report, check_config, train};
use crate::rng::Rng;

/// One training document: per-word features [words, hidden], its
/// mentions in order of first word then last, and each mention's chain.
#[derive(Debug, Clone, PartialEq)]
pub struct CorefExample {
    pub features: Vec<f32>,
    pub mentions: Vec<(u32, u32)>,
    pub chains: Vec<u32>,
}

/// The share of mentions whose decoded antecedent is right: an earlier
/// mention of its chain, or none for a mention that starts its chain
/// (or whose chain's earlier mentions are all out of reach). 1 for no
/// mentions.
pub fn antecedent_accuracy(head: &CorefRanker, examples: &[CorefExample]) -> Result<f64> {
    let hidden = head.hidden();
    let (mut right, mut total) = (0usize, 0usize);
    let (mut g, mut scores, mut out, mut w) = (Vec::new(), Vec::new(), Vec::new(), CoreWork::default());
    for (i, e) in examples.iter().enumerate() {
        if hidden == 0 || !e.features.len().is_multiple_of(hidden) || e.chains.len() != e.mentions.len() {
            return Err(invalid(format!("trainer: example {i} does not fit {hidden} features and a chain a mention")));
        }
        let n = e.features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        head.mentions(&e.features, &words, &e.mentions, &mut g, &mut w)?;
        head.scores(&g, &mut scores)?;
        head.decode(&scores, &mut out)?;
        for (a, got) in out.iter().enumerate() {
            let reach = head.antecedents().min(a);
            let linked = (1..=reach).any(|c| e.chains[a - c] == e.chains[a]);
            right += usize::from(match got {
                Some(b) => e.chains[*b as usize] == e.chains[a],
                None => !linked,
            });
            total += 1;
        }
    }
    Ok(if total == 0 { 1.0 } else { right as f64 / total as f64 })
}

impl Model for Ranker<f32> {
    type Example = CorefExample;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        Ranker::params(self).to_vec()
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        Ranker::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        Ranker::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        Ranker::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        Ranker::decays(self)
    }
    fn loss(&self, e: &CorefExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        Ranker::loss(self, &e.features, &e.mentions, &e.chains, grads, w)
    }
    fn score(&self, dev: &[CorefExample]) -> Result<f64> {
        antecedent_accuracy(&self.to_core()?, dev)
    }
}

/// Trains a coreference ranker for `spec` over features `hidden` wide on
/// `train_set`, keeping the weights with the best dev antecedent
/// accuracy.
pub fn fit_coref(
    spec: &HeadSpec,
    hidden: usize,
    train_set: &[CorefExample],
    dev: &[CorefExample],
    c: &Config,
) -> Result<(Ranker<f32>, Report)> {
    check_config(c, train_set.len())?;
    let mut rng = Rng::new(c.seed);
    let model = Ranker::<f32>::new(spec, hidden, &mut rng)?;
    // The loss checks each document before any gradient moves; check
    // dev's the same way now, so a bad dev set fails first.
    let (mut w, mut grads) = (Work::default(), model.zero_grads());
    for e in dev {
        model.loss(&e.features, &e.mentions, &e.chains, &mut grads, &mut w)?;
    }
    train(model, &mut rng, train_set, dev, c, &mut |_, _, _| Ok(()))
}
