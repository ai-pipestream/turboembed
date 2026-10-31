//! Training a relation classifier: the loop of `fit` over sequences with
//! their mentions and gold relations, scored on dev by micro relation F1
//! through the core's own classifier (`turbo::head::RelationClassifier`),
//! decoded as at inference.

use turbo::head::RelationClassifier;
use turbo::head::relation::{Mention, RelationWork};
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{Config, Model, Report, check_config, train};
use crate::relation::{GoldRelation, Relations, Work};
use crate::rng::Rng;

/// One training sequence: per-word features [words, hidden], its
/// mentions, and its gold relations between them, none labelled `none`.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationExample {
    pub features: Vec<f32>,
    pub mentions: Vec<Mention>,
    pub relations: Vec<GoldRelation>,
}

/// The counts behind relation F1 for one label: relations found and
/// right, found, and gold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub right: usize,
    pub found: usize,
    pub gold: usize,
}

impl Counts {
    /// F1 of these counts; 1 when neither side has a relation.
    pub fn f1(&self) -> f64 {
        if self.found + self.gold == 0 { 1.0 } else { 2.0 * self.right as f64 / (self.found + self.gold) as f64 }
    }
}

/// Each label's counts of `head` on `examples`, by label index (`none`'s
/// all zero): a relation counts when its mentions, direction and label
/// are a gold relation's.
pub fn relation_counts(head: &RelationClassifier, examples: &[RelationExample]) -> Result<Vec<Counts>> {
    let hidden = head.hidden();
    let mut counts = vec![Counts::default(); head.labels().len()];
    let (mut scores, mut kept, mut w) = (Vec::new(), Vec::new(), RelationWork::default());
    for (i, e) in examples.iter().enumerate() {
        if hidden == 0 || !e.features.len().is_multiple_of(hidden) {
            return Err(invalid(format!("trainer: example {i} is not rows of {hidden} features")));
        }
        let words: Vec<u32> = (0..(e.features.len() / hidden) as u32).collect();
        head.scores(&e.features, &words, &e.mentions, &mut scores, &mut w)?;
        head.decode(&scores, e.mentions.len(), &mut kept)?;
        for r in &kept {
            let c = &mut counts[r.label as usize];
            c.found += 1;
            c.right += usize::from(e.relations.contains(&GoldRelation { head: r.head, tail: r.tail, label: r.label }));
        }
        for g in &e.relations {
            let c = counts.get_mut(g.label as usize).ok_or_else(|| invalid(format!("trainer: gold {g:?}")))?;
            c.gold += 1;
        }
    }
    Ok(counts)
}

/// Micro relation F1 of `head` on `examples`, over every label but `none`.
pub fn relation_f1(head: &RelationClassifier, examples: &[RelationExample]) -> Result<f64> {
    let all = relation_counts(head, examples)?.into_iter().fold(Counts::default(), |a, c| Counts {
        right: a.right + c.right,
        found: a.found + c.found,
        gold: a.gold + c.gold,
    });
    Ok(all.f1())
}

impl Model for Relations<f32> {
    type Example = RelationExample;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        Relations::params(self).to_vec()
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        Relations::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        Relations::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        Relations::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        Relations::decays(self)
    }
    fn loss(&self, e: &RelationExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        Relations::loss(self, &e.features, &e.mentions, &e.relations, grads, w)
    }
    fn score(&self, dev: &[RelationExample]) -> Result<f64> {
        relation_f1(&self.to_core()?, dev)
    }
}

/// Trains a relation classifier for `spec` over features `hidden` wide on
/// `train_set`, keeping the weights with the best dev relation F1.
pub fn fit_relations(
    spec: &HeadSpec,
    hidden: usize,
    train_set: &[RelationExample],
    dev: &[RelationExample],
    c: &Config,
) -> Result<(Relations<f32>, Report)> {
    check_config(c, train_set.len())?;
    let mut rng = Rng::new(c.seed);
    let model = Relations::<f32>::new(spec, hidden, &mut rng)?;
    // Each example's mentions and relations are checked by the loss
    // before any gradient moves; check dev's the same way now, so a bad
    // dev set fails first.
    let mut w = Work::default();
    let mut grads = model.zero_grads();
    for e in dev {
        model.loss(&e.features, &e.mentions, &e.relations, &mut grads, &mut w)?;
    }
    train(model, &mut rng, train_set, dev, c, &mut |_, _, _| Ok(()))
}
