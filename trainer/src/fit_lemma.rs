//! Training an edit-tree lemmatizer: the trees mined from the training
//! pairs, then the loop of `fit` over sentences with gold tags and
//! lemmas, scored on dev by lemma accuracy through the core's own
//! lemmatizer (`turbo::head::EditTreeLemmatizer`).

use std::collections::HashMap;

use turbo::head::EditTreeLemmatizer;
use turbo::head::edit_tree::EditTree;
use turbo::head::lemmatizer::Work as CoreWork;
use turbo::manifest::{HeadSpec, MAX_LABELS};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{Config, Model, OnBest, Report, check_config, train};
use crate::lemma::{Lemmatizer, Work};
use crate::rng::Rng;

/// The longest tree label a head carries (manifest rule).
const MAX_LABEL: usize = 128;

/// One training sentence: per-word features [words, hidden], each word's
/// form and gold lemma, and its gold tag when the head reads tags (else
/// empty).
#[derive(Debug, Clone, PartialEq)]
pub struct LemmaExample {
    pub features: Vec<f32>,
    pub tags: Vec<u32>,
    pub forms: Vec<String>,
    pub lemmas: Vec<String>,
}

/// The trees of the training pairs that come up at least `min_freq`
/// times, as labels: most frequent first, then by label, at most the
/// labels a head holds, none longer than a label may be. A pair whose
/// tree the core would refuse on its own form (an empty lemma for a
/// word) gives none.
pub fn mine_trees<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>, min_freq: usize) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (form, lemma) in pairs {
        let tree = EditTree::mine(form, lemma);
        if tree.apply(form).is_some() {
            *counts.entry(tree.label()).or_default() += 1;
        }
    }
    let mut kept: Vec<(String, usize)> =
        counts.into_iter().filter(|(l, c)| *c >= min_freq.max(1) && l.len() <= MAX_LABEL).collect();
    kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    kept.truncate(MAX_LABELS);
    kept.into_iter().map(|(l, _)| l).collect()
}

/// A sentence with each word's gold tree among the head's, None when the
/// head does not have it.
pub(crate) struct Prepared {
    e: LemmaExample,
    gold: Vec<Option<u32>>,
}

/// Lemma accuracy of `head` on `examples`, decoded by the core exactly as
/// at inference, the gold tags standing in for the tagger's.
pub fn accuracy(head: &EditTreeLemmatizer, examples: &[LemmaExample]) -> Result<f64> {
    accuracy_of(head, examples.iter())
}

fn accuracy_of<'a>(head: &EditTreeLemmatizer, examples: impl Iterator<Item = &'a LemmaExample> + Clone) -> Result<f64> {
    let hidden = head.hidden();
    check(examples.clone(), hidden, head.spec().pos_input.as_ref().map(|p| p.labels))?;
    let (mut right, mut total) = (0usize, 0usize);
    let (mut scores, mut trees, mut w) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.forms.len();
        let words: Vec<u32> = (0..n as u32).collect();
        let tags = head.spec().pos_input.as_ref().map(|_| e.tags.as_slice());
        head.scores(&e.features, &words, tags, &mut scores, &mut w)?;
        let forms: Vec<&str> = e.forms.iter().map(String::as_str).collect();
        head.decode(&scores, &forms, &mut trees)?;
        let lemmas = head.lemmas(&trees, &forms)?;
        right += lemmas.iter().zip(&e.lemmas).filter(|(a, b)| a == b).count();
        total += n;
    }
    Ok(if total == 0 { 0.0 } else { right as f64 / total as f64 })
}

/// Each sentence has `hidden` features, a form and a lemma per word, and
/// a tag in range per word exactly when the head reads `tags` tags.
fn check<'a>(examples: impl Iterator<Item = &'a LemmaExample>, hidden: usize, tags: Option<u32>) -> Result<()> {
    for (i, e) in examples.enumerate() {
        let n = e.forms.len();
        let fits = hidden > 0
            && n.checked_mul(hidden) == Some(e.features.len())
            && e.lemmas.len() == n
            && match tags {
                Some(k) => e.tags.len() == n && e.tags.iter().all(|&t| t < k),
                None => e.tags.is_empty(),
            };
        if !fits {
            return Err(invalid(format!(
                "trainer: sentence {i} does not fit {hidden} features, a form, a lemma and a tag in range per word"
            )));
        }
    }
    Ok(())
}

impl Model for Lemmatizer<f32> {
    type Example = Prepared;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        Lemmatizer::params(self).to_vec()
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        Lemmatizer::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        Lemmatizer::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        Lemmatizer::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        Lemmatizer::decays(self)
    }
    fn loss(&self, p: &Prepared, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        let words = p.gold.iter().filter(|g| g.is_some()).count();
        Ok((Lemmatizer::loss(self, &p.e.features, &p.e.tags, &p.gold, grads, w)?, words))
    }
    fn score(&self, dev: &[Prepared]) -> Result<f64> {
        accuracy_of(&self.to_core()?, dev.iter().map(|p| &p.e))
    }
}

/// Trains a lemmatizer for `spec` (its labels the trees, as
/// [`mine_trees`] gives them) over features `hidden` wide on `train`,
/// keeping the weights with the best dev lemma accuracy. A word whose
/// tree the head does not have is left out of the loss but still scored.
pub fn fit_lemmatizer(
    spec: &HeadSpec,
    hidden: usize,
    train_set: Vec<LemmaExample>,
    dev: Vec<LemmaExample>,
    c: &Config,
) -> Result<(Lemmatizer<f32>, Report)> {
    fit_lemmatizer_with(spec, hidden, train_set, dev, c, &mut |_, _, _| Ok(()))
}

/// [`fit_lemmatizer`], calling `on_best` with the model, its epoch and
/// dev score each time the best improves (a checkpoint to publish).
pub fn fit_lemmatizer_with(
    spec: &HeadSpec,
    hidden: usize,
    train_set: Vec<LemmaExample>,
    dev: Vec<LemmaExample>,
    c: &Config,
    on_best: OnBest<'_, Lemmatizer<f32>>,
) -> Result<(Lemmatizer<f32>, Report)> {
    check_config(c, train_set.len())?;
    let tags = spec.pos_input.as_ref().map(|p| p.labels);
    check(train_set.iter(), hidden, tags)?;
    check(dev.iter(), hidden, tags)?;
    let index: HashMap<&str, u32> = spec.labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let prepare = |e: LemmaExample| {
        let gold = e
            .forms
            .iter()
            .zip(&e.lemmas)
            .map(|(f, l)| index.get(EditTree::mine(f, l).label().as_str()).copied())
            .collect();
        Prepared { e, gold }
    };
    let train_set: Vec<Prepared> = train_set.into_iter().map(prepare).collect();
    let dev: Vec<Prepared> = dev.into_iter().map(prepare).collect();
    let mut rng = Rng::new(c.seed);
    let model = Lemmatizer::<f32>::new(spec, hidden, &mut rng)?;
    train(model, &mut rng, &train_set, &dev, c, on_best)
}
