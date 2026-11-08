//! Training an edit-tree lemmatizer: the trees mined from the training
//! pairs, then the loop of `fit` over sentences with gold tags and
//! lemmas, scored on dev by lemma accuracy through the core's own
//! lemmatizer (`turbo::head::EditTreeLemmatizer`).

use std::collections::{BTreeMap, HashMap};

use turbo::head::EditTreeLemmatizer;
use turbo::head::edit_tree::EditTree;
use turbo::head::lemmatizer::Work as CoreWork;
use turbo::manifest::{HeadSpec, LayerMix, LemmaLookup, MAX_LABELS, MAX_SEEN_FORM_BYTES, SeenLemma};
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{Config, Model, OnBest, Report, check_config, train};
use crate::lemma::{Lemmatizer, Work};
use crate::mix::{check_mix_layers, mix_grad, mix_layers, weights_of};
use crate::real::Real;
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
    pub language: Option<u32>,
}

/// The trees of the training pairs that come up at least `min_freq`
/// times, as labels: most frequent first, then by label, at most the
/// labels a head holds, none longer than a label may be. Each pair's
/// tree is `EditTree::mine_cased`'s, so a capitalized form whose lemma is
/// lowercase shares the lowercase form's tree under `(l ...)`. A pair
/// whose tree the core would refuse on its own form (an empty lemma for
/// a word) gives none.
pub fn mine_trees<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>, min_freq: usize) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (form, lemma) in pairs {
        let tree = EditTree::mine_cased(form, lemma);
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

/// Unambiguous training word forms seen at least `min_freq` times, for
/// [`HeadSpec::seen_lookup`]: a form whose training occurrences all had
/// one lemma is listed with that lemma, ordered by form.
pub fn mine_seen_lookup<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>, min_freq: usize) -> LemmaLookup {
    mine_seen_lookup_tagged(pairs.into_iter().map(|(f, l)| (f, None, l)), min_freq)
}

/// [`mine_seen_lookup`] with an optional tag index per training word:
/// a form that is unambiguous across all tags is listed with `tag: None`;
/// a form that is ambiguous overall is listed with `tag: Some(t)` for each
/// tag `t` under which it had one lemma at least `min_freq` times.
pub fn mine_seen_lookup_tagged<'a>(
    triples: impl IntoIterator<Item = (&'a str, Option<u32>, &'a str)>,
    min_freq: usize,
) -> LemmaLookup {
    let min_freq = min_freq.max(1);
    let mut by_form: BTreeMap<&'a str, BTreeMap<&'a str, usize>> = BTreeMap::new();
    let mut by_tag: BTreeMap<(&'a str, u32), BTreeMap<&'a str, usize>> = BTreeMap::new();
    for (form, tag, lemma) in triples {
        if form.is_empty() || form.len() > MAX_SEEN_FORM_BYTES || lemma.is_empty() || lemma.len() > MAX_SEEN_FORM_BYTES
        {
            continue;
        }
        *by_form.entry(form).or_default().entry(lemma).or_default() += 1;
        if let Some(t) = tag {
            *by_tag.entry((form, t)).or_default().entry(lemma).or_default() += 1;
        }
    }
    let mut entries = Vec::new();
    for (&form, lemmas) in &by_form {
        if lemmas.len() == 1 {
            let (&lemma, &count) = lemmas.iter().next().expect("one lemma");
            if count >= min_freq {
                entries.push(SeenLemma { form: form.to_owned(), tag: None, lemma: lemma.to_owned() });
            }
        } else {
            for (&(f, t), tag_lemmas) in by_tag.range((form, 0)..=(form, u32::MAX)) {
                if tag_lemmas.len() == 1 {
                    let (&lemma, &count) = tag_lemmas.iter().next().expect("one lemma");
                    if count >= min_freq {
                        entries.push(SeenLemma { form: f.to_owned(), tag: Some(t), lemma: lemma.to_owned() });
                    }
                }
            }
        }
    }
    LemmaLookup { min_freq: min_freq as u32, entries }
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
    check_languages(examples.clone(), head.spec())?;
    let (mut right, mut total) = (0usize, 0usize);
    let (mut scores, mut trees, mut w) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.forms.len();
        let words: Vec<u32> = (0..n as u32).collect();
        let tags = head.spec().pos_input.as_ref().map(|_| e.tags.as_slice());
        let language = match (&head.spec().language, e.language) {
            (Some(language), Some(row)) => Some(language.languages[row as usize].as_str()),
            _ => None,
        };
        head.scores_language(&e.features, &words, tags, language, &mut scores, &mut w)?;
        let forms: Vec<&str> = e.forms.iter().map(String::as_str).collect();
        head.decode(&scores, &forms, &mut trees)?;
        let lemmas = head.lemmas_tagged(&trees, &forms, tags)?;
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

fn check_languages<'a>(examples: impl Iterator<Item = &'a LemmaExample>, spec: &HeadSpec) -> Result<()> {
    for (i, example) in examples.enumerate() {
        match (&spec.language, example.language) {
            (None, None) => {}
            (None, Some(_)) => {
                return Err(invalid(format!("trainer: sentence {i} has a language the lemmatizer does not read")));
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
        Ok((Lemmatizer::loss_language(self, &p.e.features, &p.e.tags, p.e.language, &p.gold, grads, w)?, words))
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
    check_languages(train_set.iter(), spec)?;
    check_languages(dev.iter(), spec)?;
    let index: HashMap<&str, u32> = spec.labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let prepare = |e: LemmaExample| {
        let gold = e
            .forms
            .iter()
            .zip(&e.lemmas)
            .map(|(f, l)| index.get(EditTree::mine_cased(f, l).label().as_str()).copied())
            .collect();
        Prepared { e, gold }
    };
    let train_set: Vec<Prepared> = train_set.into_iter().map(prepare).collect();
    let dev: Vec<Prepared> = dev.into_iter().map(prepare).collect();
    let mut rng = Rng::new(c.seed);
    let model = Lemmatizer::<f32>::new(spec, hidden, &mut rng)?;
    train(model, &mut rng, &train_set, &dev, c, on_best)
}

/// One training sentence's features at each mixed layer ([words, hidden]
/// each), gold tags, forms and gold lemmas.
#[derive(Debug, Clone, PartialEq)]
pub struct MixLemmaExample {
    pub layers: Vec<Vec<f32>>,
    pub tags: Vec<u32>,
    pub forms: Vec<String>,
    pub lemmas: Vec<String>,
}

/// A lemmatizer head and its learned scalar mix over encoder layers.
#[derive(Debug, Clone)]
pub struct MixLemmatizer<R: Real> {
    head: Lemmatizer<R>,
    params: Vec<Vec<R>>,
}

impl<R: Real> MixLemmatizer<R> {
    /// A seeded lemmatizer over a uniform mix of `layers` layers.
    pub fn new(spec: &HeadSpec, hidden: usize, layers: usize, rng: &mut Rng) -> Result<Self> {
        check_mix_layers(spec, layers)?;
        let plain = HeadSpec { layer_mix: None, ..spec.clone() };
        let head = Lemmatizer::new(&plain, hidden, rng)?;
        let mut params = head.params().to_vec();
        params.push(vec![R::ZERO; layers]);
        params.push(vec![R::ONE]);
        Ok(MixLemmatizer { head, params })
    }

    pub fn lemmatizer(&self) -> &Lemmatizer<R> {
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
            return Err(invalid("trainer: weights do not match the mix and its lemmatizer"));
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
        tags: &[u32],
        gold: &[Option<u32>],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<f64> {
        let k = self.head_len();
        if grads.len() != self.params.len() || grads.iter().zip(&self.params).any(|(g, p)| g.len() != p.len()) {
            return Err(invalid("trainer: gradients do not match the mix and its lemmatizer"));
        }
        let x = self.mix(layers)?;
        let mut dx = Vec::new();
        let (head, mix) = grads.split_at_mut(k);
        let loss = self.head.loss_input(&x, tags, gold, head, w, &mut dx)?;
        let [dlogits, dgamma] = mix else { unreachable!() };
        mix_grad(layers, &self.params[k], self.params[k + 1][0], &dx, dlogits, dgamma);
        Ok(loss)
    }
}

impl MixLemmatizer<f32> {
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

    /// An [`EditTreeLemmatizer`] carrying `layer_mix` and these weights.
    pub fn to_core(&self) -> Result<EditTreeLemmatizer> {
        let spec = self.spec();
        let tensors = self.tensors();
        let get = |s: &str| tensors.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
        EditTreeLemmatizer::from_tensors(&spec, self.head.hidden(), &get)
    }
}

pub(crate) struct MixPrepared {
    e: MixLemmaExample,
    gold: Vec<Option<u32>>,
}

impl Model for MixLemmatizer<f32> {
    type Example = MixPrepared;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        self.params.clone()
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        MixLemmatizer::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        let norm = opt.step(&mut self.params, grads, decay)?;
        let k = self.head_len();
        self.head.set_params(self.params[..k].to_vec())?;
        Ok(norm)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        MixLemmatizer::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        let mut d = self.head.decays();
        d.extend([false, false]);
        d
    }
    fn loss(&self, p: &MixPrepared, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        let words = p.gold.iter().filter(|g| g.is_some()).count();
        Ok((MixLemmatizer::loss(self, &p.e.layers, &p.e.tags, &p.gold, grads, w)?, words))
    }
    fn score(&self, dev: &[MixPrepared]) -> Result<f64> {
        let mixed: Vec<LemmaExample> = dev
            .iter()
            .map(|p| {
                Ok(LemmaExample {
                    features: self.mix(&p.e.layers)?,
                    tags: p.e.tags.clone(),
                    forms: p.e.forms.clone(),
                    lemmas: p.e.lemmas.clone(),
                    language: None,
                })
            })
            .collect::<Result<_>>()?;
        accuracy_of(&self.head.to_core()?, mixed.iter())
    }
}

/// Trains a lemmatizer over a learned mix of `layers` layers.
pub fn fit_lemmatizer_mix(
    spec: &HeadSpec,
    hidden: usize,
    layers: usize,
    train_set: Vec<MixLemmaExample>,
    dev: Vec<MixLemmaExample>,
    c: &Config,
    on_best: OnBest<'_, MixLemmatizer<f32>>,
) -> Result<(MixLemmatizer<f32>, Report)> {
    check_config(c, train_set.len())?;
    let mut rng = Rng::new(c.seed);
    let model = MixLemmatizer::<f32>::new(spec, hidden, layers, &mut rng)?;
    let tags = spec.pos_input.as_ref().map(|p| p.labels);
    for (i, e) in train_set.iter().chain(&dev).enumerate() {
        let n = e.forms.len();
        let first = e.layers.first().map_or(0, Vec::len);
        let ok = hidden > 0
            && e.layers.len() == layers
            && e.layers.iter().all(|l| l.len() == first)
            && n.checked_mul(hidden) == Some(first)
            && e.lemmas.len() == n
            && match tags {
                Some(k) => e.tags.len() == n && e.tags.iter().all(|&t| t < k),
                None => e.tags.is_empty(),
            };
        if !ok {
            return Err(invalid(format!(
                "trainer: sentence {i} does not fit {layers} layers of {hidden} features, a form, a lemma and a tag per word"
            )));
        }
    }
    let index: HashMap<&str, u32> = spec.labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let prepare = |e: MixLemmaExample| {
        let gold = e
            .forms
            .iter()
            .zip(&e.lemmas)
            .map(|(f, l)| index.get(EditTree::mine_cased(f, l).label().as_str()).copied())
            .collect();
        MixPrepared { e, gold }
    };
    let train_set: Vec<MixPrepared> = train_set.into_iter().map(prepare).collect();
    let dev: Vec<MixPrepared> = dev.into_iter().map(prepare).collect();
    train(model, &mut rng, &train_set, &dev, c, on_best)
}
