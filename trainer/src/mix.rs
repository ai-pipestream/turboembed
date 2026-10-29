//! A token tagger head over a learned mix of an encoder's layers (#1616):
//! ELMo's scalar mix (Peters et al. 2018), `gamma * sum_l softmax(s)_l *
//! x_l`, its weights trained with the head on per-layer features read
//! once (`turbo-train-head cache`). The mix starts uniform with gain 1;
//! neither takes weight decay.

use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{self, Config, Example, Model, OnBest, Report, check_config};
use crate::real::Real;
use crate::rng::Rng;
use crate::tagger::{Tagger, Work};

/// One training sequence's features at each mixed layer ([words, hidden]
/// each, the same words), gold labels and forms, as [`Example`].
#[derive(Debug, Clone, PartialEq)]
pub struct MixExample {
    pub layers: Vec<Vec<f32>>,
    pub labels: Vec<u32>,
    pub forms: Vec<String>,
}

/// The head and its mix: `params` holds the head's tensors, then the
/// mix's logits (one a layer), then its gain.
#[derive(Debug, Clone)]
pub struct MixTagger<R: Real> {
    tagger: Tagger<R>,
    params: Vec<Vec<R>>,
}

impl<R: Real> MixTagger<R> {
    /// A seeded head over a uniform mix of `layers` layers.
    pub fn new(spec: &HeadSpec, hidden: usize, layers: usize, rng: &mut Rng) -> Result<Self> {
        if layers == 0 {
            return Err(invalid("trainer: a mix of no layers"));
        }
        let tagger = Tagger::new(spec, hidden, rng)?;
        let mut params = tagger.params().to_vec();
        params.push(vec![R::ZERO; layers]);
        params.push(vec![R::ONE]);
        Ok(MixTagger { tagger, params })
    }

    pub fn tagger(&self) -> &Tagger<R> {
        &self.tagger
    }

    fn head_len(&self) -> usize {
        self.params.len() - 2
    }

    /// The mix's weights (the softmax of its logits) and gain.
    pub fn weights(&self) -> (Vec<R>, R) {
        let s = &self.params[self.head_len()];
        let max = s.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|v| (v.to_f64() - max).exp()).collect();
        let z: f64 = e.iter().sum();
        (e.iter().map(|v| R::from_f64(v / z)).collect(), self.params[self.head_len() + 1][0])
    }

    /// The mixed features of one sequence's layers.
    pub fn mix(&self, layers: &[Vec<R>]) -> Result<Vec<R>> {
        let (a, gamma) = self.weights();
        if layers.len() != a.len() || layers.iter().any(|l| l.len() != layers[0].len()) {
            return Err(invalid(format!("trainer: {} layers of features for a mix of {}", layers.len(), a.len())));
        }
        let mut out = vec![R::ZERO; layers[0].len()];
        for (x, &al) in layers.iter().zip(&a) {
            let k = gamma * al;
            out.iter_mut().zip(x).for_each(|(o, &v)| *o += k * v);
        }
        Ok(out)
    }

    pub fn set_params(&mut self, params: Vec<Vec<R>>) -> Result<()> {
        let lens: Vec<usize> = self.params.iter().map(Vec::len).collect();
        if params.len() != lens.len() || params.iter().zip(&lens).any(|(p, &n)| p.len() != n) {
            return Err(invalid("trainer: weights do not match the mix and its head"));
        }
        self.tagger.set_params(params[..lens.len() - 2].to_vec())?;
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
        forms: Option<&[&str]>,
        gold: &[u32],
        grads: &mut [Vec<R>],
        w: &mut Work<R>,
    ) -> Result<f64> {
        let k = self.head_len();
        if grads.len() != self.params.len() || grads.iter().zip(&self.params).any(|(g, p)| g.len() != p.len()) {
            return Err(invalid("trainer: gradients do not match the mix and its head"));
        }
        let x = self.mix(layers)?;
        let mut dx = Vec::new();
        let (head, mix) = grads.split_at_mut(k);
        let loss = self.tagger.loss_input(&x, forms, gold, head, w, &mut dx)?;
        // d/d x_l-weight: g_l = dx . x_l; the gain's is sum_l a_l g_l and
        // logit l's gamma a_l (g_l - that sum).
        let (a, gamma) = self.weights();
        let g: Vec<f64> =
            layers.iter().map(|l| l.iter().zip(&dx).map(|(&v, &d)| v.to_f64() * d.to_f64()).sum()).collect();
        let mean: f64 = a.iter().zip(&g).map(|(&al, gl)| al.to_f64() * gl).sum();
        for (l, (al, gl)) in a.iter().zip(&g).enumerate() {
            mix[0][l] += R::from_f64(gamma.to_f64() * al.to_f64() * (gl - mean));
        }
        mix[1][0] += R::from_f64(mean);
        Ok(loss)
    }
}

impl Model for MixTagger<f32> {
    type Example = MixExample;
    type Work = Work<f32>;
    fn params(&self) -> &[Vec<f32>] {
        &self.params
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        MixTagger::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        let norm = opt.step(&mut self.params, grads, decay)?;
        let k = self.head_len();
        self.tagger.set_params(self.params[..k].to_vec())?;
        Ok(norm)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        MixTagger::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        let mut d = self.tagger.decays();
        d.extend([false, false]);
        d
    }
    fn loss(&self, e: &MixExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        let forms: Option<Vec<&str>> = (!e.forms.is_empty()).then(|| e.forms.iter().map(String::as_str).collect());
        Ok((MixTagger::loss(self, &e.layers, forms.as_deref(), &e.labels, grads, w)?, e.labels.len()))
    }
    fn score(&self, dev: &[MixExample]) -> Result<f64> {
        let mixed = mixed(self, dev)?;
        fit::score(&self.tagger.to_token_tagger()?, &mixed)
    }
}

/// `examples` with their layers mixed by `m`, as the head reads them.
///
/// # Errors
///
/// An example whose layers do not fit the mix.
pub fn mixed(m: &MixTagger<f32>, examples: &[MixExample]) -> Result<Vec<Example>> {
    examples
        .iter()
        .map(|e| Ok(Example { features: m.mix(&e.layers)?, labels: e.labels.clone(), forms: e.forms.clone() }))
        .collect()
}

/// Trains a head over a learned mix of `layers` layers, as
/// [`fit::fit_with`] trains one over a single layer: the same loop, with
/// the mix's weights stepped with the head's.
///
/// # Errors
///
/// As `fit_with`, and examples whose layers do not fit the mix.
pub fn fit_mix(
    spec: &HeadSpec,
    hidden: usize,
    layers: usize,
    train: &[MixExample],
    dev: &[MixExample],
    c: &Config,
    on_best: OnBest<'_, MixTagger<f32>>,
) -> Result<(MixTagger<f32>, Report)> {
    check_config(c, train.len())?;
    let mut rng = Rng::new(c.seed);
    let model = MixTagger::<f32>::new(spec, hidden, layers, &mut rng)?;
    for e in train.iter().chain(dev) {
        let first = e.layers.first().map_or(0, Vec::len);
        if e.layers.len() != layers || e.layers.iter().any(|l| l.len() != first) || first != e.labels.len() * hidden {
            return Err(invalid(format!(
                "trainer: an example of {} layers for {} words, not {layers} of [words, {hidden}]",
                e.layers.len(),
                e.labels.len()
            )));
        }
    }
    fit::train(model, &mut rng, train, dev, c, on_best)
}
