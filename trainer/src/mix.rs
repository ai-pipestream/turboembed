//! A token tagger head over a learned mix of an encoder's layers (#1616):
//! ELMo's scalar mix (Peters et al. 2018), `gamma * sum_l softmax(s)_l *
//! x_l`, its weights trained with the head on per-layer features read
//! once (`turbo-train-head cache`). The mix starts uniform with gain 1;
//! neither takes weight decay.

use turbo::head::{TokenSplitter, TokenTagger};
use turbo::manifest::{HeadKind, HeadSpec, LayerMix};
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

pub(crate) fn check_mix_layers(spec: &HeadSpec, layers: usize) -> Result<()> {
    if layers == 0 {
        return Err(invalid("trainer: a mix of no layers"));
    }
    if let Some(m) = spec.layer_mix
        && m.layers as usize != layers
    {
        return Err(invalid(format!(
            "trainer: head {:?} has layer_mix.layers = {}, not {layers}",
            spec.name, m.layers
        )));
    }
    Ok(())
}

/// A mixed head's tensors by suffix: the head's, then the mix's logits
/// and gain, the last two of `params`.
pub(crate) fn mix_tensors(mut head: Vec<(String, Vec<f32>)>, params: &[Vec<f32>]) -> Vec<(String, Vec<f32>)> {
    let k = params.len() - 2;
    head.push(("mix.weight".into(), params[k].clone()));
    head.push(("mix.gamma".into(), params[k + 1].clone()));
    head
}

/// One optimizer step on a mixed head's `params`, then the head's own
/// tensors, all but the mix's two, handed back to it by `set_head`.
pub(crate) fn mix_step(
    opt: &mut AdamW<f32>,
    params: &mut [Vec<f32>],
    grads: &mut [Vec<f32>],
    decay: &[bool],
    set_head: impl FnOnce(Vec<Vec<f32>>) -> Result<()>,
) -> Result<f64> {
    let norm = opt.step(params, grads, decay)?;
    set_head(params[..params.len() - 2].to_vec())?;
    Ok(norm)
}

/// A mixed head's weight decay: the head's, and none for the mix's logits
/// and gain.
pub(crate) fn mix_decays(mut head: Vec<bool>) -> Vec<bool> {
    head.extend([false, false]);
    head
}

pub(crate) fn weights_of<R: Real>(logits: &[R], gamma: R) -> (Vec<R>, R) {
    let max = logits.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = logits.iter().map(|v| (v.to_f64() - max).exp()).collect();
    let z: f64 = e.iter().sum();
    (e.iter().map(|v| R::from_f64(v / z)).collect(), gamma)
}

pub(crate) fn mix_layers<R: Real>(layers: &[Vec<R>], logits: &[R], gamma: R) -> Result<Vec<R>> {
    let (a, gamma) = weights_of(logits, gamma);
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

pub(crate) fn mix_grad<R: Real>(
    layers: &[Vec<R>],
    logits: &[R],
    gamma: R,
    dx: &[R],
    dlogits: &mut [R],
    dgamma: &mut [R],
) {
    let (a, gamma) = weights_of(logits, gamma);
    let g: Vec<f64> = layers.iter().map(|l| l.iter().zip(dx).map(|(&v, &d)| v.to_f64() * d.to_f64()).sum()).collect();
    let mean: f64 = a.iter().zip(&g).map(|(&al, gl)| al.to_f64() * gl).sum();
    for (l, (al, gl)) in a.iter().zip(&g).enumerate() {
        dlogits[l] += R::from_f64(gamma.to_f64() * al.to_f64() * (gl - mean));
    }
    dgamma[0] += R::from_f64(mean);
}

impl<R: Real> MixTagger<R> {
    /// A seeded head over a uniform mix of `layers` layers.
    pub fn new(spec: &HeadSpec, hidden: usize, layers: usize, rng: &mut Rng) -> Result<Self> {
        check_mix_layers(spec, layers)?;
        let plain = HeadSpec { layer_mix: None, ..spec.clone() };
        let tagger = Tagger::new(&plain, hidden, rng)?;
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
        let [dlogits, dgamma] = mix else { unreachable!() };
        mix_grad(layers, &self.params[k], self.params[k + 1][0], &dx, dlogits, dgamma);
        Ok(loss)
    }
}

impl MixTagger<f32> {
    /// The head's spec with `layer_mix` set to this mix's layer count.
    pub fn spec(&self) -> HeadSpec {
        let layers = self.params[self.head_len()].len() as u32;
        HeadSpec { layer_mix: Some(LayerMix { layers, layer: None }), ..self.tagger.spec().clone() }
    }

    /// Each tensor of the head and its mix by suffix (`tagger::tensors`),
    /// ready for `TokenTagger::from_tensors` or a safetensors file.
    pub fn tensors(&self) -> Vec<(String, Vec<f32>)> {
        mix_tensors(self.tagger.tensors(), &self.params)
    }

    /// A [`TokenTagger`] carrying `layer_mix` and these weights.
    ///
    /// # Errors
    ///
    /// A weight that is not finite.
    pub fn to_token_tagger(&self) -> Result<TokenTagger> {
        let spec = self.spec();
        let tensors = self.tensors();
        let get = |s: &str| tensors.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
        TokenTagger::from_tensors(&spec, self.tagger.hidden(), &get)
    }

    /// A [`TokenSplitter`] carrying `layer_mix` and these weights.
    ///
    /// # Errors
    ///
    /// A weight that is not finite.
    pub fn to_token_splitter(&self) -> Result<TokenSplitter> {
        let spec = self.spec();
        let tensors = self.tensors();
        let get = |s: &str| tensors.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
        TokenSplitter::from_tensors(&spec, self.tagger.hidden(), &get)
    }
}

impl Model for MixTagger<f32> {
    type Example = MixExample;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        self.params.clone()
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        MixTagger::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        mix_step(opt, &mut self.params, grads, decay, |p| self.tagger.set_params(p))
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        MixTagger::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        mix_decays(self.tagger.decays())
    }
    fn loss(&self, e: &MixExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        let forms: Option<Vec<&str>> = (!e.forms.is_empty()).then(|| e.forms.iter().map(String::as_str).collect());
        Ok((MixTagger::loss(self, &e.layers, forms.as_deref(), &e.labels, grads, w)?, e.labels.len()))
    }
    fn score(&self, dev: &[MixExample]) -> Result<f64> {
        let mixed = mixed(self, dev)?;
        match self.tagger.spec().kind {
            HeadKind::SentenceBoundary => fit::score_boundary(&self.tagger.to_boundary()?, &mixed),
            HeadKind::TokenSplit => fit::score_split(&self.tagger.to_splitter()?, &mixed),
            _ => fit::score(&self.tagger.to_token_tagger()?, &mixed),
        }
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
        .map(|e| {
            Ok(Example {
                features: m.mix(&e.layers)?,
                labels: e.labels.clone(),
                forms: e.forms.clone(),
                language: None,
            })
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adamw::Settings;

    #[test]
    fn the_mix_tensors_follow_the_head_and_take_no_decay() {
        let head = vec![("out.weight".to_string(), vec![1.0f32])];
        let params = vec![vec![1.0], vec![0.5, 0.5], vec![2.0]];
        let names: Vec<_> = mix_tensors(head, &params).into_iter().collect();
        assert_eq!(
            names,
            [("out.weight".into(), vec![1.0]), ("mix.weight".into(), vec![0.5, 0.5]), ("mix.gamma".into(), vec![2.0])]
        );
        assert_eq!(mix_decays(vec![true, false]), [true, false, false, false]);
    }

    #[test]
    fn a_mix_step_hands_the_head_its_own_tensors_after_the_step() {
        let mut params = vec![vec![1.0f32, 1.0], vec![0.0, 0.0], vec![1.0]];
        let mut grads = vec![vec![1.0f32, 1.0], vec![0.0, 0.0], vec![0.0]];
        let mut opt = AdamW::new(Settings::default(), &[2, 2, 1]).unwrap();
        let mut given = None;
        let norm = mix_step(&mut opt, &mut params, &mut grads, &[true, false, false], |p| {
            given = Some(p);
            Ok(())
        })
        .unwrap();
        assert!(norm > 0.0);
        assert!(params[0][0] < 1.0, "the step moved the head's weights");
        assert_eq!(given, Some(vec![params[0].clone()]));
    }
}
