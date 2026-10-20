use super::{
    DenseClassifierModel, OutputKind,
    math::{self, Rng},
};
use crate::heads::{ClassifierHead, DenseLayer, HiddenActivation};
use std::sync::Arc;
use turbo_analysis::{Layer, TagSet};
use turbo_analysis_layers::Vectors;
mod optimizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Logistic,
    Mlp { hidden: u32, activation: HiddenActivation },
}
/// Minibatch Adam with coupled L2 weight decay (biases are not decayed).
#[derive(Clone, Debug)]
pub struct AdamParameters {
    pub seed: u64,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub batch_size: usize,
    pub epochs: usize,
    /// Stop after this many validation epochs without a strict improvement.
    pub patience: Option<usize>,
}
impl Default for AdamParameters {
    fn default() -> Self {
        Self { seed: 42, learning_rate: 0.01, weight_decay: 0., batch_size: 32, epochs: 200, patience: None }
    }
}
#[derive(Clone, Debug)]
pub struct DenseClassifierTrainer {
    pub architecture: Architecture,
    pub output: OutputKind,
    pub params: AdamParameters,
}
#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub model: DenseClassifierModel,
    pub epochs: usize,
    pub best_epoch: usize,
    pub training_loss: f64,
    pub validation_loss: Option<f64>,
}
impl DenseClassifierTrainer {
    /// Targets are one probability per label (softmax rows sum to one).
    /// Validation must use the same encoder and dimension. With validation,
    /// return the best validation checkpoint, including epoch zero.
    pub fn fit<L: Layer>(
        &self,
        vectors: &Vectors<L>,
        targets: &[Vec<f64>],
        tags: Arc<TagSet>,
        validation: Option<(&Vectors<L>, &[Vec<f64>])>,
    ) -> Result<TrainingResult, String> {
        let p = &self.params;
        if !p.learning_rate.is_finite()
            || p.learning_rate <= 0.
            || !p.weight_decay.is_finite()
            || p.weight_decay < 0.
            || p.batch_size == 0
            || p.epochs == 0
            || p.patience == Some(0)
            || (p.patience.is_some() && validation.is_none())
        {
            return Err("invalid Adam parameters or patience without validation".into());
        }
        vectors.check()?;
        math::validate_targets(targets, vectors.len(), tags.len(), self.output)?;
        if tags.is_empty() || tags.len() > u16::MAX as usize {
            return Err("invalid classifier labels".into());
        }
        let activation = match self.architecture {
            Architecture::Logistic => HiddenActivation::None,
            Architecture::Mlp { activation, .. }
                if matches!(activation, HiddenActivation::Relu | HiddenActivation::Gelu) =>
            {
                activation
            }
            _ => return Err("MLP activation must be ReLU or GeLU".into()),
        };
        let mut rng = Rng(p.seed);
        let dims = match self.architecture {
            Architecture::Logistic => vec![vectors.dim, tags.len() as u32],
            Architecture::Mlp { hidden, .. } => vec![vectors.dim, hidden, tags.len() as u32],
        };
        // Bound training allocations before creating parameters and Adam state.
        let mut count = 0usize;
        for pair in dims.windows(2) {
            if pair.contains(&0) {
                return Err("classifier dimensions must be positive".into());
            }
            count = (pair[0] as usize)
                .checked_mul(pair[1] as usize)
                .and_then(|n| n.checked_add(pair[1] as usize))
                .and_then(|n| count.checked_add(n))
                .ok_or("classifier dimensions overflow")?;
        }
        if count > 4_000_000 {
            return Err("classifier exceeds four million training parameters".into());
        }
        let mut layers = Vec::new();
        for pair in dims.windows(2) {
            let scale = (6. / (f64::from(pair[0]) + f64::from(pair[1]))).sqrt();
            let weights = (0..pair[0] as usize * pair[1] as usize).map(|_| (2. * rng.unit() - 1.) * scale).collect();
            layers.push(DenseLayer::new(pair[0], pair[1], weights, vec![0.; pair[1] as usize])?);
        }
        let build = |layers: &[DenseLayer]| -> Result<DenseClassifierModel, String> {
            let a = &layers[0];
            let head = if layers.len() == 1 {
                ClassifierHead::linear(
                    vectors.model.clone(),
                    vectors.dim,
                    tags.clone(),
                    self.output.mode(),
                    a.weights.clone(),
                    a.bias.clone(),
                )?
            } else {
                let b = &layers[1];
                ClassifierHead::mlp(
                    vectors.model.clone(),
                    vectors.dim,
                    a.out_dim,
                    tags.clone(),
                    self.output.mode(),
                    activation,
                    a.weights.clone(),
                    a.bias.clone(),
                    b.weights.clone(),
                    b.bias.clone(),
                )?
            };
            DenseClassifierModel::new(head)
        };
        let initial = build(&layers)?;
        initial.check_vectors(vectors)?;
        if let Some((v, y)) = validation {
            initial.check_vectors(v)?;
            math::validate_targets(y, v.len(), tags.len(), self.output)?;
        }
        let dataset_loss = |layers: &[DenseLayer], v: &Vectors<L>, y: &[Vec<f64>]| -> Result<f64, String> {
            let mut loss = 0.;
            for (x, y) in v.data.chunks_exact(v.dim as usize).zip(y) {
                let (a, _) = optimizer::activations(layers, activation, x)?;
                loss += math::loss(a.last().unwrap(), y, self.output, 1.)? / v.len() as f64;
            }
            if !loss.is_finite() {
                return Err("non-finite dataset loss".into());
            }
            Ok(loss)
        };
        let mut best_layers = layers.clone();
        let mut best_loss = match validation {
            Some((v, y)) => dataset_loss(&layers, v, y)?,
            None => f64::INFINITY,
        };
        let mut best_epoch = 0;
        let mut stale = 0;
        let mut epochs = 0;
        let mut order: Vec<_> = (0..vectors.len()).collect();
        let mut adam = optimizer::Adam::new(&layers);
        for epoch in 0..p.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            for batch in order.chunks(p.batch_size) {
                let mut grads = optimizer::zeros(&layers);
                for &i in batch {
                    let d = vectors.dim as usize;
                    optimizer::gradient(
                        &layers,
                        activation,
                        self.output,
                        &vectors.data[i * d..(i + 1) * d],
                        &targets[i],
                        &mut grads,
                    )?;
                }
                adam.step(&mut layers, &grads, batch.len(), p)?;
            }
            epochs = epoch + 1;
            if let Some((v, y)) = validation {
                let loss = dataset_loss(&layers, v, y)?;
                if loss < best_loss {
                    best_loss = loss;
                    best_layers = layers.clone();
                    best_epoch = epochs;
                    stale = 0;
                } else {
                    stale += 1;
                }
                if p.patience.is_some_and(|limit| stale >= limit) {
                    break;
                }
            }
        }
        if validation.is_some() {
            layers = best_layers;
        } else {
            best_epoch = epochs;
        }
        let training_loss = dataset_loss(&layers, vectors, targets)?;
        Ok(TrainingResult {
            model: build(&layers)?,
            epochs,
            best_epoch,
            training_loss,
            validation_loss: validation.map(|_| best_loss),
        })
    }
}

#[cfg(test)]
mod gradient_tests;
