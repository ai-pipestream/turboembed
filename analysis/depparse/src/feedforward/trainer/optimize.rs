use super::*;
use std::collections::BTreeMap;
use turbo_jvm::random::JavaRandom;

pub(super) fn optimize(
    model: &mut FeedforwardDependencyModel,
    examples: &[(Vec<usize>, usize)],
    settings: &FeedforwardSettings,
) -> Result<()> {
    let w = &mut model.weights;
    let (e, h, o) = (w.embedding_size, w.hidden_bias.len(), w.output_bias.len());
    let input = model.layout.feature_count() * e;
    let mut accum = [vec![0.0; w.hidden_weights.len()], vec![0.0; h], vec![0.0; w.output_weights.len()], vec![0.0; o]];
    let mut embedding_accum = vec![0.0; w.embeddings.len()];
    let mut gradients = accum.clone();
    let mut random = JavaRandom::new(settings.seed);
    let mut order: Vec<_> = (0..examples.len()).collect();
    if order.len() > i32::MAX as usize {
        return Err(error("too many training examples"));
    }
    let keep = 1.0 - settings.dropout;
    let mut x = vec![0.0; input];
    let mut pre = vec![0.0; h];
    let mut hidden = vec![0.0; h];
    let mut mask = vec![false; h];
    let mut probabilities = vec![0.0; o];
    let mut delta_h = vec![0.0; h];
    let mut delta_x = vec![0.0; input];
    for _ in 0..settings.epochs {
        random.shuffle(&mut order);
        for batch in order.chunks(settings.batch_size) {
            for g in &mut gradients {
                g.fill(0.0);
            }
            let mut embedding_gradients = BTreeMap::<usize, Vec<f64>>::new();
            for &index in batch {
                let (ids, gold) = &examples[index];
                for (f, &id) in ids.iter().enumerate() {
                    for d in 0..e {
                        x[f * e + d] = f64::from(w.embeddings[id * e + d]);
                    }
                }
                for j in 0..h {
                    mask[j] = random.next_double() < keep;
                    if !mask[j] {
                        pre[j] = 0.0;
                        hidden[j] = 0.0;
                        continue;
                    }
                    let mut sum = f64::from(w.hidden_bias[j]);
                    for (k, &v) in x.iter().enumerate() {
                        sum += f64::from(w.hidden_weights[j * input + k]) * v;
                    }
                    pre[j] = sum;
                    hidden[j] = model.activation.apply(sum) / keep;
                }
                for (out, p) in probabilities.iter_mut().enumerate() {
                    *p = f64::from(w.output_bias[out]);
                    for (j, &v) in hidden.iter().enumerate() {
                        *p += f64::from(w.output_weights[out * h + j]) * v;
                    }
                }
                if probabilities.iter().any(|x| !x.is_finite()) {
                    return Err(error("non-finite training score"));
                }
                let max = probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut norm = 0.0;
                for p in &mut probabilities {
                    *p = (*p - max).exp();
                    norm += *p;
                }
                for p in &mut probabilities {
                    *p /= norm;
                }
                delta_h.fill(0.0);
                delta_x.fill(0.0);
                for (out, &p) in probabilities.iter().enumerate() {
                    let delta = p - if out == *gold { 1.0 } else { 0.0 };
                    gradients[3][out] += delta;
                    for j in 0..h {
                        gradients[2][out * h + j] += delta * hidden[j];
                        delta_h[j] += delta * f64::from(w.output_weights[out * h + j]);
                    }
                }
                for j in 0..h {
                    if !mask[j] {
                        continue;
                    }
                    let delta = match model.activation {
                        Activation::Cube => delta_h[j] * 3.0 * pre[j] * pre[j] / keep,
                        Activation::Relu => {
                            if pre[j] > 0.0 {
                                delta_h[j] / keep
                            } else {
                                0.0
                            }
                        }
                    };
                    gradients[1][j] += delta;
                    for k in 0..input {
                        gradients[0][j * input + k] += delta * x[k];
                        delta_x[k] += delta * f64::from(w.hidden_weights[j * input + k]);
                    }
                }
                for (f, &id) in ids.iter().enumerate() {
                    let g = embedding_gradients.entry(id).or_insert_with(|| vec![0.0; e]);
                    for d in 0..e {
                        g[d] += delta_x[f * e + d];
                    }
                }
            }
            for (i, weights) in [&mut w.hidden_weights, &mut w.hidden_bias, &mut w.output_weights, &mut w.output_bias]
                .into_iter()
                .enumerate()
            {
                update(
                    weights,
                    &gradients[i],
                    &mut accum[i],
                    batch.len(),
                    settings,
                    if i % 2 == 0 { settings.l2 } else { 0.0 },
                )?;
            }
            for (id, g) in embedding_gradients {
                update(
                    &mut w.embeddings[id * e..(id + 1) * e],
                    &g,
                    &mut embedding_accum[id * e..(id + 1) * e],
                    batch.len(),
                    settings,
                    0.0,
                )?;
            }
        }
    }
    model.check()
}
fn update(
    weights: &mut [f32],
    gradients: &[f64],
    accum: &mut [f64],
    batch: usize,
    settings: &FeedforwardSettings,
    l2: f64,
) -> Result<()> {
    for ((weight, &g), a) in weights.iter_mut().zip(gradients).zip(accum) {
        let g = g / batch as f64 + l2 * f64::from(*weight);
        *a += g * g;
        if !g.is_finite() || !a.is_finite() {
            return Err(error("non-finite AdaGrad accumulator"));
        }
        *weight = (f64::from(*weight) - settings.learning_rate * g / (a.sqrt() + 1e-6)) as f32;
        if !weight.is_finite() {
            return Err(error("non-finite trained parameter"));
        }
    }
    Ok(())
}
