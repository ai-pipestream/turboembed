//! The optional BiLSTM adapter every word head runs over its features,
//! forward and back, with the weights at the front of the head's tensor
//! list (`lstm.{k}.{fwd,bwd}.{w,u,b}`, six a layer), as the core reads
//! them (`turbo::head` words.rs).

use crate::lstm;
use crate::real::Real;

/// The adapter's shape: layers and state width over features `hidden` wide.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub h: usize,
}

impl Shape {
    /// The width of the word vectors the adapter gives.
    pub fn width(&self) -> usize {
        if self.layers == 0 { self.hidden } else { 2 * self.h }
    }
}

/// Each layer's output and LSTM caches, kept for backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    layers: Vec<Vec<R>>,
    caches: Vec<lstm::Cache<R>>,
}

fn dir<R>(params: &[Vec<R>], k: usize, d: usize) -> lstm::Dir<'_, R> {
    let i = 6 * k + 3 * d;
    lstm::Dir { w: &params[i], u: &params[i + 1], b: &params[i + 2], reverse: d == 1 }
}

/// Runs the adapter over features [n, hidden]; the word vectors are
/// [`output`]`(features, w)`.
pub fn forward<R: Real>(s: Shape, params: &[Vec<R>], features: &[R], w: &mut Work<R>) {
    let n = features.len() / s.hidden;
    w.layers.resize_with(s.layers, Vec::new);
    w.caches.resize_with(2 * s.layers, lstm::Cache::default);
    let mut width = s.hidden;
    for k in 0..s.layers {
        let mut out = std::mem::take(&mut w.layers[k]);
        out.clear();
        out.resize(n * 2 * s.h, R::ZERO);
        let x = if k == 0 { features } else { &w.layers[k - 1] };
        for d in 0..2 {
            lstm::forward(dir(params, k, d), x, width, s.h, &mut out, 2 * s.h, d * s.h, &mut w.caches[2 * k + d]);
        }
        w.layers[k] = out;
        width = 2 * s.h;
    }
}

/// The word vectors of the last forward: the features themselves without
/// an adapter.
pub fn output<'a, R>(features: &'a [R], w: &'a Work<R>) -> &'a [R] {
    w.layers.last().map_or(features, Vec::as_slice)
}

/// Adds the adapter's weight gradients to `grads` from `dx`, the
/// gradient of its output, and returns the gradient of its input (`dx`
/// itself without an adapter) when `input` asks for it, which a head's
/// hashed embeddings take; the encoder is frozen, so otherwise the first
/// layer's input gradient is skipped and an empty vector returned.
pub fn backward<R: Real>(
    s: Shape,
    params: &[Vec<R>],
    features: &[R],
    mut dx: Vec<R>,
    grads: &mut [Vec<R>],
    w: &Work<R>,
    input: bool,
) -> Vec<R> {
    let n = features.len() / s.hidden;
    for k in (0..s.layers).rev() {
        let inp = if k == 0 { s.hidden } else { 2 * s.h };
        let x = if k == 0 { features } else { &w.layers[k - 1] };
        let want = k > 0 || input;
        let mut dprev = if want { vec![R::ZERO; n * inp] } else { Vec::new() };
        for d in 0..2 {
            let i = 6 * k + 3 * d;
            let [gw, gu, gb] = grads.get_disjoint_mut([i, i + 1, i + 2]).expect("distinct");
            let g = lstm::Grad { w: gw, u: gu, b: gb };
            let into = want.then_some(&mut dprev[..]);
            lstm::backward(dir(params, k, d), x, inp, s.h, &dx, 2 * s.h, d * s.h, &w.caches[2 * k + d], g, into);
        }
        dx = dprev;
    }
    dx
}

/// Adds `local`, one example's complete gradients (the adapter's
/// included), into `grads`, or refuses and leaves `grads` as they were
/// when a local gradient or a sum it would store is not finite.
pub fn commit<R: Real>(grads: &mut [Vec<R>], local: &[Vec<R>], what: &str) -> turbo::status::Result<()> {
    let finite = grads.iter().zip(local).all(|(g, l)| g.iter().zip(l).all(|(&a, &b)| (a + b).to_f64().is_finite()))
        && local.iter().flatten().all(|v| v.to_f64().is_finite());
    if !finite {
        return Err(turbo::status::invalid(format!("trainer: a {what} gradient or its sum is not finite")));
    }
    for (g, l) in grads.iter_mut().zip(local) {
        g.iter_mut().zip(l).for_each(|(a, &b)| *a += b);
    }
    Ok(())
}
