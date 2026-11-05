//! A BERT/RoBERTa layer: attention over absolute-position embeddings,
//! post-norm residuals, biased projections and an ungated erf GELU MLP.

use crate::real::Real;

use super::layer::Work;
use super::{ATTN_NORM, Compute, Encoder, FFN_IN, FFN_NORM, FFN_OUT, OUT, QKV, attention, dropout, math, norm};

pub(super) fn forward<R: Real>(e: &Encoder<R>, l: usize, at: Option<u64>, x: &[R], w: &mut Work<R>) {
    let (h, i) = (e.geo.hidden, e.geo.intermediate);
    let n = x.len() / h;
    w.x_in.clear();
    w.x_in.extend_from_slice(x);
    let [h0, h1, h2, h3] = &mut w.h;
    w.qkv.resize(n * 3 * h, R::ZERO);
    e.project(l, QKV, x, &mut w.qkv, h0);
    w.ctx.resize(n * h, R::ZERO);
    attention::forward(&w.qkv, h, e.geo.heads, e.window(l), e.compute == Compute::Bf16, &mut w.ctx, &mut w.attn);
    e.store(&mut w.ctx);
    w.x_mid.resize(n * h, R::ZERO);
    e.project(l, OUT, &w.ctx, &mut w.x_mid, h1);
    if let Some((d, key)) = e.mask(at, l as u32, 1) {
        dropout::forward(&mut w.x_mid, d.attention, key);
        e.store(&mut w.x_mid);
    }
    for (m, &x) in w.x_mid.iter_mut().zip(x) {
        *m += x;
    }
    e.store(&mut w.x_mid);
    let (nw, nb) = e.norm_parts(e.which_slot(l, ATTN_NORM));
    w.xn1.resize(n * h, R::ZERO);
    norm::forward(&w.x_mid, nw, nb, e.geo.eps, &mut w.xn1, &mut w.n1);
    e.store(&mut w.xn1);

    w.pre.resize(n * i, R::ZERO);
    e.project(l, FFN_IN, &w.xn1, &mut w.pre, h2);
    w.act.resize(n * i, R::ZERO);
    for (a, &p) in w.act.iter_mut().zip(&w.pre) {
        *a = math::gelu(p);
    }
    e.store(&mut w.act);
    if let Some((d, key)) = e.mask(at, l as u32, 2) {
        dropout::forward(&mut w.act, d.mlp, key);
        e.store(&mut w.act);
    }
    w.xn2.resize(n * h, R::ZERO);
    e.project(l, FFN_OUT, &w.act, &mut w.xn2, h3);
    for (o, &x) in w.xn2.iter_mut().zip(&w.xn1) {
        *o += x;
    }
    e.store(&mut w.xn2);
    let (nw, nb) = e.norm_parts(e.which_slot(l, FFN_NORM));
    w.x_out.resize(n * h, R::ZERO);
    norm::forward(&w.xn2, nw, nb, e.geo.eps, &mut w.x_out, &mut w.n2);
    e.store(&mut w.x_out);
}

pub(super) fn backward<R: Real>(
    e: &Encoder<R>,
    l: usize,
    at: Option<u64>,
    w: &Work<R>,
    dx_out: &[R],
    g: &mut [Vec<R>],
) -> Vec<R> {
    let (h, i) = (e.geo.hidden, e.geo.intermediate);
    let n = dx_out.len() / h;
    let mut dxn2 = vec![R::ZERO; n * h];
    norm_back(e, e.which_slot(l, FFN_NORM), &w.xn2, &w.n2, dx_out, &mut dxn2, g);
    e.store(&mut dxn2);

    let mut dact = vec![R::ZERO; n * i];
    e.project_back(l, FFN_OUT, &w.act, &w.h[3], &dxn2, g, &mut dact);
    if let Some((d, key)) = e.mask(at, l as u32, 2) {
        dropout::backward(&mut dact, d.mlp, key);
        e.store(&mut dact);
    }
    let mut dpre = vec![R::ZERO; n * i];
    for ((dp, &da), &p) in dpre.iter_mut().zip(&dact).zip(&w.pre) {
        *dp = da * math::gelu_grad(p);
    }
    e.store(&mut dpre);
    let mut dxn1 = vec![R::ZERO; n * h];
    e.project_back(l, FFN_IN, &w.xn1, &w.h[2], &dpre, g, &mut dxn1);
    for (d, &r) in dxn1.iter_mut().zip(&dxn2) {
        *d += r;
    }
    e.store(&mut dxn1);

    let mut dx_mid = vec![R::ZERO; n * h];
    norm_back(e, e.which_slot(l, ATTN_NORM), &w.x_mid, &w.n1, &dxn1, &mut dx_mid, g);
    e.store(&mut dx_mid);
    let mut dproj = dx_mid.clone();
    if let Some((d, key)) = e.mask(at, l as u32, 1) {
        dropout::backward(&mut dproj, d.attention, key);
        e.store(&mut dproj);
    }
    let mut dctx = vec![R::ZERO; n * h];
    e.project_back(l, OUT, &w.ctx, &w.h[1], &dproj, g, &mut dctx);
    let mut dqkv = vec![R::ZERO; n * 3 * h];
    attention::backward(&w.qkv, h, e.geo.heads, e.window(l), e.compute == Compute::Bf16, &w.attn, &dctx, &mut dqkv);
    e.store(&mut dqkv);
    let mut dx = vec![R::ZERO; n * h];
    e.project_back(l, QKV, &w.x_in, &w.h[0], &dqkv, g, &mut dx);
    for (d, &r) in dx.iter_mut().zip(&dx_mid) {
        *d += r;
    }
    e.store(&mut dx);
    dx
}

fn norm_back<R: Real>(
    e: &Encoder<R>,
    s: usize,
    x: &[R],
    cache: &norm::Cache,
    dy: &[R],
    dx: &mut [R],
    g: &mut [Vec<R>],
) {
    let (w, _) = e.norm_parts(s);
    if e.frozen {
        norm::backward(x, w, cache, dy, dx, None, None);
    } else {
        let (gw, gb) = g[s].split_at_mut(e.geo.hidden);
        norm::backward(x, w, cache, dy, dx, Some(gw), Some(gb));
    }
}
