//! BERT-family embeddings and post-norm layers on the device, using the
//! same workspaces and CPU operation order as the ModernBERT path.

use turbo::status::Result;

use super::super::{ATTN_NORM, EMB_NORM, FFN_IN, FFN_NORM, FFN_OUT, OUT, QKV, WORD, slot};
use super::pass::{Bufs, Layer};
use super::{Buf, CudaEncoder, check, ffi};

fn norm_forward(e: &CudaEncoder, s: usize, x: &Buf, n: usize, y: &Buf, c: &(Buf, Buf)) -> Result<()> {
    let h = e.geo.hidden;
    check(unsafe {
        ffi::tt_layernorm(x.f32(), e.weights[s].f32(), n as i32, h as i32, e.geo.eps, y.f32(), c.0.f64(), c.1.f64())
    })?;
    let bias = unsafe { e.weights[s].f32().add(h) };
    check(unsafe { ffi::tt_bias(y.f32(), bias, n as i32, h as i32) })?;
    e.store(y, n * h)
}

fn norm_backward(e: &CudaEncoder, s: usize, x: &Buf, n: usize, dy: &Buf, c: &(Buf, Buf), dx: &Buf) -> Result<()> {
    e.layernorm_back(x, s, n, dy, c, dx)?;
    if !e.frozen {
        let db = unsafe { e.grads[s].f32().add(e.geo.hidden) };
        check(unsafe { ffi::tt_bias_back(dy.f32(), n as i32, e.geo.hidden as i32, db) })?;
    }
    Ok(())
}

pub(super) fn embedding_forward(e: &CudaEncoder, n: usize, w: &Bufs) -> Result<()> {
    let (h, vocab) = (e.geo.hidden, e.vocab);
    check(unsafe { ffi::tt_gather(e.weights[WORD].f32(), w.ids.ptr.cast(), n as i32, h as i32, w.emb_in.f32()) })?;
    let positions: Vec<i32> = (0..n).map(|t| (e.position_offset + t) as i32).collect();
    w.out.upload(&positions)?;
    let pos = unsafe { e.weights[WORD].f32().add(vocab * h) };
    check(unsafe { ffi::tt_gather(pos, w.out.ptr.cast(), n as i32, h as i32, w.x0.f32()) })?;
    check(unsafe { ffi::tt_axpy(w.emb_in.f32(), w.x0.f32(), n * h, 1.0) })?;
    w.out.upload(&vec![0i32; n])?;
    let types = unsafe { e.weights[WORD].f32().add((vocab + e.position_rows) * h) };
    check(unsafe { ffi::tt_gather(types, w.out.ptr.cast(), n as i32, h as i32, w.x0.f32()) })?;
    check(unsafe { ffi::tt_axpy(w.emb_in.f32(), w.x0.f32(), n * h, 1.0) })?;
    norm_forward(e, EMB_NORM, &w.emb_in, n, &w.x0, &w.emb)
}

pub(super) fn embedding_backward(e: &CudaEncoder, n: usize, dx: &Buf, w: &Bufs) -> Result<()> {
    let (h, vocab) = (e.geo.hidden, e.vocab);
    w.demb.zero(n * h)?;
    norm_backward(e, EMB_NORM, &w.emb_in, n, dx, &w.emb, &w.demb)?;
    e.store(&w.demb, n * h)?;
    check(unsafe { ffi::tt_scatter_add(w.demb.f32(), w.ids.ptr.cast(), n as i32, h as i32, e.grads[WORD].f32()) })?;
    let positions: Vec<i32> = (0..n).map(|t| (e.position_offset + t) as i32).collect();
    w.emb_in.upload(&positions)?;
    let pos = unsafe { e.grads[WORD].f32().add(vocab * h) };
    check(unsafe { ffi::tt_scatter_add(w.demb.f32(), w.emb_in.ptr.cast(), n as i32, h as i32, pos) })?;
    w.emb_in.upload(&vec![0i32; n])?;
    let types = unsafe { e.grads[WORD].f32().add((vocab + e.position_rows) * h) };
    check(unsafe { ffi::tt_scatter_add(w.demb.f32(), w.emb_in.ptr.cast(), n as i32, h as i32, types) })?;
    Ok(())
}

pub(super) fn layer_forward(e: &CudaEncoder, l: usize, x: &Buf, lw: &Layer, x_out: &Buf, w: &Bufs) -> Result<()> {
    let (n, h, i) = (e.work.n, e.geo.hidden, e.geo.intermediate);
    let (nn, hh, heads) = (n as i32, h as i32, e.geo.heads as i32);
    let round = (e.compute == super::Compute::Bf16) as i32;
    e.project(l, QKV, n, x, &lw.qkv, &lw.h[0])?;
    let win = e.window(l);
    check(unsafe { ffi::tt_attention_probs(lw.qkv.f32(), nn, hh, heads, win, round, w.p.f32()) })?;
    check(unsafe { ffi::tt_attention_ctx(lw.qkv.f32(), w.p.f32(), nn, hh, heads, win, lw.ctx.f32()) })?;
    e.store(&lw.ctx, n * h)?;
    e.project(l, OUT, n, &lw.ctx, &lw.x_mid, &lw.h[1])?;
    if let Some(key) = e.mask(l as u32, 1) {
        e.drop(&lw.x_mid, n * h, e.dropout.map_or(0.0, |d| d.attention), key)?;
    }
    check(unsafe { ffi::tt_axpy(lw.x_mid.f32(), x.f32(), n * h, 1.0) })?;
    e.store(&lw.x_mid, n * h)?;
    norm_forward(e, slot(l, ATTN_NORM), &lw.x_mid, n, &lw.xn1, &lw.n1)?;

    e.project(l, FFN_IN, n, &lw.xn1, &lw.pre, &lw.h[2])?;
    check(unsafe { ffi::tt_gelu(lw.pre.f32(), nn, i as i32, lw.act.f32()) })?;
    e.store(&lw.act, n * i)?;
    if let Some(key) = e.mask(l as u32, 2) {
        e.drop(&lw.act, n * i, e.dropout.map_or(0.0, |d| d.mlp), key)?;
    }
    e.project(l, FFN_OUT, n, &lw.act, &lw.xn2, &lw.h[3])?;
    check(unsafe { ffi::tt_axpy(lw.xn2.f32(), lw.xn1.f32(), n * h, 1.0) })?;
    e.store(&lw.xn2, n * h)?;
    norm_forward(e, slot(l, FFN_NORM), &lw.xn2, n, x_out, &lw.n2)
}

pub(super) fn layer_backward(e: &CudaEncoder, l: usize, cur: usize, x_in: &Buf, lw: &Layer, w: &Bufs) -> Result<usize> {
    let (n, h, i) = (e.work.n, e.geo.hidden, e.geo.intermediate);
    let (nn, hh, heads) = (n as i32, h as i32, e.geo.heads as i32);
    let round = (e.compute == super::Compute::Bf16) as i32;
    let dx_out = &w.dx[cur];
    let dx_next = &w.dx[1 - cur];
    dx_next.zero(n * h)?;
    norm_backward(e, slot(l, FFN_NORM), &lw.xn2, n, dx_out, &lw.n2, dx_next)?;
    e.store(dx_next, n * h)?;
    w.dact.zero(n * i)?;
    e.project_back(l, FFN_OUT, n, &lw.act, &lw.h[3], dx_next, &w.dact, &w.dh)?;
    if let Some(key) = e.mask(l as u32, 2) {
        e.drop(&w.dact, n * i, e.dropout.map_or(0.0, |d| d.mlp), key)?;
    }
    w.dpre.zero(n * i)?;
    check(unsafe { ffi::tt_gelu_back(lw.pre.f32(), w.dact.f32(), nn, i as i32, w.dpre.f32()) })?;
    e.store(&w.dpre, n * i)?;
    w.dxn2.zero(n * h)?;
    e.project_back(l, FFN_IN, n, &lw.xn1, &lw.h[2], &w.dpre, &w.dxn2, &w.dh)?;
    check(unsafe { ffi::tt_axpy(w.dxn2.f32(), dx_next.f32(), n * h, 1.0) })?;
    e.store(&w.dxn2, n * h)?;

    w.dsrc.zero(n * h)?;
    norm_backward(e, slot(l, ATTN_NORM), &lw.x_mid, n, &w.dxn2, &lw.n1, &w.dsrc)?;
    e.store(&w.dsrc, n * h)?;
    let dproj = match e.mask(l as u32, 1) {
        Some(key) => {
            w.dproj.copy_from(&w.dsrc, n * h)?;
            e.drop(&w.dproj, n * h, e.dropout.map_or(0.0, |d| d.attention), key)?;
            &w.dproj
        }
        None => &w.dsrc,
    };
    w.dctx.zero(n * h)?;
    e.project_back(l, OUT, n, &lw.ctx, &lw.h[1], dproj, &w.dctx, &w.dh)?;
    let win = e.window(l);
    let (qkv, p, ds, dctx, dqkv) = (lw.qkv.f32(), w.p.f32(), w.ds.f32(), w.dctx.f32(), w.dqkv.f32());
    check(unsafe { ffi::tt_attention_probs(qkv, nn, hh, heads, win, round, p) })?;
    check(unsafe { ffi::tt_attention_ds(qkv, p, dctx, nn, hh, heads, win, ds) })?;
    w.dqkv.zero(n * 3 * h)?;
    check(unsafe { ffi::tt_attention_dq(qkv, ds, nn, hh, heads, win, dqkv) })?;
    check(unsafe { ffi::tt_attention_dkv(qkv, p, ds, dctx, nn, hh, heads, win, dqkv) })?;
    e.store(&w.dqkv, n * 3 * h)?;
    w.dxn2.zero(n * h)?;
    e.project_back(l, QKV, n, x_in, &lw.h[0], &w.dqkv, &w.dxn2, &w.dh)?;
    check(unsafe { ffi::tt_axpy(w.dxn2.f32(), w.dsrc.f32(), n * h, 1.0) })?;
    e.store(&w.dxn2, n * h)?;
    dx_next.copy_from(&w.dxn2, n * h)?;
    Ok(1 - cur)
}
