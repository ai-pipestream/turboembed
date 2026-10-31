//! One sequence forward and backward on the device, in the CPU
//! encoder's order (`pass.rs`, `layer.rs`), over a workspace that grows
//! to the longest sequence seen.

use std::time::Instant;

use turbo::status::{Result, invalid};

use super::super::layer::PROJECTIONS;
use super::super::{ATTN_NORM, Compute, EMB_NORM, FFN_IN, FFN_NORM, FFN_OUT, FINAL_NORM, OUT, QKV, WORD, slot};
use super::super::{Geometry, Lora, dropout};
use super::{Buf, CudaEncoder, check, ffi};

/// What one layer's forward keeps for its backward.
pub(super) struct Layer {
    xn1: Buf,
    qkv: Buf,
    ctx: Buf,
    x_mid: Buf,
    xn2: Buf,
    pre: Buf,
    act: Buf,
    x_out: Buf,
    /// Each adapter's low-rank product, in `PROJECTIONS` order.
    h: [Buf; 4],
    n1: (Buf, Buf),
    n2: (Buf, Buf),
}

/// The buffers of a forward and its backward, for `cap` tokens.
#[derive(Default)]
pub(super) struct Workspace {
    cap: usize,
    /// The last forward's length and place.
    n: usize,
    at: Option<u64>,
    bufs: Option<Bufs>,
}

pub(super) struct Bufs {
    ids: Buf,
    emb_in: Buf,
    x0: Buf,
    out: Buf,
    emb: (Buf, Buf),
    fin: (Buf, Buf),
    layers: Vec<Layer>,
    /// The probabilities and the score gradients, [heads, n, n].
    p: Buf,
    ds: Buf,
    /// The residual stream's gradient, two to alternate between layers.
    dx: [Buf; 2],
    dy: Buf,
    dsrc: Buf,
    dqkv: Buf,
    dctx: Buf,
    dproj: Buf,
    dxn2: Buf,
    dpre: Buf,
    dact: Buf,
    demb: Buf,
    dh: Buf,
}

/// Seconds a forward and a backward took, launch to completion.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timing {
    pub forward: f64,
    pub backward: f64,
}

impl Workspace {
    /// Room for `n` tokens, kept when there is.
    pub(super) fn ensure(&mut self, geo: &Geometry, lora: Option<Lora>, n: usize) -> Result<()> {
        if n <= self.cap && self.bufs.is_some() {
            return Ok(());
        }
        let (h, i, rank) = (geo.hidden, geo.intermediate, lora.map_or(0, |l| l.rank));
        let stats = || Ok::<_, turbo::status::Error>((Buf::doubles(n)?, Buf::doubles(n)?));
        let mut layers = Vec::with_capacity(geo.layers);
        for _ in 0..geo.layers {
            layers.push(Layer {
                xn1: Buf::floats(n * h)?,
                qkv: Buf::floats(n * 3 * h)?,
                ctx: Buf::floats(n * h)?,
                x_mid: Buf::floats(n * h)?,
                xn2: Buf::floats(n * h)?,
                pre: Buf::floats(n * 2 * i)?,
                act: Buf::floats(n * i)?,
                x_out: Buf::floats(n * h)?,
                h: [Buf::floats(n * rank)?, Buf::floats(n * rank)?, Buf::floats(n * rank)?, Buf::floats(n * rank)?],
                n1: stats()?,
                n2: stats()?,
            });
        }
        self.bufs = Some(Bufs {
            ids: Buf::new(n * 4)?,
            emb_in: Buf::floats(n * h)?,
            x0: Buf::floats(n * h)?,
            out: Buf::floats(n * h)?,
            emb: stats()?,
            fin: stats()?,
            layers,
            p: Buf::floats(geo.heads * n * n)?,
            ds: Buf::floats(geo.heads * n * n)?,
            dx: [Buf::floats(n * h)?, Buf::floats(n * h)?],
            dy: Buf::floats(n * h)?,
            dsrc: Buf::floats(n * h)?,
            dqkv: Buf::floats(n * 3 * h)?,
            dctx: Buf::floats(n * h)?,
            dproj: Buf::floats(n * h)?,
            dxn2: Buf::floats(n * h)?,
            dpre: Buf::floats(n * 2 * i)?,
            dact: Buf::floats(n * i)?,
            demb: Buf::floats(n * h)?,
            dh: Buf::floats(n * rank)?,
        });
        self.cap = n;
        Ok(())
    }
}

impl CudaEncoder {
    /// The states [n, hidden] of `ids` into `out`, with no dropout.
    pub fn forward(&mut self, ids: &[u32], out: &mut Vec<f32>) -> Result<Timing> {
        self.run(ids, None, out)
    }

    /// As [`forward`](Self::forward), training: dropout, when the encoder
    /// has it, drops under the masks keyed by `at`, as the CPU's does.
    pub fn forward_at(&mut self, ids: &[u32], at: u64, out: &mut Vec<f32>) -> Result<Timing> {
        self.run(ids, Some(at), out)
    }

    fn run(&mut self, ids: &[u32], at: Option<u64>, out: &mut Vec<f32>) -> Result<Timing> {
        self.check_ids(ids)?;
        let n = ids.len();
        let h = self.geo.hidden;
        self.work.ensure(&self.geo, self.lora, n)?;
        self.work.n = n;
        self.work.at = at;
        let started = Instant::now();
        let w = self.work.bufs.as_ref().expect("ensured");
        let ids32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        w.ids.upload(&ids32)?;
        let (nn, hh) = (n as i32, h as i32);
        check(unsafe { ffi::tt_gather(self.weights[WORD].f32(), w.ids.ptr.cast(), nn, hh, w.emb_in.f32()) })?;
        self.layernorm(&w.emb_in, EMB_NORM, n, &w.x0, &w.emb)?;
        if let Some(key) = self.mask(u32::MAX, 0) {
            self.drop(&w.x0, n * h, self.dropout.map_or(0.0, |d| d.embedding), key)?;
        }
        for l in 0..self.geo.layers {
            let x = if l == 0 { &w.x0 } else { &w.layers[l - 1].x_out };
            self.layer_forward(l, x, &w.layers[l], w)?;
        }
        let last = w.layers.last().map_or(&w.x0, |l| &l.x_out);
        self.layernorm(last, FINAL_NORM, n, &w.out, &w.fin)?;
        self.sync()?;
        out.resize(n * h, 0.0);
        w.out.download(out)?;
        Ok(Timing { forward: started.elapsed().as_secs_f64(), backward: 0.0 })
    }

    /// Adds to the device's gradients those of the weights from `dout`,
    /// the gradient of the last forward's states.
    pub fn backward(&mut self, dout: &[f32]) -> Result<Timing> {
        let (n, h) = (self.work.n, self.geo.hidden);
        if n == 0 || dout.len() != n * h {
            return Err(invalid("encoder: the gradient is not shaped as the last forward's states"));
        }
        let started = Instant::now();
        let w = self.work.bufs.as_ref().expect("a forward ran");
        w.dy.upload(dout)?;
        self.store(&w.dy, n * h)?;
        let last = w.layers.last().map_or(&w.x0, |l| &l.x_out);
        w.dx[0].zero(n * h)?;
        self.layernorm_back(last, FINAL_NORM, n, &w.dy, &w.fin, &w.dx[0])?;
        let mut cur = 0;
        for l in (0..self.geo.layers).rev() {
            cur = self.layer_backward(l, cur, w)?;
        }
        if !self.frozen {
            if let Some(key) = self.mask(u32::MAX, 0) {
                self.drop(&w.dx[cur], n * h, self.dropout.map_or(0.0, |d| d.embedding), key)?;
            }
            w.demb.zero(n * h)?;
            self.layernorm_back(&w.emb_in, EMB_NORM, n, &w.dx[cur], &w.emb, &w.demb)?;
            let (nn, hh) = (n as i32, h as i32);
            check(unsafe { ffi::tt_scatter_add(w.demb.f32(), w.ids.ptr.cast(), nn, hh, self.grads[WORD].f32()) })?;
        }
        self.sync()?;
        Ok(Timing { forward: 0.0, backward: started.elapsed().as_secs_f64() })
    }

    /// The mask key of branch `branch` of layer `l` in the last forward,
    /// when it trained with dropout.
    fn mask(&self, l: u32, branch: u32) -> Option<u64> {
        let d = self.dropout?;
        let at = self.work.at?;
        Some(dropout::key(d.seed, at, l, branch))
    }

    fn drop(&self, b: &Buf, count: usize, p: f64, key: u64) -> Result<()> {
        check(unsafe { ffi::tt_dropout(b.f32(), count, p, key) })?;
        self.store(b, count)
    }

    /// y = norm(x) over slot `s`'s weight, the statistics into `c`.
    fn layernorm(&self, x: &Buf, s: usize, n: usize, y: &Buf, c: &(Buf, Buf)) -> Result<()> {
        let h = self.geo.hidden;
        let w = self.weights[s].f32();
        check(unsafe {
            ffi::tt_layernorm(x.f32(), w, n as i32, h as i32, self.geo.eps, y.f32(), c.0.f64(), c.1.f64())
        })?;
        self.store(y, n * h)
    }

    /// dx += the gradient of `layernorm` from dy; the weight's gradient
    /// unless the base is frozen.
    fn layernorm_back(&self, x: &Buf, s: usize, n: usize, dy: &Buf, c: &(Buf, Buf), dx: &Buf) -> Result<()> {
        let h = self.geo.hidden;
        let dw = if self.frozen { std::ptr::null_mut() } else { self.grads[s].f32() };
        let w = self.weights[s].f32();
        let (mean, rstd) = (c.0.f64(), c.1.f64());
        check(unsafe { ffi::tt_layernorm_back(x.f32(), w, mean, rstd, dy.f32(), n as i32, h as i32, dx.f32(), dw) })?;
        self.store(dx, n * h)
    }

    /// c = alpha op(a) op(b) + beta c, with the shapes as `tt_gemm` takes.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        t: (bool, bool),
        ab: (f32, f32),
    ) -> Result<()> {
        let (m, n, k) = (m as i32, n as i32, k as i32);
        check(unsafe { ffi::tt_gemm(a, b, c, m, n, k, t.0 as i32, t.1 as i32, ab.0, ab.1) })
    }

    /// The width of the input and the output of projection `which`.
    fn widths(&self, which: usize) -> (usize, usize) {
        let (h, i) = (self.geo.hidden, self.geo.intermediate);
        match which {
            QKV => (h, 3 * h),
            OUT => (h, h),
            FFN_IN => (h, 2 * i),
            _ => (i, h),
        }
    }

    /// The slots of adapter A and B of projection `which` of layer `l`.
    fn lora_slots(&self, l: usize, which: usize) -> (usize, usize) {
        let p = PROJECTIONS.iter().position(|&w| w == which).expect("a projection");
        let a = 3 + 6 * self.geo.layers + 8 * (l - self.lora_from) + 2 * p;
        (a, a + 1)
    }

    /// y [n, out] = x W^T (plus the adapter) for projection `which` of layer `l`.
    fn project(&self, l: usize, which: usize, n: usize, x: &Buf, y: &Buf, hb: &Buf) -> Result<()> {
        let (inp, out) = self.widths(which);
        self.gemm(x.f32(), self.weights[slot(l, which)].f32(), y.f32(), n, out, inp, (false, true), (1.0, 0.0))?;
        self.store(y, n * out)?;
        if let Some(a) = self.adapter(l) {
            let (sa, sb) = self.lora_slots(l, which);
            let s = (a.alpha / a.rank as f64) as f32;
            self.gemm(x.f32(), self.weights[sa].f32(), hb.f32(), n, a.rank, inp, (false, true), (1.0, 0.0))?;
            self.gemm(hb.f32(), self.weights[sb].f32(), y.f32(), n, out, a.rank, (false, true), (s, 1.0))?;
            self.store(y, n * out)?;
            self.store(hb, n * a.rank)?;
        }
        Ok(())
    }

    /// The gradients of `project` from dy: the weight's unless the base is
    /// frozen, the adapter's, and the input's into dx (zero on entry).
    #[allow(clippy::too_many_arguments)]
    fn project_back(
        &self,
        l: usize,
        which: usize,
        n: usize,
        x: &Buf,
        hb: &Buf,
        dy: &Buf,
        dx: &Buf,
        dh: &Buf,
    ) -> Result<()> {
        let (inp, out) = self.widths(which);
        let w = self.weights[slot(l, which)].f32();
        if !self.frozen {
            let g = self.grads[slot(l, which)].f32();
            self.gemm(dy.f32(), x.f32(), g, out, inp, n, (true, false), (1.0, 1.0))?;
        }
        self.gemm(dy.f32(), w, dx.f32(), n, inp, out, (false, false), (1.0, 1.0))?;
        if let Some(a) = self.adapter(l) {
            let (sa, sb) = self.lora_slots(l, which);
            let s = (a.alpha / a.rank as f64) as f32;
            let (wa, wb) = (self.weights[sa].f32(), self.weights[sb].f32());
            let (ga, gb) = (self.grads[sa].f32(), self.grads[sb].f32());
            self.gemm(dy.f32(), wb, dh.f32(), n, a.rank, out, (false, false), (s, 0.0))?;
            self.gemm(dy.f32(), hb.f32(), gb, out, a.rank, n, (true, false), (s, 1.0))?;
            self.gemm(dh.f32(), x.f32(), ga, a.rank, inp, n, (true, false), (1.0, 1.0))?;
            self.gemm(dh.f32(), wa, dx.f32(), n, inp, a.rank, (false, false), (1.0, 1.0))?;
        }
        self.store(dx, n * inp)
    }

    /// The rotary table's cos and sin for layer `l`.
    fn rope(&self, l: usize) -> (*const f32, *const f32) {
        let half = self.positions * self.pairs;
        let base = if self.geo.global(l) { 0 } else { 2 * half };
        unsafe { (self.table.f32().add(base), self.table.f32().add(base + half)) }
    }

    fn layer_forward(&self, l: usize, x: &Buf, lw: &Layer, w: &Bufs) -> Result<()> {
        let (n, h, i) = (self.work.n, self.geo.hidden, self.geo.intermediate);
        let (nn, hh, heads) = (n as i32, h as i32, self.geo.heads as i32);
        let round = (self.compute == Compute::Bf16) as i32;
        let src = if l == 0 {
            x
        } else {
            self.layernorm(x, slot(l, ATTN_NORM), n, &lw.xn1, &lw.n1)?;
            &lw.xn1
        };
        self.project(l, QKV, n, src, &lw.qkv, &lw.h[0])?;
        let (cs, sn) = self.rope(l);
        let pairs = self.pairs as i32;
        check(unsafe { ffi::tt_rope(lw.qkv.f32(), nn, 3 * hh, 2 * hh, pairs, cs, sn, 0) })?;
        self.store(&lw.qkv, n * 3 * h)?;
        let win = self.window(l);
        check(unsafe { ffi::tt_attention_probs(lw.qkv.f32(), nn, hh, heads, win, round, w.p.f32()) })?;
        check(unsafe { ffi::tt_attention_ctx(lw.qkv.f32(), w.p.f32(), nn, hh, heads, win, lw.ctx.f32()) })?;
        self.store(&lw.ctx, n * h)?;
        self.project(l, OUT, n, &lw.ctx, &lw.x_mid, &lw.h[1])?;
        if let Some(key) = self.mask(l as u32, 1) {
            self.drop(&lw.x_mid, n * h, self.dropout.map_or(0.0, |d| d.attention), key)?;
        }
        check(unsafe { ffi::tt_axpy(lw.x_mid.f32(), x.f32(), n * h, 1.0) })?;
        self.store(&lw.x_mid, n * h)?;
        self.layernorm(&lw.x_mid, slot(l, FFN_NORM), n, &lw.xn2, &lw.n2)?;
        self.project(l, FFN_IN, n, &lw.xn2, &lw.pre, &lw.h[2])?;
        check(unsafe { ffi::tt_geglu(lw.pre.f32(), nn, i as i32, lw.act.f32()) })?;
        self.store(&lw.act, n * i)?;
        if let Some(key) = self.mask(l as u32, 2) {
            self.drop(&lw.act, n * i, self.dropout.map_or(0.0, |d| d.mlp), key)?;
        }
        self.project(l, FFN_OUT, n, &lw.act, &lw.x_out, &lw.h[3])?;
        check(unsafe { ffi::tt_axpy(lw.x_out.f32(), lw.x_mid.f32(), n * h, 1.0) })?;
        self.store(&lw.x_out, n * h)
    }

    /// The gradient of layer `l`'s input from that of its output in
    /// `w.dx[cur]`, into the other buffer, whose index it returns.
    fn layer_backward(&self, l: usize, cur: usize, w: &Bufs) -> Result<usize> {
        let (n, h, i) = (self.work.n, self.geo.hidden, self.geo.intermediate);
        let (nn, hh, heads) = (n as i32, h as i32, self.geo.heads as i32);
        let round = (self.compute == Compute::Bf16) as i32;
        let lw = &w.layers[l];
        let dx_out = &w.dx[cur];
        let dx_mid = &w.dx[1 - cur];
        w.dact.zero(n * i)?;
        self.project_back(l, FFN_OUT, n, &lw.act, &lw.h[3], dx_out, &w.dact, &w.dh)?;
        if let Some(key) = self.mask(l as u32, 2) {
            self.drop(&w.dact, n * i, self.dropout.map_or(0.0, |d| d.mlp), key)?;
        }
        w.dpre.zero(n * 2 * i)?;
        check(unsafe { ffi::tt_geglu_back(lw.pre.f32(), w.dact.f32(), nn, i as i32, w.dpre.f32()) })?;
        self.store(&w.dpre, n * 2 * i)?;
        w.dxn2.zero(n * h)?;
        self.project_back(l, FFN_IN, n, &lw.xn2, &lw.h[2], &w.dpre, &w.dxn2, &w.dh)?;
        dx_mid.copy_from(dx_out, n * h)?;
        self.layernorm_back(&lw.x_mid, slot(l, FFN_NORM), n, &w.dxn2, &lw.n2, dx_mid)?;
        let dproj = match self.mask(l as u32, 1) {
            Some(key) => {
                w.dproj.copy_from(dx_mid, n * h)?;
                self.drop(&w.dproj, n * h, self.dropout.map_or(0.0, |d| d.attention), key)?;
                &w.dproj
            }
            None => dx_mid,
        };
        w.dctx.zero(n * h)?;
        self.project_back(l, OUT, n, &lw.ctx, &lw.h[1], dproj, &w.dctx, &w.dh)?;
        let win = self.window(l);
        let (qkv, p, ds, dctx, dqkv) = (lw.qkv.f32(), w.p.f32(), w.ds.f32(), w.dctx.f32(), w.dqkv.f32());
        check(unsafe { ffi::tt_attention_probs(qkv, nn, hh, heads, win, round, p) })?;
        check(unsafe { ffi::tt_attention_ds(qkv, p, dctx, nn, hh, heads, win, ds) })?;
        w.dqkv.zero(n * 3 * h)?;
        check(unsafe { ffi::tt_attention_dq(qkv, ds, nn, hh, heads, win, dqkv) })?;
        check(unsafe { ffi::tt_attention_dkv(qkv, p, ds, dctx, nn, hh, heads, win, dqkv) })?;
        self.store(&w.dqkv, n * 3 * h)?;
        let (cs, sn) = self.rope(l);
        check(unsafe { ffi::tt_rope(dqkv, nn, 3 * hh, 2 * hh, self.pairs as i32, cs, sn, 1) })?;
        self.store(&w.dqkv, n * 3 * h)?;
        w.dsrc.zero(n * h)?;
        let x_in = if l == 0 { &w.x0 } else { &w.layers[l - 1].x_out };
        let src = if l == 0 { x_in } else { &lw.xn1 };
        self.project_back(l, QKV, n, src, &lw.h[0], &w.dqkv, &w.dsrc, &w.dh)?;
        if l == 0 {
            check(unsafe { ffi::tt_axpy(dx_mid.f32(), w.dsrc.f32(), n * h, 1.0) })?;
        } else {
            self.layernorm_back(x_in, slot(l, ATTN_NORM), n, &w.dsrc, &lw.n1, dx_mid)?;
        }
        self.store(dx_mid, n * h)?;
        Ok(1 - cur)
    }
}
