//! One encoder layer, ModernBERT here and BERT/RoBERTa in `bert`, with
//! everything backward needs kept. ModernBERT computes
//! x += attn(attn_norm(x)), x += mlp(ffn_norm(x)), the
//! attention norm absent in layer 0, Q and K rotated by the layer's base,
//! the window on a local layer, and a gated MLP; each projection with
//! its adapter on a layer that has them; in training, dropout on
//! attention's output and on the MLP's activated hidden.

use std::borrow::Cow;

use crate::real::Real;

use super::{
    ATTN_NORM, Compute, Encoder, FFN_IN, FFN_NORM, FFN_OUT, OUT, QKV, attention, bert, dropout, gated, lora, matmul,
    norm, slot,
};

/// The projections an adapter may sit on, in `lora_slot` order.
pub(super) const PROJECTIONS: [usize; 4] = [QKV, OUT, FFN_IN, FFN_OUT];

/// What one layer's forward keeps for its backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    /// The layer's input, the residual stream as it came in.
    pub x_in: Vec<R>,
    pub(super) n1: norm::Cache,
    pub(super) xn1: Vec<R>,
    /// Q and K rotated, and V.
    pub(super) qkv: Vec<R>,
    pub(super) attn: attention::Cache<R>,
    pub(super) ctx: Vec<R>,
    /// The residual after attention.
    pub x_mid: Vec<R>,
    pub(super) n2: norm::Cache,
    pub(super) xn2: Vec<R>,
    pub(super) pre: Vec<R>,
    pub(super) act: Vec<R>,
    /// The layer's output.
    pub x_out: Vec<R>,
    /// Each adapter's low-rank product, in `PROJECTIONS` order.
    pub(super) h: [Vec<R>; 4],
}

impl<R: Real> Encoder<R> {
    /// The width of the input and the output of projection `which`.
    pub(super) fn widths(&self, which: usize) -> (usize, usize) {
        let (h, i) = (self.geo.hidden, self.geo.intermediate);
        match which {
            QKV => (h, 3 * h),
            OUT => (h, h),
            FFN_IN => (h, if self.is_modern() { 2 * i } else { i }),
            _ => (i, h),
        }
    }

    /// y = x W^T + b (plus the adapter) for projection `which` of layer `l`.
    pub(super) fn project(&self, l: usize, which: usize, x: &[R], y: &mut [R], h: &mut Vec<R>) {
        let (inp, out) = self.widths(which);
        let (weights, bias) = self.projection(l, which);
        matmul::forward(x, weights, inp, out, y);
        for row in y.chunks_exact_mut(out) {
            for (y, &b) in row.iter_mut().zip(bias) {
                *y += b;
            }
        }
        self.store(y);
        if let Some(a) = self.adapter(l) {
            let (sa, sb) = self.lora_slots(l, which);
            let p = self.weights();
            lora::forward(x, &p[sa], &p[sb], inp, a.rank, a.scale(), self.rounds(), y, h);
            self.store(y);
        }
    }

    /// The gradients of `project` from dy, including its packed bias.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_back(
        &self,
        l: usize,
        which: usize,
        x: &[R],
        h: &[R],
        dy: &[R],
        g: &mut [Vec<R>],
        dx: &mut [R],
    ) {
        let (inp, out) = self.widths(which);
        let (weights, _) = self.projection(l, which);
        if !self.frozen {
            let (gw, gb) = g[slot(l, which)].split_at_mut(inp * out);
            matmul::grad_w(x, dy, inp, out, gw);
            for row in dy.chunks_exact(out) {
                for (b, &d) in gb.iter_mut().zip(row) {
                    *b += d;
                }
            }
        }
        matmul::grad_x(weights, dy, inp, out, dx);
        if let Some(a) = self.adapter(l) {
            let (sa, sb) = self.lora_slots(l, which);
            let p = self.weights();
            let [da, db] = g.get_disjoint_mut([sa, sb]).expect("an adapter's two slots are distinct");
            lora::backward(x, &p[sa], &p[sb], h, inp, a.rank, a.scale(), dy, da, db, Some(dx));
        }
        self.store(dx);
    }

    fn rounds(&self) -> bool {
        self.compute == Compute::Bf16
    }

    fn nothing(&self) -> &[R] {
        &self.empty
    }

    /// Layer `l` over x [n, hidden], into `w.x_out`; `at` keys the
    /// dropout masks of a training forward.
    pub(super) fn layer_forward(&self, l: usize, at: Option<u64>, x: &[R], w: &mut Work<R>) {
        if !self.is_modern() {
            return bert::forward(self, l, at, x, w);
        }
        let (h, i) = (self.geo.hidden, self.geo.intermediate);
        let n = x.len() / h;
        let global = self.geo.global(l);
        w.x_in.clear();
        w.x_in.extend_from_slice(x);
        let [h0, h1, h2, h3] = &mut w.h;
        let src: &[R] = if l == 0 {
            x
        } else {
            w.xn1.resize(n * h, R::ZERO);
            norm::forward(x, &self.weights()[slot(l, ATTN_NORM)], self.nothing(), self.geo.eps, &mut w.xn1, &mut w.n1);
            self.store(&mut w.xn1);
            &w.xn1
        };
        w.qkv.resize(n * 3 * h, R::ZERO);
        self.project(l, QKV, src, &mut w.qkv, h0);
        self.table.apply(global, &mut w.qkv, 3 * h, 2 * h);
        self.store(&mut w.qkv);
        w.ctx.resize(n * h, R::ZERO);
        attention::forward(&w.qkv, h, self.geo.heads, self.window(l), self.rounds(), &mut w.ctx, &mut w.attn);
        self.store(&mut w.ctx);
        w.x_mid.resize(n * h, R::ZERO);
        self.project(l, OUT, &w.ctx, &mut w.x_mid, h1);
        if let Some((d, key)) = self.mask(at, l as u32, 1) {
            dropout::forward(&mut w.x_mid, d.attention, key);
            self.store(&mut w.x_mid);
        }
        for (m, &x) in w.x_mid.iter_mut().zip(x) {
            *m += x;
        }
        self.store(&mut w.x_mid);
        w.xn2.resize(n * h, R::ZERO);
        norm::forward(
            &w.x_mid,
            &self.weights()[slot(l, FFN_NORM)],
            self.nothing(),
            self.geo.eps,
            &mut w.xn2,
            &mut w.n2,
        );
        self.store(&mut w.xn2);
        w.pre.resize(n * 2 * i, R::ZERO);
        self.project(l, FFN_IN, &w.xn2, &mut w.pre, h2);
        w.act.resize(n * i, R::ZERO);
        gated::forward(&w.pre, i, &mut w.act);
        self.store(&mut w.act);
        if let Some((d, key)) = self.mask(at, l as u32, 2) {
            dropout::forward(&mut w.act, d.mlp, key);
            self.store(&mut w.act);
        }
        w.x_out.resize(n * h, R::ZERO);
        self.project(l, FFN_OUT, &w.act, &mut w.x_out, h3);
        for (o, &m) in w.x_out.iter_mut().zip(&w.x_mid) {
            *o += m;
        }
        self.store(&mut w.x_out);
    }

    /// The gradient of layer `l`'s input from that of its output, with
    /// the layer's weight gradients added to `g`; `at` is the forward's.
    pub(super) fn layer_backward(
        &self,
        l: usize,
        at: Option<u64>,
        w: &Work<R>,
        dx_out: &[R],
        g: &mut [Vec<R>],
    ) -> Vec<R> {
        if !self.is_modern() {
            return bert::backward(self, l, at, w, dx_out, g);
        }
        let (h, i) = (self.geo.hidden, self.geo.intermediate);
        let n = dx_out.len() / h;
        let mut dact = vec![R::ZERO; n * i];
        self.project_back(l, FFN_OUT, &w.act, &w.h[3], dx_out, g, &mut dact);
        if let Some((d, key)) = self.mask(at, l as u32, 2) {
            dropout::backward(&mut dact, d.mlp, key);
            self.store(&mut dact);
        }
        let mut dpre = vec![R::ZERO; n * 2 * i];
        gated::backward(&w.pre, i, &dact, &mut dpre);
        self.store(&mut dpre);
        let mut dxn2 = vec![R::ZERO; n * h];
        self.project_back(l, FFN_IN, &w.xn2, &w.h[2], &dpre, g, &mut dxn2);
        let mut dx_mid = dx_out.to_vec();
        let norm_w = (!self.frozen).then_some(slot(l, FFN_NORM));
        norm::backward(
            &w.x_mid,
            &self.weights()[slot(l, FFN_NORM)],
            &w.n2,
            &dxn2,
            &mut dx_mid,
            norm_w.map(|s| &mut g[s][..]),
            None,
        );
        self.store(&mut dx_mid);
        let mut dctx = vec![R::ZERO; n * h];
        let dproj: Cow<[R]> = match self.mask(at, l as u32, 1) {
            Some((d, key)) => {
                let mut dropped = dx_mid.clone();
                dropout::backward(&mut dropped, d.attention, key);
                self.store(&mut dropped);
                Cow::Owned(dropped)
            }
            None => Cow::Borrowed(&dx_mid),
        };
        self.project_back(l, OUT, &w.ctx, &w.h[1], &dproj, g, &mut dctx);
        let mut dqkv = vec![R::ZERO; n * 3 * h];
        attention::backward(&w.qkv, h, self.geo.heads, self.window(l), self.rounds(), &w.attn, &dctx, &mut dqkv);
        self.store(&mut dqkv);
        self.table.apply_grad(self.geo.global(l), &mut dqkv, 3 * h, 2 * h);
        self.store(&mut dqkv);
        let mut dsrc = vec![R::ZERO; n * h];
        let src: &[R] = if l == 0 { &w.x_in } else { &w.xn1 };
        self.project_back(l, QKV, src, &w.h[0], &dqkv, g, &mut dsrc);
        let mut dx_in = dx_mid;
        if l == 0 {
            for (d, s) in dx_in.iter_mut().zip(dsrc) {
                *d += s;
            }
        } else {
            let norm_w = (!self.frozen).then_some(slot(l, ATTN_NORM));
            let dw = norm_w.map(|s| &mut g[s][..]);
            norm::backward(&w.x_in, &self.weights()[slot(l, ATTN_NORM)], &w.n1, &dsrc, &mut dx_in, dw, None);
        }
        self.store(&mut dx_in);
        dx_in
    }
}
