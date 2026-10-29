//! A ModernBERT encoder in training (stage R2, #1519): forward over one
//! sequence of token ids with everything backward needs kept, and
//! backward to every weight, generic over [`Real`] as the R1 ops are.
//! The numerics are the core's CPU encoder's (`turbo::cpu`): the same
//! cos/sin table, erf GELU, biased-variance LayerNorm summed in f64, the
//! window |i - j| <= window / 2 on local layers, 1 / sqrt(d) scores.
//!
//! The weights are a list of tensors in a fixed order (`slot`), as the
//! R1 heads hold theirs, so `AdamW` steps them as it does a head's: the
//! word table, the embedding norm, the final norm, then six a layer
//! (attention norm, QKV, output, MLP norm, MLP in, MLP out); with
//! adapters, eight more a layer (A and B for each projection) after
//! them. With adapters the base is frozen: its gradients are not
//! computed, and `trainable` marks only the adapters.
//!
//! One sequence at a time, on one thread, with the probabilities of
//! every head kept: the reference the later legs (batches, CUDA,
//! checkpointing, bf16) are held to. No sum is split, so a run gives the
//! same bytes every time.

use std::collections::BTreeMap;

use turbo::manifest::TensorRole;
use turbo::status::{Result, invalid};

use crate::real::Real;
use crate::rng::Rng;

pub mod attention;
pub mod gated;
mod layer;
mod load;
pub mod lora;
pub mod math;
pub mod matmul;
pub mod norm;
pub mod rope;

pub use layer::Work as LayerWork;

/// A tensor as a safetensors file holds it: name, shape and values.
pub type Named = (String, Vec<u64>, Vec<f32>);

/// The encoder's shape and the knobs the manifest fixes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub eps: f64,
    /// The keys each side of a query on a local layer, `window / 2` of
    /// the manifest; usize::MAX with no local layer.
    pub window: usize,
    /// Layer l attends to the whole sequence iff `l % global_every == 0`;
    /// 0 makes every layer global.
    pub global_every: usize,
    pub max_positions: usize,
    pub global_theta: f64,
    pub local_theta: f64,
}

impl Geometry {
    /// Whether layer `l` attends to the whole sequence.
    pub fn global(&self, l: usize) -> bool {
        self.global_every == 0 || l.is_multiple_of(self.global_every)
    }

    /// A head's width.
    pub fn d(&self) -> usize {
        self.hidden / self.heads
    }

    fn check(&self) -> Result<()> {
        let bad = |why: &str| Err(invalid(format!("encoder: {why}")));
        if self.hidden == 0 || self.heads == 0 || !self.hidden.is_multiple_of(self.heads) {
            return bad("the heads must divide the hidden width");
        }
        if !self.d().is_multiple_of(2) {
            return bad("a head's width must be even to rotate");
        }
        if self.intermediate == 0 || self.max_positions == 0 {
            return bad("the intermediate width and the positions must be at least 1");
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return bad("eps must be positive and finite");
        }
        if !(self.global_theta > 1.0 && self.local_theta > 1.0) {
            return bad("the rotary bases must be above 1");
        }
        Ok(())
    }
}

/// The adapters' rank and alpha (Hu et al. 2021): each projection gains
/// `alpha / rank` times its low-rank product.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lora {
    pub rank: usize,
    pub alpha: f64,
}

impl Lora {
    fn scale<R: Real>(&self) -> R {
        R::from_f64(self.alpha / self.rank as f64)
    }
}

pub(super) const WORD: usize = 0;
pub(super) const EMB_NORM: usize = 1;
pub(super) const FINAL_NORM: usize = 2;
const BASE: usize = 3;
const PER_LAYER: usize = 6;
pub(super) const ATTN_NORM: usize = 0;
pub(super) const QKV: usize = 1;
pub(super) const OUT: usize = 2;
pub(super) const FFN_NORM: usize = 3;
pub(super) const FFN_IN: usize = 4;
pub(super) const FFN_OUT: usize = 5;

/// The slot of tensor `which` (ATTN_NORM to FFN_OUT) of layer `l`.
pub(super) fn slot(l: usize, which: usize) -> usize {
    BASE + PER_LAYER * l + which
}

/// A ModernBERT encoder's weights, with or without adapters.
#[derive(Debug, Clone)]
pub struct Encoder<R: Real> {
    geo: Geometry,
    vocab: usize,
    params: Vec<Vec<R>>,
    lora: Option<Lora>,
    /// Whether the base weights take no gradient (with adapters).
    frozen: bool,
    table: rope::Table<R>,
    /// An empty bias, for the norms.
    empty: Vec<R>,
}

/// What a forward keeps for its backward.
#[derive(Debug, Default)]
pub struct Work<R> {
    ids: Vec<u32>,
    /// The word rows before the embedding norm, and after it.
    emb_in: Vec<R>,
    emb: norm::Cache,
    x0: Vec<R>,
    layers: Vec<LayerWork<R>>,
    fin: norm::Cache,
    /// The final states, [n, hidden].
    pub out: Vec<R>,
}

impl<R: Real> Encoder<R> {
    /// A seeded encoder: projections uniform in +-1 / sqrt(in), the word
    /// table in +-0.1, norms 1.
    pub fn random(geo: Geometry, vocab: usize, rng: &mut Rng) -> Result<Self> {
        geo.check()?;
        if vocab == 0 {
            return Err(invalid("encoder: a vocabulary of no words"));
        }
        let (h, i) = (geo.hidden, geo.intermediate);
        let mut params =
            vec![(0..vocab * h).map(|_| R::from_f64(rng.uniform(0.1))).collect(), vec![R::ONE; h], vec![R::ONE; h]];
        for l in 0..geo.layers {
            let mut matrix = |out: usize, inp: usize| -> Vec<R> {
                let limit = 1.0 / (inp as f64).sqrt();
                (0..out * inp).map(|_| R::from_f64(rng.uniform(limit))).collect()
            };
            let (qkv, out, ffn_in, ffn_out) = (matrix(3 * h, h), matrix(h, h), matrix(2 * i, h), matrix(h, i));
            params.push(if l == 0 { Vec::new() } else { vec![R::ONE; h] });
            params.extend([qkv, out, vec![R::ONE; h], ffn_in, ffn_out]);
        }
        Self::assemble(geo, vocab, params)
    }

    /// An encoder over `params` in slot order, which must fit `geo`.
    pub fn assemble(geo: Geometry, vocab: usize, params: Vec<Vec<R>>) -> Result<Self> {
        geo.check()?;
        let table = rope::Table::new(geo.global_theta, geo.local_theta, geo.max_positions, geo.d())?;
        let e = Encoder { geo, vocab, params, lora: None, frozen: false, table, empty: Vec::new() };
        let want = e.shapes();
        if e.params.len() != want.len() || e.params.iter().zip(&want).any(|(p, n)| p.len() != *n) {
            return Err(invalid("encoder: the weights do not fit the geometry"));
        }
        if e.params.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("encoder: a weight is not finite"));
        }
        Ok(e)
    }

    /// Adds seeded adapters of `lora` to every projection and freezes
    /// the base.
    pub fn with_lora(mut self, lora: Lora, rng: &mut Rng) -> Result<Self> {
        if self.lora.is_some() {
            return Err(invalid("encoder: already has adapters"));
        }
        if lora.rank == 0 || !(lora.alpha.is_finite() && lora.alpha > 0.0) {
            return Err(invalid("encoder: an adapter needs a rank of at least 1 and a positive alpha"));
        }
        for _ in 0..self.geo.layers {
            for which in layer::PROJECTIONS {
                let (inp, out) = self.widths(which);
                let (a, b) = lora::init(inp, out, lora.rank, rng);
                self.params.extend([a, b]);
            }
        }
        self.lora = Some(lora);
        self.frozen = true;
        Ok(self)
    }

    /// The slots of adapter A and B of projection `which` of layer `l`.
    fn lora_slots(&self, l: usize, which: usize) -> (usize, usize) {
        let p = layer::PROJECTIONS.iter().position(|&w| w == which).expect("a projection");
        let a = BASE + PER_LAYER * self.geo.layers + 8 * l + 2 * p;
        (a, a + 1)
    }

    /// Each base slot's element count.
    fn shapes(&self) -> Vec<usize> {
        let (h, i) = (self.geo.hidden, self.geo.intermediate);
        let mut out = vec![self.vocab * h, h, h];
        for l in 0..self.geo.layers {
            out.extend([if l == 0 { 0 } else { h }, 3 * h * h, h * h, h, 2 * i * h, h * i]);
        }
        out
    }

    /// The keys each side of a query on layer `l`.
    fn window(&self, l: usize) -> usize {
        if self.geo.global(l) { usize::MAX } else { self.geo.window }
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geo
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn lora(&self) -> Option<Lora> {
        self.lora
    }

    /// The weights in slot order: the base, then the adapters.
    pub fn params(&self) -> &[Vec<R>] {
        &self.params
    }

    /// The weights, to step in place: what is written must stay finite,
    /// as [`set_params`](Self::set_params) checks.
    pub fn params_mut(&mut self) -> &mut [Vec<R>] {
        &mut self.params
    }

    /// Replaces the weights with `params`, shaped as [`params`](Self::params).
    pub fn set_params(&mut self, params: Vec<Vec<R>>) -> Result<()> {
        if params.len() != self.params.len() || params.iter().zip(&self.params).any(|(a, b)| a.len() != b.len()) {
            return Err(invalid("encoder: the weights do not match the encoder's"));
        }
        if params.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("encoder: a weight is not finite"));
        }
        self.params = params;
        Ok(())
    }

    /// Zero gradients shaped as the weights.
    pub fn grads(&self) -> Vec<Vec<R>> {
        self.params.iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    /// Which slots the optimizer moves: every one, or the adapters alone
    /// when the base is frozen.
    pub fn trainable(&self) -> Vec<bool> {
        let base = BASE + PER_LAYER * self.geo.layers;
        (0..self.params.len()).map(|s| !self.frozen || s >= base).collect()
    }

    /// Which slots take weight decay: the matrices, not the norms.
    pub fn decay(&self) -> Vec<bool> {
        let base = BASE + PER_LAYER * self.geo.layers;
        (0..self.params.len())
            .map(|s| s == WORD || s >= base || (s >= BASE && !matches!((s - BASE) % PER_LAYER, ATTN_NORM | FFN_NORM)))
            .collect()
    }

    /// The states [n, hidden] of `ids`, into `w.out`. Refuses an empty
    /// sequence, one past the positions, and an id past the vocabulary.
    pub fn forward(&self, ids: &[u32], w: &mut Work<R>) -> Result<()> {
        let h = self.geo.hidden;
        if ids.is_empty() || ids.len() > self.geo.max_positions {
            return Err(invalid(format!("encoder: {} tokens, of 1 to {}", ids.len(), self.geo.max_positions)));
        }
        if let Some(id) = ids.iter().find(|&&id| id as usize >= self.vocab) {
            return Err(invalid(format!("encoder: token id {id} is past the {} words", self.vocab)));
        }
        let n = ids.len();
        w.ids.clear();
        w.ids.extend_from_slice(ids);
        w.emb_in.clear();
        for &id in ids {
            w.emb_in.extend_from_slice(&self.params[WORD][id as usize * h..][..h]);
        }
        w.x0.resize(n * h, R::ZERO);
        norm::forward(&w.emb_in, &self.params[EMB_NORM], &self.empty, self.geo.eps, &mut w.x0, &mut w.emb);
        w.layers.resize_with(self.geo.layers, LayerWork::default);
        for l in 0..self.geo.layers {
            let (done, rest) = w.layers.split_at_mut(l);
            let x: &[R] = if l == 0 { &w.x0 } else { &done[l - 1].x_out };
            self.layer_forward(l, x, &mut rest[0]);
        }
        let last: &[R] = w.layers.last().map_or(&w.x0, |l| &l.x_out);
        w.out.resize(n * h, R::ZERO);
        norm::forward(last, &self.params[FINAL_NORM], &self.empty, self.geo.eps, &mut w.out, &mut w.fin);
        Ok(())
    }

    /// Adds to `g` (shaped as [`grads`](Self::grads)) the gradients of
    /// the weights from `dout`, the gradient of the states of the last
    /// [`forward`](Self::forward) into `w`.
    pub fn backward(&self, w: &Work<R>, dout: &[R], g: &mut [Vec<R>]) -> Result<()> {
        let h = self.geo.hidden;
        let n = w.ids.len();
        if n == 0 || dout.len() != n * h {
            return Err(invalid("encoder: the gradient is not shaped as the last forward's states"));
        }
        if g.len() != self.params.len() || g.iter().zip(&self.params).any(|(a, b)| a.len() != b.len()) {
            return Err(invalid("encoder: the gradients are not shaped as the weights"));
        }
        let last: &[R] = w.layers.last().map_or(&w.x0, |l| &l.x_out);
        let mut dx = vec![R::ZERO; n * h];
        let dw = (!self.frozen).then_some(FINAL_NORM);
        norm::backward(last, &self.params[FINAL_NORM], &w.fin, dout, &mut dx, dw.map(|s| &mut g[s][..]), None);
        for l in (0..self.geo.layers).rev() {
            dx = self.layer_backward(l, &w.layers[l], &dx, g);
        }
        if self.frozen {
            return Ok(());
        }
        let mut demb = vec![R::ZERO; n * h];
        let (gw, gn) = g.split_at_mut(EMB_NORM);
        norm::backward(&w.emb_in, &self.params[EMB_NORM], &w.emb, &dx, &mut demb, Some(&mut gn[0]), None);
        for (t, &id) in w.ids.iter().enumerate() {
            for (d, &v) in gw[WORD][id as usize * h..][..h].iter_mut().zip(&demb[t * h..][..h]) {
                *d += v;
            }
        }
        Ok(())
    }

    /// The base weights with the adapters folded in, W + s B A, in slot
    /// order: what a bundle carries.
    pub fn merged(&self) -> Vec<Vec<R>> {
        let base = BASE + PER_LAYER * self.geo.layers;
        let mut out: Vec<Vec<R>> = self.params[..base].to_vec();
        let Some(a) = self.lora else {
            return out;
        };
        for l in 0..self.geo.layers {
            for which in layer::PROJECTIONS {
                let (inp, _) = self.widths(which);
                let (sa, sb) = self.lora_slots(l, which);
                let (ma, mb) = (&self.params[sa], &self.params[sb]);
                for (wo, bo) in out[slot(l, which)].chunks_exact_mut(inp).zip(mb.chunks_exact(a.rank)) {
                    for (&b, ar) in bo.iter().zip(ma.chunks_exact(inp)) {
                        for (w, &av) in wo.iter_mut().zip(ar) {
                            *w += a.scale::<R>() * b * av;
                        }
                    }
                }
            }
        }
        out
    }

    /// The merged weights as f32 tensors named by `names` (a bundle
    /// artifact's `tensor_names`, with `{layer}`), shaped as the core
    /// reads them: what a bundle's weights file holds.
    pub fn named(&self, names: &BTreeMap<TensorRole, String>) -> Result<Vec<Named>> {
        let (h, i) = (self.geo.hidden as u64, self.geo.intermediate as u64);
        let merged = self.merged();
        let mut out = Vec::new();
        for (role, layer, s, shape) in load::roles(self.geo.layers, self.vocab as u64, h, i) {
            if role == TensorRole::AttnLnWeight && layer == Some(0) {
                continue;
            }
            let name = load::name(names, role, layer)?;
            out.push((name, shape, merged[s].iter().map(|v| v.to_f64() as f32).collect()));
        }
        Ok(out)
    }
}
