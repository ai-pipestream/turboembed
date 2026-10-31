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
//! adapters, eight more a layer that has them (A and B for each
//! projection) after them. With adapters the base is frozen: its
//! gradients are neither computed nor stored (`grads` leaves those slots
//! empty), and `trainable` marks only the adapters. Adapters go on every
//! layer (`with_lora`) or on the top ones (`with_lora_top`), as #1519
//! starts.
//!
//! One sequence at a time, on one thread; attention recomputes its
//! probabilities in backward rather than keeping them, and dropout
//! (`with_dropout`, applied by `forward_at`) draws its masks again from
//! their keys. No sum is split,
//! so a run gives the same bytes every time. Batches, threads and the
//! training loop are in `tuned`; the later legs (CUDA, bf16) are held to
//! this one.

use turbo::status::{Result, invalid};

use crate::real::Real;
use crate::rng::Rng;

pub mod attention;
pub mod dropout;
mod export;
pub mod gated;
mod layer;
mod load;
pub mod lora;
pub mod math;
pub mod matmul;
pub mod norm;
pub mod pool;
pub mod rope;
pub mod tuned;

pub use dropout::Dropout;
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
    /// The first layer with adapters.
    lora_from: usize,
    dropout: Option<Dropout>,
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
    /// The place of the last forward's example in its run, when it
    /// trained with dropout.
    at: Option<u64>,
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
        let e = Encoder {
            geo,
            vocab,
            params,
            lora: None,
            lora_from: 0,
            dropout: None,
            frozen: false,
            table,
            empty: Vec::new(),
        };
        let want = e.shapes();
        if e.params.len() != want.len() || e.params.iter().zip(&want).any(|(p, n)| p.len() != *n) {
            return Err(invalid("encoder: the weights do not fit the geometry"));
        }
        if e.params.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("encoder: a weight is not finite"));
        }
        Ok(e)
    }

    /// Adds seeded adapters of `lora` to every projection of every layer
    /// and freezes the base.
    pub fn with_lora(self, lora: Lora, rng: &mut Rng) -> Result<Self> {
        let layers = self.geo.layers;
        self.with_lora_top(lora, layers, rng)
    }

    /// Adds seeded adapters of `lora` to every projection of the top
    /// `top` layers (1 to all of them) and freezes the base, the lower
    /// layers included.
    pub fn with_lora_top(mut self, lora: Lora, top: usize, rng: &mut Rng) -> Result<Self> {
        if self.lora.is_some() {
            return Err(invalid("encoder: already has adapters"));
        }
        if lora.rank == 0 || !(lora.alpha.is_finite() && lora.alpha > 0.0) {
            return Err(invalid("encoder: an adapter needs a rank of at least 1 and a positive alpha"));
        }
        if top == 0 || top > self.geo.layers {
            return Err(invalid(format!("encoder: adapters on {top} layers, of 1 to {}", self.geo.layers)));
        }
        self.lora_from = self.geo.layers - top;
        for _ in self.lora_from..self.geo.layers {
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

    /// Drops as `d` says on a forward that gives its place in the run
    /// ([`forward_at`](Self::forward_at)); [`forward`](Self::forward)
    /// drops nothing, as inference does. Refuses a rate outside [0, 1).
    pub fn with_dropout(mut self, d: Dropout) -> Result<Self> {
        if !d.valid() {
            return Err(invalid("encoder: a dropout rate lies in [0, 1)"));
        }
        self.dropout = Some(d);
        Ok(self)
    }

    pub fn dropout(&self) -> Option<Dropout> {
        self.dropout
    }

    /// The dropout rates and mask key of branch `branch` of layer `l`
    /// (u32::MAX for the embedding) in the forward `w` holds, when it
    /// trained with dropout.
    pub(super) fn mask(&self, w_at: Option<u64>, l: u32, branch: u32) -> Option<(Dropout, u64)> {
        let d = self.dropout?;
        let at = w_at?;
        Some((d, dropout::key(d.seed, at, l, branch)))
    }

    /// The adapters of layer `l`, when it has them.
    pub(super) fn adapter(&self, l: usize) -> Option<Lora> {
        self.lora.filter(|_| l >= self.lora_from)
    }

    /// The slots of adapter A and B of projection `which` of layer `l`.
    pub(super) fn lora_slots(&self, l: usize, which: usize) -> (usize, usize) {
        let p = layer::PROJECTIONS.iter().position(|&w| w == which).expect("a projection");
        debug_assert!(l >= self.lora_from);
        let a = BASE + PER_LAYER * self.geo.layers + 8 * (l - self.lora_from) + 2 * p;
        (a, a + 1)
    }

    /// How many slots the base takes; the adapters follow.
    pub(super) fn base_slots(&self) -> usize {
        BASE + PER_LAYER * self.geo.layers
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

    /// The first layer with adapters (every layer below it is frozen
    /// base alone); the layer count without adapters.
    pub fn lora_from(&self) -> usize {
        if self.lora.is_some() { self.lora_from } else { self.geo.layers }
    }

    /// Whether the base weights take no gradient (the encoder has
    /// adapters).
    pub fn frozen(&self) -> bool {
        self.frozen
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

    /// Zero gradients shaped as the weights, but for the base's slots of
    /// a frozen encoder, which take no gradient and stay empty.
    pub fn grads(&self) -> Vec<Vec<R>> {
        let base = self.base_slots();
        let len = |s: usize, p: &Vec<R>| if self.frozen && s < base { 0 } else { p.len() };
        self.params.iter().enumerate().map(|(s, p)| vec![R::ZERO; len(s, p)]).collect()
    }

    /// Whether `g` is shaped as [`grads`](Self::grads) gives it.
    pub(super) fn shaped(&self, g: &[Vec<R>]) -> bool {
        let base = self.base_slots();
        g.len() == self.params.len()
            && g.iter()
                .zip(&self.params)
                .enumerate()
                .all(|(s, (a, b))| a.len() == b.len() || (self.frozen && s < base && a.is_empty()))
    }

    /// Which slots the optimizer moves: every one, or the adapters alone
    /// when the base is frozen.
    pub fn trainable(&self) -> Vec<bool> {
        let base = self.base_slots();
        (0..self.params.len()).map(|s| !self.frozen || s >= base).collect()
    }

    /// Which slots take weight decay: the matrices, not the norms.
    pub fn decay(&self) -> Vec<bool> {
        let base = self.base_slots();
        (0..self.params.len())
            .map(|s| s == WORD || s >= base || (s >= BASE && !matches!((s - BASE) % PER_LAYER, ATTN_NORM | FFN_NORM)))
            .collect()
    }

    /// The states [n, hidden] of `ids`, into `w.out`, with no dropout:
    /// what inference computes. Refuses an empty sequence, one past the
    /// positions, and an id past the vocabulary.
    pub fn forward(&self, ids: &[u32], w: &mut Work<R>) -> Result<()> {
        self.run(ids, None, w)
    }

    /// As [`forward`](Self::forward), training: dropout, when the encoder
    /// has it, drops under masks keyed by `at`, the example's place in
    /// the run, so the same place gives the same masks on any thread.
    pub fn forward_at(&self, ids: &[u32], at: u64, w: &mut Work<R>) -> Result<()> {
        self.run(ids, Some(at), w)
    }

    fn run(&self, ids: &[u32], at: Option<u64>, w: &mut Work<R>) -> Result<()> {
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
        w.at = at;
        w.emb_in.clear();
        for &id in ids {
            w.emb_in.extend_from_slice(&self.params[WORD][id as usize * h..][..h]);
        }
        w.x0.resize(n * h, R::ZERO);
        norm::forward(&w.emb_in, &self.params[EMB_NORM], &self.empty, self.geo.eps, &mut w.x0, &mut w.emb);
        if let Some((d, key)) = self.mask(at, u32::MAX, 0) {
            dropout::forward(&mut w.x0, d.embedding, key);
        }
        w.layers.resize_with(self.geo.layers, LayerWork::default);
        for l in 0..self.geo.layers {
            let (done, rest) = w.layers.split_at_mut(l);
            let x: &[R] = if l == 0 { &w.x0 } else { &done[l - 1].x_out };
            self.layer_forward(l, at, x, &mut rest[0]);
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
        if !self.shaped(g) {
            return Err(invalid("encoder: the gradients are not shaped as the weights"));
        }
        let last: &[R] = w.layers.last().map_or(&w.x0, |l| &l.x_out);
        let mut dx = vec![R::ZERO; n * h];
        let dw = (!self.frozen).then_some(FINAL_NORM);
        norm::backward(last, &self.params[FINAL_NORM], &w.fin, dout, &mut dx, dw.map(|s| &mut g[s][..]), None);
        for l in (0..self.geo.layers).rev() {
            dx = self.layer_backward(l, w.at, &w.layers[l], &dx, g);
        }
        if self.frozen {
            return Ok(());
        }
        if let Some((d, key)) = self.mask(w.at, u32::MAX, 0) {
            dropout::backward(&mut dx, d.embedding, key);
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
}
