//! One sequence forward and backward through the encoder, with the
//! places bfloat16 compute rounds marked: every op's output in forward,
//! every activation's gradient in backward, the weights as the shadow
//! holds them; the sums inside each op run in the single.

use turbo::status::{Result, invalid};

use crate::real::Real;

use super::layer::Work as LayerWork;
use super::{EMB_NORM, Encoder, FINAL_NORM, WORD, dropout, norm};

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
            w.emb_in.extend_from_slice(&self.weights()[WORD][id as usize * h..][..h]);
        }
        w.x0.resize(n * h, R::ZERO);
        norm::forward(&w.emb_in, &self.weights()[EMB_NORM], &self.empty, self.geo.eps, &mut w.x0, &mut w.emb);
        self.store(&mut w.x0);
        if let Some((d, key)) = self.mask(at, u32::MAX, 0) {
            dropout::forward(&mut w.x0, d.embedding, key);
            self.store(&mut w.x0);
        }
        w.layers.resize_with(self.geo.layers, LayerWork::default);
        for l in 0..self.geo.layers {
            let (done, rest) = w.layers.split_at_mut(l);
            let x: &[R] = if l == 0 { &w.x0 } else { &done[l - 1].x_out };
            self.layer_forward(l, at, x, &mut rest[0]);
        }
        let last: &[R] = w.layers.last().map_or(&w.x0, |l| &l.x_out);
        w.out.resize(n * h, R::ZERO);
        norm::forward(last, &self.weights()[FINAL_NORM], &self.empty, self.geo.eps, &mut w.out, &mut w.fin);
        self.store(&mut w.out);
        Ok(())
    }

    /// Adds to `g` (shaped as [`grads`](Self::grads)) the gradients of
    /// the weights from `dout`, the gradient of the states of the last
    /// [`forward`](Self::forward) into `w`; under bfloat16 `dout` is
    /// read rounded to it.
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
        let dout = self.stored(dout);
        let mut dx = vec![R::ZERO; n * h];
        let dw = (!self.frozen).then_some(FINAL_NORM);
        norm::backward(last, &self.weights()[FINAL_NORM], &w.fin, &dout, &mut dx, dw.map(|s| &mut g[s][..]), None);
        self.store(&mut dx);
        for l in (0..self.geo.layers).rev() {
            dx = self.layer_backward(l, w.at, &w.layers[l], &dx, g);
        }
        if self.frozen {
            return Ok(());
        }
        if let Some((d, key)) = self.mask(w.at, u32::MAX, 0) {
            dropout::backward(&mut dx, d.embedding, key);
            self.store(&mut dx);
        }
        let mut demb = vec![R::ZERO; n * h];
        let (gw, gn) = g.split_at_mut(EMB_NORM);
        norm::backward(&w.emb_in, &self.weights()[EMB_NORM], &w.emb, &dx, &mut demb, Some(&mut gn[0]), None);
        self.store(&mut demb);
        for (t, &id) in w.ids.iter().enumerate() {
            for (d, &v) in gw[WORD][id as usize * h..][..h].iter_mut().zip(&demb[t * h..][..h]) {
                *d += v;
            }
        }
        Ok(())
    }
}
