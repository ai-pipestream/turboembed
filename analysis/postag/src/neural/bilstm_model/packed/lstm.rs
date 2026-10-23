//! One LSTM direction over a batch of sequences on the CPU backend's
//! packed product (`turbo::cpu::Matrix`). The input projections are made
//! by the caller in one product over every step; here only the recurrent
//! product runs step by step, over every sequence still running, so a
//! step is one product of `[active, h]` by `U^T` however many sentences
//! or words are in the batch. Each output row of the packed product is one
//! chain of multiply-adds, so a sequence's states do not depend on what
//! else is in the batch.

use turbo::cpu::{Matrix, Mode};

use crate::neural::NeuralError;
use crate::neural::lstm::LstmLayer;

/// The rows of `parts` (each `[n_i][k]`) one after another as f32, packed.
pub(super) fn pack(parts: &[&[Vec<f64>]], k: usize) -> Result<Matrix, NeuralError> {
    let mut flat = Vec::new();
    for part in parts {
        for row in part.iter() {
            if row.len() != k {
                return Err(NeuralError::InvalidArgument(format!("packed row of {} values, not {k}", row.len())));
            }
            flat.extend(row.iter().map(|&v| v as f32));
        }
    }
    Matrix::from_rows(&[&flat], k).map_err(allocation)
}

pub(super) fn allocation(bytes: usize) -> NeuralError {
    NeuralError::InvalidArgument(format!("packing a BiLSTM weight matrix failed ({bytes} bytes)"))
}

pub(super) fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The reusable buffers of the recurrences of one call.
#[derive(Default)]
pub(super) struct Steps {
    gates: Vec<f32>,
    state: Vec<f32>,
    cell: Vec<f32>,
}

/// The recurrent weights `U [4h, h]` of one direction, packed.
pub(super) struct Recurrent {
    u: Matrix,
    h: usize,
}

impl Recurrent {
    pub(super) fn new(layer: &LstmLayer) -> Result<Self, NeuralError> {
        let h = layer.hidden_size();
        Ok(Self { u: pack(&[layer.u()], h)?, h })
    }

    pub(super) fn hidden(&self) -> usize {
        self.h
    }

    /// Runs the direction over sequences whose lengths `lens` are sorted
    /// longest first. `pre(s, t)` is the `[4h]` input pre-activation
    /// (bias included) of step `t` of sequence `s`; `emit(s, t, state)`
    /// receives the state after that step. Gates are `(i, f, g, o)` as in
    /// [`LstmLayer`]; the first step has no recurrent product (the state
    /// is zero), as in Java.
    pub(super) fn run<'a>(
        &self,
        lens: &[usize],
        pre: impl Fn(usize, usize) -> &'a [f32],
        mut emit: impl FnMut(usize, usize, &[f32]),
        work: &mut Steps,
    ) {
        debug_assert!(lens.windows(2).all(|w| w[0] >= w[1]), "lengths sorted longest first");
        let (h, g) = (self.h, 4 * self.h);
        let Steps { gates, state, cell } = work;
        for v in [&mut *state, &mut *cell] {
            v.clear();
            v.resize(lens.len() * h, 0.0);
        }
        gates.clear();
        gates.resize(lens.len() * g, 0.0);
        for t in 0..lens.first().copied().unwrap_or(0) {
            let active = lens.partition_point(|&l| l > t);
            for s in 0..active {
                gates[s * g..(s + 1) * g].copy_from_slice(&pre(s, t)[..g]);
            }
            if t > 0 {
                self.u.apply(state, h, active, None, gates, g, Mode::Residual);
            }
            for s in 0..active {
                let gs = &gates[s * g..(s + 1) * g];
                let (st, ce) = (&mut state[s * h..(s + 1) * h], &mut cell[s * h..(s + 1) * h]);
                for j in 0..h {
                    let i = sigmoid(gs[j]);
                    let f = sigmoid(gs[h + j]);
                    let c = f * ce[j] + i * gs[2 * h + j].tanh();
                    ce[j] = c;
                    st[j] = sigmoid(gs[3 * h + j]) * c.tanh();
                }
                emit(s, t, st);
            }
        }
    }
}
