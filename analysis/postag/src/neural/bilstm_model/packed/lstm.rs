//! The tagger's LSTM weights packed for the CPU backend: the stacked
//! products as `turbo::cpu::Matrix`, and each BiLSTM as one
//! `turbo::cpu::lstm::Lstm` over both directions (#859), which runs every
//! sequence of a call in one recurrent product per step.

use turbo::cpu::Matrix;
use turbo::cpu::lstm::{Lstm, Weights};
use turbo::status::{Error, INVALID_ARGUMENT};

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

/// A forward and a backward [`LstmLayer`] as one two-direction [`Lstm`]:
/// an output row is the forward state, then the backward one. Gates are
/// `(i, f, g, o)` in both. A weight past f32's range is refused.
pub(super) fn bidirectional(fwd: &LstmLayer, bwd: &LstmLayer) -> Result<Lstm, NeuralError> {
    let flat = |m: &[Vec<f64>]| -> Vec<f32> { m.iter().flatten().map(|&v| v as f32).collect() };
    let (wf, uf, bf) = (flat(fwd.w()), flat(fwd.u()), to_f32(fwd.b()));
    let (wb, ub, bb) = (flat(bwd.w()), flat(bwd.u()), to_f32(bwd.b()));
    let dirs = [Weights { w: &wf, u: &uf, b: &bf, reverse: false }, Weights { w: &wb, u: &ub, b: &bb, reverse: true }];
    Lstm::new(fwd.input_size(), fwd.hidden_size(), &dirs).map_err(|e| NeuralError::InvalidArgument(e.message))
}

/// A refused run as the tagger reports it. The lengths passed are checked
/// positive and sorted, so INVALID_ARGUMENT is a value past f32's range,
/// which the tagger answers from the f64 reference.
pub(super) fn run_error(e: Error) -> NeuralError {
    if e.code == INVALID_ARGUMENT {
        NeuralError::NonFiniteScore(e.message)
    } else {
        NeuralError::InvalidArgument(e.message)
    }
}
