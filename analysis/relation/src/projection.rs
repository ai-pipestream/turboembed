//! Immutable shared Matrix projections and caller-owned reusable storage.

use turbo::cpu::{Matrix, Mode};
use turbo::head::Linear;

/// Reusable projection scratch, owned by a caller, never shared by heads.
/// Retain it between requests to retain capacities. Model heads remain immutable.
#[derive(Default)]
pub struct Scratch {
    pub(crate) buffers: [Vec<f32>; 8],
}

pub(crate) struct Projection {
    pub in_dim: usize,
    pub out_dim: usize,
    matrix: Matrix,
    bias: Vec<f32>,
}

impl Projection {
    pub fn new(l: &Linear) -> Result<Self, String> {
        let matrix = Matrix::from_rows(&[l.weight()], l.in_dim)
            .map_err(|bytes| format!("packing a {}x{} layer needs {bytes} bytes", l.out_dim, l.in_dim))?;
        Ok(Self { in_dim: l.in_dim, out_dim: l.out_dim, matrix, bias: l.bias().to_vec() })
    }

    pub fn rows_into(&self, x: &[f32], n: usize, out: &mut Vec<f32>) {
        assert_eq!(Some(x.len()), n.checked_mul(self.in_dim), "rows of the input width");
        out.resize(n.checked_mul(self.out_dim).expect("projection output overflow"), 0.0);
        self.matrix.apply(x, self.in_dim, n, Some(&self.bias), out, self.out_dim, Mode::Plain);
    }
}

pub(crate) fn relu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = x.max(0.0));
}

pub(crate) fn mlp_into(p: &[Projection; 2], x: &[f32], n: usize, tmp: &mut Vec<f32>, out: &mut Vec<f32>) {
    p[0].rows_into(x, n, tmp);
    relu(tmp);
    p[1].rows_into(tmp, n, out);
}

#[cfg(test)]
mod tests;
