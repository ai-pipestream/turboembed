//! An LSTM over a batch of sequences of different lengths, in F32 on the
//! CPU's packed kernels, for the sequence taggers' BiLSTMs.
//!
//! The rows of every sequence lie one after another, the longest first:
//! sequence s holds rows off[s]..off[s] + len[s] of the input and the
//! output. One product computes every direction's input projections for
//! every row (the input weights of all directions stacked); then each step
//! runs one recurrent product over the sequences still running at that
//! step, which are a prefix of the batch since it is sorted longest first,
//! and one pass of the gates. Gates are (i, f, g, o), each `hidden` wide.
//! A reverse direction reads each sequence from its last row.
//!
//! Each output is one chain of products and gates over its own sequence's
//! rows, so a sequence's states do not depend on the others in the batch.

use super::kernels::{Matrix, Mode};
use crate::head::sigmoid;
use crate::status::{Result, invalid};

/// One direction's recurrent weights and its place in the stacked input
/// projection.
struct Direction {
    reverse: bool,
    /// U, [4 * hidden, hidden].
    recurrent: Matrix,
}

/// Checked, packed weights of one or more directions over one input.
pub struct Lstm {
    input: usize,
    hidden: usize,
    /// Every direction's W, [directions * 4 * hidden, input], and bias.
    projection: Matrix,
    bias: Vec<f32>,
    directions: Vec<Direction>,
}

/// One direction's weights as an upstream LSTM holds them: W
/// [4 * hidden, input] and U [4 * hidden, hidden] row-major, gates (i, f,
/// g, o), and the bias [4 * hidden] (the input and recurrent biases
/// summed, where a checkpoint keeps two).
pub struct Weights<'a> {
    pub w: &'a [f32],
    pub u: &'a [f32],
    pub b: &'a [f32],
    pub reverse: bool,
}

/// What a run computes in, kept across runs so that a run of the same or
/// smaller shape allocates nothing.
#[derive(Default)]
pub struct Work {
    /// [rows, directions * 4 * hidden] input projections.
    projected: Vec<f32>,
    /// [batch, 4 * hidden] recurrent products of one step.
    step: Vec<f32>,
    /// [batch, hidden] each sequence's state and cell, in batch order.
    state: Vec<f32>,
    cell: Vec<f32>,
}

fn grow(v: &mut Vec<f32>, n: usize) -> Result<&mut [f32]> {
    if v.len() < n {
        v.try_reserve_exact(n - v.len()).map_err(|_| invalid("lstm workspace allocation failed"))?;
        v.resize(n, 0.0);
    }
    Ok(&mut v[..n])
}

impl Lstm {
    /// The directions in `dirs`, over inputs `input` wide with states
    /// `hidden` wide. Every weight must be finite and of its shape.
    pub fn new(input: usize, hidden: usize, dirs: &[Weights<'_>]) -> Result<Lstm> {
        let gates = hidden.checked_mul(4).filter(|_| input > 0 && hidden > 0);
        let Some(gates) = gates else {
            return Err(invalid(format!("lstm: input {input} and hidden {hidden} must be positive")));
        };
        if dirs.is_empty() {
            return Err(invalid("lstm: no direction"));
        }
        for (i, d) in dirs.iter().enumerate() {
            let shapes = [("w", d.w.len(), gates * input), ("u", d.u.len(), gates * hidden), ("b", d.b.len(), gates)];
            for (name, got, want) in shapes {
                if got != want {
                    return Err(invalid(format!("lstm direction {i}: {name} has {got} values, expected {want}")));
                }
            }
            if !d.w.iter().chain(d.u).chain(d.b).all(|v| v.is_finite()) {
                return Err(invalid(format!("lstm direction {i}: a weight is not finite")));
            }
        }
        let oom = |_| invalid("lstm weight packing allocation failed");
        let parts: Vec<&[f32]> = dirs.iter().map(|d| d.w).collect();
        let projection = Matrix::from_rows(&parts, input).map_err(oom)?;
        let mut directions = Vec::with_capacity(dirs.len());
        for d in dirs {
            directions
                .push(Direction { reverse: d.reverse, recurrent: Matrix::from_rows(&[d.u], hidden).map_err(oom)? });
        }
        let bias = dirs.iter().flat_map(|d| d.b.iter().copied()).collect();
        Ok(Lstm { input, hidden, projection, bias, directions })
    }

    pub fn input(&self) -> usize {
        self.input
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The width of an output row: every direction's state, in the order
    /// the directions were given.
    pub fn width(&self) -> usize {
        self.directions.len() * self.hidden
    }

    /// Every direction over the sequences of `lengths` (each at least 1,
    /// the longest first), whose rows are `x` [sum, input], into `out`
    /// [sum, width]: row r's direction k state at columns k * hidden to
    /// (k + 1) * hidden.
    ///
    /// Lengths out of order are refused, not sorted: a caller holding
    /// sequences in another order sorts them longest first (a stable sort
    /// keeps equal lengths in place), lays their rows out in that order,
    /// and puts the output rows back. A sequence's states do not depend on
    /// where it sits in the batch, so the order changes nothing else.
    pub fn run(&self, x: &[f32], lengths: &[usize], out: &mut [f32], w: &mut Work) -> Result<()> {
        let (h, g) = (self.hidden, 4 * self.hidden);
        let rows = lengths.iter().try_fold(0usize, |a, &n| a.checked_add(n));
        let rows = rows.ok_or_else(|| invalid("lstm: lengths overflow"))?;
        if lengths.contains(&0) || lengths.windows(2).any(|p| p[0] < p[1]) {
            return Err(invalid("lstm: lengths must be at least 1 and the longest first"));
        }
        if Some(x.len()) != rows.checked_mul(self.input) || Some(out.len()) != rows.checked_mul(self.width()) {
            return Err(invalid(format!(
                "lstm: {} input and {} output values for {rows} rows of {} and {}",
                x.len(),
                out.len(),
                self.input,
                self.width()
            )));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(invalid("lstm: an input is not finite"));
        }
        if rows == 0 {
            return Ok(());
        }
        let (batch, dg) = (lengths.len(), self.directions.len() * g);
        let starts: Vec<usize> = lengths.iter().scan(0, |a, &n| Some(std::mem::replace(a, *a + n))).collect();
        let projected = grow(&mut w.projected, rows * dg)?;
        self.projection.apply(x, self.input, rows, Some(&self.bias), projected, dg, Mode::Plain);
        let projected = &w.projected[..rows * dg];
        grow(&mut w.step, batch * g)?;
        grow(&mut w.state, batch * h)?;
        grow(&mut w.cell, batch * h)?;
        let width = self.width();
        for (k, d) in self.directions.iter().enumerate() {
            w.state[..batch * h].fill(0.0);
            w.cell[..batch * h].fill(0.0);
            let mut active = batch;
            for t in 0..lengths[0] {
                while lengths[active - 1] <= t {
                    active -= 1;
                }
                if t > 0 {
                    d.recurrent.apply(&w.state, h, active, None, &mut w.step, g, Mode::Plain);
                }
                for s in 0..active {
                    let at = starts[s] + if d.reverse { lengths[s] - 1 - t } else { t };
                    let a = &projected[at * dg + k * g..][..g];
                    let rec = &w.step[s * g..][..g];
                    let (state, cell) = (&mut w.state[s * h..][..h], &mut w.cell[s * h..][..h]);
                    let row = &mut out[at * width + k * h..][..h];
                    for j in 0..h {
                        let gate = |q: usize| a[q * h + j] + if t > 0 { rec[q * h + j] } else { 0.0 };
                        let c = sigmoid(gate(1)) * cell[j] + sigmoid(gate(0)) * gate(2).tanh();
                        let v = sigmoid(gate(3)) * c.tanh();
                        if !v.is_finite() {
                            return Err(invalid(format!("lstm: a state is not finite at row {at}")));
                        }
                        (cell[j], state[j], row[j]) = (c, v, v);
                    }
                }
            }
        }
        Ok(())
    }
}
