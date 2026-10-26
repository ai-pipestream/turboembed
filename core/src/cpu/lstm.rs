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
//! A reverse direction reads each sequence from its last row. A caller
//! that already holds the input projections (a table of distinct rows,
//! by id) runs the recurrences alone with `Lstm::run_gathered`.
//!
//! Each output is one chain of products and gates over its own sequence's
//! rows, so a sequence's states do not depend on the others in the batch.

use super::kernels::{Matrix, Mode};
use crate::head::sigmoid;
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, OUT_OF_MEMORY, Result, invalid};

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
    /// Each sequence's first row.
    starts: Vec<usize>,
}

/// A run refused for what the caller passed; the weights are not at fault.
fn argument(message: impl Into<String>) -> Error {
    Error::new(INVALID_ARGUMENT, message)
}

/// Whether every value is finite: a branch-free fold in blocks, so the
/// check vectorizes instead of stopping at each value.
fn finite(v: &[f32]) -> bool {
    v.chunks(64).all(|c| c.iter().fold(true, |a, x| a & x.is_finite()))
}

fn grow(v: &mut Vec<f32>, n: usize) -> Result<&mut [f32]> {
    if v.len() < n {
        v.try_reserve_exact(n - v.len()).map_err(|_| Error::new(OUT_OF_MEMORY, "lstm workspace allocation failed"))?;
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
        let (Some(w), Some(u)) = (gates.checked_mul(input), gates.checked_mul(hidden)) else {
            return Err(invalid(format!("lstm: input {input} and hidden {hidden} make weights past usize")));
        };
        if dirs.is_empty() {
            return Err(invalid("lstm: no direction"));
        }
        for (i, d) in dirs.iter().enumerate() {
            let shapes = [("w", d.w.len(), w), ("u", d.u.len(), u), ("b", d.b.len(), gates)];
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
    ///
    /// Refused: lengths of 0, out of order or overflowing, and non-finite
    /// inputs or states, with INVALID_ARGUMENT; `x` or `out` of the wrong
    /// size, with INVALID_SHAPE; a workspace that cannot grow, with
    /// OUT_OF_MEMORY.
    pub fn run(&self, x: &[f32], lengths: &[usize], out: &mut [f32], w: &mut Work) -> Result<()> {
        let rows = self.rows(lengths, out)?;
        if Some(x.len()) != rows.checked_mul(self.input) {
            let message = format!("lstm: {} input values for {rows} rows of {}", x.len(), self.input);
            return Err(Error::new(INVALID_SHAPE, message));
        }
        if !finite(x) {
            return Err(argument("lstm: an input is not finite"));
        }
        if rows == 0 {
            return Ok(());
        }
        let dg = self.projected_width();
        let mut projected = std::mem::take(&mut w.projected);
        let result = grow(&mut projected, rows * dg).and_then(|p| {
            self.projection.apply(x, self.input, rows, Some(&self.bias), p, dg, Mode::Plain);
            let p = &*p;
            self.recur(|r| &p[r * dg..][..dg], lengths, out, w)
        });
        w.projected = projected;
        result
    }

    /// The width of a row of input projections: every direction's four
    /// gates, in the order the directions were given.
    pub fn projected_width(&self) -> usize {
        self.directions.len() * 4 * self.hidden
    }

    /// [`run`](Self::run) from input projections the caller made once
    /// per distinct row: row r of the sequences reads row `index[r]` of
    /// `table` [distinct, projected_width], each row's `W x + b` of every
    /// direction, the directions in the order given (a character table
    /// read by character id, say). Nothing is copied out of the table.
    /// Refused as `run` refuses, and a non-finite table value with
    /// INVALID_ARGUMENT; a table of the wrong width, or an index of the
    /// wrong length or past the table, with INVALID_SHAPE.
    pub fn run_gathered(
        &self,
        table: &[f32],
        index: &[usize],
        lengths: &[usize],
        out: &mut [f32],
        w: &mut Work,
    ) -> Result<()> {
        let (rows, pw) = (self.rows(lengths, out)?, self.projected_width());
        let distinct = table.len() / pw;
        if !table.len().is_multiple_of(pw) || index.len() != rows || index.iter().any(|&i| i >= distinct) {
            let message =
                format!("lstm: a table of {} values and {} indices for {rows} rows of {pw}", table.len(), index.len());
            return Err(Error::new(INVALID_SHAPE, message));
        }
        if !finite(table) {
            return Err(argument("lstm: a projection is not finite"));
        }
        if rows == 0 {
            return Ok(());
        }
        self.recur(|r| &table[index[r] * pw..][..pw], lengths, out, w)
    }

    /// The rows of `lengths`, checked, and `out` checked against them.
    fn rows(&self, lengths: &[usize], out: &[f32]) -> Result<usize> {
        let rows = lengths.iter().try_fold(0usize, |a, &n| a.checked_add(n));
        let rows = rows.ok_or_else(|| argument("lstm: lengths overflow"))?;
        if lengths.contains(&0) || lengths.windows(2).any(|p| p[0] < p[1]) {
            return Err(argument("lstm: lengths must be at least 1 and the longest first"));
        }
        if Some(out.len()) != rows.checked_mul(self.width()) {
            let message = format!("lstm: {} output values for {rows} rows of {}", out.len(), self.width());
            return Err(Error::new(INVALID_SHAPE, message));
        }
        Ok(rows)
    }

    /// The recurrences over checked, nonempty `lengths`; `projected(r)` is
    /// row r's projections of every direction.
    fn recur<'a>(
        &self,
        projected: impl Fn(usize) -> &'a [f32],
        lengths: &[usize],
        out: &mut [f32],
        w: &mut Work,
    ) -> Result<()> {
        let (h, g) = (self.hidden, 4 * self.hidden);
        let batch = lengths.len();
        w.starts.clear();
        w.starts.extend(lengths.iter().scan(0, |a, &n| Some(std::mem::replace(a, *a + n))));
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
                let at = |s: usize| {
                    let (start, n) = (w.starts[s], lengths[s]);
                    start + if d.reverse { n - 1 - t } else { t }
                };
                // Each step's gates: the input projection, plus the
                // recurrent product added onto it after the first step.
                for s in 0..active {
                    w.step[s * g..][..g].copy_from_slice(&projected(at(s))[k * g..][..g]);
                }
                if t > 0 {
                    d.recurrent.apply(&w.state, h, active, None, &mut w.step, g, Mode::Residual);
                }
                for s in 0..active {
                    let gs = &w.step[s * g..][..g];
                    // A gate past f32's range would saturate to a finite
                    // state; refuse it as a non-finite state is refused.
                    if !finite(gs) {
                        return Err(argument(format!("lstm: a gate is not finite at row {}", at(s))));
                    }
                    let (state, cell) = (&mut w.state[s * h..][..h], &mut w.cell[s * h..][..h]);
                    for j in 0..h {
                        let c = sigmoid(gs[h + j]) * cell[j] + sigmoid(gs[j]) * gs[2 * h + j].tanh();
                        (cell[j], state[j]) = (c, sigmoid(gs[3 * h + j]) * c.tanh());
                    }
                    if !finite(state) {
                        return Err(argument(format!("lstm: a state is not finite at row {}", at(s))));
                    }
                    out[at(s) * width + k * h..][..h].copy_from_slice(state);
                }
            }
        }
        Ok(())
    }
}
