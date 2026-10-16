//! Single-direction LSTM layer (`LstmLayer`) with forward and backward passes.

use crate::neural::{JavaRng, NeuralError};

const ACCUMULATE_BLOCK: usize = 32;

/// Maximum hidden width before `4 * hidden_size` overflows a 32-bit signed integer.
pub const MAX_HIDDEN_SIZE: usize = (i32::MAX / 4) as usize;

/// Forward-pass intermediate activations for backpropagation through time.
#[derive(Clone, Debug)]
pub struct ForwardCache {
    pub(crate) i_gate: Vec<Vec<f64>>,
    pub(crate) f_gate: Vec<Vec<f64>>,
    pub(crate) g_gate: Vec<Vec<f64>>,
    pub(crate) o_gate: Vec<Vec<f64>>,
    pub(crate) c: Vec<Vec<f64>>,
    pub(crate) tanh_c: Vec<Vec<f64>>,
    pub(crate) h: Vec<Vec<f64>>,
    pub(crate) d_gates: Vec<Vec<f64>>,
}

impl ForwardCache {
    /// Allocates a forward cache for `steps` time steps and `hidden_size` hidden units.
    pub fn of(steps: usize, hidden_size: usize) -> Self {
        Self {
            i_gate: vec![vec![0.0; hidden_size]; steps],
            f_gate: vec![vec![0.0; hidden_size]; steps],
            g_gate: vec![vec![0.0; hidden_size]; steps],
            o_gate: vec![vec![0.0; hidden_size]; steps],
            c: vec![vec![0.0; hidden_size]; steps],
            tanh_c: vec![vec![0.0; hidden_size]; steps],
            h: vec![vec![0.0; hidden_size]; steps],
            d_gates: vec![vec![0.0; 4 * hidden_size]; steps],
        }
    }
}

/// Parameter gradient accumulators for an [`LstmLayer`].
#[derive(Clone, Debug)]
pub struct LstmGradients {
    pub(crate) dw: Vec<Vec<f64>>,
    pub(crate) du: Vec<Vec<f64>>,
    pub(crate) db: Vec<f64>,
}

impl LstmGradients {
    /// Allocates zeroed gradient accumulators for `(input_size, hidden_size)`.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            dw: vec![vec![0.0; input_size]; 4 * hidden_size],
            du: vec![vec![0.0; hidden_size]; 4 * hidden_size],
            db: vec![0.0; 4 * hidden_size],
        }
    }

    /// Zeros all accumulated parameter gradients.
    pub fn zero(&mut self) {
        for row in &mut self.dw {
            row.fill(0.0);
        }
        for row in &mut self.du {
            row.fill(0.0);
        }
        self.db.fill(0.0);
    }

    /// Input-to-gate weight gradients `[4 * hidden_size][input_size]`.
    pub fn dw(&self) -> &[Vec<f64>] {
        &self.dw
    }

    /// Recurrent-to-gate weight gradients `[4 * hidden_size][hidden_size]`.
    pub fn du(&self) -> &[Vec<f64>] {
        &self.du
    }

    /// Gate bias gradients `[4 * hidden_size]`.
    pub fn db(&self) -> &[f64] {
        &self.db
    }
}

/// Single-direction LSTM layer with stacked gates `(i, f, g, o)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LstmLayer {
    input_size: usize,
    hidden_size: usize,
    w: Vec<Vec<f64>>,
    u: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl LstmLayer {
    /// Initializes a new LSTM layer with Xavier-uniform weights and unit forget-gate biases.
    pub fn new(input_size: usize, hidden_size: usize, rng: &mut JavaRng) -> Result<Self, NeuralError> {
        if input_size == 0 || hidden_size == 0 || hidden_size > MAX_HIDDEN_SIZE {
            return Err(NeuralError::InvalidArgument(format!(
                "inputSize ({input_size}) must be positive and hiddenSize ({hidden_size}) must be in [1, {MAX_HIDDEN_SIZE}]"
            )));
        }
        let w_limit = (6.0 / (input_size as f64 + 4.0 * hidden_size as f64)).sqrt();
        let u_limit = (6.0 / (hidden_size as f64 + 4.0 * hidden_size as f64)).sqrt();
        let mut w = vec![vec![0.0; input_size]; 4 * hidden_size];
        let mut u = vec![vec![0.0; hidden_size]; 4 * hidden_size];
        let mut b = vec![0.0; 4 * hidden_size];
        for r in 0..(4 * hidden_size) {
            for val in &mut w[r] {
                *val = (rng.next_double() * 2.0 - 1.0) * w_limit;
            }
            for val in &mut u[r] {
                *val = (rng.next_double() * 2.0 - 1.0) * u_limit;
            }
        }
        for val in &mut b[hidden_size..(2 * hidden_size)] {
            *val = 1.0;
        }
        Ok(Self { input_size, hidden_size, w, u, b })
    }

    /// Constructs an LSTM layer from pre-existing weight matrices and bias vector.
    pub fn of_weights(
        input_size: usize,
        hidden_size: usize,
        w: Vec<Vec<f64>>,
        u: Vec<Vec<f64>>,
        b: Vec<f64>,
    ) -> Result<Self, NeuralError> {
        if input_size == 0 || hidden_size == 0 || hidden_size > MAX_HIDDEN_SIZE {
            return Err(NeuralError::InvalidArgument(format!(
                "inputSize ({input_size}) must be positive and hiddenSize ({hidden_size}) must be in [1, {MAX_HIDDEN_SIZE}]"
            )));
        }
        let gates = 4 * hidden_size;
        if w.len() != gates || u.len() != gates || b.len() != gates {
            return Err(NeuralError::InvalidArgument("gate row count mismatch".into()));
        }
        for r in 0..gates {
            if w[r].len() != input_size || u[r].len() != hidden_size {
                return Err(NeuralError::InvalidArgument("gate column count mismatch".into()));
            }
        }
        Ok(Self { input_size, hidden_size, w, u, b })
    }

    /// Width of each input step vector.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Width of the hidden state and cell state.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Input-to-gate weights `[4 * hidden_size][input_size]`.
    pub fn w(&self) -> &[Vec<f64>] {
        &self.w
    }

    /// Mutable input-to-gate weights `[4 * hidden_size][input_size]`.
    pub fn w_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.w
    }

    /// Recurrent-to-gate weights `[4 * hidden_size][hidden_size]`.
    pub fn u(&self) -> &[Vec<f64>] {
        &self.u
    }

    /// Mutable recurrent-to-gate weights `[4 * hidden_size][hidden_size]`.
    pub fn u_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.u
    }

    /// Gate biases `[4 * hidden_size]`.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Mutable gate biases `[4 * hidden_size]`.
    pub fn b_mut(&mut self) -> &mut [f64] {
        &mut self.b
    }

    /// Allocates a new gradient accumulator matching this layer's dimensions.
    pub fn new_gradients(&self) -> LstmGradients {
        LstmGradients::new(self.input_size, self.hidden_size)
    }

    /// Runs the LSTM forward over `xs`, optionally populating `cache` for backpropagation.
    pub fn run(&self, xs: &[Vec<f64>], mut cache: Option<&mut ForwardCache>) -> Result<Vec<Vec<f64>>, NeuralError> {
        if xs.is_empty() {
            return Err(NeuralError::InvalidArgument("xs must not be empty".into()));
        }
        let steps = xs.len();
        let h_size = self.hidden_size;
        let in_size = self.input_size;
        let mut h = vec![vec![0.0; h_size]; steps];
        let mut c_prev = vec![0.0; h_size];
        let mut c_curr = vec![0.0; h_size];
        let mut gates = vec![0.0; 4 * h_size];

        for t in 0..steps {
            let x = &xs[t];
            gates.copy_from_slice(&self.b);
            for r in 0..(4 * h_size) {
                let w_row = &self.w[r];
                let mut sum = 0.0;
                for k in 0..in_size {
                    sum += w_row[k] * x[k];
                }
                if t > 0 {
                    let u_row = &self.u[r];
                    let h_prev = &h[t - 1];
                    for k in 0..h_size {
                        sum += u_row[k] * h_prev[k];
                    }
                }
                gates[r] += sum;
            }
            for j in 0..h_size {
                let ig = sigmoid(gates[j]);
                let fg = sigmoid(gates[h_size + j]);
                let gg = gates[2 * h_size + j].tanh();
                let og = sigmoid(gates[3 * h_size + j]);
                let cj = fg * c_prev[j] + ig * gg;
                let tanh_cj = cj.tanh();
                let hj = og * tanh_cj;
                c_curr[j] = cj;
                h[t][j] = hj;
                if let Some(ref mut c) = cache {
                    c.i_gate[t][j] = ig;
                    c.f_gate[t][j] = fg;
                    c.g_gate[t][j] = gg;
                    c.o_gate[t][j] = og;
                    c.c[t][j] = cj;
                    c.tanh_c[t][j] = tanh_cj;
                    c.h[t][j] = hj;
                }
            }
            std::mem::swap(&mut c_prev, &mut c_curr);
        }
        Ok(h)
    }

    /// Backpropagates `d_hs` through time, writing input gradients to `d_xs` and accumulating parameter gradients into `grads`.
    pub fn backward(
        &self,
        xs: &[Vec<f64>],
        cache: &mut ForwardCache,
        d_hs: &[Vec<f64>],
        d_xs: &mut [Vec<f64>],
        grads: &mut LstmGradients,
    ) {
        let steps = xs.len();
        let h_size = self.hidden_size;
        let in_size = self.input_size;
        let gates_size = 4 * h_size;
        let mut dh_next = vec![0.0; h_size];
        let mut dc_next = vec![0.0; h_size];

        for t in (0..steps).rev() {
            let d_gates = &mut cache.d_gates[t];
            let dx = &mut d_xs[t];
            dx.fill(0.0);
            for j in 0..h_size {
                let dh = d_hs[t][j] + dh_next[j];
                let og = cache.o_gate[t][j];
                let tanh_c = cache.tanh_c[t][j];
                let dc = dh * og * (1.0 - tanh_c * tanh_c) + dc_next[j];

                let ig = cache.i_gate[t][j];
                let fg = cache.f_gate[t][j];
                let gg = cache.g_gate[t][j];
                let c_prev = if t > 0 { cache.c[t - 1][j] } else { 0.0 };

                let di = dc * gg * ig * (1.0 - ig);
                let df = dc * c_prev * fg * (1.0 - fg);
                let dg = dc * ig * (1.0 - gg * gg);
                let d_o = dh * tanh_c * og * (1.0 - og);

                d_gates[j] = di;
                d_gates[h_size + j] = df;
                d_gates[2 * h_size + j] = dg;
                d_gates[3 * h_size + j] = d_o;

                dc_next[j] = dc * fg;
            }
            dh_next.fill(0.0);
            for r in 0..gates_size {
                let dg = d_gates[r];
                grads.db[r] += dg;
                let w_row = &self.w[r];
                for k in 0..in_size {
                    dx[k] += w_row[k] * dg;
                }
                let u_row = &self.u[r];
                for k in 0..h_size {
                    dh_next[k] += u_row[k] * dg;
                }
            }
        }

        let mut block_end = steps as isize - 1;
        while block_end >= 0 {
            let block_start = (block_end - ACCUMULATE_BLOCK as isize + 1).max(0) as usize;
            let end_usize = block_end as usize;
            for r in 0..gates_size {
                let dw_row = &mut grads.dw[r];
                let du_row = &mut grads.du[r];
                for t in (block_start..=end_usize).rev() {
                    let dg = cache.d_gates[t][r];
                    let x = &xs[t];
                    for k in 0..in_size {
                        dw_row[k] += dg * x[k];
                    }
                    if t > 0 {
                        let h_prev = &cache.h[t - 1];
                        for k in 0..h_size {
                            du_row[k] += dg * h_prev[k];
                        }
                    }
                }
            }
            block_end -= ACCUMULATE_BLOCK as isize;
        }
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT_SIZE: usize = 3;
    const HIDDEN_SIZE: usize = 4;
    const STEPS: usize = 5;
    const EPSILON: f64 = 1e-6;
    const TOLERANCE: f64 = 1e-6;

    fn random_sequence(rng: &mut JavaRng) -> Vec<Vec<f64>> {
        let mut xs = vec![vec![0.0; INPUT_SIZE]; STEPS];
        for row in &mut xs {
            for val in row {
                *val = rng.next_gaussian() * 0.5;
            }
        }
        xs
    }

    fn random_projection(rng: &mut JavaRng) -> Vec<Vec<f64>> {
        let mut proj = vec![vec![0.0; HIDDEN_SIZE]; STEPS];
        for row in &mut proj {
            for val in row {
                *val = rng.next_gaussian() * 0.5;
            }
        }
        proj
    }

    fn loss(layer: &LstmLayer, xs: &[Vec<f64>], proj: &[Vec<f64>]) -> f64 {
        let h = layer.run(xs, None).unwrap();
        let mut total = 0.0;
        for t in 0..xs.len() {
            for j in 0..HIDDEN_SIZE {
                total += proj[t][j] * h[t][j];
            }
        }
        total
    }

    fn assert_close(analytic: f64, numerical: f64, what: &str) {
        let scale = 1.0f64.max(analytic.abs().max(numerical.abs()));
        assert!((analytic - numerical).abs() <= TOLERANCE * scale, "{what}: analytic={analytic} numerical={numerical}");
    }

    #[test]
    fn input_and_parameter_gradients_match_finite_differences() {
        let mut rng = JavaRng::new(42);
        let mut layer = LstmLayer::new(INPUT_SIZE, HIDDEN_SIZE, &mut JavaRng::new(7)).unwrap();
        let mut xs = random_sequence(&mut rng);
        let proj = random_projection(&mut rng);

        let mut cache = ForwardCache::of(STEPS, HIDDEN_SIZE);
        layer.run(&xs, Some(&mut cache)).unwrap();
        let mut d_xs = vec![vec![0.0; INPUT_SIZE]; STEPS];
        let mut grads = layer.new_gradients();
        layer.backward(&xs, &mut cache, &proj, &mut d_xs, &mut grads);

        for t in 0..STEPS {
            for k in 0..INPUT_SIZE {
                let orig = xs[t][k];
                xs[t][k] = orig + EPSILON;
                let plus = loss(&layer, &xs, &proj);
                xs[t][k] = orig - EPSILON;
                let minus = loss(&layer, &xs, &proj);
                xs[t][k] = orig;
                let num = (plus - minus) / (2.0 * EPSILON);
                assert_close(d_xs[t][k], num, &format!("d_xs[{t}][{k}]"));
            }
        }

        for r in 0..(4 * HIDDEN_SIZE) {
            for k in 0..INPUT_SIZE {
                let orig = layer.w[r][k];
                layer.w[r][k] = orig + EPSILON;
                let plus = loss(&layer, &xs, &proj);
                layer.w[r][k] = orig - EPSILON;
                let minus = loss(&layer, &xs, &proj);
                layer.w[r][k] = orig;
                let num = (plus - minus) / (2.0 * EPSILON);
                assert_close(grads.dw()[r][k], num, &format!("dw[{r}][{k}]"));
            }
            for k in 0..HIDDEN_SIZE {
                let orig = layer.u[r][k];
                layer.u[r][k] = orig + EPSILON;
                let plus = loss(&layer, &xs, &proj);
                layer.u[r][k] = orig - EPSILON;
                let minus = loss(&layer, &xs, &proj);
                layer.u[r][k] = orig;
                let num = (plus - minus) / (2.0 * EPSILON);
                assert_close(grads.du()[r][k], num, &format!("du[{r}][{k}]"));
            }
            let orig = layer.b[r];
            layer.b[r] = orig + EPSILON;
            let plus = loss(&layer, &xs, &proj);
            layer.b[r] = orig - EPSILON;
            let minus = loss(&layer, &xs, &proj);
            layer.b[r] = orig;
            let num = (plus - minus) / (2.0 * EPSILON);
            assert_close(grads.db()[r], num, &format!("db[{r}]"));
        }
    }

    #[test]
    fn rejects_empty_sequence_and_zeros_gradients() {
        let layer = LstmLayer::new(INPUT_SIZE, HIDDEN_SIZE, &mut JavaRng::new(1)).unwrap();
        assert!(layer.run(&[], None).is_err());
        let mut grads = layer.new_gradients();
        grads.db[0] = 1.0;
        grads.zero();
        assert!(grads.db().iter().all(|&v| v == 0.0));
    }
}
