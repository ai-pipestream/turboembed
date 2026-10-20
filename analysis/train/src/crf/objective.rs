use turbo_sequence::crf::{CrfConstraints, CrfGradient, CrfLattice, CrfScores, CrfSequenceModel};

#[derive(Clone, Debug)]
pub struct IndexedSequence {
    pub features: Vec<Vec<(usize, f64)>>,
    pub outcomes: Vec<usize>,
}
pub struct CrfObjective {
    features: usize,
    mask: CrfConstraints,
    samples: Vec<IndexedSequence>,
    l2: f64,
    dimension: usize,
}
impl CrfObjective {
    pub fn new(features: usize, mask: CrfConstraints, samples: Vec<IndexedSequence>, l2: f64) -> Result<Self, String> {
        let dimension = CrfSequenceModel::parameter_count(mask.labels(), features).map_err(|e| e.to_string())?;
        if !l2.is_finite() || l2 < 0.0 || samples.is_empty() {
            return Err("invalid CRF objective".into());
        }
        for s in &samples {
            if s.features.len().checked_mul(mask.labels()).is_none_or(|n| n > 16_000_000) {
                return Err("sequence capacity exceeded".into());
            }
            if s.features.len() != s.outcomes.len() || !mask.accepts(&s.outcomes) {
                return Err("invalid gold sequence".into());
            }
            if s.features.iter().flatten().any(|&(f, v)| f >= features || !v.is_finite()) {
                return Err("invalid indexed feature".into());
            }
        }
        Ok(Self { features, mask, samples, l2, dimension })
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Complete regularized negative log likelihood and analytic gradient.
    pub fn value_gradient(&self, x: &[f64], g: &mut [f64]) -> Result<f64, String> {
        if x.len() != self.dimension || g.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return Err("invalid objective parameters".into());
        }
        g.fill(0.0);
        let n = self.mask.labels();
        let at = self.features * n;
        let mut loss = 0.0;
        for s in &self.samples {
            let mut emissions = vec![vec![0.0; n]; s.features.len()];
            for (row, out) in s.features.iter().zip(&mut emissions) {
                for &(f, v) in row {
                    for y in 0..n {
                        out[y] += v * x[f * n + y];
                    }
                }
            }
            let lattice = CrfLattice::new(
                CrfScores {
                    emissions: &emissions,
                    transitions: &x[at..at + n * n],
                    start: &x[at + n * n..at + n * n + n],
                    end: &x[at + n * n + n..],
                },
                &self.mask,
            )
            .map_err(|e| e.to_string())?;
            let CrfGradient { loss: value, emission, transition, start, end } =
                lattice.loss_gradient(&s.outcomes).map_err(|e| e.to_string())?;
            loss += value;
            for (row, grad) in s.features.iter().zip(&emission) {
                for &(f, v) in row {
                    for y in 0..n {
                        g[f * n + y] += v * grad[y];
                    }
                }
            }
            for (out, v) in g[at..].iter_mut().zip(transition.iter().chain(&start).chain(&end)) {
                *out += v;
            }
        }
        for (g, &v) in g.iter_mut().zip(x) {
            loss += 0.5 * self.l2 * v * v;
            *g += self.l2 * v;
        }
        if !loss.is_finite() || g.iter().any(|v| !v.is_finite()) {
            return Err("non-finite CRF objective".into());
        }
        Ok(loss)
    }
}
