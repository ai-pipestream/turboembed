use super::{CrfConstraints, CrfError, Result, finite, invalid};

/// Dense scores, transition entries in previous-label-major order.
pub struct CrfScores<'a> {
    pub emissions: &'a [Vec<f64>],
    pub transitions: &'a [f64],
    pub start: &'a [f64],
    pub end: &'a [f64],
}
/// Negative log likelihood and its derivatives, in model layout order.
pub struct CrfGradient {
    pub loss: f64,
    pub emission: Vec<Vec<f64>>,
    pub transition: Vec<f64>,
    pub start: Vec<f64>,
    pub end: Vec<f64>,
}
/// Forward/backward tables over only legal paths. Emission row centering
/// avoids losing the normalizer for large common emission offsets.
pub struct CrfLattice {
    pub(crate) emissions: Vec<Vec<f64>>,
    pub(crate) transitions: Vec<f64>,
    pub(crate) start: Vec<f64>,
    pub(crate) end: Vec<f64>,
    pub(crate) mask: CrfConstraints,
    pub(crate) alpha: Vec<Vec<f64>>,
    pub(crate) beta: Vec<Vec<f64>>,
    pub(crate) z: f64,
    offset: f64,
}
pub(crate) fn logsum(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(f64::NEG_INFINITY, |a, b| {
        if a == f64::NEG_INFINITY {
            b
        } else if b == f64::NEG_INFINITY {
            a
        } else {
            let m = a.max(b);
            m + (-(a - b).abs()).exp().ln_1p()
        }
    })
}
impl CrfLattice {
    pub fn new(scores: CrfScores<'_>, mask: &CrfConstraints) -> Result<Self> {
        let n = mask.labels();
        let t = scores.emissions.len();
        if scores.transitions.len() != n * n
            || scores.start.len() != n
            || scores.end.len() != n
            || scores.emissions.iter().any(|r| r.len() != n)
        {
            return Err(invalid("score dimensions"));
        }
        if t.checked_mul(n).is_none_or(|v| v > 16_000_000) {
            return Err(CrfError::Capacity);
        }
        for &x in scores.emissions.iter().flatten().chain(scores.transitions).chain(scores.start).chain(scores.end) {
            finite(x)?;
        }
        let mut emissions = scores.emissions.to_vec();
        let mut offset = 0.0;
        for row in &mut emissions {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            offset = finite(offset + max)?;
            for x in row {
                *x = finite(*x - max)?;
            }
        }
        let mut a = Self {
            emissions,
            transitions: scores.transitions.to_vec(),
            start: scores.start.to_vec(),
            end: scores.end.to_vec(),
            mask: mask.clone(),
            alpha: vec![vec![f64::NEG_INFINITY; n]; t],
            beta: vec![vec![f64::NEG_INFINITY; n]; t],
            z: 0.0,
            offset,
        };
        if t == 0 {
            return Ok(a);
        }
        for y in 0..n {
            if mask.start[y] {
                a.alpha[0][y] = finite(a.start[y] + a.emissions[0][y])?;
            }
            if mask.end[y] {
                a.beta[t - 1][y] = a.end[y];
            }
        }
        for i in 1..t {
            for y in 0..n {
                let sum = logsum(
                    (0..n)
                        .filter(|&p| mask.allows_transition(p, y))
                        .map(|p| a.alpha[i - 1][p] + a.transitions[p * n + y]),
                );
                if sum != f64::NEG_INFINITY {
                    a.alpha[i][y] = finite(sum + a.emissions[i][y])?;
                }
            }
        }
        a.z = logsum((0..n).filter(|&y| mask.end[y]).map(|y| a.alpha[t - 1][y] + a.end[y]));
        if a.z == f64::NEG_INFINITY {
            return Err(CrfError::NoPath);
        }
        finite(a.z)?;
        for i in (0..t - 1).rev() {
            for p in 0..n {
                let sum = logsum(
                    (0..n)
                        .filter(|&y| mask.allows_transition(p, y))
                        .map(|y| a.transitions[p * n + y] + a.emissions[i + 1][y] + a.beta[i + 1][y]),
                );
                if sum != f64::NEG_INFINITY {
                    a.beta[i][p] = finite(sum)?;
                }
            }
        }
        Ok(a)
    }
    pub fn log_partition(&self) -> Result<f64> {
        finite(self.z + self.offset)
    }
    pub fn marginals(&self) -> Vec<Vec<f64>> {
        self.alpha
            .iter()
            .zip(&self.beta)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x + y - self.z).exp().clamp(0.0, 1.0)).collect())
            .collect()
    }
    pub fn transition_marginal(&self, position: usize, from: usize, to: usize) -> Result<f64> {
        let n = self.mask.labels();
        if position == 0 || position >= self.alpha.len() || from >= n || to >= n {
            return Err(invalid("transition position"));
        }
        if !self.mask.allows_transition(from, to) {
            return Ok(0.0);
        }
        Ok((self.alpha[position - 1][from]
            + self.transitions[from * n + to]
            + self.emissions[position][to]
            + self.beta[position][to]
            - self.z)
            .exp()
            .clamp(0.0, 1.0))
    }
    /// Probability of fixing a contiguous sequence of labels. A BIO entity
    /// additionally excludes a continuation immediately after its last token.
    pub fn span_probability(&self, start: usize, labels: &[usize], forbid_next: Option<usize>) -> Result<f64> {
        let end = start.checked_add(labels.len()).ok_or(CrfError::Capacity)?;
        let n = self.mask.labels();
        if labels.is_empty()
            || end > self.alpha.len()
            || labels.iter().any(|&y| y >= n)
            || forbid_next.is_some_and(|y| y >= n)
        {
            return Err(invalid("span labels"));
        }
        let mut score = self.alpha[start][labels[0]];
        for j in 1..labels.len() {
            let (p, y) = (labels[j - 1], labels[j]);
            if !self.mask.allows_transition(p, y) {
                return Ok(0.0);
            }
            score += self.transitions[p * n + y] + self.emissions[start + j][y];
        }
        let last = *labels.last().unwrap();
        let suffix = if end == self.alpha.len() {
            self.beta[end - 1][last]
        } else {
            logsum(
                (0..n)
                    .filter(|&y| Some(y) != forbid_next && self.mask.allows_transition(last, y))
                    .map(|y| self.transitions[last * n + y] + self.emissions[end][y] + self.beta[end][y]),
            )
        };
        Ok((score + suffix - self.z).exp().clamp(0.0, 1.0))
    }
    /// Negative log likelihood and derivatives in emission, transition,
    /// start, end order. Forbidden parameters have exactly zero gradient.
    pub fn loss_gradient(&self, gold: &[usize]) -> Result<CrfGradient> {
        if gold.len() != self.alpha.len() || !self.mask.accepts(gold) {
            return Err(invalid("gold sequence"));
        }
        let n = self.mask.labels();
        let mut emission = self.marginals();
        let mut transition = vec![0.0; n * n];
        let mut start = vec![0.0; n];
        let mut end = vec![0.0; n];
        if gold.is_empty() {
            return Ok(CrfGradient { loss: 0.0, emission, transition, start, end });
        }
        start.copy_from_slice(&emission[0]);
        end.copy_from_slice(emission.last().unwrap());
        start[gold[0]] -= 1.0;
        end[*gold.last().unwrap()] -= 1.0;
        let mut score = self.start[gold[0]] + self.end[*gold.last().unwrap()];
        for (i, &y) in gold.iter().enumerate() {
            score += self.emissions[i][y];
            emission[i][y] -= 1.0;
            if i > 0 {
                score += self.transitions[gold[i - 1] * n + y];
                transition[gold[i - 1] * n + y] -= 1.0;
                for p in 0..n {
                    for q in 0..n {
                        transition[p * n + q] += self.transition_marginal(i, p, q)?;
                    }
                }
            }
        }
        Ok(CrfGradient { loss: finite(self.z - score)?, emission, transition, start, end })
    }
}
