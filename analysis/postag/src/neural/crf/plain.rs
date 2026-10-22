//! The linear-chain CRF decode in plain f64, the default numerics' host
//! side: Viterbi by max-plus and the marginals by forward-backward in log
//! space. [`CrfScorer`](super::CrfScorer) gives the JVM's bits by summing
//! exactly and is the opt-in parity path; this one rounds each sum as it
//! goes, so it can pick another path only where two paths' scores are
//! within rounding of each other, and runs far faster.

/// `log(sum(exp(v)))` over `v`, from its maximum.
fn log_sum_exp(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let max = v.clone().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return max;
    }
    max + v.map(|x| (x - max).exp()).sum::<f64>().ln()
}

/// The highest-scoring tag sequence; a tie goes to the lower tag index, as
/// in [`CrfScorer::viterbi`](super::CrfScorer::viterbi). The dimensions
/// are the caller's to check.
pub(crate) fn viterbi(emissions: &[Vec<f64>], transitions: &[Vec<f64>], start: &[f64], end: &[f64]) -> Vec<usize> {
    let (steps, tags) = (emissions.len(), start.len());
    if steps == 0 {
        return Vec::new();
    }
    let mut back = vec![0usize; steps * tags];
    let mut prev: Vec<f64> = (0..tags).map(|j| start[j] + emissions[0][j]).collect();
    let mut cur = vec![0.0f64; tags];
    for t in 1..steps {
        for k in 0..tags {
            let (mut best, mut arg) = (prev[0] + transitions[0][k], 0);
            for j in 1..tags {
                let s = prev[j] + transitions[j][k];
                if s > best {
                    (best, arg) = (s, j);
                }
            }
            cur[k] = best + emissions[t][k];
            back[t * tags + k] = arg;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let mut last = 0;
    for j in 1..tags {
        if prev[j] + end[j] > prev[last] + end[last] {
            last = j;
        }
    }
    let mut path = vec![0usize; steps];
    path[steps - 1] = last;
    for t in (1..steps).rev() {
        path[t - 1] = back[t * tags + path[t]];
    }
    path
}

/// `P(y_t = k | x)` for every token and tag, as
/// [`CrfScorer::marginals`](super::CrfScorer::marginals) defines them.
pub(crate) fn marginals(emissions: &[Vec<f64>], transitions: &[Vec<f64>], start: &[f64], end: &[f64]) -> Vec<Vec<f64>> {
    let (steps, tags) = (emissions.len(), start.len());
    if steps == 0 {
        return Vec::new();
    }
    let mut alpha = vec![vec![0.0f64; tags]; steps];
    for j in 0..tags {
        alpha[0][j] = start[j] + emissions[0][j];
    }
    for t in 1..steps {
        for k in 0..tags {
            let a = &alpha[t - 1];
            alpha[t][k] = emissions[t][k] + log_sum_exp((0..tags).map(|j| a[j] + transitions[j][k]));
        }
    }
    let mut beta = vec![vec![0.0f64; tags]; steps];
    beta[steps - 1].copy_from_slice(end);
    for t in (0..steps - 1).rev() {
        for j in 0..tags {
            let (e, b) = (&emissions[t + 1], &beta[t + 1]);
            beta[t][j] = log_sum_exp((0..tags).map(|k| transitions[j][k] + e[k] + b[k]));
        }
    }
    let z = log_sum_exp((0..tags).map(|j| alpha[steps - 1][j] + end[j]));
    alpha.iter().zip(&beta).map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a + b - z).exp()).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::super::CrfScorer;
    use super::*;
    use crate::neural::JavaRng;

    fn random(rng: &mut JavaRng, rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
        (0..rows).map(|_| (0..cols).map(|_| (rng.next_double() * 2.0 - 1.0) * scale).collect()).collect()
    }

    #[test]
    fn plain_decode_agrees_with_the_exact_scorer() {
        let mut rng = JavaRng::new(206);
        for (steps, tags) in [(1, 1), (1, 5), (2, 3), (7, 17), (40, 17), (3, 50)] {
            let e = random(&mut rng, steps, tags, 4.0);
            let tr = random(&mut rng, tags, tags, 2.0);
            let (st, en) = (random(&mut rng, 1, tags, 1.0).remove(0), random(&mut rng, 1, tags, 1.0).remove(0));
            assert_eq!(viterbi(&e, &tr, &st, &en), CrfScorer::viterbi(&e, &tr, &st, &en).unwrap(), "{steps}x{tags}");
            let exact = CrfScorer::marginals(&e, &tr, &st, &en).unwrap();
            for (p, q) in marginals(&e, &tr, &st, &en).iter().flatten().zip(exact.iter().flatten()) {
                assert!((p - q).abs() < 1e-12, "{steps}x{tags}: {p} against {q}");
            }
        }
        assert!(viterbi(&[], &[], &[], &[]).is_empty() && marginals(&[], &[], &[], &[]).is_empty());
    }

    #[test]
    fn ties_go_to_the_lower_tag() {
        let e = vec![vec![0.0; 3]; 4];
        let tr = vec![vec![0.0; 3]; 3];
        let z = vec![0.0; 3];
        assert_eq!(viterbi(&e, &tr, &z, &z), vec![0; 4]);
        assert_eq!(viterbi(&e, &tr, &z, &z), CrfScorer::viterbi(&e, &tr, &z, &z).unwrap());
    }
}
