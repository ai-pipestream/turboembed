//! Exact linear-chain CRF scoring (`CrfScoreTable` and `CrfScorer`).
//!
//! `CrfScoreTable` stores exact sums of finite IEEE-754 `f64` values as
//! scaled 128-bit two's-complement integers when all stored values share a
//! common binary exponent window of under 128 bits, and promotes to a
//! two's-complement `i32`-word representation anchored at `2^-1074` when
//! dynamic range exceeds 128 bits. `difference(a, b)` returns the
//! round-half-to-even `f64` value of `table[a] - table[b]`.
//!
//! `CrfScorer` uses `CrfScoreTable` for exact Viterbi path comparison and
//! backward-forward marginal computation so that large path scores that share
//! high-order bits do not lose low-order transition or emission differences.

mod table;
#[cfg(test)]
mod tests;
mod words;

use crate::neural::NeuralError;

pub use table::CrfScoreTable;

/// Linear-chain CRF decoding, marginal probabilities, and negative log-likelihood gradients.
pub struct CrfScorer;

impl CrfScorer {
    /// The tag count shared by every input, or an error naming the first dimension that breaks it:
    /// `start` and `end` set the tag count, every `emissions` row and every `transitions` row and
    /// its width must match it, and a sequence to score needs at least one tag.
    fn check_dimensions(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
    ) -> Result<usize, NeuralError> {
        let num_tags = start.len();
        let bad = |detail: String| NeuralError::InvalidArgument(detail);
        if end.len() != num_tags {
            return Err(bad(format!("CRF end weights {} != {} tags", end.len(), num_tags)));
        }
        if !emissions.is_empty() && num_tags == 0 {
            return Err(bad("CRF scoring with no tags".into()));
        }
        if transitions.len() != num_tags || transitions.iter().any(|row| row.len() != num_tags) {
            return Err(bad(format!("CRF transitions must be {num_tags}x{num_tags}")));
        }
        if let Some(t) = emissions.iter().position(|row| row.len() != num_tags) {
            return Err(bad(format!("CRF emissions at {t} hold {} of {num_tags} tags", emissions[t].len())));
        }
        Ok(num_tags)
    }

    /// Finds the highest-scoring tag index sequence by Viterbi decoding.
    pub fn viterbi(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
    ) -> Result<Vec<usize>, NeuralError> {
        let steps = emissions.len();
        if steps == 0 {
            return Ok(Vec::new());
        }
        let num_tags = Self::check_dimensions(emissions, transitions, start, end)?;
        let mut backpointers = vec![vec![0usize; num_tags]; steps];
        let mut previous = CrfScoreTable::new(num_tags + 1);
        let mut current = CrfScoreTable::new(num_tags + 1);
        let mut candidate = CrfScoreTable::new(1);
        let mut exponents = [0i32; 4];
        let mut mantissas = [0i64; 4];

        for j in 0..num_tags {
            let mut count = 0;
            count = previous.add_term(&mut exponents, &mut mantissas, count, j, start[j])?;
            count = previous.add_term(&mut exponents, &mut mantissas, count, j, emissions[0][j])?;
            if steps == 1 {
                count = previous.add_term(&mut exponents, &mut mantissas, count, j, end[j])?;
            }
            previous.store(j, &exponents, &mantissas, count);
        }

        for t in 1..steps {
            let is_last = t == steps - 1;
            current.reset(num_tags + 1);
            for k in 0..num_tags {
                let mut best_prev = 0usize;
                for j in 0..num_tags {
                    candidate.reset(1);
                    candidate.copy_from(0, &previous, j);
                    let mut count = 1;
                    count = candidate.add_term(&mut exponents, &mut mantissas, count, 0, transitions[j][k])?;
                    count = candidate.add_term(&mut exponents, &mut mantissas, count, 0, emissions[t][k])?;
                    if is_last {
                        count = candidate.add_term(&mut exponents, &mut mantissas, count, 0, end[k])?;
                    }
                    candidate.store_with_base(0, 0, &exponents, &mantissas, count);
                    if j == 0 {
                        current.copy_from(k, &candidate, 0);
                    } else {
                        current.copy_from(num_tags, &candidate, 0);
                        if current.difference(num_tags, k) > 0.0 {
                            current.copy_from(k, &candidate, 0);
                            best_prev = j;
                        }
                    }
                }
                backpointers[t][k] = best_prev;
            }
            std::mem::swap(&mut previous, &mut current);
        }

        let mut best_last = 0usize;
        for j in 1..num_tags {
            if previous.difference(j, best_last) > 0.0 {
                best_last = j;
            }
        }
        let mut path = vec![0usize; steps];
        path[steps - 1] = best_last;
        for t in (1..steps).rev() {
            path[t - 1] = backpointers[t][path[t]];
        }
        Ok(path)
    }

    /// Computes the marginal posterior probability `P(y_t = k | x)` for every token and tag.
    pub fn marginals(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
    ) -> Result<Vec<Vec<f64>>, NeuralError> {
        let (_, marginals) = Self::infer(emissions, transitions, start, end, None, None)?;
        Ok(marginals)
    }

    /// Computes negative log-likelihood loss and accumulates gradients into `d_emissions`, `d_transitions`, `d_start`, and `d_end`.
    #[allow(clippy::too_many_arguments)]
    pub fn loss_and_gradients(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
        gold: &[usize],
        d_emissions: &mut [Vec<f64>],
        d_transitions: &mut [Vec<f64>],
        d_start: &mut [f64],
        d_end: &mut [f64],
    ) -> Result<f64, NeuralError> {
        let steps = emissions.len();
        if steps == 0 {
            return Ok(0.0);
        }
        let num_tags = Self::check_dimensions(emissions, transitions, start, end)?;
        if gold.len() != steps {
            return Err(NeuralError::InvalidArgument(format!("CRF gold sequence {} for {steps} tokens", gold.len())));
        }
        if let Some(t) = gold.iter().position(|&g| g >= num_tags) {
            return Err(NeuralError::InvalidArgument(format!("CRF gold tag {} at {t} of {num_tags} tags", gold[t])));
        }
        let (loss, marginals) = Self::infer(emissions, transitions, start, end, Some(gold), Some(d_transitions))?;
        for t in 0..steps {
            for k in 0..num_tags {
                d_emissions[t][k] += marginals[t][k];
            }
            d_emissions[t][gold[t]] -= 1.0;
            if t > 0 {
                d_transitions[gold[t - 1]][gold[t]] -= 1.0;
            }
        }
        for k in 0..num_tags {
            d_start[k] += marginals[0][k];
            d_end[k] += marginals[steps - 1][k];
        }
        d_start[gold[0]] -= 1.0;
        d_end[gold[steps - 1]] -= 1.0;
        Ok(loss)
    }

    fn infer(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
        gold: Option<&[usize]>,
        mut transition_grads: Option<&mut [Vec<f64>]>,
    ) -> Result<(f64, Vec<Vec<f64>>), NeuralError> {
        let steps = emissions.len();
        if steps == 0 {
            return Ok((0.0, Vec::new()));
        }
        let num_tags = Self::check_dimensions(emissions, transitions, start, end)?;
        let mut best: Vec<CrfScoreTable> = (0..steps).map(|_| CrfScoreTable::new(num_tags)).collect();
        let mut log_mass = vec![vec![0.0f64; num_tags]; steps];
        let mut exponents = [0i32; 4];
        let mut mantissas = [0i64; 4];

        for k in 0..num_tags {
            let count = best[steps - 1].add_term(&mut exponents, &mut mantissas, 0, k, end[k])?;
            best[steps - 1].store(k, &exponents, &mantissas, count);
        }

        let mut next = CrfScoreTable::new(num_tags);
        let mut candidates = CrfScoreTable::new(num_tags);
        let mut weights = vec![0.0f64; num_tags];
        for t in (0..steps - 1).rev() {
            next.reset(num_tags);
            for k in 0..num_tags {
                next.copy_from(k, &best[t + 1], k);
                let count = next.add_term(&mut exponents, &mut mantissas, 1, k, emissions[t + 1][k])?;
                next.store_with_base(k, k, &exponents, &mantissas, count);
            }
            let mut largest_mass = f64::NEG_INFINITY;
            for j in 0..num_tags {
                candidates.reset(num_tags);
                let mut best_next = 0usize;
                for k in 0..num_tags {
                    candidates.copy_from(k, &next, k);
                    let count = candidates.add_term(&mut exponents, &mut mantissas, 1, k, transitions[j][k])?;
                    candidates.store_with_base(k, k, &exponents, &mantissas, count);
                    if k > 0 && candidates.difference(k, best_next) > 0.0 {
                        best_next = k;
                    }
                }
                best[t].copy_from(j, &candidates, best_next);
                for k in 0..num_tags {
                    weights[k] = candidates.difference(k, best_next) + log_mass[t + 1][k];
                }
                log_mass[t][j] = Self::log_sum_exp(&weights);
                if log_mass[t][j] > largest_mass {
                    largest_mass = log_mass[t][j];
                }
            }
            for j in 0..num_tags {
                log_mass[t][j] -= largest_mass;
            }
        }

        let mut probabilities = vec![vec![0.0f64; num_tags]; steps];
        candidates.reset(num_tags);
        let mut best_first = 0usize;
        for k in 0..num_tags {
            candidates.copy_from(k, &best[0], k);
            let mut count = 1;
            count = candidates.add_term(&mut exponents, &mut mantissas, count, k, start[k])?;
            count = candidates.add_term(&mut exponents, &mut mantissas, count, k, emissions[0][k])?;
            candidates.store_with_base(k, k, &exponents, &mantissas, count);
            if k > 0 && candidates.difference(k, best_first) > 0.0 {
                best_first = k;
            }
        }
        for k in 0..num_tags {
            probabilities[0][k] = candidates.difference(k, best_first) + log_mass[0][k];
        }
        let mut loss = Self::normalize_in_place(&mut probabilities[0], gold.map(|g| g[0]));
        let mut conditional = vec![0.0f64; num_tags];
        let mut candidate = CrfScoreTable::new(2);
        for t in 0..steps - 1 {
            candidates.reset(num_tags);
            for k in 0..num_tags {
                candidates.copy_from(k, &best[t + 1], k);
                let count = candidates.add_term(&mut exponents, &mut mantissas, 1, k, emissions[t + 1][k])?;
                candidates.store_with_base(k, k, &exponents, &mantissas, count);
            }
            for j in 0..num_tags {
                for k in 0..num_tags {
                    candidate.reset(2);
                    candidate.copy_from(0, &candidates, k);
                    let count = candidate.add_term(&mut exponents, &mut mantissas, 1, 0, transitions[j][k])?;
                    candidate.store_with_base(0, 0, &exponents, &mantissas, count);
                    candidate.copy_from(1, &best[t], j);
                    conditional[k] = candidate.difference(0, 1) + log_mass[t + 1][k];
                }
                let selected_gold = gold.and_then(|g| if g[t] == j { Some(g[t + 1]) } else { None });
                loss += Self::normalize_in_place(&mut conditional, selected_gold);
                for k in 0..num_tags {
                    let probability = probabilities[t][j] * conditional[k];
                    probabilities[t + 1][k] += probability;
                    if let Some(tg) = transition_grads.as_deref_mut() {
                        tg[j][k] += probability;
                    }
                }
            }
            let mut total = 0.0f64;
            for &value in &probabilities[t + 1] {
                total += value;
            }
            for k in 0..num_tags {
                probabilities[t + 1][k] /= total;
            }
        }
        Ok((loss, probabilities))
    }

    fn normalize_in_place(values: &mut [f64], gold: Option<usize>) -> f64 {
        let mut max = f64::NEG_INFINITY;
        for &v in values.iter() {
            if v > max {
                max = v;
            }
        }
        let selected = gold.map_or(max, |g| values[g]);
        let mut total = 0.0f64;
        for v in values.iter_mut() {
            *v = (*v - max).exp();
            total += *v;
        }
        for v in values.iter_mut() {
            *v /= total;
        }
        if gold.is_some() { (max - selected) + total.ln() } else { 0.0 }
    }

    fn log_sum_exp(xs: &[f64]) -> f64 {
        let mut max = f64::NEG_INFINITY;
        for &x in xs {
            if x > max {
                max = x;
            }
        }
        let mut sum = 0.0f64;
        for &x in xs {
            sum += (x - max).exp();
        }
        max + sum.ln()
    }
}
