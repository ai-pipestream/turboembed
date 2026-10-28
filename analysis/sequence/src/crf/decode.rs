use super::{CrfError, CrfLattice, CrfPath, Result, finite};

#[derive(Clone)]
struct Cell {
    score: f64,
    previous: usize,
    rank: usize,
}
impl CrfLattice {
    /// Exact k-best constrained Viterbi using k backpointers per cell.
    /// Ties use label and predecessor order, independent of hash iteration.
    pub fn best_paths(&self, k: usize) -> Result<Vec<CrfPath>> {
        let (t, n) = (self.emissions.len(), self.mask.labels());
        if k == 0 {
            return Ok(Vec::new());
        }
        if k > 4096 || t.checked_mul(n).and_then(|x| x.checked_mul(k)).is_none_or(|x| x > 16_000_000) {
            return Err(CrfError::Capacity);
        }
        if t == 0 {
            return Ok(vec![CrfPath { labels: Vec::new(), log_probability: 0.0, token_probabilities: Vec::new() }]);
        }
        let mut cells = vec![vec![Vec::<Cell>::new(); n]; t];
        for (y, cell) in cells[0].iter_mut().enumerate() {
            if self.mask.start[y] {
                cell.push(Cell { score: finite(self.start[y] + self.emissions[0][y])?, previous: 0, rank: 0 });
            }
        }
        for i in 1..t {
            for y in 0..n {
                let mut candidates = Vec::new();
                for (p, previous) in cells[i - 1].iter().enumerate() {
                    if !self.mask.allows_transition(p, y) {
                        continue;
                    }
                    for (rank, prior) in previous.iter().enumerate() {
                        candidates.push(Cell {
                            score: finite(prior.score + self.transitions[p * n + y] + self.emissions[i][y])?,
                            previous: p,
                            rank,
                        });
                    }
                }
                candidates.sort_by(|a, b| {
                    b.score.total_cmp(&a.score).then(a.previous.cmp(&b.previous)).then(a.rank.cmp(&b.rank))
                });
                candidates.truncate(k);
                cells[i][y] = candidates;
            }
        }
        let mut finals = Vec::new();
        for (y, cell) in cells[t - 1].iter().enumerate() {
            if self.mask.end[y] {
                for (rank, c) in cell.iter().enumerate() {
                    finals.push((finite(c.score + self.end[y])?, y, rank));
                }
            }
        }
        finals.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        finals.truncate(k);
        let marginals = self.marginals();
        Ok(finals
            .into_iter()
            .map(|(score, mut label, mut rank)| {
                let mut labels = vec![0; t];
                for i in (0..t).rev() {
                    labels[i] = label;
                    let c = &cells[i][label][rank];
                    label = c.previous;
                    rank = c.rank;
                }
                let token_probabilities = labels.iter().enumerate().map(|(i, &y)| marginals[i][y]).collect();
                CrfPath { labels, log_probability: (score - self.z).min(0.0), token_probabilities }
            })
            .collect())
    }
}
