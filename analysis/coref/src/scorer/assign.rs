//! The Hungarian method for the CEAF alignments.

/// The largest total similarity of any perfect matching of a square
/// matrix, by the Hungarian method with potentials, as
/// `CorefScorer.maximumAssignment` computes it.
pub fn maximum_assignment(similarity: &[Vec<f64>]) -> f64 {
    let n = similarity.len();
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut matched_row = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        matched_row[0] = i;
        let mut column = 0;
        let mut minimum = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[column] = true;
            let row = matched_row[column];
            let mut delta = f64::INFINITY;
            let mut next = 0;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cost = -similarity[row - 1][j - 1] - u[row] - v[j];
                if cost < minimum[j] {
                    minimum[j] = cost;
                    way[j] = column;
                }
                if minimum[j] < delta {
                    delta = minimum[j];
                    next = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[matched_row[j]] += delta;
                    v[j] -= delta;
                } else {
                    minimum[j] -= delta;
                }
            }
            column = next;
            if matched_row[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            matched_row[column] = matched_row[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    (1..=n).map(|j| similarity[matched_row[j] - 1][j - 1]).sum()
}

#[cfg(test)]
mod tests {
    use super::maximum_assignment;

    #[test]
    fn finds_the_best_matching_not_the_greedy_one() {
        let m = vec![vec![3.0, 2.0, 0.0], vec![3.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
        assert_eq!(maximum_assignment(&m), 6.0);
        assert_eq!(maximum_assignment(&[]), 0.0);
        assert_eq!(maximum_assignment(&[vec![0.5]]), 0.5);
    }
}
