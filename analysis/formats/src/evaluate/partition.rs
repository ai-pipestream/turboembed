//! `CrossValidationPartitioner`: the folds of n-fold cross-validation.
//!
//! The samples are not shuffled. Fold `t` of `n` tests on the samples at
//! the places `i` with `i % n == t` and trains on the rest, as the
//! partitioner's streams read them. With one partition the training
//! stream skips a sample before each one it returns, as it does before
//! every test sample, so it trains on the samples at odd places, all of
//! which it also tests on; OpenNLP does the same.

/// One fold: the places of its training samples and of its test samples,
/// in the order the streams return them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fold {
    /// The fold's number, from 0: the place of its test samples modulo
    /// the number of partitions.
    pub index: usize,
    pub training: Vec<usize>,
    pub test: Vec<usize>,
}

impl Fold {
    /// The training samples of `samples`.
    pub fn training_of<'s, T>(&self, samples: &'s [T]) -> impl Iterator<Item = &'s T> {
        self.training.iter().map(move |&i| &samples[i])
    }

    /// The test samples of `samples`.
    pub fn test_of<'s, T>(&self, samples: &'s [T]) -> impl Iterator<Item = &'s T> {
        self.test.iter().map(move |&i| &samples[i])
    }
}

/// The folds over `len` samples in `partitions` partitions, one per
/// partition; none for 0 partitions.
#[derive(Clone, Debug)]
pub struct CrossValidation {
    len: usize,
    partitions: usize,
    next: usize,
}

impl CrossValidation {
    pub fn new(len: usize, partitions: usize) -> CrossValidation {
        CrossValidation { len, partitions, next: 0 }
    }
}

impl Iterator for CrossValidation {
    type Item = Fold;

    fn next(&mut self) -> Option<Fold> {
        if self.next >= self.partitions {
            return None;
        }
        let (n, t, len) = (self.partitions, self.next, self.len);
        self.next += 1;
        // `TrainingSampleStream.read`: skip one sample when the place read
        // is a test place, then return the next.
        let mut training = Vec::new();
        let mut at = 0;
        loop {
            if at % n == t {
                at += 1;
            }
            if at >= len {
                break;
            }
            training.push(at);
            at += 1;
        }
        // `TestSampleStream.read`: skip to the next test place.
        let test = (t..len).step_by(n).collect();
        Some(Fold { index: t, training, test })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_by_place() {
        let folds: Vec<Fold> = CrossValidation::new(7, 3).collect();
        assert_eq!(folds.len(), 3);
        assert_eq!(folds[0].test, [0, 3, 6]);
        assert_eq!(folds[0].training, [1, 2, 4, 5]);
        assert_eq!(folds[2].test, [2, 5]);
        assert_eq!(folds[2].training, [0, 1, 3, 4, 6]);
        let samples = ["a", "b", "c", "d", "e", "f", "g"];
        assert_eq!(folds[1].test_of(&samples).copied().collect::<Vec<_>>(), ["b", "e"]);
    }

    #[test]
    fn one_partition_trains_on_odd_places() {
        let folds: Vec<Fold> = CrossValidation::new(5, 1).collect();
        assert_eq!(folds, [Fold { index: 0, training: vec![1, 3], test: vec![0, 1, 2, 3, 4] }]);
        assert_eq!(CrossValidation::new(5, 0).count(), 0);
    }
}
