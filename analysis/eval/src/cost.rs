//! What a pipeline cost: time per stage, tokens embedded, and how long
//! its chunks are.

use serde::{Deserialize, Serialize};

use crate::Result;

/// Wall time of each stage over the whole set, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageTimes {
    pub analysis_ms: f64,
    pub chunking_ms: f64,
    pub embedding_ms: f64,
    pub retrieval_ms: f64,
}

/// The smallest, median, 90th percentile and largest of a set of lengths.
///
/// Percentiles are by nearest rank: with the n values sorted ascending
/// and counted from 1, the p percentile is the value at rank ceil(p n).
/// So each figure is a length some chunk actually has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Distribution {
    pub min: u64,
    pub p50: u64,
    pub p90: u64,
    pub max: u64,
}

impl Distribution {
    /// The distribution of `values`; an error when there are none.
    pub fn of(values: &[u64]) -> Result<Distribution> {
        if values.is_empty() {
            return Err("no values to take a distribution of".into());
        }
        let mut v = values.to_vec();
        v.sort_unstable();
        // ceil(p n) with p in percent, in integers, so no rounding is
        // involved.
        let rank = |percent: usize| (percent * v.len()).div_ceil(100).max(1);
        Ok(Distribution { min: v[0], p50: v[rank(50) - 1], p90: v[rank(90) - 1], max: v[v.len() - 1] })
    }

    /// True when min <= p50 <= p90 <= max.
    pub fn is_ordered(&self) -> bool {
        self.min <= self.p50 && self.p50 <= self.p90 && self.p90 <= self.max
    }
}

/// The cost of one pipeline on one set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    pub wall: StageTimes,
    /// Tokens the embedder was given, queries and chunks together.
    pub tokens_embedded: u64,
    /// Chunks the chunker made over the set's documents.
    pub chunks: u64,
    /// Chunk lengths in UTF-8 bytes of the source text.
    pub chunk_bytes: Distribution,
    /// Chunk lengths in the embedder's tokens.
    pub chunk_tokens: Distribution,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_by_hand() {
        // Sorted: 1 2 3 4 5 6 7 8 9 10 (n = 10).
        // p50: rank ceil(5) = 5, value 5. p90: rank ceil(9) = 9, value 9.
        let v: Vec<u64> = (1..=10).rev().collect();
        assert_eq!(Distribution::of(&v).unwrap(), Distribution { min: 1, p50: 5, p90: 9, max: 10 });
        // n = 3, sorted 10 20 30: p50 rank ceil(1.5) = 2, value 20;
        // p90 rank ceil(2.7) = 3, value 30.
        assert_eq!(Distribution::of(&[30, 10, 20]).unwrap(), Distribution { min: 10, p50: 20, p90: 30, max: 30 });
        // One value: every figure is that value.
        assert_eq!(Distribution::of(&[7]).unwrap(), Distribution { min: 7, p50: 7, p90: 7, max: 7 });
        assert!(Distribution::of(&[]).is_err());
    }
}
