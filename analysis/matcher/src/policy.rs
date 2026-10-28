//! Opt-in edit-distance and phonetic matching policy shared by analyzers.

use crate::phonetic::{double_metaphone, soundex};

/// An invalid [`FuzzyMatchPolicy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuzzyPolicyError(&'static str);

impl std::fmt::Display for FuzzyPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for FuzzyPolicyError {}

/// How a candidate may differ from an exact string match.
///
/// Distance is measured in Unicode scalar values with adjacent transposition
/// counted as one edit. Exact matches always score 1.0. Other matches lose
/// `edit_penalty` per edit; a phonetic match counts as one more edit than the
/// configured maximum. A policy with zero distance and no phonetic fallback is
/// disabled, preserving the exact-only behavior of its caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FuzzyMatchPolicy {
    max_edit_distance: u8,
    min_token_len: usize,
    phonetic_fallback: bool,
    edit_penalty: f32,
}

impl FuzzyMatchPolicy {
    /// Creates a checked policy allowing at most two edits.
    pub fn new(
        max_edit_distance: u8,
        min_token_len: usize,
        phonetic_fallback: bool,
        edit_penalty: f32,
    ) -> Result<Self, FuzzyPolicyError> {
        if max_edit_distance > 2 {
            return Err(FuzzyPolicyError("fuzzy edit distance must be in 0..=2"));
        }
        if !edit_penalty.is_finite() || !(0.0..=1.0).contains(&edit_penalty) {
            return Err(FuzzyPolicyError("fuzzy edit penalty must be finite and in [0, 1]"));
        }
        Ok(Self { max_edit_distance, min_token_len, phonetic_fallback, edit_penalty })
    }

    /// Whether anything beyond an exact string match is admitted.
    pub fn is_enabled(&self) -> bool {
        self.max_edit_distance != 0 || self.phonetic_fallback
    }

    pub fn max_edit_distance(&self) -> u8 {
        self.max_edit_distance
    }

    pub fn min_token_len(&self) -> usize {
        self.min_token_len
    }

    pub fn phonetic_fallback(&self) -> bool {
        self.phonetic_fallback
    }

    pub fn edit_penalty(&self) -> f32 {
        self.edit_penalty
    }

    /// A deterministic hash of every policy field.
    pub fn fingerprint(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        h.u8(self.max_edit_distance).usize(self.min_token_len).bool(self.phonetic_fallback).f32(self.edit_penalty);
        h.finish()
    }

    /// Scores a candidate, or rejects it when the policy does not admit it.
    pub fn score(&self, query: &str, candidate: &str) -> Option<f32> {
        if query == candidate {
            return Some(1.0);
        }
        if !self.is_enabled()
            || query.is_empty()
            || candidate.is_empty()
            || query.chars().count() < self.min_token_len
            || candidate.chars().count() < self.min_token_len
        {
            return None;
        }
        if let Some(distance) = damerau_distance(query, candidate, self.max_edit_distance) {
            let penalty = f32::from(distance) * self.edit_penalty;
            return Some((1.0 - penalty).clamp(0.0, 1.0));
        }
        if self.phonetic_fallback && phonetic_equal(query, candidate) {
            let penalty = (f32::from(self.max_edit_distance) + 1.0) * self.edit_penalty;
            return Some((1.0 - penalty).clamp(0.0, 1.0));
        }
        None
    }
}

impl Default for FuzzyMatchPolicy {
    fn default() -> Self {
        Self { max_edit_distance: 0, min_token_len: 4, phonetic_fallback: false, edit_penalty: 0.1 }
    }
}

fn phonetic_equal(a: &str, b: &str) -> bool {
    let sa = soundex(a);
    let sb = soundex(b);
    if sa[0] != b'0' && sa == sb {
        return true;
    }
    let (pa, aa) = double_metaphone(a);
    let (pb, ab) = double_metaphone(b);
    let present = |key: &[u8; 8]| key.iter().any(|&c| c != 0);
    [pa == pb && present(&pa), pa == ab && present(&pa), aa == pb && present(&aa), aa == ab && present(&aa)]
        .into_iter()
        .any(std::convert::identity)
}

pub(crate) fn damerau_distance(a: &str, b: &str, max: u8) -> Option<u8> {
    let max = usize::from(max);
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    let inf = max + 1;
    let mut two = vec![inf; b.len() + 1];
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![inf; b.len() + 1];
    for (i, &ac) in a.iter().enumerate() {
        curr.fill(inf);
        curr[0] = i + 1;
        let lo = (i + 1).saturating_sub(max).max(1);
        let hi = (i + 1).saturating_add(max).min(b.len());
        for j in lo..=hi {
            let substitution = prev[j - 1] + usize::from(ac != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(substitution);
            if i > 0 && j > 1 && ac == b[j - 2] && a[i - 1] == b[j - 1] {
                curr[j] = curr[j].min(two[j - 2] + 1);
            }
        }
        if curr.iter().copied().min().unwrap_or(inf) > max {
            return None;
        }
        std::mem::swap(&mut two, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }
    (prev[b.len()] <= max).then_some(prev[b.len()] as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_validates_and_scores_edits_and_length() {
        assert!(FuzzyMatchPolicy::new(3, 4, false, 0.1).is_err());
        assert!(FuzzyMatchPolicy::new(2, 4, false, f32::NAN).is_err());
        let exact = FuzzyMatchPolicy::default();
        assert_eq!(exact.score("same", "same"), Some(1.0));
        assert_eq!(exact.score("same", "smoe"), None);

        let fuzzy = FuzzyMatchPolicy::new(2, 4, false, 0.1).unwrap();
        assert_eq!(fuzzy.score("Califronia", "California"), Some(0.9));
        assert_eq!(fuzzy.score("abc", "acb"), None);
        assert_eq!(fuzzy.score("ab", "abcd"), None);
        assert_eq!(fuzzy.score("a🙂b", "ab"), None);
        assert_eq!(fuzzy.score("abcd🙂", "abdc🙂"), Some(0.9));
    }

    #[test]
    fn phonetic_fallback_does_not_equate_empty_keys() {
        let fuzzy = FuzzyMatchPolicy::new(0, 4, true, 0.1).unwrap();
        assert_eq!(fuzzy.score("Kathryn", "Catherine"), Some(0.9));
        assert_eq!(fuzzy.score("", ""), Some(1.0));
        assert_eq!(fuzzy.score("🙂🙂🙂🙂", "🙃🙃🙃🙃"), None);
    }
}
