//! Fuzzy string matching scorers (normalized Levenshtein and Jaro-Winkler)
//! over folded gazetteer match keys and place names.

use std::collections::HashSet;

use crate::entry::GazetteerEntry;
use crate::geocode::by_prior_view;
use crate::index::{GazetteerIndex, with_key};
use crate::store::Entry;

const JARO_WINKLER_PREFIX_SCALE: f64 = 0.1;
const JARO_WINKLER_MAX_PREFIX: usize = 4;
const JARO_WINKLER_BOOST_THRESHOLD: f64 = 0.7;

/// Character-level normalized Levenshtein similarity in `[0.0, 1.0]`.
///
/// Returns `1.0` when both strings are equal (including both empty), and
/// `1.0 - (edit_distance / max(a.chars().count(), b.chars().count()))` otherwise.
pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    let max_len = m.max(n);
    if max_len == 0 {
        return 1.0;
    }
    if m == 0 || n == 0 {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for (i, &ca) in a_chars.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b_chars.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    let dist = prev[n];
    (1.0 - (dist as f64 / max_len as f64)).clamp(0.0, 1.0)
}

/// Character-level Jaro similarity in `[0.0, 1.0]`.
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    if m == 0 || n == 0 {
        return 0.0;
    }
    let match_dist = (m.max(n) / 2).saturating_sub(1);
    let mut a_matched = vec![false; m];
    let mut b_matched = vec![false; n];
    let mut matches = 0usize;

    for (i, &ca) in a_chars.iter().enumerate() {
        let start = i.saturating_sub(match_dist);
        let end = (i + match_dist + 1).min(n);
        for j in start..end {
            if !b_matched[j] && ca == b_chars[j] {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }

    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..m {
        if !a_matched[i] {
            continue;
        }
        while !b_matched[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m_f = matches as f64;
    let t_f = (transpositions as f64) / 2.0;
    ((m_f / m as f64) + (m_f / n as f64) + ((m_f - t_f) / m_f)) / 3.0
}

/// Character-level Jaro-Winkler similarity in `[0.0, 1.0]`.
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let jaro = jaro_similarity(a, b);
    if jaro < JARO_WINKLER_BOOST_THRESHOLD {
        return jaro;
    }
    let prefix = a.chars().zip(b.chars()).take(JARO_WINKLER_MAX_PREFIX).take_while(|(ca, cb)| ca == cb).count();
    (jaro + (prefix as f64) * JARO_WINKLER_PREFIX_SCALE * (1.0 - jaro)).clamp(0.0, 1.0)
}

/// Which string similarity metric [`FuzzyStringScorer`] evaluates on folded
/// gazetteer match keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FuzzyMetric {
    /// Mean of normalized Levenshtein and Jaro-Winkler similarity.
    #[default]
    Combined,
    /// Normalized Levenshtein similarity (`1 - d / max_len`).
    NormalizedLevenshtein,
    /// Jaro-Winkler similarity.
    JaroWinkler,
}

impl FuzzyMetric {
    /// Scores two pre-folded match keys in `[0.0, 1.0]`.
    #[inline]
    pub fn score_folded(self, a: &str, b: &str) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        if a == b {
            return 1.0;
        }
        match self {
            FuzzyMetric::NormalizedLevenshtein => normalized_levenshtein(a, b),
            FuzzyMetric::JaroWinkler => jaro_winkler(a, b),
            FuzzyMetric::Combined => 0.5 * (normalized_levenshtein(a, b) + jaro_winkler(a, b)),
        }
    }
}

/// Scores surface mentions against gazetteer entries by folding both with
/// [`crate::Folder`] and taking the maximum similarity across the entry's
/// canonical name and alternate names.
#[derive(Clone, Copy, Debug)]
pub struct FuzzyStringScorer {
    metric: FuzzyMetric,
    min_similarity: f64,
}

impl Default for FuzzyStringScorer {
    fn default() -> Self {
        FuzzyStringScorer { metric: FuzzyMetric::Combined, min_similarity: 0.80 }
    }
}

impl FuzzyStringScorer {
    /// Creates a fuzzy scorer using `metric` and `min_similarity` in `[0.0, 1.0]`.
    pub fn new(metric: FuzzyMetric, min_similarity: f64) -> FuzzyStringScorer {
        FuzzyStringScorer { metric, min_similarity: min_similarity.clamp(0.0, 1.0) }
    }

    pub fn metric(&self) -> FuzzyMetric {
        self.metric
    }

    pub fn min_similarity(&self) -> f64 {
        self.min_similarity
    }

    /// Scores two raw strings after folding both into gazetteer match keys.
    pub fn score_names(&self, query: &str, candidate: &str) -> f64 {
        with_key(query, |q_key| {
            if q_key.is_empty() {
                return 0.0;
            }
            let mut folder = crate::fold::Folder::new();
            let c_key = folder.fold(candidate);
            self.metric.score_folded(q_key, &c_key)
        })
    }

    /// Scores `query` against a borrowed gazetteer [`Entry`] across its
    /// canonical name and all alternate names, returning a value in `[0.0, 1.0]`.
    pub fn score_entry(&self, query: &str, entry: &Entry<'_>) -> f64 {
        with_key(query, |q_key| {
            if q_key.is_empty() {
                return 0.0;
            }
            let mut folder = crate::fold::Folder::new();
            let mut key_buf = String::new();
            folder.fold_into(entry.name(), &mut key_buf);
            let mut best = self.metric.score_folded(q_key, &key_buf);
            if best >= 1.0 {
                return 1.0;
            }
            for alt in entry.alternate_names() {
                folder.fold_into(alt, &mut key_buf);
                let s = self.metric.score_folded(q_key, &key_buf);
                if s > best {
                    best = s;
                    if best >= 1.0 {
                        return 1.0;
                    }
                }
            }
            best
        })
    }

    /// Scores `query` against an owned [`GazetteerEntry`].
    pub fn score_owned_entry(&self, query: &str, entry: &GazetteerEntry) -> f64 {
        with_key(query, |q_key| {
            if q_key.is_empty() {
                return 0.0;
            }
            let mut folder = crate::fold::Folder::new();
            let mut key_buf = String::new();
            folder.fold_into(entry.name(), &mut key_buf);
            let mut best = self.metric.score_folded(q_key, &key_buf);
            if best >= 1.0 {
                return 1.0;
            }
            for alt in entry.alternate_names() {
                folder.fold_into(alt, &mut key_buf);
                let s = self.metric.score_folded(q_key, &key_buf);
                if s > best {
                    best = s;
                    if best >= 1.0 {
                        return 1.0;
                    }
                }
            }
            best
        })
    }
}

impl GazetteerIndex {
    pub(crate) fn lookup_fuzzy(
        &self,
        name: &str,
        scorer: &FuzzyStringScorer,
        min_similarity: f64,
        limit: usize,
    ) -> Vec<(Entry<'_>, f64)> {
        if limit == 0 {
            return Vec::new();
        }
        let threshold = min_similarity.clamp(0.0, 1.0);
        with_key(name, |q_key| {
            if q_key.is_empty() {
                return Vec::new();
            }
            let mut seen_records: HashSet<u32> = HashSet::new();
            let mut matches: Vec<(Entry<'_>, f64)> = Vec::new();

            // Exact match key hits first (similarity 1.0).
            for &r in self.records_of_key(q_key) {
                if seen_records.insert(r) {
                    matches.push((Entry::new(&self.store, r), 1.0));
                }
            }

            if threshold < 1.0 && matches.len() < limit {
                let mut key_hits: Vec<(u32, f64)> = Vec::new();
                for k in 0..self.names.len() as u32 {
                    let cand_key = self.names.key(k);
                    if cand_key == q_key {
                        continue;
                    }
                    let sim = scorer.metric().score_folded(q_key, cand_key);
                    if sim >= threshold {
                        key_hits.push((k, sim));
                    }
                }
                key_hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                for (k, sim) in key_hits {
                    let start = self.postings_start[k as usize] as usize;
                    let end = self.postings_start[k as usize + 1] as usize;
                    for &r in &self.postings[start..end] {
                        if seen_records.insert(r) {
                            matches.push((Entry::new(&self.store, r), sim));
                        }
                    }
                }
            }

            matches.sort_by(|(ea, sa), (eb, sb)| sb.total_cmp(sa).then_with(|| by_prior_view(ea, eb)));
            matches.truncate(limit);
            matches
        })
    }
}
