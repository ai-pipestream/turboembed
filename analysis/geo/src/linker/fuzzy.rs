//! Fuzzy string matching scorers (normalized Levenshtein and Jaro-Winkler)
//! over folded gazetteer match keys and place names.

mod jaro;
use jaro::{ascii_pair, jaro_ascii, jw_ascii};

use std::cell::RefCell;
use std::collections::HashSet;

use crate::entry::GazetteerEntry;
use crate::fold::Folder;
use crate::geocode::by_prior_view;
use crate::index::{GazetteerIndex, with_key};
use crate::store::Entry;

const JARO_WINKLER_PREFIX_SCALE: f64 = 0.1;
const JARO_WINKLER_MAX_PREFIX: usize = 4;
const JARO_WINKLER_BOOST_THRESHOLD: f64 = 0.7;

#[derive(Default)]
struct FuzzyWorkspace {
    folder: Folder,
    key_buf: String,
    a_chars: Vec<char>,
    b_chars: Vec<char>,
    prev: Vec<usize>,
    curr: Vec<usize>,
    a_matched: Vec<bool>,
    b_matched: Vec<bool>,
}

thread_local! {
    static WORKSPACE: RefCell<FuzzyWorkspace> = RefCell::new(FuzzyWorkspace::default());
}

fn with_workspace<R>(f: impl FnOnce(&mut FuzzyWorkspace) -> R) -> R {
    WORKSPACE.with(|cell| match cell.try_borrow_mut() {
        Ok(mut guard) => f(&mut guard),
        Err(_) => f(&mut FuzzyWorkspace::default()),
    })
}

#[inline]
fn char_len_fast(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.chars().count() }
}

/// Upper bound on [`FuzzyMetric::score_folded`] given only the character
/// counts `m` and `n` of two non-empty strings.
#[inline]
fn metric_upper_bound(metric: FuzzyMetric, m: usize, n: usize) -> f64 {
    if m == 0 || n == 0 {
        return 0.0;
    }
    if m == n {
        return 1.0;
    }
    let min_l = m.min(n) as f64;
    let max_l = m.max(n) as f64;
    let lev_ub = (1.0 - ((max_l - min_l) / max_l)).clamp(0.0, 1.0);
    let raw_ub = match metric {
        FuzzyMetric::NormalizedLevenshtein => lev_ub,
        FuzzyMetric::JaroWinkler => {
            let jaro_ub = ((min_l / m as f64) + (min_l / n as f64) + 1.0) / 3.0;
            if jaro_ub < JARO_WINKLER_BOOST_THRESHOLD {
                jaro_ub
            } else {
                (jaro_ub + (JARO_WINKLER_MAX_PREFIX as f64) * JARO_WINKLER_PREFIX_SCALE * (1.0 - jaro_ub))
                    .clamp(0.0, 1.0)
            }
        }
        FuzzyMetric::Combined => {
            let jaro_ub = ((min_l / m as f64) + (min_l / n as f64) + 1.0) / 3.0;
            let jw_ub = if jaro_ub < JARO_WINKLER_BOOST_THRESHOLD {
                jaro_ub
            } else {
                (jaro_ub + (JARO_WINKLER_MAX_PREFIX as f64) * JARO_WINKLER_PREFIX_SCALE * (1.0 - jaro_ub))
                    .clamp(0.0, 1.0)
            };
            0.5 * (lev_ub + jw_ub)
        }
    };
    (raw_ub + 1e-12).min(1.0)
}

fn lev_slice(a: &[char], b: &[char], prev: &mut Vec<usize>, curr: &mut Vec<usize>) -> f64 {
    let m = a.len();
    let n = b.len();
    let max_len = m.max(n);
    if max_len == 0 {
        return 1.0;
    }
    if m == 0 || n == 0 {
        return 0.0;
    }
    prev.clear();
    prev.extend(0..=n);
    curr.resize(n + 1, 0);
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(prev, curr);
    }
    let dist = prev[n];
    (1.0 - (dist as f64 / max_len as f64)).clamp(0.0, 1.0)
}

fn jaro_slice(a: &[char], b: &[char], a_matched: &mut Vec<bool>, b_matched: &mut Vec<bool>) -> f64 {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return 0.0;
    }
    let match_dist = (m.max(n) / 2).saturating_sub(1);
    a_matched.clear();
    a_matched.resize(m, false);
    b_matched.clear();
    b_matched.resize(n, false);
    let mut matches = 0usize;

    for (i, &ca) in a.iter().enumerate() {
        let start = i.saturating_sub(match_dist);
        let end = (i + match_dist + 1).min(n);
        for j in start..end {
            if !b_matched[j] && ca == b[j] {
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
        if a[i] != b[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m_f = matches as f64;
    let t_f = (transpositions as f64) / 2.0;
    ((m_f / m as f64) + (m_f / n as f64) + ((m_f - t_f) / m_f)) / 3.0
}

fn jw_slice(a: &[char], b: &[char], a_matched: &mut Vec<bool>, b_matched: &mut Vec<bool>) -> f64 {
    let jaro = jaro_slice(a, b, a_matched, b_matched);
    if jaro < JARO_WINKLER_BOOST_THRESHOLD {
        return jaro;
    }
    let prefix = a.iter().zip(b.iter()).take(JARO_WINKLER_MAX_PREFIX).take_while(|(ca, cb)| ca == cb).count();
    (jaro + (prefix as f64) * JARO_WINKLER_PREFIX_SCALE * (1.0 - jaro)).clamp(0.0, 1.0)
}

impl FuzzyWorkspace {
    fn load_a(&mut self, a: &str) {
        self.a_chars.clear();
        self.a_chars.extend(a.chars());
    }

    fn score_with_loaded_a(&mut self, metric: FuzzyMetric, a: &str, b: &str) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        if a == b {
            return 1.0;
        }
        self.b_chars.clear();
        self.b_chars.extend(b.chars());
        match metric {
            FuzzyMetric::NormalizedLevenshtein => {
                lev_slice(&self.a_chars, &self.b_chars, &mut self.prev, &mut self.curr)
            }
            FuzzyMetric::JaroWinkler => {
                jw_slice(&self.a_chars, &self.b_chars, &mut self.a_matched, &mut self.b_matched)
            }
            FuzzyMetric::Combined => {
                let l = lev_slice(&self.a_chars, &self.b_chars, &mut self.prev, &mut self.curr);
                let j = jw_slice(&self.a_chars, &self.b_chars, &mut self.a_matched, &mut self.b_matched);
                0.5 * (l + j)
            }
        }
    }
}

/// Character-level normalized Levenshtein similarity in `[0.0, 1.0]`.
///
/// Returns `1.0` when both strings are equal (including both empty), and
/// `1.0 - (edit_distance / max(a.chars().count(), b.chars().count()))` otherwise.
pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    with_workspace(|ws| {
        ws.load_a(a);
        ws.b_chars.clear();
        ws.b_chars.extend(b.chars());
        lev_slice(&ws.a_chars, &ws.b_chars, &mut ws.prev, &mut ws.curr)
    })
}

/// Character-level Jaro similarity in `[0.0, 1.0]`.
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if let Some((a, b)) = ascii_pair(a, b) {
        return jaro_ascii(a, b);
    }
    with_workspace(|ws| {
        ws.load_a(a);
        ws.b_chars.clear();
        ws.b_chars.extend(b.chars());
        jaro_slice(&ws.a_chars, &ws.b_chars, &mut ws.a_matched, &mut ws.b_matched)
    })
}

/// Character-level Jaro-Winkler similarity in `[0.0, 1.0]`.
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if let Some((a, b)) = ascii_pair(a, b) {
        return jw_ascii(a, b);
    }
    with_workspace(|ws| {
        ws.load_a(a);
        ws.b_chars.clear();
        ws.b_chars.extend(b.chars());
        jw_slice(&ws.a_chars, &ws.b_chars, &mut ws.a_matched, &mut ws.b_matched)
    })
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
        with_workspace(|ws| {
            ws.load_a(a);
            ws.score_with_loaded_a(self, a, b)
        })
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
    /// A `NaN` threshold is sanitized to `1.0` (exact match only).
    pub fn new(metric: FuzzyMetric, min_similarity: f64) -> FuzzyStringScorer {
        let clamped = if min_similarity.is_nan() { 1.0 } else { min_similarity.clamp(0.0, 1.0) };
        FuzzyStringScorer { metric, min_similarity: clamped }
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
            with_workspace(|ws| {
                ws.folder.fold_into(candidate, &mut ws.key_buf);
                ws.load_a(q_key);
                let c_key = ws.key_buf.clone();
                ws.score_with_loaded_a(self.metric, q_key, &c_key)
            })
        })
    }

    /// Scores `query` against a borrowed gazetteer [`Entry`] across its
    /// canonical name and all alternate names, returning a value in `[0.0, 1.0]`.
    pub fn score_entry(&self, query: &str, entry: &Entry<'_>) -> f64 {
        with_key(query, |q_key| {
            if q_key.is_empty() {
                return 0.0;
            }
            with_workspace(|ws| {
                ws.load_a(q_key);
                let mut key_buf = std::mem::take(&mut ws.key_buf);
                ws.folder.fold_into(entry.name(), &mut key_buf);
                let mut best = ws.score_with_loaded_a(self.metric, q_key, &key_buf);
                if best < 1.0 {
                    for alt in entry.alternate_names() {
                        ws.folder.fold_into(alt, &mut key_buf);
                        let s = ws.score_with_loaded_a(self.metric, q_key, &key_buf);
                        if s > best {
                            best = s;
                            if best >= 1.0 {
                                break;
                            }
                        }
                    }
                }
                ws.key_buf = key_buf;
                best
            })
        })
    }

    /// Scores `query` against an owned [`GazetteerEntry`].
    pub fn score_owned_entry(&self, query: &str, entry: &GazetteerEntry) -> f64 {
        with_key(query, |q_key| {
            if q_key.is_empty() {
                return 0.0;
            }
            with_workspace(|ws| {
                ws.load_a(q_key);
                let mut key_buf = std::mem::take(&mut ws.key_buf);
                ws.folder.fold_into(entry.name(), &mut key_buf);
                let mut best = ws.score_with_loaded_a(self.metric, q_key, &key_buf);
                if best < 1.0 {
                    for alt in entry.alternate_names() {
                        ws.folder.fold_into(alt, &mut key_buf);
                        let s = ws.score_with_loaded_a(self.metric, q_key, &key_buf);
                        if s > best {
                            best = s;
                            if best >= 1.0 {
                                break;
                            }
                        }
                    }
                }
                ws.key_buf = key_buf;
                best
            })
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
        if limit == 0 || min_similarity.is_nan() || min_similarity > 1.0 {
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
                let metric = scorer.metric();
                let key_hits = with_workspace(|ws| {
                    ws.load_a(q_key);
                    let m = ws.a_chars.len();
                    let mut hits: Vec<(u32, f64)> = Vec::new();
                    for k in 0..self.names.len() as u32 {
                        let cand_key = self.names.key(k);
                        if cand_key == q_key {
                            continue;
                        }
                        let n = char_len_fast(cand_key);
                        if metric_upper_bound(metric, m, n) < threshold {
                            continue;
                        }
                        let sim = ws.score_with_loaded_a(metric, q_key, cand_key);
                        if sim >= threshold {
                            hits.push((k, sim));
                        }
                    }
                    hits
                });

                let mut sorted_hits = key_hits;
                sorted_hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let mut cutoff_sim: Option<f64> = None;
                for (k, sim) in sorted_hits {
                    if let Some(cut) = cutoff_sim
                        && sim < cut
                    {
                        break;
                    }
                    let start = self.postings_start[k as usize] as usize;
                    let end = self.postings_start[k as usize + 1] as usize;
                    for &r in &self.postings[start..end] {
                        if seen_records.insert(r) {
                            matches.push((Entry::new(&self.store, r), sim));
                            if matches.len() >= limit && cutoff_sim.is_none() {
                                cutoff_sim = Some(sim);
                            }
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
