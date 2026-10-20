//! Candidate n-grams, their YAKE score, and the deduplication pass
//! (Campos et al. 2020, sections 3.3 and 4): `S(kw) = prod(S(w)) /
//! (KF * (1 + sum(S(w))))`, lower is better, with the interior stop word
//! correction, then containment and similarity dedup.

use std::collections::HashMap;

use turbo_analysis::Span;

use crate::ranked::RankedKeyword;
use crate::{Dedup, Stopwords};

use super::features::{Fold, TAG_D, TAG_U};

/// One distinct candidate between enumeration and ranking: its term ids
/// (the table's names are the folded forms), its frequency `KF`, and
/// every occurrence span in document order.
pub(crate) struct Accum {
    ids: Vec<u32>,
    kf: u64,
    occurrences: Vec<Span>,
}

/// A candidate after scoring: `score` is the raw YAKE score, lower
/// better, always finite and positive.
pub(crate) struct Scored {
    accum: Accum,
    score: f64,
}

/// The span one window covers: from the first token's start to the last
/// token's end.
fn span_of(fold: &Fold<'_>, i: usize, len: usize) -> Span {
    Span::new(fold.tok_start[i], fold.tok_end[i + len - 1])
}

/// Slides every window of `min_n..=max_n` tokens over each sentence and
/// keeps those whose tokens are all content-tagged and whose first and
/// last terms are not stop words, keyed by their term id list, never by
/// a joined string (so a token holding a comma is never re-split).
/// Stop words inside a window are kept, for the scoring correction.
pub(crate) fn enumerate(fold: &Fold<'_>, stopwords: &Stopwords, min_n: usize, max_n: usize) -> Vec<Accum> {
    let n_tokens = fold.term_of.len();
    let mut by_ids: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut accums: Vec<Accum> = Vec::new();
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(fold.sentence_count);
    let mut start = 0usize;
    for s in 0..fold.sentence_count {
        let mut end = start;
        while end < n_tokens && fold.sentence_of[end] as usize == s {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    for &(a, b) in &ranges {
        for len in min_n..=max_n {
            if len > b - a {
                break;
            }
            for i in a..=b - len {
                if fold.tag_of[i..i + len].iter().any(|&t| t == TAG_U || t == TAG_D) {
                    continue;
                }
                if stopwords.contains(&fold.table.names[fold.term_of[i] as usize])
                    || stopwords.contains(&fold.table.names[fold.term_of[i + len - 1] as usize])
                {
                    continue;
                }
                let ids = &fold.term_of[i..i + len];
                match by_ids.get(ids) {
                    Some(&ix) => {
                        let acc = &mut accums[ix];
                        acc.kf += 1;
                        acc.occurrences.push(span_of(fold, i, len));
                    }
                    None => {
                        by_ids.insert(ids.to_vec(), accums.len());
                        accums.push(Accum { ids: ids.to_vec(), kf: 1, occurrences: vec![span_of(fold, i, len)] });
                    }
                }
            }
        }
    }
    accums
}

/// Scores every candidate. A non-stop-word term multiplies `prod` by its
/// `TScore` and adds it to `sum`; an interior stop word `s` between
/// `prev` and `next` contributes the bigram probability
/// `rightCo(prev)[s] / TF(prev) * rightCo(s)[next] / TF(s)` (0 when a
/// neighbour is absent from the term table, per the design's javadoc
/// rule, never a panic), multiplying `prod` by `2 - p` and subtracting
/// `1 - p` from `sum`. `score = prod / (KF * (1 + sum))`; a candidate
/// whose denominator does not hold together is dropped rather than
/// ranked on a NaN.
pub(crate) fn score(fold: &Fold<'_>, tscore: &[f64], accums: Vec<Accum>, stopwords: &Stopwords) -> Vec<Scored> {
    let right_co = |a: u32, b: u32| fold.right.get(&(a, b)).copied().unwrap_or(0) as f64;
    let tf = |id: u32| fold.table.tf[id as usize] as f64;
    accums
        .into_iter()
        .filter_map(|accum| {
            let mut prod = 1.0f64;
            let mut sum = 0.0f64;
            for k in 0..accum.ids.len() {
                let id = accum.ids[k];
                if !stopwords.contains(&fold.table.names[id as usize]) {
                    let s = tscore[id as usize];
                    prod *= s;
                    sum += s;
                } else if k > 0 && k + 1 < accum.ids.len() {
                    let prev = accum.ids[k - 1];
                    let next = accum.ids[k + 1];
                    let bi = (right_co(prev, id) / tf(prev)) * (right_co(id, next) / tf(id));
                    prod *= 2.0 - bi;
                    sum -= 1.0 - bi;
                }
            }
            let denom = accum.kf as f64 * (1.0 + sum);
            let score = prod / denom;
            (score.is_finite() && score > 0.0).then_some(Scored { accum, score })
        })
        .collect()
}

/// Whether `needle` is a contiguous subsequence of `hay`.
fn contains(hay: &[u32], needle: &[u32]) -> bool {
    !needle.is_empty() && needle.len() <= hay.len() && hay.windows(needle.len()).any(|w| w == needle)
}

/// Sorts ascending by score (ties by earlier first occurrence, then the
/// term list, so the pass is deterministic), drops candidates past
/// `threshold`, drops a candidate contained in a survivor or containing
/// one, or whose similarity with a survivor reaches the dedup threshold,
/// cuts to `top_n`, and normalizes the raw scores into a salience in
/// `[0, 1]` (the best surviving candidate scores 1; with one survivor or
/// equal scores, 1 for all).
pub(crate) fn rank(
    fold: &Fold<'_>,
    scored: Vec<Scored>,
    dedup: Dedup,
    threshold: f64,
    top_n: usize,
) -> Vec<RankedKeyword> {
    let mut kept: Vec<Scored> = scored.into_iter().filter(|s| s.score <= threshold).collect();
    kept.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.accum.occurrences[0].start.cmp(&b.accum.occurrences[0].start))
            .then_with(|| a.accum.ids.cmp(&b.accum.ids))
    });
    let mut survivors: Vec<Scored> = Vec::with_capacity(kept.len());
    'next: for cand in kept {
        let names =
            |ids: &[u32]| -> Vec<&str> { ids.iter().map(|&id| fold.table.names[id as usize].as_str()).collect() };
        let cand_names = names(&cand.accum.ids);
        for kept in &survivors {
            let kept_names = names(&kept.accum.ids);
            if contains(&kept.accum.ids, &cand.accum.ids)
                || contains(&cand.accum.ids, &kept.accum.ids)
                || dedup.similarity.similarity(&cand_names, &kept_names) >= dedup.threshold
            {
                continue 'next;
            }
        }
        survivors.push(cand);
    }
    survivors.truncate(top_n);
    let (min, max) = survivors.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), s| (lo.min(s.score), hi.max(s.score)));
    survivors
        .into_iter()
        .enumerate()
        .map(|(rank, s)| {
            let salience = if max > min { (max - s.score) / (max - min) } else { 1.0 };
            let phrase =
                s.accum.ids.iter().map(|&id| fold.table.names[id as usize].as_str()).collect::<Vec<_>>().join(" ");
            RankedKeyword { phrase, terms: s.accum.ids.len() as u16, salience, occurrences: s.accum.occurrences, rank }
        })
        .collect()
}
