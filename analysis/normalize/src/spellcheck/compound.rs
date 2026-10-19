//! Compound correction: a whole phrase at once, with words wrongly run
//! together split and words wrongly split merged, a port of OpenNLP's
//! `SymSpell.lookupCompound`.
//!
//! Beyond the corrected phrase, the port reports where each word of the
//! output came from: the input token it corrects or splits, or the two
//! input tokens it merges (with the whitespace between them). That is what
//! lets the normalizer write the correction as a view with an exact
//! alignment.

use std::borrow::Cow;
use std::ops::Range;

use turbo_unicode::props::white_space;

use super::suggestion::{Suggestion, Verbosity};
use super::symspell::{LookupError, SymSpell, saturated_add};

/// The correction of a whole phrase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compound {
    /// The corrected phrase: the parts joined by single spaces.
    pub term: String,
    /// The distance from the input, without the whitespace at its ends, to
    /// `term`.
    pub edit_distance: usize,
    /// The naive Bayes estimate of the phrase's count: N times the product
    /// of each part's count over N.
    pub count: u64,
    /// The parts of `term`, in order.
    pub parts: Vec<CompoundPart>,
}

/// One word, or one split pair of words, of a compound correction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompoundPart {
    /// The input it replaces: one token, or two tokens and the whitespace
    /// between them when it merges them. Byte offsets in the input.
    pub source: Range<usize>,
    /// Where it is in `Compound::term`.
    pub output: Range<usize>,
    /// Its edit distance as OpenNLP scores it: the lookup distance, one
    /// more for a merge, the maximum plus one for a token left as it is.
    pub edit_distance: usize,
    pub count: u64,
}

/// A part while the phrase is being corrected.
struct Part<'a> {
    term: Cow<'a, str>,
    edit_distance: usize,
    count: u64,
    first: usize,
    last: usize,
}

impl<'a> Part<'a> {
    fn of(s: &Suggestion<'a>, token: usize) -> Part<'a> {
        Part { term: Cow::Borrowed(s.term), edit_distance: s.edit_distance, count: s.count, first: token, last: token }
    }
}

/// The tokens of `text`: maximal runs of characters without the Unicode
/// White_Space property, as byte ranges.
pub(crate) fn tokens(text: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (white_space(c), start) {
            (true, Some(s)) => {
                out.push(s..i);
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(s..text.len());
    }
    out
}

/// OpenNLP's estimate for a token nothing corrects: 10^7 / 10^length.
fn estimated_count(token: &str) -> u64 {
    (10_000_000.0 / 10f64.powi(token.chars().count() as i32)) as i64 as u64
}

/// OpenNLP's estimate for the second token of a merge check when it has no
/// correction: 10 / 10^length.
fn combi_estimated_count(token: &str) -> u64 {
    (10.0 / 10f64.powi(token.chars().count() as i32)) as i64 as u64
}

impl SymSpell {
    /// The best correction of `input` as a phrase, each token corrected
    /// within `max_edit_distance`.
    pub fn lookup_compound(&self, input: &str, max_edit_distance: usize) -> Result<Compound, LookupError> {
        self.check(max_edit_distance)?;
        Ok(self.compound_within(input, max_edit_distance))
    }

    /// `lookup_compound` for a distance already checked against the index.
    pub(crate) fn compound_within(&self, input: &str, max: usize) -> Compound {
        let tokens = tokens(input);
        let n_max = self.corpus_word_count();
        let mut parts: Vec<Part<'_>> = Vec::with_capacity(tokens.len());
        let mut last_combi = false;
        for (i, range) in tokens.iter().enumerate() {
            let token = &input[range.clone()];
            let suggestions = self.lookup_within(token, Verbosity::Top, max);
            // A space wrongly put between this token and the previous one.
            if i > 0 && !last_combi && self.merges(input, &tokens, i, &suggestions, &mut parts, max, n_max) {
                last_combi = true;
                continue;
            }
            last_combi = false;
            let single = token.chars().nth(1).is_none();
            // A correct token, or a single character, is never split.
            if let Some(best) = suggestions.first()
                && (best.edit_distance == 0 || single)
            {
                parts.push(Part::of(best, i));
                continue;
            }
            let split = if single { None } else { self.best_split(token, i, suggestions.first(), max, n_max) };
            parts.push(split.unwrap_or_else(|| Part {
                term: Cow::Borrowed(token),
                edit_distance: max + 1,
                count: estimated_count(token),
                first: i,
                last: i,
            }));
        }
        self.assemble(input, &tokens, parts, n_max)
    }

    /// Replaces the last part with the merge of tokens `i - 1` and `i` when
    /// the merged word scores better than the two apart.
    #[allow(clippy::too_many_arguments)]
    fn merges<'a>(
        &'a self,
        input: &'a str,
        tokens: &[Range<usize>],
        i: usize,
        suggestions: &[Suggestion<'a>],
        parts: &mut [Part<'a>],
        max: usize,
        n_max: u64,
    ) -> bool {
        let previous = &input[tokens[i - 1].clone()];
        let token = &input[tokens[i].clone()];
        let joined = format!("{previous}{token}");
        let combi = self.lookup_within(&joined, Verbosity::Top, max);
        let Some(combined) = combi.first() else { return false };
        let best1 = parts.last_mut().expect("a part for every earlier token");
        let (d2, c2) = match suggestions.first() {
            Some(s) => (s.edit_distance, s.count),
            None => (max + 1, combi_estimated_count(token)),
        };
        let distance1 = best1.edit_distance + d2;
        let better = combined.edit_distance + 1 < distance1
            || (combined.edit_distance + 1 == distance1
                && combined.count as f64 > best1.count as f64 / n_max as f64 * c2 as f64);
        if better {
            *best1 = Part {
                term: Cow::Borrowed(combined.term),
                edit_distance: combined.edit_distance + 1,
                count: combined.count,
                first: i - 1,
                last: i,
            };
        }
        better
    }

    /// The best way to split `token` in two words, with the token's own
    /// correction (if any) as the candidate to beat.
    fn best_split<'a>(
        &'a self,
        token: &str,
        index: usize,
        own: Option<&Suggestion<'a>>,
        max: usize,
        n_max: u64,
    ) -> Option<Part<'a>> {
        let mut best = own.map(|s| Part::of(s, index));
        let metric = self.config.metric();
        for (j, _) in token.char_indices().skip(1) {
            let Some(s1) = self.lookup_within(&token[..j], Verbosity::Top, max).first().copied() else { continue };
            let Some(s2) = self.lookup_within(&token[j..], Verbosity::Top, max).first().copied() else { continue };
            let split = format!("{} {}", s1.term, s2.term);
            // Joining two words spends an edit on the space, so the split
            // may be further than the maximum; it is capped, not dropped.
            let distance = super::distance::EditDistance::distance(&metric, token, &split, max).unwrap_or(max + 1);
            if let Some(b) = &best {
                if distance > b.edit_distance {
                    continue;
                }
                if distance < b.edit_distance {
                    best = None;
                }
            }
            let count = self.split_count(token, &split, &s1, &s2, own, n_max);
            if best.as_ref().is_none_or(|b| count > b.count) {
                best =
                    Some(Part { term: Cow::Owned(split), edit_distance: distance, count, first: index, last: index });
            }
        }
        best
    }

    /// The count of a split: the bigram's count, raised so that a split
    /// that rebuilds the token or agrees with its correction outranks the
    /// correction; or, for a pair that is not a known bigram, the naive
    /// Bayes estimate capped below every real bigram.
    fn split_count(
        &self,
        token: &str,
        split: &str,
        s1: &Suggestion<'_>,
        s2: &Suggestion<'_>,
        own: Option<&Suggestion<'_>>,
        n_max: u64,
    ) -> u64 {
        let rebuilds =
            token.len() == s1.term.len() + s2.term.len() && token.starts_with(s1.term) && token.ends_with(s2.term);
        match self.bigrams.get(split) {
            Some(&count) => match own {
                Some(best) if rebuilds => count.max(saturated_add(best.count, 2)),
                Some(best) if s1.term == best.term || s2.term == best.term => count.max(saturated_add(best.count, 1)),
                Some(_) => count,
                None if rebuilds => count.max(saturated_add(s1.count.max(s2.count), 2)),
                None => count,
            },
            None => {
                let estimate = s1.count as f64 / n_max as f64 * s2.count as f64;
                self.min_bigram_count().min(estimate as i64 as u64)
            }
        }
    }

    /// Joins the parts into the phrase and scores it.
    fn assemble(&self, input: &str, tokens: &[Range<usize>], parts: Vec<Part<'_>>, n_max: u64) -> Compound {
        let mut term = String::new();
        let mut out = Vec::with_capacity(parts.len());
        let mut product = n_max as f64;
        let n = n_max.max(1) as f64;
        for part in parts {
            // OpenNLP puts a space before a part only when the phrase so far
            // is not empty, which matters only for an empty dictionary word.
            if !term.is_empty() {
                term.push(' ');
            }
            let start = term.len();
            term.push_str(&part.term);
            product *= part.count as f64 / n;
            out.push(CompoundPart {
                source: tokens[part.first].start..tokens[part.last].end,
                output: start..term.len(),
                edit_distance: part.edit_distance,
                count: part.count,
            });
        }
        let trimmed = input.trim_matches(white_space);
        let metric = self.config.metric();
        let edit_distance =
            super::distance::EditDistance::distance(&metric, trimmed, &term, usize::MAX).expect("no bound");
        Compound { term, edit_distance, count: product as i64 as u64, parts: out }
    }
}
