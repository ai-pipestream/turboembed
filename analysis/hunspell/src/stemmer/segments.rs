//! Compound word byte-span decomposition for position-length token lattices
//! (issue #2340).

use std::ops::Range;

use crate::dictionary::case::{CaseType, case_type};

use super::results::Results;
use super::{Analysis, Stemmer};

/// One decomposed part of a compound word, carrying its UTF-8 byte sub-range
/// within the input word and its dictionary-normalized stem (`OCONV` applied).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompoundSegment {
    pub byte_range: Range<usize>,
    pub stem: String,
}

impl<'d> Stemmer<'d> {
    /// Decomposes `word` into compound parts when the dictionary has no direct
    /// or affixed non-compound reading and decomposes it into two or more
    /// parts, mapping each part onto valid UTF-8 byte boundaries of `word`.
    pub fn decompose_word(&self, word: &str) -> Option<Vec<CompoundSegment>> {
        let trimmed_len = word.trim_end_matches('.').len();
        if trimmed_len == 0 {
            return None;
        }
        let target = &word[..trimmed_len];
        let input = self.dictionary.input_form(target);
        if input.is_empty() {
            return None;
        }
        let d = self.dictionary;
        if !d.compounds_declared() || d.first_forbidden(d.lookup(&input)) {
            return None;
        }
        let mut results = Results::for_compound_segments();
        let all_caps = case_type(&input) == CaseType::Upper;
        let variants = self.variants(&input);
        for variant in &variants {
            self.analyze_variant(variant, &mut Analysis { surface: &input, variant, results: &mut results, all_caps });
        }
        if !results.is_empty() || results.forbidden {
            return None;
        }
        for variant in &variants {
            self.decompose(variant, &input, &mut results);
            if results.first_compound.is_some() {
                break;
            }
        }
        let split = results.first_compound?;
        map_parts_to_word(target, word.len(), &split, |s| d.output_form(s).into_owned())
    }
}

fn map_parts_to_word(
    target: &str,
    full_word_len: usize,
    split: &[(String, String)],
    mut to_output: impl FnMut(&str) -> String,
) -> Option<Vec<CompoundSegment>> {
    if split.len() < 2 {
        return None;
    }
    let char_starts: Vec<usize> = target.char_indices().map(|(i, _)| i).chain(std::iter::once(target.len())).collect();
    let total_chars = char_starts.len() - 1;
    if total_chars < split.len() {
        return None;
    }
    let part_char_counts: Vec<usize> = split.iter().map(|(surf, _)| surf.chars().count()).collect();
    let sum_chars: usize = part_char_counts.iter().sum();

    let mut out = Vec::with_capacity(split.len());
    if sum_chars == total_chars {
        let mut cursor = 0usize;
        for (idx, ((_, stem), &cnt)) in split.iter().zip(&part_char_counts).enumerate() {
            if cnt == 0 {
                return None;
            }
            let b_start = char_starts[cursor];
            let b_end = if idx + 1 == split.len() { full_word_len } else { char_starts[cursor + cnt] };
            out.push(CompoundSegment { byte_range: b_start..b_end, stem: to_output(stem) });
            cursor += cnt;
        }
        return Some(out);
    }

    // Handle SIMPLIFIEDTRIPLE or CHECKCOMPOUNDPATTERN junctions where expanded
    // part surfaces carry one extra restored character at a junction boundary.
    let mut cursor = 0usize;
    for (idx, ((surf, stem), &cnt)) in split.iter().zip(&part_char_counts).enumerate() {
        let remaining_parts = split.len() - 1 - idx;
        if cursor >= total_chars || total_chars - cursor < remaining_parts + 1 {
            return None;
        }
        let max_end = total_chars - remaining_parts;
        let end_char = if idx + 1 == split.len() { total_chars } else { (cursor + cnt).min(max_end).max(cursor + 1) };
        // When a simplified triple restored the first character of this part,
        // overlapping by 1 character keeps the surface slice anchored cleanly
        // while staying non-decreasing in (start, end).
        let start_char = if idx > 0
            && cursor > 0
            && end_char - cursor < cnt
            && surf
                .chars()
                .next()
                .zip(target[char_starts[cursor - 1]..].chars().next())
                .is_some_and(|(a, b)| a.eq_ignore_ascii_case(&b))
        {
            cursor - 1
        } else {
            cursor
        };
        let b_start = char_starts[start_char];
        let b_end = if idx + 1 == split.len() { full_word_len } else { char_starts[end_char] };
        if b_start >= b_end {
            return None;
        }
        out.push(CompoundSegment { byte_range: b_start..b_end, stem: to_output(stem) });
        cursor = end_char;
    }
    Some(out)
}
