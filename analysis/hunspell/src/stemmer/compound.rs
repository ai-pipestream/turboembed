//! Reading a word as a compound: OpenNLP's `decompose` and the searches
//! under it.
//!
//! With compound flags declared, the word is split into parts of at least
//! `COMPOUNDMIN` characters, each read as an entry (or an affixed entry)
//! the flags admit at its place. A `CHECKCOMPOUNDPATTERN` replacement and
//! a `SIMPLIFIEDTRIPLE` letter are put back first where the word may have
//! lost them. Each `COMPOUNDRULE` is searched on its own. One budget of
//! [`PART_CHECK_BUDGET`] part checks covers one call of `decompose`.
//!
//! The case of a junction is judged on the input as written (the case
//! source) where the variant read has its characters at the same places.
//! Offsets are bytes of the variant; the case source's are found through
//! the characters' indices.

use std::rc::Rc;

use crate::compound_rule::CompoundRule;
use crate::dictionary::rules::CompoundPosition;
use crate::hash::FoldMap;
use crate::settings::{CompoundPattern, Side};
use crate::text::{is_upper, utf16_len};

use super::results::{CompoundPart, Results};
use super::{Analysis, MAX_COMPOUND_PARTS, PART_CHECK_BUDGET, Stemmer};

/// A boundary a search must split at: where a pattern's replacement or a
/// triple letter was put back, with the pattern (none for a triple).
#[derive(Copy, Clone)]
struct Junction<'d> {
    offset: usize,
    pattern: Option<&'d CompoundPattern>,
}

/// The byte offset of each character of `text`, and its length.
fn char_offsets(text: &str) -> Vec<usize> {
    let mut offsets: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    offsets.push(text.len());
    offsets
}

/// The offset in `case` of the character at byte `offset` of `word`; both
/// have the same characters at the same indices.
fn case_offset(word: &str, case: &str, offset: usize) -> usize {
    if word.len() == case.len() && word.is_ascii() {
        return offset;
    }
    let index = word[..offset].chars().count();
    case.char_indices().nth(index).map_or(case.len(), |(i, _)| i)
}

/// Whether the case source's characters stand where the variant's do.
/// OpenNLP takes the input as the case source when its UTF-16 length is
/// the variant's; the port asks also for the same character count and the
/// same UTF-16 width at each index, without which Java's offsets would cut
/// a character in two. Case variants map character to character, so the
/// two conditions agree on every variant the stemmer makes but the sharp
/// s ones, whose lengths differ anyway.
fn aligned(surface: &str, word: &str) -> bool {
    utf16_len(surface) == utf16_len(word)
        && surface.chars().count() == word.chars().count()
        && surface.chars().zip(word.chars()).all(|(a, b)| a.len_utf16() == b.len_utf16())
}

impl<'d> Stemmer<'d> {
    /// Reads a case variant as a compound. `surface` is the input the
    /// variant came from.
    pub(crate) fn decompose(&self, word: &str, surface: &str, results: &mut Results) {
        let d = self.dictionary;
        if d.first_forbidden(d.lookup(word)) || self.rejects_compound_text(word) {
            return;
        }
        let case_source = if aligned(surface, word) { surface } else { word };
        let mut budget = PART_CHECK_BUDGET;
        if d.positional_compounds_declared() {
            self.search_spelling(word, case_source, results, &mut budget, &[]);
            for pattern in d.compound_patterns.iter() {
                let Some(replacement) = pattern.replacement.as_deref().filter(|r| !r.is_empty()) else { continue };
                let mut found = word.find(replacement);
                while let Some(at) = found {
                    if budget <= 0 {
                        break;
                    }
                    let end = at + replacement.len();
                    let (case_at, case_end) = (case_offset(word, case_source, at), case_offset(word, case_source, end));
                    let expanded = format!("{}{}{}{}", &word[..at], pattern.end, pattern.begin, &word[end..]);
                    let expanded_case = format!(
                        "{}{}{}{}",
                        &case_source[..case_at],
                        pattern.end,
                        pattern.begin,
                        &case_source[case_end..]
                    );
                    let junction = Junction { offset: at + pattern.end.len(), pattern: Some(pattern) };
                    self.search_spelling(&expanded, &expanded_case, results, &mut budget, &[junction]);
                    let next = at + word[at..].chars().next().map_or(1, char::len_utf8);
                    found = word[next..].find(replacement).map(|i| i + next);
                }
            }
            if d.settings.simplified_triple {
                self.search_triples(word, case_source, 0, &[], results, &mut budget);
            }
        }
        if !d.compound_rules.is_empty() {
            let offsets = char_offsets(word);
            let case_offsets = char_offsets(case_source);
            for rule in d.compound_rules.iter() {
                let mut search =
                    RuleSearch { word, case: case_source, offsets: &offsets, case_offsets: &case_offsets, rule };
                self.search_rule(&mut search, 0, &mut Vec::new(), results, &mut budget);
            }
        }
    }

    /// Whether the text a compound level splits may not be split: a
    /// `CHECKCOMPOUNDREP` replacement, or a space put between two of its
    /// characters, makes a word that reads without compounding.
    fn rejects_compound_text(&self, text: &str) -> bool {
        let d = self.dictionary;
        if d.rejects_compound_replacement(text, |w| self.is_noncompound_form(w)) {
            return true;
        }
        // without this bound every split of a long word would be read again
        // at every compound level
        if !d.may_read_spaced_form(utf16_len(text) + 1) {
            return false;
        }
        let mut spaced = String::with_capacity(text.len() + 1);
        for (at, _) in text.char_indices().skip(1) {
            spaced.clear();
            spaced.push_str(&text[..at]);
            spaced.push(' ');
            spaced.push_str(&text[at..]);
            if self.is_noncompound_form(&spaced) {
                return true;
            }
        }
        false
    }

    /// Whether `word` has a direct or affixed reading.
    fn is_noncompound_form(&self, word: &str) -> bool {
        if word.is_empty() || !self.dictionary.may_read(utf16_len(word)) {
            return false;
        }
        let mut results = Results::new(false);
        for variant in &self.variants(word) {
            self.analyze_variant(
                variant,
                &mut Analysis { surface: word, variant, results: &mut results, all_caps: false },
            );
        }
        !results.is_empty()
    }

    /// Puts back the third letter of a run of two that `SIMPLIFIEDTRIPLE`
    /// may have dropped, requiring a junction before or after it, at each
    /// such run from byte `from` on.
    fn search_triples(
        &self,
        word: &str,
        surface: &str,
        from: usize,
        junctions: &[Junction<'d>],
        results: &mut Results,
        budget: &mut i32,
    ) {
        if junctions.len() >= MAX_COMPOUND_PARTS - 1 {
            return;
        }
        let mut at = from;
        while at < word.len() && *budget > 0 {
            let c = word[at..].chars().next().expect("at is a character boundary");
            let width = c.len_utf8();
            let next = at + width;
            if next < word.len()
                && word[next..].starts_with(c)
                && (at == 0 || !word[..at].ends_with(c))
                && (next + width == word.len() || !word[next + width..].starts_with(c))
            {
                let case_next = case_offset(word, surface, next);
                let expanded = format!("{}{c}{}", &word[..next], &word[next..]);
                let expanded_case = format!("{}{c}{}", &surface[..case_next], &surface[case_next..]);
                for boundary in [next, next + width] {
                    let left = *budget;
                    *budget -= 1;
                    if left <= 0 {
                        return;
                    }
                    let mut required = junctions.to_vec();
                    required.push(Junction { offset: boundary, pattern: None });
                    self.search_spelling(&expanded, &expanded_case, results, budget, &required);
                    self.search_triples(&expanded, &expanded_case, next + 2 * width, &required, results, budget);
                }
            }
            at = next;
        }
    }

    /// Splits one spelling into parts, at every required junction.
    fn search_spelling(
        &self,
        word: &str,
        surface: &str,
        results: &mut Results,
        budget: &mut i32,
        junctions: &[Junction<'d>],
    ) {
        let offsets = char_offsets(word);
        let count = offsets.len() - 1;
        if count < 2 * self.dictionary.settings.compound_min {
            return;
        }
        if junctions.iter().any(|j| j.offset == 0 || j.offset >= word.len()) {
            return;
        }
        let case_offsets = char_offsets(surface);
        let mut search = PartSearch {
            word,
            case: surface,
            offsets: &offsets,
            case_offsets: &case_offsets,
            junctions,
            remainder_checks: vec![RemainderCheck::Unknown; count + 1],
            readings: FoldMap::default(),
        };
        self.search(&mut search, 0, &mut Vec::new(), results, budget);
    }

    /// Extends a split with each admissible part from character
    /// `from_point`, and the rest recursively. The junction into the part
    /// keeps to `CHECKCOMPOUNDCASE` and `CHECKCOMPOUNDTRIPLE`; a split
    /// reaching the end adds every part's readings.
    fn search(
        &self,
        s: &mut PartSearch<'_, 'd>,
        from_point: usize,
        parts: &mut Vec<CompoundPart<'d>>,
        results: &mut Results,
        budget: &mut i32,
    ) {
        let d = self.dictionary;
        let from = s.offsets[from_point];
        let current = s.junctions.iter().rev().find(|j| j.offset == from).copied();
        if parts.len() >= MAX_COMPOUND_PARTS
            || (from > 0 && self.violates_boundary_checks(s, from_point, current.is_some_and(|j| j.pattern.is_none())))
        {
            return;
        }
        let min = d.settings.compound_min;
        let first = from == 0;
        let ends = s.offsets.len() - 1;
        if ends - from_point < min {
            return;
        }
        // the remainder is split into further parts, so it must pass the
        // same text checks as the whole input
        if !first {
            if s.remainder_checks[from_point] == RemainderCheck::Unknown {
                s.remainder_checks[from_point] = if self.rejects_compound_text(&s.word[from..]) {
                    RemainderCheck::Rejected
                } else {
                    RemainderCheck::Allowed
                };
            }
            if s.remainder_checks[from_point] == RemainderCheck::Rejected {
                return;
            }
        }
        for end_point in from_point + min..=ends {
            if *budget <= 0 {
                return;
            }
            let end = s.offsets[end_point];
            let last = end == s.word.len();
            let spans_junction = s.junctions.iter().any(|j| from < j.offset && end > j.offset);
            if (first && last) || spans_junction || (!last && ends - end_point < min) {
                continue;
            }
            *budget -= 1;
            let part = &s.word[from..end];
            let position = if first {
                CompoundPosition::Begin
            } else if last {
                CompoundPosition::End
            } else {
                CompoundPosition::Middle
            };
            let case_part = &s.case[s.case_offsets[from_point]..s.case_offsets[end_point]];
            // many splits reach the same part; each still counts against the
            // budget, as in OpenNLP, but its readings are found once
            let readings = match s.readings.get(&(from_point, end_point)) {
                Some(r) => Rc::clone(r),
                None => {
                    let r: Rc<[CompoundPart<'d>]> = self
                        .part_readings(part, case_part, position, first, last, results.hyphenated_first_part)
                        .into();
                    s.readings.insert((from_point, end_point), Rc::clone(&r));
                    r
                }
            };
            for candidate in readings.iter().cloned() {
                if let Some(left) = parts.last()
                    && self.rejects_junction(left, &candidate, current.and_then(|j| j.pattern), last)
                {
                    continue;
                }
                parts.push(candidate);
                if last {
                    let mut units = 0;
                    let mut syllables = 0;
                    for (i, selected) in parts.iter().enumerate() {
                        let flags = d.flags(selected.entry);
                        units += d.compound_units(flags, &selected.affixes);
                        syllables +=
                            d.compound_syllables(&selected.surface, flags, &selected.affixes, i == parts.len() - 1);
                    }
                    let closing = parts.last().expect("the candidate was pushed");
                    if (results.hyphenated_first_part || d.compound_size_allowed(syllables, units))
                        && d.compound_case_allowed(d.flags(closing.entry), s.case)
                    {
                        results.add_compound(d, parts);
                    }
                } else {
                    self.search(s, end_point, parts, results, budget);
                }
                parts.pop();
            }
        }
    }

    /// Whether the junction before character `point` breaks
    /// `CHECKCOMPOUNDCASE` (an upper case letter on either side, in the
    /// case source) or `CHECKCOMPOUNDTRIPLE` (three equal letters across
    /// it, unless the junction put a simplified triple back).
    fn violates_boundary_checks(&self, s: &PartSearch<'_, '_>, point: usize, allow_triple: bool) -> bool {
        let settings = &self.dictionary.settings;
        let from = s.offsets[point];
        let before = s.word[..from].chars().next_back().expect("from is inside the word");
        let after = s.word[from..].chars().next().expect("from is inside the word");
        if settings.check_compound_case {
            let case_from = s.case_offsets[point];
            let case_before = s.case[..case_from].chars().next_back();
            let case_after = s.case[case_from..].chars().next();
            if case_before.is_some_and(is_upper) || case_after.is_some_and(is_upper) {
                return true;
            }
        }
        if !allow_triple && settings.check_compound_triple && before == after {
            let before_start = from - before.len_utf8();
            let after_end = from + after.len_utf8();
            if (before_start > 0 && s.word[..before_start].ends_with(after))
                || (after_end < s.word.len() && s.word[after_end..].starts_with(after))
            {
                return true;
            }
        }
        false
    }

    /// Whether a declaration forbids `right` after `left`:
    /// `CHECKCOMPOUNDDUP` a closing part repeating the one before it, a
    /// junction put back from a pattern unless that pattern's flags hold,
    /// and any other junction some pattern describes.
    fn rejects_junction(
        &self,
        left: &CompoundPart<'_>,
        right: &CompoundPart<'_>,
        allowed: Option<&CompoundPattern>,
        last: bool,
    ) -> bool {
        let d = self.dictionary;
        if last && d.settings.check_compound_dup && left.root == right.root {
            return true;
        }
        let l = Side { text: &left.surface, flags: d.flags(left.entry), affixes: &left.affixes };
        let r = Side { text: &right.surface, flags: d.flags(right.entry), affixes: &right.affixes };
        match allowed {
            Some(pattern) => !pattern.matches(&l, &r),
            None => d.compound_patterns.iter().any(|p| p.matches(&l, &r)),
        }
    }

    /// Searches the splits a `COMPOUNDRULE` admits, from character
    /// `from_point`. The flags of the parts' entries, one homonym each,
    /// must follow the rule.
    fn search_rule(
        &self,
        s: &mut RuleSearch<'_, 'd>,
        from_point: usize,
        parts: &mut Vec<CompoundPart<'d>>,
        results: &mut Results,
        budget: &mut i32,
    ) {
        let d = self.dictionary;
        let ends = s.offsets.len() - 1;
        let min = d.settings.compound_min;
        if parts.len() >= MAX_COMPOUND_PARTS || ends - from_point < min {
            return;
        }
        let mut end_point = from_point + min;
        while end_point <= ends && *budget > 0 {
            let last = end_point == ends;
            if !(from_point == 0 && last) {
                *budget -= 1;
                let part = &s.word[s.offsets[from_point]..s.offsets[end_point]];
                let case_part = &s.case[s.case_offsets[from_point]..s.case_offsets[end_point]];
                for candidate in self.rule_parts(part, case_part, last) {
                    if last && d.settings.check_compound_dup && parts.last().is_some_and(|p| p.root == candidate.root) {
                        continue;
                    }
                    parts.push(candidate);
                    if s.rule.matches(parts.iter().map(|p| d.flags(p.entry)), last) {
                        if last {
                            results.add_compound(d, parts);
                        } else {
                            self.search_rule(s, end_point, parts, results, budget);
                        }
                    }
                    parts.pop();
                }
            }
            if last {
                break;
            }
            end_point += 1;
        }
    }
}

/// Whether the remainder from a character may be split, once known.
#[derive(Copy, Clone, PartialEq, Eq)]
enum RemainderCheck {
    Unknown,
    Allowed,
    Rejected,
}

/// The state of one positional compound search.
struct PartSearch<'a, 'd> {
    word: &'a str,
    /// The case source, with the same characters at the same indices.
    case: &'a str,
    /// The byte offset of each character of the word and of the case
    /// source, and their lengths.
    offsets: &'a [usize],
    case_offsets: &'a [usize],
    junctions: &'a [Junction<'d>],
    remainder_checks: Vec<RemainderCheck>,
    /// The readings of each part tried, by its first and end character.
    readings: FoldMap<(usize, usize), Rc<[CompoundPart<'d>]>>,
}

/// The state of one `COMPOUNDRULE` search.
struct RuleSearch<'a, 'd> {
    word: &'a str,
    case: &'a str,
    offsets: &'a [usize],
    case_offsets: &'a [usize],
    rule: &'d CompoundRule,
}
