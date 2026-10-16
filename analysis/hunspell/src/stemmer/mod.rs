//! Stemming against a dictionary: OpenNLP's `HunspellStemmer`.
//!
//! A word is read in each of its case variants: as a listed entry, with one
//! suffix, one prefix, both (a cross product), or a further affix that a
//! continuation class allows undone. When none of that reads it, it is
//! split into compound parts, and then at its word breaks (by default a
//! hyphen), each piece read the same way.

mod affixes;
mod compound;
mod parts;
mod results;
mod variants;

use std::collections::HashMap;
use std::rc::Rc;

use crate::affix::Affix;
use crate::dictionary::Dictionary;
use crate::dictionary::case::{CaseType, case_type};
use crate::dictionary::{EntryId, Homonyms};

use results::Results;

/// The most compound part checks of one compound search, and of one word's
/// break search.
const PART_CHECK_BUDGET: i32 = 2048;

/// The most parts of one compound, and the deepest nesting of word breaks.
const MAX_COMPOUND_PARTS: usize = 64;

/// A stemmer over a [`Dictionary`]. It holds no state of its own, so one
/// can be shared between threads like the dictionary.
#[derive(Copy, Clone, Debug)]
pub struct Stemmer<'d> {
    dictionary: &'d Dictionary,
}

impl<'d> Stemmer<'d> {
    pub fn new(dictionary: &'d Dictionary) -> Stemmer<'d> {
        Stemmer { dictionary }
    }

    pub fn dictionary(&self) -> &'d Dictionary {
        self.dictionary
    }

    /// The distinct stems of `word`, the word's own entry first, with the
    /// output conversion (`OCONV`) applied; empty when the dictionary does
    /// not read the word. OpenNLP's `stemAll` returns the word itself in
    /// that case.
    pub fn stems(&self, word: &str) -> Vec<String> {
        let input = self.dictionary.input_form(word);
        let found = self.find_word(&input, false);
        let mut out = results::OrderedSet::default();
        for stem in found.iter() {
            out.insert(self.dictionary.output_form(stem).into_owned());
        }
        out.into_vec()
    }

    /// The first of [`Stemmer::stems`], `None` when there is none. OpenNLP's
    /// `stem` returns the word itself in that case.
    pub fn stem(&self, word: &str) -> Option<String> {
        self.stems(word).into_iter().next()
    }

    /// The morphological analyses of `word`: each a space separated list
    /// of Hunspell fields (`st:walk po:verb is:past`), a compound's parts
    /// each introduced by `pa:` and the part's text. At most 2048, without
    /// the output conversion; empty when the dictionary does not read the
    /// word.
    pub fn analyze(&self, word: &str) -> Vec<String> {
        let input = self.dictionary.input_form(word);
        self.find_word(&input, true).to_vec()
    }

    /// The readings of a whole input. Trailing periods are removed first so
    /// that an abbreviation finds its entry; when the shorter form has no
    /// reading, one period is put back for entries listed with it.
    fn find_word(&self, input: &str, morphological: bool) -> Rc<[String]> {
        let end = input.trim_end_matches('.').len();
        if end == 0 {
            // without this a rule that only strips could restore its strip
            // text onto nothing
            return Rc::from([]);
        }
        let found = self.find_stems(&input[..end], 0, &mut { PART_CHECK_BUDGET }, &mut HashMap::new(), morphological);
        if found.is_empty() && end < input.len() {
            return self.find_stems(
                &input[..end + 1],
                0,
                &mut { PART_CHECK_BUDGET },
                &mut HashMap::new(),
                morphological,
            );
        }
        found
    }

    /// The readings of an input or of a piece between word breaks.
    fn find_stems(
        &self,
        input: &str,
        depth: usize,
        budget: &mut i32,
        cache: &mut HashMap<String, Rc<[String]>>,
        morphological: bool,
    ) -> Rc<[String]> {
        if input.is_empty() || depth >= MAX_COMPOUND_PARTS || *budget <= 0 {
            return Rc::from([]);
        }
        if let Some(cached) = cache.get(input) {
            return Rc::clone(cached);
        }
        let d = self.dictionary;
        if d.first_forbidden(d.lookup(input)) {
            return Rc::from([]);
        }
        let mut results = Results::new(morphological);
        let all_caps = case_type(input) == CaseType::Upper;
        let variants = self.variants(input);
        for variant in &variants {
            self.analyze_variant(variant, &mut Analysis { surface: input, variant, results: &mut results, all_caps });
        }
        // a forbidden direct or affixed reading forbids the spelling, so no
        // compound or break reading is tried
        if results.is_empty() && !results.forbidden && d.compounds_declared() {
            for variant in &variants {
                self.decompose(variant, input, &mut results);
            }
        }
        if results.is_empty() && !results.forbidden {
            self.read_breaks(input, depth, budget, cache, &mut results);
        }
        let found: Rc<[String]> = results.values.into_vec().into();
        cache.insert(input.to_string(), Rc::clone(&found));
        found
    }

    /// Reads `input` at each place a `BREAK` separator may split it: the
    /// piece after a separator bound to the start, the piece before one
    /// bound to the end, or both pieces of an inner one.
    fn read_breaks(
        &self,
        input: &str,
        depth: usize,
        budget: &mut i32,
        cache: &mut HashMap<String, Rc<[String]>>,
        results: &mut Results,
    ) {
        let morphological = results.morphological;
        for word_break in self.dictionary.word_breaks.iter() {
            if *budget <= 0 {
                break;
            }
            let (start, end, separator) = (word_break.at_start, word_break.at_end, &*word_break.separator);
            let mut found = input.find(separator);
            while let Some(at) = found {
                if *budget <= 0 {
                    break;
                }
                let after = at + separator.len();
                found = input[after..].find(separator).map(|i| i + after);
                if (start && at != 0)
                    || (end && after != input.len())
                    || (!start && at == 0)
                    || (!end && after == input.len())
                {
                    continue;
                }
                *budget -= 1;
                if start && !end {
                    let right = self.find_stems(&input[after..], depth + 1, budget, cache, morphological);
                    results.add_all(right.iter().cloned());
                } else if end && !start {
                    let left = self.find_stems(&input[..at], depth + 1, budget, cache, morphological);
                    results.add_all(left.iter().cloned());
                } else if !start {
                    let mut left = self.find_stems(&input[..at], depth + 1, budget, cache, morphological);
                    if left.is_empty() && separator == "-" && self.dictionary.hungarian {
                        left = self.hyphenated_first_part(&input[..at], morphological);
                    }
                    if !left.is_empty() {
                        let right = self.find_stems(&input[after..], depth + 1, budget, cache, morphological);
                        if !right.is_empty() {
                            results.add_broken(&input[..at], &left, &input[after..], &right);
                        }
                    }
                }
            }
        }
    }

    /// The readings of the part of a Hungarian word before a hyphen: a
    /// listed word ending in the hyphen, or a compound under the relaxed
    /// rules for such a part.
    fn hyphenated_first_part(&self, text: &str, morphological: bool) -> Rc<[String]> {
        let mut results = Results::new(morphological);
        let hyphenated = format!("{text}-");
        let all_caps = case_type(text) == CaseType::Upper;
        for variant in &self.variants(&hyphenated) {
            self.analyze_variant(
                variant,
                &mut Analysis { surface: &hyphenated, variant, results: &mut results, all_caps },
            );
        }
        if results.is_empty() && !results.forbidden && self.dictionary.compounds_declared() {
            results.hyphenated_first_part = true;
            for variant in &self.variants(text) {
                self.decompose(variant, text, &mut results);
            }
        }
        results.values.into_vec().into()
    }
}

/// The reading of one case variant of an input.
pub(crate) struct Analysis<'a> {
    /// The input after conversion.
    surface: &'a str,
    /// The case variant read.
    variant: &'a str,
    results: &'a mut Results,
    /// The input is all upper case, so the hidden capitalized forms of
    /// mixed-case entries match too.
    all_caps: bool,
}

impl Analysis<'_> {
    fn lookup<'d>(&self, dictionary: &'d Dictionary, word: &str) -> Homonyms<'d> {
        dictionary.lookup_with_hidden(word, self.all_caps)
    }

    /// Adds the stems or analyses of a reading that keeps to the case
    /// restrictions.
    fn add(&mut self, dictionary: &Dictionary, root: &str, entry: EntryId, affixes: &[&Affix]) {
        if dictionary.accepts_case(dictionary.flags(entry), self.surface, self.variant, affixes) {
            if self.results.morphological {
                self.results.add_all(dictionary.morphological_analyses(root, entry, false, affixes));
            } else {
                self.results.add_all(dictionary.morphological_stems(root, entry, affixes));
            }
        }
    }

    /// Notes that undoing the affix of `flag` reached a forbidden entry.
    fn note_forbidden(&mut self, dictionary: &Dictionary, entry: EntryId, flag: crate::flag::Flag) {
        if dictionary.forbids_affixed(dictionary.flags(entry), flag) {
            self.results.forbidden = true;
        }
    }
}
