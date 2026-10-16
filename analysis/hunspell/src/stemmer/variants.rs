//! The case variants a word is read in: OpenNLP's `variants`.

use crate::dictionary::case::DOTTED_CAPITAL_I;
use crate::text::{is_lower, is_title, is_upper};

use super::Stemmer;

/// The most variants the sharp s alternatives may bring the list to.
const MAX_CASE_VARIANTS: usize = 64;

/// Adds `value` to `list` unless the list holds it.
fn push_new(list: &mut Vec<String>, value: String) {
    if !list.contains(&value) {
        list.push(value);
    }
}

impl Stemmer<'_> {
    /// The variants in the order they are read, since the first reading
    /// found is the stem: the word itself; for an all upper case word or
    /// one capitalized with the rest lower case, its lower case (an all
    /// upper case word also capitalized, and with an apostrophe the part
    /// after it capitalized, for the elided articles of Catalan, French and
    /// Italian); for a capitalized word with a further capital, the word
    /// with its first letter lower case.
    pub(crate) fn variants(&self, surface: &str) -> Vec<String> {
        let case = self.dictionary.case;
        let mut variants = vec![surface.to_string()];
        let (mut upper, mut lower_after_first, mut all_upper, mut first_upper) = (false, true, true, false);
        let (mut uppers, mut letters) = (0usize, 0usize);
        for c in surface.chars() {
            if is_lower(c) {
                all_upper = false;
                letters += 1;
            } else if is_upper(c) || is_title(c) {
                if letters > 0 {
                    lower_after_first = false;
                }
                first_upper |= letters == 0;
                upper = true;
                uppers += 1;
                letters += 1;
            }
        }
        if upper && (all_upper || lower_after_first) {
            let lowered = case.lower(surface);
            if all_upper
                && let Some(apostrophe) = lowered.find('\'')
                && apostrophe > 0
                && apostrophe < lowered.len() - 1
            {
                let elided = format!("{}{}", &lowered[..=apostrophe], case.upper_initial(&lowered[apostrophe + 1..]));
                let capitalized = case.upper_initial(&elided);
                push_new(&mut variants, elided);
                push_new(&mut variants, capitalized);
            }
            if all_upper {
                push_new(&mut variants, lowered.clone());
                push_new(&mut variants, case.upper_initial(&lowered));
                if surface.starts_with(DOTTED_CAPITAL_I) {
                    // capitalizing keeps a dotted capital I, so an entry
                    // such as İnci is found outside the Turkic languages too
                    let first = lowered.chars().next().map_or(0, char::len_utf8);
                    push_new(&mut variants, format!("{DOTTED_CAPITAL_I}{}", &lowered[first..]));
                }
                if self.dictionary.settings.check_sharps {
                    self.add_sharp_variants(&lowered, 0, &mut variants);
                }
            } else {
                push_new(&mut variants, lowered);
            }
        } else if first_upper && uppers > 1 && !all_upper {
            push_new(&mut variants, case.lower_initial(surface));
        }
        variants
    }

    /// Adds the forms with `ss` written `ß` from byte `from` on, each also
    /// capitalized, one replacement set at a time, until the list holds
    /// [`MAX_CASE_VARIANTS`].
    fn add_sharp_variants(&self, word: &str, from: usize, variants: &mut Vec<String>) {
        if variants.len() >= MAX_CASE_VARIANTS {
            return;
        }
        let case = self.dictionary.case;
        let mut found = word[from..].find("ss").map(|i| i + from);
        while let Some(at) = found {
            let changed = format!("{}ß{}", &word[..at], &word[at + 2..]);
            let capitalized = case.upper_initial(&changed);
            push_new(variants, changed.clone());
            push_new(variants, capitalized);
            // after the ß, which is two bytes where Java counts one unit
            self.add_sharp_variants(&changed, at + 'ß'.len_utf8(), variants);
            if variants.len() >= MAX_CASE_VARIANTS {
                return;
            }
            found = word[at + 1..].find("ss").map(|i| i + at + 1);
        }
    }
}
