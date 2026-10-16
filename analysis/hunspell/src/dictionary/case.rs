//! Letter case as the dictionary sees it: the capitalization classes of a
//! word, and lower and upper case with the Turkic mapping of `i` when
//! `LANG` names Turkish, Azerbaijani or Crimean Tatar.

use crate::text::{is_lower, is_title, is_upper, lower_char, push_lowercase, upper_unit};

/// The capitalization class of a word: OpenNLP's `CaseType` (`NOCAP`,
/// `INITCAP`, `ALLCAP`, `HUHINITCAP`, `HUHCAP`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CaseType {
    /// No upper case letter.
    Lower,
    /// One upper case letter, the first character.
    Initial,
    /// Every letter upper case.
    Upper,
    /// An upper case first letter and another upper case letter among
    /// lower case ones.
    MixedInitial,
    /// Upper case letters after a lower case first letter.
    Mixed,
}

/// The class of `word`. A title case letter counts as upper case; only the
/// first character can make the initial upper case.
pub(crate) fn case_type(word: &str) -> CaseType {
    let (mut letters, mut uppers, mut first_upper) = (0usize, 0usize, false);
    for (i, c) in word.char_indices() {
        if is_lower(c) {
            letters += 1;
        } else if is_upper(c) || is_title(c) {
            first_upper |= i == 0;
            uppers += 1;
            letters += 1;
        }
    }
    if uppers == 0 {
        CaseType::Lower
    } else if uppers == 1 && first_upper {
        CaseType::Initial
    } else if uppers == letters {
        CaseType::Upper
    } else if first_upper {
        CaseType::MixedInitial
    } else {
        CaseType::Mixed
    }
}

/// The dotted capital I, upper case `i` in the Turkic languages.
pub(crate) const DOTTED_CAPITAL_I: char = '\u{130}';
/// The dotless small i, lower case `I` in the Turkic languages.
const DOTLESS_SMALL_I: char = '\u{131}';

/// How the dictionary maps case.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CaseMapping {
    pub(crate) turkic: bool,
}

impl CaseMapping {
    /// The mapping `LANG` selects: Turkic for `tr`, `az` and `crh`, alone or
    /// followed by `_` and a region.
    pub(crate) fn for_language(language: &str) -> CaseMapping {
        let is = |code: &str| language == code || language.strip_prefix(code).is_some_and(|r| r.starts_with('_'));
        CaseMapping { turkic: is("tr") || is("az") || is("crh") }
    }

    /// The dictionary's lower case: per character, with `I` to `ı` and `İ`
    /// to `i` in Turkic.
    pub(crate) fn lower(self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        if self.turkic {
            out.extend(text.chars().map(|c| match c {
                'I' => DOTLESS_SMALL_I,
                DOTTED_CAPITAL_I => 'i',
                c => lower_char(c),
            }));
        } else {
            push_lowercase(text, &mut out);
        }
        out
    }

    /// `word` with its first character upper case, `i` to `İ` and `ı` to
    /// `I` in Turkic. `word` is not empty.
    pub(crate) fn upper_initial(self, word: &str) -> String {
        let first = word.chars().next().expect("the word is not empty");
        let upper = match first {
            'i' if self.turkic => DOTTED_CAPITAL_I,
            DOTLESS_SMALL_I if self.turkic => 'I',
            c => upper_unit(c),
        };
        let mut out = String::with_capacity(word.len() + 1);
        out.push(upper);
        out.push_str(&word[first.len_utf8()..]);
        out
    }

    /// `word` with its first character lower case. `word` is not empty.
    pub(crate) fn lower_initial(self, word: &str) -> String {
        let first = word.chars().next().expect("the word is not empty");
        let (head, rest) = word.split_at(first.len_utf8());
        let mut out = self.lower(head);
        out.push_str(rest);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert_eq!(case_type("walk"), CaseType::Lower);
        assert_eq!(case_type("Walk"), CaseType::Initial);
        assert_eq!(case_type("WALK"), CaseType::Upper);
        assert_eq!(case_type("McDonald"), CaseType::MixedInitial);
        assert_eq!(case_type("eBook"), CaseType::Mixed);
        assert_eq!(case_type("A1"), CaseType::Initial);
        assert_eq!(case_type("1A"), CaseType::Upper);
        assert_eq!(case_type(""), CaseType::Lower);
    }

    #[test]
    fn turkic_case() {
        let t = CaseMapping::for_language("tr_TR");
        assert!(t.turkic);
        assert!(!CaseMapping::for_language("trx").turkic);
        assert_eq!(t.lower("DİYARBAKIR"), "diyarbakır");
        assert_eq!(t.upper_initial("inci"), "\u{130}nci");
        let d = CaseMapping { turkic: false };
        assert_eq!(d.lower("İ"), "i");
        assert_eq!(d.upper_initial("\u{10428}x"), "\u{10428}x");
        assert_eq!(d.lower_initial("ABC"), "aBC");
    }
}
