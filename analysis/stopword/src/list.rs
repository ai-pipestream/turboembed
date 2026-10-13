//! The bundled lists and the lookup.

use turbo_stopword_layers::StopwordLanguage;
use turbo_unicode::java::to_lower_char;

mod data {
    include!(concat!(env!("OUT_DIR"), "/lists.rs"));
}

use data::MAX_BYTES;

static LISTS: [StopwordList; 11] = [
    StopwordList { language: StopwordLanguage::Bulgarian, words: &data::BG },
    StopwordList { language: StopwordLanguage::Danish, words: &data::DA },
    StopwordList { language: StopwordLanguage::German, words: &data::DE },
    StopwordList { language: StopwordLanguage::English, words: &data::EN },
    StopwordList { language: StopwordLanguage::Spanish, words: &data::ES },
    StopwordList { language: StopwordLanguage::Finnish, words: &data::FI },
    StopwordList { language: StopwordLanguage::French, words: &data::FR },
    StopwordList { language: StopwordLanguage::Italian, words: &data::IT },
    StopwordList { language: StopwordLanguage::Dutch, words: &data::NL },
    StopwordList { language: StopwordLanguage::Portuguese, words: &data::PT },
    StopwordList { language: StopwordLanguage::Russian, words: &data::RU },
];

/// One bundled list: the filter `StopwordLists.forLanguage` returns.
#[derive(Debug)]
pub struct StopwordList {
    language: StopwordLanguage,
    /// Sorted by UTF-8 bytes, without duplicates.
    words: &'static [&'static str],
}

impl StopwordList {
    /// The list OpenNLP bundles for `language`.
    pub fn bundled(language: StopwordLanguage) -> &'static StopwordList {
        &LISTS[language as usize]
    }

    pub fn language(&self) -> StopwordLanguage {
        self.language
    }

    /// The words, sorted by UTF-8 bytes (for these lists, which hold no
    /// character outside the Basic Multilingual Plane, the order of
    /// `String.compareTo`). They are the set `StopwordFilter.stopwords()`
    /// returns, as written in the file: every one is already lowercase.
    pub fn words(&self) -> &'static [&'static str] {
        self.words
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// `StopwordFilter.isStopword(token)`: whether `token`, lowercased one
    /// character at a time with `Character.toLowerCase(int)`, is a word of
    /// the list. There is no other folding: accents, `ß` against `ss` and
    /// Unicode normalization forms all count as different words. The crate
    /// documentation says why this is exactly what OpenNLP's
    /// case-insensitive lookup does for these lists.
    pub fn contains(&self, token: &str) -> bool {
        // Lowercased text has at least one byte per character and the input
        // at most four, so a token longer than this cannot lowercase to a
        // word of the list.
        if token.len() > 4 * MAX_BYTES {
            return false;
        }
        if token.len() <= MAX_BYTES && token.bytes().all(|b| b.is_ascii() && !b.is_ascii_uppercase()) {
            return self.find(token.as_bytes());
        }
        let mut buf = [0u8; MAX_BYTES];
        let mut n = 0;
        for c in token.chars() {
            let lower = to_lower_char(c);
            let end = n + lower.len_utf8();
            if end > MAX_BYTES {
                return false;
            }
            lower.encode_utf8(&mut buf[n..end]);
            n = end;
        }
        self.find(&buf[..n])
    }

    /// Appends a flag per token to `out`: whether it is a stopword.
    pub fn flag<'t>(&self, tokens: impl IntoIterator<Item = &'t str>, out: &mut Vec<bool>) {
        out.extend(tokens.into_iter().map(|t| self.contains(t)));
    }

    fn find(&self, lower: &[u8]) -> bool {
        self.words.binary_search_by(|w| w.as_bytes().cmp(lower)).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_unicode::java::to_upper_char;

    #[test]
    fn lists_are_sorted_lowercase_and_in_language_order() {
        for (i, list) in LISTS.iter().enumerate() {
            assert_eq!(list.language() as usize, i);
            assert!(!list.is_empty());
            assert!(list.words.windows(2).all(|w| w[0] < w[1]), "{:?}", list.language);
            for w in list.words {
                assert!(w.chars().all(|c| to_lower_char(c) == c), "{w} is not lowercase");
                assert!(w.len() <= MAX_BYTES);
            }
        }
        let sizes: Vec<usize> = LISTS.iter().map(StopwordList::len).collect();
        // Finnish has three words twice in its file (minä, sinä, sillä).
        assert_eq!(sizes, [190, 94, 231, 174, 308, 229, 154, 279, 101, 203, 159]);
    }

    #[test]
    fn lookup_lowercases_per_character_and_folds_nothing_else() {
        let en = StopwordList::bundled(StopwordLanguage::English);
        // U+0130 lowercases to a plain i with Character.toLowerCase.
        for yes in ["the", "THE", "The", "tHe", "\u{130}T"] {
            assert!(en.contains(yes), "{yes}");
        }
        for no in ["", "the ", " the", "th", "thee", "\u{17F}o", "\u{131}t", "the\u{1D41A}"] {
            assert!(!en.contains(no), "{no:?}");
        }
        let de = StopwordList::bundled(StopwordLanguage::German);
        assert!(de.contains("da\u{DF}"));
        assert!(de.contains("DA\u{1E9E}"));
        assert!(!de.contains("DASS"));
        assert!(de.contains("\u{DC}BER"));
        assert!(!de.contains("u\u{308}ber"));
        assert!(!de.contains("uber"));
        let ru = StopwordList::bundled(StopwordLanguage::Russian);
        assert!(ru.contains("\u{412}\u{42B}"));
        assert!(!ru.contains("\u{1C80}\u{44B}"));
        // U+212B ANGSTROM SIGN lowercases to U+00E5, two bytes from three.
        let da = StopwordList::bundled(StopwordLanguage::Danish);
        assert!(da.contains("P\u{212B}"));
        let long = "a".repeat(4 * MAX_BYTES + 1);
        assert!(!en.contains(&long));
        let mut flags = Vec::new();
        en.flag(["The", "cat", "is", "here"], &mut flags);
        assert_eq!(flags, [true, false, true, true]);
    }

    /// `String.hashCode` of the per-character lowercase form.
    fn java_hash(chars: impl Iterator<Item = char>) -> u32 {
        let mut h = 0u32;
        let mut units = [0u16; 2];
        for c in chars {
            for &u in c.encode_utf16(&mut units).iter() {
                h = h.wrapping_mul(31).wrapping_add(u32::from(u));
            }
        }
        h
    }

    /// OpenNLP finds a word in a `HashSet` whose hash is that of the
    /// lowercased word and whose equality is `String.compareToIgnoreCase`,
    /// which also equates a few characters that lowercase differently
    /// (U+017F with `s`, U+1C80 with `в`). Such a token matches in Java only
    /// when its lowercased hash equals the word's. This checks every
    /// spelling of every word that differs from it only in those characters
    /// and finds no equal hash, so lowercasing and comparing, as
    /// `contains` does, gives Java's answer for every token.
    #[test]
    fn no_spelling_that_only_compare_to_ignore_case_equates_shares_a_hash() {
        // Characters whose lowercase differs from the lowercase of their
        // uppercase, by what that lowercase of the uppercase is.
        let mut irregular: Vec<(char, char)> = Vec::new();
        for c in (0..=0x10FFFF).filter_map(char::from_u32) {
            let fold = to_lower_char(to_upper_char(c));
            if fold != to_lower_char(c) {
                irregular.push((fold, to_lower_char(c)));
            }
        }
        assert!(irregular.contains(&('s', '\u{17F}')) && irregular.contains(&('\u{432}', '\u{1C80}')));
        let mut checked = 0;
        for list in &LISTS {
            for w in list.words {
                let base: Vec<char> = w.chars().collect();
                let options: Vec<Vec<char>> = base
                    .iter()
                    .map(|&c| std::iter::once(c).chain(irregular.iter().filter(|p| p.0 == c).map(|p| p.1)).collect())
                    .collect();
                let hash = java_hash(base.iter().copied());
                let mut pick = vec![0usize; base.len()];
                loop {
                    let mut i = 0;
                    while i < pick.len() && pick[i] + 1 == options[i].len() {
                        pick[i] = 0;
                        i += 1;
                    }
                    if i == pick.len() {
                        break;
                    }
                    pick[i] += 1;
                    checked += 1;
                    let variant = pick.iter().zip(&options).map(|(&k, o)| o[k]);
                    assert_ne!(java_hash(variant), hash, "{w}");
                }
            }
        }
        assert!(checked > 3000, "{checked}");
    }
}
