//! ftfy's badness heuristic (`BADNESS_RE` in ftfy's badness.py): whether
//! a text holds a sequence of characters far more likely to be mojibake
//! than meant. Each arm below is one alternative of the regex, in its
//! order, matched at every position; the character classes are the
//! generated sets in `classes`. Ported from ftfy 6.3.1 by Robyn Speer
//! (Apache-2.0; see NOTICE).

use super::classes::*;

/// The classes a character is in.
pub(crate) fn class(c: char) -> u64 {
    let cp = u32::from(c);
    match CLASSES.binary_search_by_key(&cp, |&(k, _)| k) {
        Ok(i) => CLASSES[i].1,
        Err(_) => 0,
    }
}

/// Python's `\s` for str patterns: `str.isspace`, which is White_Space
/// plus the information separators U+001C to U+001F.
fn is_space(c: char) -> bool {
    turbo_unicode::props::white_space(c) || ('\u{1C}'..='\u{1F}').contains(&c)
}

/// Python's `\w` for str patterns: letters, numbers and the underscore.
fn is_word(c: char) -> bool {
    let gc = turbo_unicode::category::general_category(c);
    gc.is_letter() || gc.is_number() || c == '_'
}

/// A text as ftfy's patterns see it: characters with their classes.
pub(crate) struct Scan<'a> {
    chars: &'a [char],
    classes: Vec<u64>,
}

impl<'a> Scan<'a> {
    pub(crate) fn new(chars: &'a [char]) -> Scan<'a> {
        Scan { chars, classes: chars.iter().map(|&c| class(c)).collect() }
    }

    pub(crate) fn len(&self) -> usize {
        self.chars.len()
    }

    /// Whether character `i` exists and is in one of `classes`.
    pub(crate) fn has(&self, i: usize, classes: u64) -> bool {
        self.classes.get(i).is_some_and(|&m| m & classes != 0)
    }

    /// Whether character `i` exists and passes `f`.
    fn is(&self, i: usize, f: impl Fn(char) -> bool) -> bool {
        self.chars.get(i).is_some_and(|&c| f(c))
    }

    /// Whether character `i` is `c`.
    fn eq(&self, i: usize, c: char) -> bool {
        self.chars.get(i) == Some(&c)
    }

    /// Whether some alternative of ftfy's `BADNESS_RE` matches at `i`.
    fn bad_at(&self, i: usize) -> bool {
        let at = |j: usize, classes: u64| self.has(i + j, classes);
        let is = |j: usize, f: fn(char) -> bool| self.is(i + j, f);
        let eq = |j: usize, c: char| self.eq(i + j, c);
        let c = self.chars[i];
        at(0, C1)
            || (at(0, BEFORE_BAD) && at(1, BAD))
            || (c.is_ascii_alphabetic() && at(1, COMMON_CASED) && at(2, BAD))
            || (at(0, BAD) && at(1, AFTER_BAD))
            || (at(0, BEFORE_UPPER_ACCENTED) && at(1, UPPER_ACCENTED))
            || (at(0, BEFORE_LOWER_ACCENTED) && at(1, LOWER_ACCENTED))
            || (at(0, BEFORE_CURRENCY) && at(1, CURRENCY))
            || (is_space(c) && at(1, UPPER_ACCENTED) && at(2, CURRENCY))
            || (at(0, UPPER_ACCENTED_BOX) && at(1, NUMERIC_LAW))
            || (at(0, BEFORE_START_NUMERIC) && at(1, START_PUNCT) && at(2, NUMERIC))
            || (at(0, BEFORE_END_START) && at(1, END_PUNCT) && at(2, START_PUNCT))
            || (at(0, CURRENCY_NUMERIC_BOX) && at(1, START_PUNCT))
            || (c.is_ascii_lowercase() && at(1, UPPER_ACCENTED) && at(2, START_PUNCT_CURRENCY))
            || (at(0, BOX) && at(1, KAOMOJI))
            || (at(0, BEFORE_BOX) && at(1, BOX))
            || (at(0, BOX) && at(1, END_PUNCT))
            || (at(0, ACCENTED) && at(1, PUNCT) && is(2, is_word))
            // The ligature œ when not followed by an unaccented Latin letter.
            || (matches!(c, 'Œ' | 'œ') && is(1, |d| !d.is_ascii_alphabetic()))
            // Degree signs after capital letters.
            || (at(0, UPPER_ACCENTED) && eq(1, '°'))
            // Common windows-1252 two-character mojibake.
            || (matches!(c, 'Â' | 'Ã' | 'Î' | 'Ð') && at(1, AFTER_CAP_A_TILDE))
            || (c == '×' && is(1, |d| matches!(d, '²' | '³')))
            // windows-1252 mojibake of Arabic words, four characters.
            || (matches!(c, 'Ø' | 'Ù')
                && at(1, ARABIC_SECOND)
                && is(2, |d| matches!(d, 'Ø' | 'Ù'))
                && at(3, ARABIC_SECOND))
            // windows-1252 starts of three-byte South Asian letters.
            || (c == 'à' && is(1, |d| matches!(d, '²' | 'µ' | '¹' | '¼' | '½' | '¾')))
            // Mac OS Roman mojibake.
            || (c == '√'
                && is(1, |d| matches!(d, '±' | '∂' | '†' | '≠' | '®' | '™' | '´' | '≤' | '≥' | '¥' | 'µ' | 'ø')))
            || (c == '≈' && is(1, |d| matches!(d, '°' | '¢')))
            || (c == '‚'
                && eq(1, 'Ä')
                && is(2, |d| matches!(d, 'ì' | 'î' | 'ï' | 'ò' | 'ô' | 'ú' | 'ù' | 'û' | '†' | '°' | '¢' | 'π')))
            || (c == '‚' && is(1, |d| matches!(d, 'â' | 'ó')) && is(2, |d| matches!(d, 'à' | 'ä' | '°' | 'ê')))
            // windows-1251 mojibake of the U+2000 block.
            || (c == 'в' && eq(1, 'Ђ'))
            // windows-1251 mojibake of Latin-1 or Cyrillic, three characters.
            || (is_cyrillic_lead(c) && at(1, CYRILLIC_MIDDLE) && is(2, is_cyrillic_lead))
            // windows-1251 over Latin-1 over windows-1252, near a Latin letter.
            || (c == 'Г'
                && eq(1, 'ў')
                && eq(2, 'В')
                && eq(3, 'Ђ')
                && eq(4, 'В')
                && is(5, |d| d != '\n')
                && is(6, |d| d.is_ascii_alphabetic() || d == ' '))
            // windows-1252 encodings of 'à' and 'á', and of U+00A0 itself.
            || (c == 'Ã' && is(1, |d| matches!(d, '\u{A0}' | '¡')))
            || (c.is_ascii_lowercase()
                && ((is(1, is_space) && is(2, is_cap_a_tilde_or_circumflex) && eq(3, ' '))
                    || (is(1, is_cap_a_tilde_or_circumflex) && eq(2, ' '))))
            || (i == 0 && is_cap_a_tilde_or_circumflex(c) && eq(1, ' '))
            // Â standing for nothing before a character encoded as itself.
            || (at(0, BEFORE_CAP_A_CIRCUMFLEX) && eq(1, 'Â') && at(2, AFTER_CAP_A_CIRCUMFLEX))
            // windows-1253 mojibake of the U+2000 block.
            || (c == 'β' && eq(1, '€') && is(2, |d| matches!(d, '™' | '\u{A0}' | 'Ά' | '\u{AD}' | '®' | '°')))
            // windows-1253 mojibake of Latin-1 or Greek, three characters.
            || (is_greek_lead(c) && at(1, GREEK_MIDDLE) && is(2, is_greek_lead))
            // windows-1257 mojibake of the U+2000 block.
            || (c == 'ā' && eq(1, '€'))
    }
}

fn is_cyrillic_lead(c: char) -> bool {
    matches!(c, '\u{412}' | '\u{413}' | '\u{420}' | '\u{421}')
}

fn is_greek_lead(c: char) -> bool {
    matches!(c, '\u{392}' | '\u{393}' | '\u{39E}' | '\u{39F}')
}

fn is_cap_a_tilde_or_circumflex(c: char) -> bool {
    matches!(c, 'Ã' | 'Â')
}

/// ftfy's `is_bad`: whether any alternative of `BADNESS_RE` matches.
pub(crate) fn is_bad_chars(chars: &[char]) -> bool {
    let scan = Scan::new(chars);
    (0..scan.len()).any(|i| scan.bad_at(i))
}

/// Whether `text` looks like it holds mojibake, by ftfy's heuristic.
pub fn is_bad(text: &str) -> bool {
    is_bad_chars(&text.chars().collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mojibake_is_bad_and_plain_text_is_not() {
        assert!(is_bad("Ã©tÃ©"));
        assert!(is_bad("â€œquotedâ€\u{9D}"));
        assert!(is_bad("\u{80}"));
        assert!(!is_bad("été"));
        assert!(!is_bad("plain ASCII"));
        assert!(!is_bad(""));
    }

    #[test]
    fn the_start_anchor_holds_only_at_the_start() {
        assert!(is_bad("Ã x"));
        // One character in, "[a-z]\s?[ÃÂ][ ]" needs a lowercase letter
        // first; "1" is not one.
        assert!(!is_bad("1Ã x"));
    }
}
