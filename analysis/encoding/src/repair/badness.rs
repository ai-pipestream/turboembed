//! ftfy's badness heuristic (`BADNESS_RE` in ftfy's badness.py): whether
//! a text holds a sequence of characters far more likely to be mojibake
//! than meant. Each arm below is one alternative of the regex, in its
//! order, matched at every position; the character classes are the
//! generated sets in `classes`. Ported from ftfy 6.3.1 by Robyn Speer
//! (Apache-2.0; see NOTICE).

use super::classes::*;
use turbo_unicode::whitespace::is_python_space;

/// The classes of each ASCII character, so the commonest lookups skip
/// the search.
static ASCII_CLASSES: [u64; 128] = {
    let mut table = [0; 128];
    let mut i = 0;
    while i < CLASSES.len() {
        let (cp, classes) = CLASSES[i];
        if cp < 128 {
            table[cp as usize] = classes;
        }
        i += 1;
    }
    table
};

/// The classes a character is in.
pub(crate) fn class(c: char) -> u64 {
    let cp = u32::from(c);
    if cp < 128 {
        return ASCII_CLASSES[cp as usize];
    }
    match CLASSES.binary_search_by_key(&cp, |&(k, _)| k) {
        Ok(i) => CLASSES[i].1,
        Err(_) => 0,
    }
}

/// Python's `\w` for str patterns: letters, numbers and the underscore.
fn is_word(c: char) -> bool {
    let gc = turbo_unicode::category::general_category(c);
    gc.is_letter() || gc.is_number() || c == '_'
}

/// The most characters one alternative of `BADNESS_RE` looks at.
const LOOK: usize = 7;

/// The characters from one position on, as far as any alternative looks,
/// with their classes, each looked up once as it enters the window. The
/// scan moves this window along the text, so it holds no table of
/// classes for the whole text.
struct Window {
    chars: [char; LOOK],
    classes: [u64; LOOK],
    len: usize,
}

impl Window {
    fn push(&mut self, c: char) {
        self.chars[self.len] = c;
        self.classes[self.len] = class(c);
        self.len += 1;
    }

    fn advance(&mut self) {
        self.chars.copy_within(1..self.len, 0);
        self.classes.copy_within(1..self.len, 0);
        self.len -= 1;
    }

    /// Whether character `j` of the window exists and is in one of `classes`.
    fn has(&self, j: usize, classes: u64) -> bool {
        j < self.len && self.classes[j] & classes != 0
    }

    /// Whether character `j` of the window exists and passes `f`.
    fn is(&self, j: usize, f: impl Fn(char) -> bool) -> bool {
        j < self.len && f(self.chars[j])
    }

    /// Whether character `j` of the window is `c`.
    fn eq(&self, j: usize, c: char) -> bool {
        j < self.len && self.chars[j] == c
    }

    /// Whether some alternative of ftfy's `BADNESS_RE` matches at the
    /// window's first character, `first` when that is the text's first.
    fn bad_here(&self, first: bool) -> bool {
        let at = |j: usize, classes: u64| self.has(j, classes);
        let is = |j: usize, f: fn(char) -> bool| self.is(j, f);
        let eq = |j: usize, c: char| self.eq(j, c);
        let c = self.chars[0];
        at(0, C1)
            || (at(0, BEFORE_BAD) && at(1, BAD))
            || (c.is_ascii_alphabetic() && at(1, COMMON_CASED) && at(2, BAD))
            || (at(0, BAD) && at(1, AFTER_BAD))
            || (at(0, BEFORE_UPPER_ACCENTED) && at(1, UPPER_ACCENTED))
            || (at(0, BEFORE_LOWER_ACCENTED) && at(1, LOWER_ACCENTED))
            || (at(0, BEFORE_CURRENCY) && at(1, CURRENCY))
            || (is_python_space(c) && at(1, UPPER_ACCENTED) && at(2, CURRENCY))
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
                && ((is(1, is_python_space) && is(2, is_cap_a_tilde_or_circumflex) && eq(3, ' '))
                    || (is(1, is_cap_a_tilde_or_circumflex) && eq(2, ' '))))
            || (first && is_cap_a_tilde_or_circumflex(c) && eq(1, ' '))
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

/// ftfy's `is_bad` over a stream of characters: whether any
/// alternative of `BADNESS_RE` matches.
fn any_bad(mut chars: impl Iterator<Item = char>) -> bool {
    let mut w = Window { chars: ['\0'; LOOK], classes: [0; LOOK], len: 0 };
    for c in chars.by_ref().take(LOOK) {
        w.push(c);
    }
    let mut first = true;
    while w.len > 0 {
        if w.bad_here(first) {
            return true;
        }
        first = false;
        w.advance();
        if let Some(c) = chars.next() {
            w.push(c);
        }
    }
    false
}

/// ftfy's `is_bad`: whether any alternative of `BADNESS_RE` matches.
pub(crate) fn is_bad_chars(chars: &[char]) -> bool {
    any_bad(chars.iter().copied())
}

/// Whether `text` looks like it holds mojibake, by ftfy's heuristic.
pub fn is_bad(text: &str) -> bool {
    any_bad(text.chars())
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
