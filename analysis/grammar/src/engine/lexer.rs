//! One pass over a text view that cuts it into lexemes: words, numerals
//! and single punctuation characters.
//!
//! The lexer works on the view's characters, not on a token layer: the
//! spans a grammar reports start and end on character boundaries that a
//! tokenizer may not have cut at (`10k`, `twenty-one`, `4/5`).
//!
//! - A word is a run of alphabetic characters and marks. Its lowercase
//!   form goes into one buffer shared by all lexemes.
//! - A numeral is a chain of ASCII digits with separators, read by the
//!   grammar's [`NumeralStyle`]; a leading dot or comma after whitespace
//!   or at the start of the text opens one too (`.25`).
//! - Any other character that is not whitespace is a punctuation lexeme
//!   of its own, with its lowercase form in the buffer, so a lexicon can
//!   hold `%` or `½`.
//!
//! A whitespace-delimited chunk that holds `://` or starts with `www.`
//! is a web address and yields no lexemes at all.
//!
//! A grammar can ask for two more things through [`WordForms`]: that the
//! lowercase form of a word drop its accents (`miércoles` is looked up as
//! `miercoles`), and that a word be cut into glued words (`tresmillonesimo`
//! into `tres` and `millonesimo`). Spans stay those of the text as written.

use crate::dec::Dec;

use super::numeral::{NumeralStyle, Reading, digits_end};

/// What a lexeme is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Word,
    Numeral,
    Punct,
}

/// One lexeme: a byte span of the text and a range of the lowercase
/// buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Lexeme {
    pub start: u32,
    pub end: u32,
    pub kind: Kind,
    /// No whitespace (and no skipped web address) between this lexeme and
    /// the one before it.
    pub glued: bool,
    /// The lowercase form's range in [`Lexed::lower`].
    pub low: (u32, u32),
    /// For a numeral, its index in [`Lexed::numerals`].
    pub numeral: u32,
}

impl Lexeme {
    /// Whether the lexeme is a word or a numeral: two of these glued
    /// together are parts of one written token.
    pub fn is_alnum(&self) -> bool {
        self.kind != Kind::Punct
    }
}

/// The lexemes of one text, with their buffers. Kept between texts so the
/// buffers are reused.
#[derive(Clone, Debug, Default)]
pub struct Lexed {
    pub lexemes: Vec<Lexeme>,
    /// Lowercase forms of every lexeme, back to back.
    pub lower: String,
    /// The value of every numeral lexeme, in order.
    pub numerals: Vec<Dec>,
    digits: Vec<u8>,
    /// For the word being read: the offset in [`Lexed::lower`] and in the
    /// text where each of its characters starts.
    bounds: Vec<(u32, u32)>,
    cuts: Vec<u32>,
}

/// How a grammar wants its words: accents folded, and compounds cut.
pub trait WordForms {
    /// Whether the lowercase form of a word drops its accents (see
    /// [`fold`]).
    fn folds(&self) -> bool {
        false
    }

    /// Byte offsets into `word`, a lowercase form, where it is cut into
    /// glued words, in increasing order; none leaves it whole. Every offset
    /// is a character boundary inside the word.
    fn split(&self, _word: &str, _cuts: &mut Vec<u32>) {}
}

/// The forms of the plain lexer: no folding, no cutting.
pub struct PlainWords;

impl WordForms for PlainWords {}

/// A lowercase letter without its accent: `á` is `a`, `ü` is `u`, `º` is
/// `o`. `ñ` and `ç` are letters of their own and stay.
pub fn fold(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ä' | 'ã' | 'å' | 'ª' => 'a',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ò' | 'ó' | 'ô' | 'ö' | 'õ' | 'º' => 'o',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ý' | 'ÿ' => 'y',
        _ => c,
    }
}

/// Whether `c` is a combining accent that folding drops.
fn is_accent_mark(c: char) -> bool {
    ('\u{300}'..='\u{36f}').contains(&c)
}

/// Whether `c` belongs in a word.
fn is_word_char(c: char) -> bool {
    turbo_unicode::props::alphabetic(c) || turbo_unicode::category::general_category(c).is_mark()
}

/// Whether ASCII byte `b` is `White_Space`.
fn ascii_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t'..=b'\r')
}

/// The character at byte `i`, which is a character boundary.
fn char_at(text: &str, i: usize) -> char {
    let b = text.as_bytes()[i];
    if b < 0x80 { char::from(b) } else { text[i..].chars().next().expect("a character at a boundary") }
}

/// The end of the whitespace-delimited chunk starting at `i`.
fn chunk_end(text: &str, mut i: usize) -> usize {
    let b = text.as_bytes();
    while i < b.len() {
        if b[i] < 0x80 {
            if ascii_space(b[i]) {
                return i;
            }
            i += 1;
        } else {
            let c = char_at(text, i);
            if turbo_unicode::props::white_space(c) {
                return i;
            }
            i += c.len_utf8();
        }
    }
    i
}

/// Whether the chunk is a web address.
fn is_address(chunk: &str) -> bool {
    let b = chunk.as_bytes();
    (b.len() > 4 && b[..4].eq_ignore_ascii_case(b"www.")) || b.windows(3).any(|w| w == b"://")
}

impl Lexed {
    /// Cuts `text` into lexemes, replacing what the buffers held.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes.
    pub fn lex(&mut self, text: &str, style: &NumeralStyle) {
        self.lex_words(text, style, &PlainWords);
    }

    /// Cuts `text` into lexemes with the word forms `words` asks for.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes.
    pub fn lex_words(&mut self, text: &str, style: &NumeralStyle, words: &dyn WordForms) {
        let folds = words.folds();
        assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
        self.lexemes.clear();
        self.lower.clear();
        self.numerals.clear();
        let b = text.as_bytes();
        let mut i = 0;
        let mut glued = false;
        // The end of a chain whose digit runs are read one by one.
        let mut plain_until = 0;
        while i < b.len() {
            let c = char_at(text, i);
            if if c.is_ascii() { ascii_space(b[i]) } else { turbo_unicode::props::white_space(c) } {
                i += c.len_utf8();
                glued = false;
                continue;
            }
            if !glued {
                let end = chunk_end(text, i);
                if is_address(&text[i..end]) {
                    i = end;
                    continue;
                }
            }
            let start = i;
            if c.is_ascii_digit() {
                if i < plain_until {
                    i = digits_end(b, i);
                    self.push_numeral(text, start, i, Dec::from_digits(&b[start..i], b""), glued);
                } else {
                    match style.read(text, i, &mut self.digits) {
                        Reading::One { end, value } => {
                            i = end;
                            self.push_numeral(text, start, end, value, glued);
                        }
                        Reading::List { end } => {
                            plain_until = end;
                            i = digits_end(b, i);
                            self.push_numeral(text, start, i, Dec::from_digits(&b[start..i], b""), glued);
                        }
                    }
                }
            } else if (c == '.' || c == ',') && !glued && b.get(i + 1).is_some_and(u8::is_ascii_digit) && {
                // A fraction without its integer part, `.25`, unless more
                // separated digits follow (`.1.2`).
                let e = digits_end(b, i + 1);
                !(matches!(b.get(e), Some(b'.' | b',')) && b.get(e + 1).is_some_and(u8::is_ascii_digit))
            } {
                i = digits_end(b, i + 1);
                let value = Dec::from_digits(b"", &b[start + 1..i]);
                self.push_numeral(text, start, i, value, glued);
            } else if is_word_char(c) {
                let low_start = self.lower.len();
                self.bounds.clear();
                while i < b.len() {
                    self.bounds.push((self.lower.len() as u32, i as u32));
                    if b[i].is_ascii_alphabetic() {
                        self.lower.push(char::from(b[i].to_ascii_lowercase()));
                        i += 1;
                        continue;
                    }
                    let c = char_at(text, i);
                    if c.is_ascii() || !is_word_char(c) {
                        self.bounds.pop();
                        break;
                    }
                    i += c.len_utf8();
                    if !folds {
                        self.lower.extend(c.to_lowercase());
                    } else if is_accent_mark(c) {
                        // `n` and a combining tilde is `ñ`; other accents go.
                        if c == '\u{303}' && self.lower.ends_with('n') {
                            self.lower.pop();
                            self.lower.push('ñ');
                        }
                    } else {
                        self.lower.extend(c.to_lowercase().map(fold));
                    }
                }
                self.push_word(start, i, glued, low_start, words);
            } else {
                let low_start = self.lower.len();
                self.lower.extend(c.to_lowercase());
                i += c.len_utf8();
                self.push(start, i, Kind::Punct, glued, low_start, u32::MAX);
            }
            glued = true;
        }
    }

    /// Pushes the word `start..end`, cut where `words` says.
    fn push_word(&mut self, start: usize, end: usize, glued: bool, low_start: usize, words: &dyn WordForms) {
        self.cuts.clear();
        words.split(&self.lower[low_start..], &mut self.cuts);
        if self.cuts.is_empty() {
            self.push(start, end, Kind::Word, glued, low_start, u32::MAX);
            return;
        }
        let (mut text_at, mut low_at, mut glue) = (start, low_start, glued);
        for k in 0..self.cuts.len() {
            let low_cut = low_start + self.cuts[k] as usize;
            // A dropped accent shares its offset with the next letter; the
            // cut goes before that letter.
            let Some(&(_, text_cut)) = self.bounds.iter().rev().find(|b| b.0 as usize == low_cut) else { continue };
            let text_cut = text_cut as usize;
            if low_cut <= low_at || text_cut <= text_at {
                continue;
            }
            self.lexemes.push(Lexeme {
                start: text_at as u32,
                end: text_cut as u32,
                kind: Kind::Word,
                glued: glue,
                low: (low_at as u32, low_cut as u32),
                numeral: u32::MAX,
            });
            (text_at, low_at, glue) = (text_cut, low_cut, true);
        }
        self.lexemes.push(Lexeme {
            start: text_at as u32,
            end: end as u32,
            kind: Kind::Word,
            glued: glue,
            low: (low_at as u32, self.lower.len() as u32),
            numeral: u32::MAX,
        });
    }

    fn push_numeral(&mut self, text: &str, start: usize, end: usize, value: Dec, glued: bool) {
        let low_start = self.lower.len();
        self.lower.push_str(&text[start..end]);
        let ix = self.numerals.len() as u32;
        self.numerals.push(value);
        self.push(start, end, Kind::Numeral, glued, low_start, ix);
    }

    fn push(&mut self, start: usize, end: usize, kind: Kind, glued: bool, low_start: usize, numeral: u32) {
        self.lexemes.push(Lexeme {
            start: start as u32,
            end: end as u32,
            kind,
            glued,
            low: (low_start as u32, self.lower.len() as u32),
            numeral,
        });
    }

    pub fn len(&self) -> usize {
        self.lexemes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lexemes.is_empty()
    }

    /// The lowercase form of lexeme `i`.
    pub fn word(&self, i: usize) -> &str {
        let (s, e) = self.lexemes[i].low;
        &self.lower[s as usize..e as usize]
    }

    /// Lexeme `i` as written.
    pub fn raw<'t>(&self, text: &'t str, i: usize) -> &'t str {
        let l = &self.lexemes[i];
        &text[l.start as usize..l.end as usize]
    }

    /// The value of lexeme `i` when it is a numeral.
    pub fn numeral(&self, i: usize) -> Option<Dec> {
        self.numerals.get(self.lexemes[i].numeral as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EN: NumeralStyle =
        NumeralStyle { mixed_separators: true, space_groups: true, indian_groups: true, decimal_comma: false };

    fn lex(text: &str) -> Vec<(String, Kind, bool)> {
        let mut l = Lexed::default();
        l.lex(text, &EN);
        (0..l.len()).map(|i| (l.word(i).to_string(), l.lexemes[i].kind, l.lexemes[i].glued)).collect()
    }

    fn words(text: &str) -> Vec<String> {
        lex(text).into_iter().map(|(w, _, _)| w).collect()
    }

    #[test]
    fn words_are_lowercased_and_glue_is_recorded() {
        use Kind::*;
        assert_eq!(
            lex("Twenty-ONE  10k"),
            [
                ("twenty".into(), Word, false),
                ("-".into(), Punct, true),
                ("one".into(), Word, true),
                ("10".into(), Numeral, false),
                ("k".into(), Word, true),
            ]
        );
    }

    #[test]
    fn spans_are_byte_offsets() {
        let mut l = Lexed::default();
        let text = "é 40\u{202f}000½";
        l.lex(text, &EN);
        let spans: Vec<(u32, u32)> = l.lexemes.iter().map(|x| (x.start, x.end)).collect();
        assert_eq!(spans, [(0, 2), (3, 11), (11, 13)]);
        assert_eq!(l.numeral(1).unwrap().to_string(), "40000");
        assert_eq!(l.word(2), "½");
        assert_eq!(l.raw(text, 1), "40\u{202f}000");
        assert_eq!(l.numeral(0), None);
    }

    #[test]
    fn lists_split_into_integers() {
        assert_eq!(words("192.168.1.2"), ["192", ".", "168", ".", "1", ".", "2"]);
        let mut l = Lexed::default();
        l.lex("192.168.1.2", &EN);
        let v: Vec<String> = l.numerals.iter().map(Dec::to_string).collect();
        assert_eq!(v, ["192", "168", "1", "2"]);
        assert_eq!(words("1,234.5 and 1.2"), ["1,234.5", "and", "1.2"]);
    }

    #[test]
    fn a_leading_separator_opens_a_numeral_after_space_only() {
        assert_eq!(words(".08"), [".08"]);
        assert_eq!(words("the .25ml"), ["the", ".25", "ml"]);
        assert_eq!(words("a.25"), ["a", ".", "25"]);
        assert_eq!(words(" .1.2"), [".", "1.2"]);
    }

    #[test]
    fn web_addresses_are_skipped() {
        assert_eq!(words("go to https://x.example/a?q=30%25 now"), ["go", "to", "now"]);
        assert_eq!(words("www.example.com/5 x"), ["x"]);
        assert_eq!(words("www. 5"), ["www", ".", "5"]);
        let l = lex("a http://x b");
        assert!(!l[1].2, "the lexeme after an address is not glued");
    }

    /// Folds accents and cuts `dosmil` after `dos`.
    struct Folding;

    impl WordForms for Folding {
        fn folds(&self) -> bool {
            true
        }

        fn split(&self, word: &str, cuts: &mut Vec<u32>) {
            if let Some(rest) = word.strip_prefix("dos")
                && !rest.is_empty()
            {
                cuts.push(3);
            }
        }
    }

    fn lex_folded(text: &str) -> Lexed {
        let mut l = Lexed::default();
        l.lex_words(text, &EN, &Folding);
        l
    }

    #[test]
    fn folding_drops_accents_and_keeps_spans() {
        let text = "Miércoles 1º Año a\u{301}n\u{303}o";
        let l = lex_folded(text);
        let words: Vec<&str> = (0..l.len()).map(|i| l.word(i)).collect();
        assert_eq!(words, ["miercoles", "1", "o", "año", "año"]);
        assert_eq!(l.raw(text, 0), "Miércoles");
        assert_eq!(l.raw(text, 2), "º");
        assert_eq!(l.raw(text, 4), "a\u{301}n\u{303}o");
        assert_eq!(fold('Ü'), 'Ü', "fold takes a lowercase letter");
        assert_eq!(fold('ü'), 'u');
    }

    #[test]
    fn a_split_word_gives_glued_pieces_with_exact_spans() {
        let text = "x Dósmilésimo";
        let l = lex_folded(text);
        let got: Vec<(&str, &str, bool)> =
            (0..l.len()).map(|i| (l.word(i), l.raw(text, i), l.lexemes[i].glued)).collect();
        assert_eq!(got, [("x", "x", false), ("dos", "Dós", false), ("milesimo", "milésimo", true)]);
        assert_eq!(words("dosmil"), ["dosmil"], "the plain lexer does not cut");
    }

    #[test]
    fn marks_stay_in_words() {
        assert_eq!(words("cafe\u{301} 5"), ["cafe\u{301}", "5"]);
    }
}
