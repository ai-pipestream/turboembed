//! `StringPattern` and `FeatureGeneratorUtil.tokenFeature`: the shape of a
//! token (all lower case, two digits, an initial capital and so on) and
//! the short class name the feature generators write for it.
//!
//! OpenNLP has two versions, chosen here by [`PatternMode`]:
//!
//! - [`PatternMode::Utf16Units`] is the classification of OpenNLP 1.x,
//!   2.x and 3.x at the commit this crate follows: one UTF-16 code unit at
//!   a time, with `Character.getType(char)`. A letter outside the Basic
//!   Multilingual Plane is two surrogate units that are not letters, and
//!   a combining mark ends the "all lower case" and "all capitals"
//!   classes. Every published model was trained with it.
//! - [`PatternMode::CodePoints`] is the classification of the
//!   `ready/string-pattern-unicode` branch of the OpenNLP fork, where it
//!   is the default (`CompatibilityMode.CURRENT`) and the first version is
//!   kept as `CompatibilityMode.LEGACY`: one code point at a time, with
//!   `Character.isUpperCase(int)` for capitals, combining marks continuing
//!   the letter before them, a letter without case (Arabic, Hangul)
//!   keeping a token lower case, and a malformed surrogate clearing every
//!   whole-token class. Its `cp` class accepts any upper case code point
//!   before the period, where the first version accepts only `A` to `Z`,
//!   `Ä`, `Ö` and `Ü`.
//!
//! A model gives the same features only with the version it was trained
//! with, so the mode is a property of the model, not a preference.
//!
//! Both take the token as UTF-16 code units, which is what the Java reads;
//! [`StringPattern::recognize`] walks a `&str` without copying it and
//! [`StringPattern::recognize_utf16`] takes units that may hold lone
//! surrogates (a piece of a token cut by `SimpleTokenizer`, say).

use turbo_jvm::chars::Utf16Units;
use turbo_unicode::GeneralCategory;
use turbo_unicode::Script;
use turbo_unicode::java::code_point::{general_category, is_upper_case, script};

use crate::java::{is_high, is_low};

/// Which version of `StringPattern` to reproduce; see the module
/// documentation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum PatternMode {
    /// One UTF-16 code unit at a time (OpenNLP before 3.0, and 3.x at the
    /// commit this crate follows).
    #[default]
    Utf16Units,
    /// One code point at a time (the `ready/string-pattern-unicode`
    /// branch).
    CodePoints,
}

const INITIAL_CAPITAL_LETTER: u16 = 1;
const ALL_CAPITAL_LETTER: u16 = 1 << 1;
const ALL_LOWERCASE_LETTER: u16 = 1 << 2;
const ALL_LETTERS: u16 = 1 << 3;
const ALL_DIGIT: u16 = 1 << 4;
const ALL_HIRAGANA: u16 = 1 << 5;
const ALL_KATAKANA: u16 = 1 << 6;
const CONTAINS_PERIOD: u16 = 1 << 7;
const CONTAINS_COMMA: u16 = 1 << 8;
const CONTAINS_SLASH: u16 = 1 << 9;
const CONTAINS_DIGIT: u16 = 1 << 10;
const CONTAINS_HYPHEN: u16 = 1 << 11;
const CONTAINS_LETTERS: u16 = 1 << 12;

/// The middle dot, the prolonged sound mark and the wave dash, which may
/// join Hiragana or Katakana within a word.
fn is_japanese_connector(cp: u32) -> bool {
    matches!(cp, 0x30FB | 0x30FC | 0x301C)
}

fn is_letter(category: GeneralCategory) -> bool {
    category.is_letter()
}

/// The classes of one token.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StringPattern {
    bits: u16,
    digits: u32,
    /// The token's length in UTF-16 code units, `String.length()`.
    units: usize,
    /// Whether the token is a capital followed by a period and nothing
    /// else, by the definition of the mode it was recognized in.
    cap_period: bool,
}

impl StringPattern {
    /// Classifies `token`.
    pub fn recognize(token: &str, mode: PatternMode) -> StringPattern {
        StringPattern::recognize_iter(Utf16Units::new(token).map(|u| u.unit), mode)
    }

    /// Classifies a token given as UTF-16 code units.
    pub fn recognize_utf16(units: &[u16], mode: PatternMode) -> StringPattern {
        StringPattern::recognize_iter(units.iter().copied(), mode)
    }

    fn recognize_iter(units: impl Iterator<Item = u16>, mode: PatternMode) -> StringPattern {
        match mode {
            PatternMode::Utf16Units => by_units(units),
            PatternMode::CodePoints => by_code_points(units),
        }
    }

    /// `isAllLetter`: every character is a letter.
    pub fn is_all_letter(&self) -> bool {
        self.bits & ALL_LETTERS != 0
    }

    /// `isInitialCapitalLetter`: the first character is a capital letter.
    pub fn is_initial_capital_letter(&self) -> bool {
        self.bits & INITIAL_CAPITAL_LETTER != 0
    }

    /// `isAllCapitalLetter`.
    pub fn is_all_capital_letter(&self) -> bool {
        self.bits & ALL_CAPITAL_LETTER != 0
    }

    /// `isAllLowerCaseLetter`.
    pub fn is_all_lower_case_letter(&self) -> bool {
        self.bits & ALL_LOWERCASE_LETTER != 0
    }

    /// `isAllDigit`.
    pub fn is_all_digit(&self) -> bool {
        self.bits & ALL_DIGIT != 0
    }

    /// `isAllHiragana`.
    pub fn is_all_hiragana(&self) -> bool {
        self.bits & ALL_HIRAGANA != 0
    }

    /// `isAllKatakana`.
    pub fn is_all_katakana(&self) -> bool {
        self.bits & ALL_KATAKANA != 0
    }

    /// `digits`: the number of decimal digits.
    pub fn digits(&self) -> u32 {
        self.digits
    }

    /// `containsPeriod`.
    pub fn contains_period(&self) -> bool {
        self.bits & CONTAINS_PERIOD != 0
    }

    /// `containsComma`.
    pub fn contains_comma(&self) -> bool {
        self.bits & CONTAINS_COMMA != 0
    }

    /// `containsSlash`.
    pub fn contains_slash(&self) -> bool {
        self.bits & CONTAINS_SLASH != 0
    }

    /// `containsDigit`.
    pub fn contains_digit(&self) -> bool {
        self.bits & CONTAINS_DIGIT != 0
    }

    /// `containsHyphen`.
    pub fn contains_hyphen(&self) -> bool {
        self.bits & CONTAINS_HYPHEN != 0
    }

    /// `containsLetters`.
    pub fn contains_letters(&self) -> bool {
        self.bits & CONTAINS_LETTERS != 0
    }

    /// `FeatureGeneratorUtil.tokenFeature`: the first class that matches,
    /// in this order: `jah` (all Hiragana), `jak` (all Katakana), `lc`
    /// (all lower case), `2d` and `4d` (two or four digits), then for a
    /// token with a digit `an` (with letters), `dd` (with a hyphen), `ds`
    /// (a slash), `dc` (a comma), `dp` (a period) or `num`; then `sc` and
    /// `ac` (one or more capitals and nothing else), `cp` (a capital and a
    /// period), `ic` (an initial capital) and `other`.
    pub fn token_class(&self) -> &'static str {
        if self.is_all_hiragana() {
            "jah"
        } else if self.is_all_katakana() {
            "jak"
        } else if self.is_all_lower_case_letter() {
            "lc"
        } else if self.digits == 2 {
            "2d"
        } else if self.digits == 4 {
            "4d"
        } else if self.contains_digit() {
            if self.contains_letters() {
                "an"
            } else if self.contains_hyphen() {
                "dd"
            } else if self.contains_slash() {
                "ds"
            } else if self.contains_comma() {
                "dc"
            } else if self.contains_period() {
                "dp"
            } else {
                "num"
            }
        } else if self.is_all_capital_letter() {
            if self.units == 1 { "sc" } else { "ac" }
        } else if self.cap_period {
            "cp"
        } else if self.is_initial_capital_letter() {
            "ic"
        } else {
            "other"
        }
    }
}

/// `FeatureGeneratorUtil.tokenFeature(token)`.
pub fn token_feature(token: &str, mode: PatternMode) -> &'static str {
    StringPattern::recognize(token, mode).token_class()
}

/// `FeatureGeneratorUtil.tokenFeature` of a token given as UTF-16 units.
pub fn token_feature_utf16(units: &[u16], mode: PatternMode) -> &'static str {
    StringPattern::recognize_utf16(units, mode).token_class()
}

/// The first version: one UTF-16 unit at a time. An empty token keeps
/// every whole-token class, so it is "all Hiragana" (`jah`).
fn by_units(units: impl Iterator<Item = u16>) -> StringPattern {
    let mut bits = ALL_CAPITAL_LETTER | ALL_LOWERCASE_LETTER | ALL_DIGIT | ALL_LETTERS | ALL_HIRAGANA | ALL_KATAKANA;
    let mut digits = 0;
    let mut head = [0u16; 2];
    let mut i = 0;
    let mut units = units.peekable();
    while let Some(ch) = units.next() {
        if i < 2 {
            head[i] = ch;
        }
        let category = general_category(u32::from(ch));
        if is_letter(category) {
            bits |= CONTAINS_LETTERS;
            bits &= !ALL_DIGIT;
            if category == GeneralCategory::Lu {
                if i == 0 {
                    bits |= INITIAL_CAPITAL_LETTER;
                }
                bits &= !ALL_LOWERCASE_LETTER;
            } else {
                bits &= !ALL_CAPITAL_LETTER;
            }
        } else {
            bits &= !(ALL_LETTERS | ALL_CAPITAL_LETTER | ALL_LOWERCASE_LETTER);
            if category == GeneralCategory::Nd {
                bits |= CONTAINS_DIGIT;
                bits &= !(ALL_HIRAGANA | ALL_KATAKANA);
                digits += 1;
            } else {
                bits &= !ALL_DIGIT;
            }
            bits |= contains_flag(u32::from(ch));
        }

        // `Character.UnicodeScript.of(token.codePointAt(i))`: a high
        // surrogate followed by a low one is read as the pair; any other
        // unit, a low surrogate included, as itself.
        let mut cp = u32::from(ch);
        if is_high(ch)
            && let Some(&next) = units.peek()
            && is_low(next)
        {
            cp = 0x1_0000 + ((cp - 0xD800) << 10) + (u32::from(next) - 0xDC00);
        }
        match script(cp) {
            Script::Common => {
                if !is_japanese_connector(u32::from(ch)) {
                    bits &= !(ALL_HIRAGANA | ALL_KATAKANA);
                }
            }
            Script::Latin => bits &= !(ALL_HIRAGANA | ALL_KATAKANA),
            Script::Han => bits &= !(ALL_HIRAGANA | ALL_KATAKANA | ALL_LOWERCASE_LETTER),
            Script::Hiragana => bits &= !(ALL_KATAKANA | ALL_LOWERCASE_LETTER),
            Script::Katakana => bits &= !(ALL_HIRAGANA | ALL_LOWERCASE_LETTER),
            _ => {}
        }
        i += 1;
    }
    let cap_period = i == 2 && matches!(head[0], 0x41..=0x5A | 0xC4 | 0xD6 | 0xDC) && head[1] == u16::from(b'.');
    StringPattern { bits, digits, units: i, cap_period }
}

/// The `CONTAINS_*` flag of a punctuation character the classes look for.
fn contains_flag(cp: u32) -> u16 {
    match cp {
        0x2C => CONTAINS_COMMA,
        0x2E => CONTAINS_PERIOD,
        0x2F => CONTAINS_SLASH,
        0x2D => CONTAINS_HYPHEN,
        _ => 0,
    }
}

/// The branch's version: one code point at a time.
fn by_code_points(units: impl Iterator<Item = u16>) -> StringPattern {
    let mut all_capital = true;
    let mut all_lowercase = true;
    let mut all_letters = true;
    let mut all_digits = true;
    let mut all_hiragana = true;
    let mut all_katakana = true;
    let mut initial_capital = false;
    let mut contains_letters = false;
    let mut contains_digit = false;
    let mut malformed = false;
    let mut mark_can_continue_word = false;
    let mut japanese_run_started = false;
    let mut bits = 0u16;
    let mut digits = 0;
    // The first three units, for the `cp` class.
    let mut head = [0u16; 3];
    let mut offset = 0;
    let mut units = units.peekable();
    while let Some(ch) = units.next() {
        let at = offset;
        offset += 1;
        if at < 3 {
            head[at] = ch;
        }
        let pair_low = if is_high(ch) { units.peek().copied().filter(|&next| is_low(next)) } else { None };
        if (is_high(ch) && pair_low.is_none()) || is_low(ch) {
            malformed = true;
            all_capital = false;
            all_lowercase = false;
            all_letters = false;
            all_digits = false;
            all_hiragana = false;
            all_katakana = false;
            mark_can_continue_word = false;
            continue;
        }
        let cp = match pair_low {
            Some(low) => {
                units.next();
                if offset < 3 {
                    head[offset] = low;
                }
                offset += 1;
                0x1_0000 + ((u32::from(ch) - 0xD800) << 10) + (u32::from(low) - 0xDC00)
            }
            None => u32::from(ch),
        };
        let category = general_category(cp);
        let is_mark = matches!(category, GeneralCategory::Mn | GeneralCategory::Mc | GeneralCategory::Me);
        if is_letter(category) {
            let uppercase = is_upper_case(cp);
            if at == 0 {
                initial_capital = uppercase;
            }
            contains_letters = true;
            all_capital &= uppercase;
            if all_lowercase {
                // A letter without case keeps a word lower case; Han and
                // kana letters end it.
                let s = script(cp);
                all_lowercase = !uppercase && s != Script::Han && s != Script::Hiragana && s != Script::Katakana;
            }
            all_digits = false;
            mark_can_continue_word = true;
        } else {
            all_letters = false;
            let continuing_mark = is_mark && mark_can_continue_word;
            if !continuing_mark {
                all_capital = false;
                all_lowercase = false;
            }
            if category == GeneralCategory::Nd {
                contains_digit = true;
                digits += 1;
                mark_can_continue_word = false;
            } else if !continuing_mark {
                all_digits = false;
                mark_can_continue_word = false;
            }
            bits |= contains_flag(cp);
        }

        if all_hiragana || all_katakana {
            let s = script(cp);
            if s == Script::Hiragana {
                all_katakana = false;
                japanese_run_started = true;
            } else if s == Script::Katakana {
                all_hiragana = false;
                japanese_run_started = true;
            } else if is_japanese_connector(cp) && japanese_run_started {
                mark_can_continue_word = true;
            } else if !is_mark || !mark_can_continue_word {
                all_hiragana = false;
                all_katakana = false;
            }
        }
    }
    let length = offset;
    if length == 0 {
        return StringPattern { bits: 0, digits: 0, units: 0, cap_period: false };
    }
    if malformed {
        initial_capital = false;
    }
    for (set, flag) in [
        (initial_capital, INITIAL_CAPITAL_LETTER),
        (all_capital, ALL_CAPITAL_LETTER),
        (all_lowercase, ALL_LOWERCASE_LETTER),
        (all_letters, ALL_LETTERS),
        (all_digits, ALL_DIGIT),
        (all_hiragana, ALL_HIRAGANA),
        (all_katakana, ALL_KATAKANA),
        (contains_letters, CONTAINS_LETTERS),
        (contains_digit, CONTAINS_DIGIT),
    ] {
        if set {
            bits |= flag;
        }
    }
    StringPattern { bits, digits, units: length, cap_period: cap_period_by_code_point(&head, length) }
}

/// The branch's `isCapPeriod`: one upper case code point, then a period,
/// then the end of the token.
fn cap_period_by_code_point(head: &[u16; 3], length: usize) -> bool {
    let (first, width) = if is_high(head[0]) && is_low(head[1]) {
        (0x1_0000 + ((u32::from(head[0]) - 0xD800) << 10) + (u32::from(head[1]) - 0xDC00), 2)
    } else {
        (u32::from(head[0]), 1)
    };
    is_upper_case(first) && width + 1 == length && head[width] == u16::from(b'.')
}
