//! What a pattern can test about a token: string attributes read from the
//! text or from a layer aligned to the tokens, and flags.
//!
//! The lexical attributes and flags follow spaCy's `spacy/lang/lex_attrs.py`
//! (MIT), which defines them with Python's `str` methods; the Unicode
//! properties those methods read come from `turbo-unicode`.

use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::props;

/// A string attribute of a token. Pattern tests compare it with strings;
/// a phrase matcher matches phrases over one of them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Attr {
    /// The token's text in its view: spaCy's `ORTH` and `TEXT`. Tokens
    /// made on a normalized view (`nfkc`, `dehyphenated`) carry that
    /// view's text, so this is how a pattern matches normalized forms.
    Text,
    /// The text in lowercase (`str::to_lowercase`, as Python's
    /// `str.lower`): spaCy's `LOWER`.
    Lower,
    /// The first character: spaCy's `PREFIX`.
    Prefix,
    /// The last three characters: spaCy's `SUFFIX`.
    Suffix,
    /// The word shape, spaCy's `SHAPE`: letters as `X` or `x`, digits as
    /// `d`, other characters as themselves, a run of one kind cut after
    /// four ("Xxxxx" for "Hello", "dd.d" for "12.5"), and `LONG` for 100
    /// characters or more.
    Shape,
    /// A normalized form per token the caller supplies
    /// ([`crate::Input::with_norms`]): spaCy's `NORM`.
    Norm,
    /// The lemma, from `opennlp:lemmas`.
    Lemma,
    /// The stem, from `turbo:stems`.
    Stem,
    /// The part-of-speech tag, from `opennlp:pos`: spaCy's `TAG` and `POS`
    /// (one tag set per layer here).
    Pos,
    /// The kind of the entity that covers the token, from
    /// `opennlp:entities`: spaCy's `ENT_TYPE`. When entities overlap, the
    /// first one in the layer that covers the token gives it.
    Entity,
}

impl Attr {
    pub(crate) const ALL: [Attr; 10] = [
        Attr::Text,
        Attr::Lower,
        Attr::Prefix,
        Attr::Suffix,
        Attr::Shape,
        Attr::Norm,
        Attr::Lemma,
        Attr::Stem,
        Attr::Pos,
        Attr::Entity,
    ];

    /// The attribute's value for a string standing for a token, when the
    /// string's own text gives it: the text, its lowercase, prefix,
    /// suffix and shape. The others are values already (a lemma, a tag).
    pub(crate) fn of_text<'a>(self, text: &'a str, scratch: &'a mut String) -> &'a str {
        match self {
            Attr::Lower => {
                lower_into(text, scratch);
                scratch
            }
            Attr::Prefix => prefix(text),
            Attr::Suffix => suffix(text),
            Attr::Shape => {
                shape_into(text, scratch);
                scratch
            }
            _ => text,
        }
    }
}

/// A true-or-false property of a token.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Flag {
    /// Every character a letter (Python's `str.isalpha`): spaCy's `IS_ALPHA`.
    Alpha,
    /// Every character below U+0080: `IS_ASCII`.
    Ascii,
    /// Every character a decimal digit (general category Nd; Python's
    /// `str.isdigit` also counts superscripts and circled digits): `IS_DIGIT`.
    Digit,
    /// Python's `str.islower`: `IS_LOWER`.
    Lower,
    /// Python's `str.isupper`: `IS_UPPER`.
    Upper,
    /// Python's `str.istitle`: `IS_TITLE`.
    Title,
    /// Every character punctuation (general category P): `IS_PUNCT`.
    Punct,
    /// Python's `str.isspace`: `IS_SPACE`.
    Space,
    /// One of `( ) [ ] { } < >`: `IS_BRACKET`.
    Bracket,
    /// A quotation mark of spaCy's list: `IS_QUOTE`.
    Quote,
    /// An opening bracket or quotation mark of spaCy's list: `IS_LEFT_PUNCT`.
    LeftPunct,
    /// A closing bracket or quotation mark of spaCy's list: `IS_RIGHT_PUNCT`.
    RightPunct,
    /// Every character a currency symbol (general category Sc): `IS_CURRENCY`.
    Currency,
    /// spaCy's language-neutral `like_num`: digits after an optional sign,
    /// with `,` and `.` ignored, or a fraction of two such numbers. The
    /// number words some languages add are not here.
    LikeNum,
    /// Whitespace follows the token in its view: spaCy's `SPACY`.
    SpaceAfter,
    /// The token starts a sentence: `IS_SENT_START`. Needs the tokens'
    /// sentence column.
    SentStart,
    /// The token is a stopword, from `turbo:stopwords`: `IS_STOP`.
    Stop,
    /// The token overlaps a number mention of `turbo:numbers`.
    Number,
}

impl Flag {
    /// The flags a token's own text decides; the rest need its context
    /// or a layer.
    pub(crate) fn text_decides(self) -> bool {
        !matches!(self, Flag::SpaceAfter | Flag::SentStart | Flag::Stop | Flag::Number)
    }

    #[inline]
    pub(crate) fn bit(self) -> u32 {
        1 << self as u32
    }
}

/// What a phrase matcher matches phrases over: one string attribute, or
/// one flag (a phrase over [`Flag::Punct`] matches any run of tokens whose
/// punctuation flags are the phrase's).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Field {
    Attr(Attr),
    Flag(Flag),
}

/// Where a token is in the entities that cover it, as spaCy's `ENT_IOB`:
/// the first token of an entity, a later one, or in none. When entities
/// overlap, the first one in the layer that covers the token decides.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Iob {
    Begin,
    Inside,
    Outside,
}

/// A comparison of a token's length in characters with a number.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    #[inline]
    pub(crate) fn holds(self, a: u32, b: u32) -> bool {
        match self {
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
        }
    }
}

/// `text` in lowercase, written over `out`. ASCII text is lowered byte by
/// byte; other text goes through `str::to_lowercase`, which maps a final
/// sigma as Python's `str.lower` does.
pub(crate) fn lower_into(text: &str, out: &mut String) {
    out.clear();
    if text.is_ascii() {
        out.extend(text.bytes().map(|b| b.to_ascii_lowercase() as char));
    } else {
        out.push_str(&text.to_lowercase());
    }
}

/// The first character, or nothing for an empty token.
pub(crate) fn prefix(text: &str) -> &str {
    text.chars().next().map_or("", |c| &text[..c.len_utf8()])
}

/// The last three characters, or the whole token when shorter.
pub(crate) fn suffix(text: &str) -> &str {
    let at = text.char_indices().rev().nth(2).map_or(0, |(i, _)| i);
    &text[at..]
}

/// spaCy's `word_shape`, written over `out`.
pub(crate) fn shape_into(text: &str, out: &mut String) {
    out.clear();
    if text.chars().nth(99).is_some() {
        out.push_str("LONG");
        return;
    }
    let mut last = None;
    let mut run = 0;
    for c in text.chars() {
        let shape = if is_alpha(c) {
            if is_upper_char(c) { 'X' } else { 'x' }
        } else if is_digit(c) {
            'd'
        } else {
            c
        };
        if Some(shape) == last {
            run += 1;
        } else {
            run = 0;
            last = Some(shape);
        }
        if run < 4 {
            out.push(shape);
        }
    }
}

fn is_alpha(c: char) -> bool {
    use GeneralCategory::*;
    matches!(general_category(c), Lu | Ll | Lt | Lm | Lo)
}

fn is_digit(c: char) -> bool {
    general_category(c) == GeneralCategory::Nd
}

fn is_title_char(c: char) -> bool {
    general_category(c) == GeneralCategory::Lt
}

/// A single character's `str.isupper`.
fn is_upper_char(c: char) -> bool {
    props::uppercase(c) && !props::lowercase(c) && !is_title_char(c)
}

fn all(text: &str, f: impl Fn(char) -> bool) -> bool {
    text.chars().all(f)
}

/// Python's `str.islower`: no uppercase or titlecase character, and at
/// least one lowercase one.
fn is_lower(text: &str) -> bool {
    let mut cased = false;
    for c in text.chars() {
        if props::uppercase(c) || is_title_char(c) {
            return false;
        }
        cased |= props::lowercase(c);
    }
    cased
}

/// Python's `str.isupper`.
fn is_upper(text: &str) -> bool {
    let mut cased = false;
    for c in text.chars() {
        if props::lowercase(c) || is_title_char(c) {
            return false;
        }
        cased |= props::uppercase(c);
    }
    cased
}

/// Python's `str.istitle`: uppercase and titlecase characters only after
/// uncased ones, lowercase ones only after cased ones, and one cased
/// character at least.
fn is_title(text: &str) -> bool {
    let (mut cased, mut after_cased) = (false, false);
    for c in text.chars() {
        if props::uppercase(c) || is_title_char(c) {
            if after_cased {
                return false;
            }
            (after_cased, cased) = (true, true);
        } else if props::lowercase(c) {
            if !after_cased {
                return false;
            }
            (after_cased, cased) = (true, true);
        } else {
            after_cased = false;
        }
    }
    cased
}

/// Python's `str.isspace`: the White_Space property, and the separators
/// U+001C to U+001F, which Python also counts.
fn is_space(text: &str) -> bool {
    !text.is_empty() && all(text, |c| props::white_space(c) || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn digits(text: &str) -> bool {
    !text.is_empty() && all(text, is_digit)
}

fn like_num(text: &str) -> bool {
    let text = text.strip_prefix(['+', '-', '±', '~']).unwrap_or(text);
    if text.contains([',', '.']) {
        let kept: String = text.chars().filter(|&c| c != ',' && c != '.').collect();
        return like_num_plain(&kept);
    }
    like_num_plain(text)
}

fn like_num_plain(text: &str) -> bool {
    if digits(text) {
        return true;
    }
    let mut parts = text.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(num), Some(den), None) => digits(num) && digits(den),
        _ => false,
    }
}

const QUOTES: [&str; 19] =
    ["\"", "'", "`", "«", "»", "‘", "’", "‚", "‛", "“", "”", "„", "‟", "‹", "›", "❮", "❯", "''", "``"];
const LEFT: [&str; 16] = ["(", "[", "{", "<", "\"", "'", "«", "‘", "‚", "‛", "“", "„", "‟", "‹", "❮", "``"];
const RIGHT: [&str; 12] = [")", "]", "}", ">", "\"", "'", "»", "’", "”", "›", "❯", "''"];

/// The flags a token's own text decides, one bit per [`Flag`]. As in
/// spaCy, an empty token is ASCII, punctuation and currency (an empty
/// loop is true) and nothing else.
pub(crate) fn text_flags(text: &str) -> u32 {
    let mut bits = 0;
    let mut set = |f: Flag, on: bool| {
        if on {
            bits |= f.bit();
        }
    };
    set(Flag::Alpha, !text.is_empty() && all(text, is_alpha));
    set(Flag::Ascii, text.is_ascii());
    set(Flag::Digit, digits(text));
    set(Flag::Lower, is_lower(text));
    set(Flag::Upper, is_upper(text));
    set(Flag::Title, is_title(text));
    set(Flag::Punct, all(text, |c| general_category(c).is_punctuation()));
    set(Flag::Space, is_space(text));
    set(Flag::Bracket, matches!(text, "(" | ")" | "[" | "]" | "{" | "}" | "<" | ">"));
    set(Flag::Quote, QUOTES.contains(&text));
    set(Flag::LeftPunct, LEFT.contains(&text));
    set(Flag::RightPunct, RIGHT.contains(&text));
    set(Flag::Currency, all(text, |c| general_category(c) == GeneralCategory::Sc));
    set(Flag::LikeNum, like_num(text));
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(s: &str) -> String {
        let mut out = String::new();
        shape_into(s, &mut out);
        out
    }

    fn has(text: &str, f: Flag) -> bool {
        text_flags(text) & f.bit() != 0
    }

    #[test]
    fn shapes_follow_spacy() {
        // spaCy's test_lexeme and test_text cases.
        assert_eq!(shape("Nasa"), "Xxxx");
        assert_eq!(shape("capitalized"), "xxxx");
        assert_eq!(shape("999999999"), "dddd");
        assert_eq!(shape("C3P0"), "XdXd");
        assert_eq!(shape(","), ",");
        assert_eq!(shape("\n"), "\n");
        assert_eq!(shape("``,-"), "``,-");
        assert_eq!(shape("12.5"), "dd.d");
        assert_eq!(shape(&"a".repeat(100)), "LONG");
        assert_eq!(shape(&"a".repeat(99)), "xxxx");
    }

    #[test]
    fn affixes_count_characters() {
        assert_eq!(prefix("héllo"), "h");
        assert_eq!(prefix("éa"), "é");
        assert_eq!(prefix(""), "");
        assert_eq!(suffix("héllo"), "llo");
        assert_eq!(suffix("aé"), "aé");
        assert_eq!(suffix("naïve"), "ïve");
    }

    #[test]
    fn flags_follow_python_string_methods() {
        assert!(has("Hello", Flag::Title) && has("Hello", Flag::Alpha));
        assert!(!has("HeLLo", Flag::Title));
        assert!(has("A1", Flag::Upper) && !has("1", Flag::Upper));
        assert!(has("a1", Flag::Lower) && !has("aB", Flag::Lower));
        assert!(has("Hello World", Flag::Title) && !has("They're", Flag::Title));
        assert!(has("...", Flag::Punct) && has("¿", Flag::Punct) && !has("a.", Flag::Punct));
        assert!(has("\n", Flag::Space) && has("\u{1c}", Flag::Space) && !has("\u{200b}", Flag::Space));
        assert!(has("$", Flag::Currency) && has("€", Flag::Currency));
        assert!(has("``", Flag::Quote) && has("``", Flag::LeftPunct) && has("''", Flag::RightPunct));
        assert!(has("[", Flag::Bracket) && !has("[[", Flag::Bracket));
        assert!(has("١٢", Flag::Digit) && !has("²", Flag::Digit));
        assert!(has("", Flag::Ascii) && has("", Flag::Punct) && !has("", Flag::Alpha));
    }

    #[test]
    fn like_num_follows_spacy() {
        for yes in ["10", "1,000", "10.5", "+3", "-2", "±1", "~4", "1/2", "3.0/4"] {
            assert!(has(yes, Flag::LikeNum), "{yes}");
        }
        for no in ["ten", "1/2/3", "/2", "1e5", "", "-", "1-2"] {
            assert!(!has(no, Flag::LikeNum), "{no}");
        }
    }

    #[test]
    fn lowercase_maps_final_sigma() {
        let mut out = String::new();
        lower_into("ΟΔΟΣ", &mut out);
        assert_eq!(out, "οδος");
        lower_into("MiXeD", &mut out);
        assert_eq!(out, "mixed");
    }
}
