//! What a pattern can test on a token: a string attribute (its text, its
//! lowercase form, its lemma, its tag) or a yes or no flag (alphabetic,
//! punctuation, a stopword, the first of its sentence).

use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::normalize::{Form, normalize};
use turbo_unicode::{casefold, java, props};

/// A string a token has: its text, a form computed from the text, or the
/// value a layer gives it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Attr {
    /// The token's text (spaCy's `ORTH`).
    Text,
    /// The text lowercased as `String.toLowerCase(Locale.ROOT)` does, on
    /// Unicode 18 data (spaCy's `LOWER`).
    Lower,
    /// A form for matching that ignores case and compatibility variants:
    /// the full case folding of the text's NFKD form, put in NFKC, with
    /// default ignorable code points removed. `ﬁnal`, `FINAL` and `final`
    /// have one form. Not spaCy's `NORM`, which is the lowercase form with
    /// per-language exceptions.
    Norm,
    /// The word shape as spaCy's `word_shape` writes it: `x` for a
    /// lowercase letter, `X` for an uppercase one, `d` for a decimal digit,
    /// any other character as itself, a run of one symbol cut after four,
    /// and `LONG` for text of 100 characters or more. `Apple3` is `Xxxxxd`.
    Shape,
    /// The token's lemma, from `opennlp:lemmas`.
    Lemma,
    /// The token's stem, from `turbo:stems`.
    Stem,
    /// The token's part-of-speech tag, from `opennlp:pos` (spaCy's `TAG`, or
    /// `POS` when the tagger writes universal tags).
    Pos,
    /// The kind of the entity the token is in, from `opennlp:entities`, or
    /// the empty string outside every entity (spaCy's `ENT_TYPE`).
    Entity,
    /// A string column the caller added to the sequence under this name.
    Column(&'static str),
}

/// A yes or no property of a token. Each one that reads the text holds
/// for a token whose every character has the property; none holds for an
/// empty token.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Flag {
    /// Every character a letter (general category L).
    Alpha,
    /// Every character ASCII.
    Ascii,
    /// Every character a decimal digit (general category Nd). Python's
    /// `isdigit`, which spaCy uses, also takes superscript digits.
    Digit,
    /// At least one lowercase character and no uppercase or titlecase one,
    /// as Python's `islower`.
    Lower,
    /// At least one uppercase character and no lowercase or titlecase one,
    /// as Python's `isupper`.
    Upper,
    /// Python's `istitle`: at least one cased character, each uppercase or
    /// titlecase character after an uncased one, each lowercase character
    /// after a cased one.
    Title,
    /// Every character punctuation (general category P).
    Punct,
    /// Every character `White_Space`.
    Space,
    /// One opening or closing bracket character (general category Ps or
    /// Pe), or `<` or `>`.
    Bracket,
    /// Every character a quotation mark (`Quotation_Mark`) or a grave
    /// accent, which is used as one.
    Quote,
    /// Every character a currency symbol (general category Sc).
    Currency,
    /// The token reads as a number written in digits: an optional sign
    /// (`+`, `-`, `±`, `~`), then decimal digits with `,` and `.` between
    /// them, or a fraction of two such digit strings. When the sequence has
    /// `turbo:numbers`, a token inside a number mention holds as well,
    /// which covers numbers written in words.
    LikeNum,
    /// A stopword, from `turbo:stopwords`.
    Stop,
    /// The first token of its sentence.
    SentStart,
    /// Followed by whitespace in the text.
    SpaceAfter,
}

impl Flag {
    pub(crate) const COUNT: usize = 15;

    pub(crate) fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

const _: () = assert!(Flag::SpaceAfter as usize + 1 == Flag::COUNT);

/// The flags read from a token's text: a bit per `Flag`, at `Flag::bit`.
pub(crate) fn lexical_flags(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }
    let mut all = Flag::Alpha.bit()
        | Flag::Ascii.bit()
        | Flag::Digit.bit()
        | Flag::Punct.bit()
        | Flag::Space.bit()
        | Flag::Quote.bit()
        | Flag::Currency.bit();
    // Python's islower, isupper and istitle, one pass.
    let (mut lower, mut upper, mut title_case, mut cased_any) = (false, false, false, false);
    let mut title = true;
    let mut previous_cased = false;
    for c in text.chars() {
        let cat = general_category(c);
        if !cat.is_letter() {
            all &= !Flag::Alpha.bit();
        }
        if !c.is_ascii() {
            all &= !Flag::Ascii.bit();
        }
        if cat != GeneralCategory::Nd {
            all &= !Flag::Digit.bit();
        }
        if !cat.is_punctuation() {
            all &= !Flag::Punct.bit();
        }
        if !props::white_space(c) {
            all &= !Flag::Space.bit();
        }
        if !(props::quotation_mark(c) || c == '`') {
            all &= !Flag::Quote.bit();
        }
        if cat != GeneralCategory::Sc {
            all &= !Flag::Currency.bit();
        }
        let is_title = cat == GeneralCategory::Lt;
        let is_upper = props::uppercase(c) && !is_title;
        let is_lower = props::lowercase(c);
        lower |= is_lower;
        upper |= is_upper;
        title_case |= is_title;
        if is_upper || is_title {
            title &= !previous_cased;
            previous_cased = true;
            cased_any = true;
        } else if is_lower {
            title &= previous_cased;
            previous_cased = true;
            cased_any = true;
        } else {
            previous_cased = false;
        }
    }
    if lower && !upper && !title_case {
        all |= Flag::Lower.bit();
    }
    if upper && !lower && !title_case {
        all |= Flag::Upper.bit();
    }
    if title && cased_any {
        all |= Flag::Title.bit();
    }
    if is_bracket(text) {
        all |= Flag::Bracket.bit();
    }
    if like_num(text) {
        all |= Flag::LikeNum.bit();
    }
    all
}

fn is_bracket(text: &str) -> bool {
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => {
            matches!(c, '<' | '>') || matches!(general_category(c), GeneralCategory::Ps | GeneralCategory::Pe)
        }
        _ => false,
    }
}

/// Digits with `,` and `.` between them, after an optional sign; or two
/// such digit strings around one `/`.
fn like_num(text: &str) -> bool {
    let text = text.strip_prefix(['+', '-', '±', '~']).unwrap_or(text);
    match text.split_once('/') {
        Some((a, b)) => digits(a, false) && digits(b, false),
        None => digits(text, true),
    }
}

fn digits(text: &str, separators: bool) -> bool {
    let mut any = false;
    let mut after_separator = false;
    for c in text.chars() {
        if general_category(c) == GeneralCategory::Nd {
            any = true;
            after_separator = false;
        } else if separators && matches!(c, ',' | '.') && any && !after_separator {
            after_separator = true;
        } else {
            return false;
        }
    }
    any && !after_separator
}

/// Writes `Attr::Lower` of `text` to `out`.
pub(crate) fn lower(text: &str, out: &mut String) {
    java::to_lowercase(text, out);
}

/// Writes `Attr::Norm` of `text` to `out`, using `tmp` for the steps.
pub(crate) fn norm(text: &str, out: &mut String, tmp: &mut String) {
    tmp.clear();
    normalize(Form::Nfkd, text, tmp);
    let mut folded = String::with_capacity(tmp.len());
    casefold::fold(tmp, &mut folded);
    folded.retain(|c| !props::default_ignorable(c));
    normalize(Form::Nfkc, &folded, out);
}

/// Writes `Attr::Shape` of `text` to `out`.
pub(crate) fn shape(text: &str, out: &mut String) {
    if text.chars().count() >= 100 {
        out.push_str("LONG");
        return;
    }
    let mut last = None;
    let mut run = 0;
    for c in text.chars() {
        let cat = general_category(c);
        let s = if cat.is_letter() {
            if props::uppercase(c) { 'X' } else { 'x' }
        } else if cat == GeneralCategory::Nd {
            'd'
        } else {
            c
        };
        if Some(s) == last {
            run += 1;
        } else {
            run = 0;
            last = Some(s);
        }
        if run < 4 {
            out.push(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has(text: &str, f: Flag) -> bool {
        lexical_flags(text) & f.bit() != 0
    }

    #[test]
    fn case_flags_follow_python() {
        assert!(has("hello", Flag::Lower) && !has("hello", Flag::Upper) && !has("hello", Flag::Title));
        assert!(has("HELLO", Flag::Upper) && !has("HELLO", Flag::Title));
        assert!(has("Hello", Flag::Title) && !has("Hello", Flag::Lower));
        assert!(has("Hello World", Flag::Title));
        assert!(!has("HeLLo", Flag::Title));
        assert!(has("a1", Flag::Lower) && !has("12", Flag::Lower) && !has("12", Flag::Upper));
        // A titlecase digraph is neither lower nor upper, and is title.
        assert!(has("\u{01C5}", Flag::Title) && !has("\u{01C5}", Flag::Upper));
        assert!(has("ß", Flag::Lower));
    }

    #[test]
    fn character_class_flags() {
        assert!(has("Straße", Flag::Alpha) && !has("Straße", Flag::Ascii));
        assert!(!has("a-b", Flag::Alpha));
        assert!(has("٣٤", Flag::Digit) && !has("²", Flag::Digit));
        assert!(has("...", Flag::Punct) && has("«", Flag::Punct) && !has("$", Flag::Punct));
        assert!(has("$", Flag::Currency) && has("€", Flag::Currency));
        assert!(has("\u{00A0} ", Flag::Space));
        assert!(has("«", Flag::Quote) && has("``", Flag::Quote) && has("\"", Flag::Quote));
        assert!(has("(", Flag::Bracket) && has("<", Flag::Bracket) && has("】", Flag::Bracket));
        assert!(!has("((", Flag::Bracket));
        assert_eq!(lexical_flags(""), 0);
    }

    #[test]
    fn numbers_in_digits() {
        for t in ["10", "-3", "1,000.5", "±2", "3/4", "٣"] {
            assert!(like_num(t), "{t}");
        }
        for t in ["", ",1", "1,", "1,,0", "1/", "ten", "1/2/3", "+"] {
            assert!(!like_num(t), "{t}");
        }
    }

    #[test]
    fn shapes_follow_spacy() {
        let s = |t: &str| {
            let mut out = String::new();
            shape(t, &mut out);
            out
        };
        assert_eq!(s("Apple3"), "Xxxxxd");
        assert_eq!(s("Hellooooo"), "Xxxxx");
        assert_eq!(s("1999"), "dddd");
        assert_eq!(s("U.S.A."), "X.X.X.");
        assert_eq!(s("Ärger"), "Xxxxx");
        assert_eq!(s(&"a".repeat(100)), "LONG");
    }

    #[test]
    fn norm_folds_case_and_compatibility() {
        let n = |t: &str| {
            let (mut out, mut tmp) = (String::new(), String::new());
            norm(t, &mut out, &mut tmp);
            out
        };
        assert_eq!(n("\u{FB01}nal"), "final");
        assert_eq!(n("FINAL"), "final");
        assert_eq!(n("Straße"), "strasse");
        assert_eq!(n("so\u{00AD}ft"), "soft");
        assert_eq!(n("Ｆｕｌｌ"), "full");
        assert_eq!(n("e\u{0301}"), "é");
    }
}
