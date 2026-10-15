//! Finding definitions in one sentence: the candidates around each
//! parenthesis and the match of a short form against its long form
//! candidate, as Schwartz and Hearst (2003) describe them and as their
//! reference implementation (`ExtractAbbrev.java`) applies them.

use turbo_analysis::Span;
use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::java::{is_letter, is_letter_or_digit, to_lower_char};

/// One definition: a short form, its long form, and where the
/// parenthesis that held one of them closes. Every offset is a UTF-8 byte
/// offset into the text the sentence spans are in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Definition {
    pub short: Span,
    pub long: Span,
    /// One past the closing parenthesis. A later mention starts at or
    /// after it.
    pub end: u32,
}

impl Definition {
    /// Where the definition starts: its short form or its long form,
    /// whichever comes first.
    pub fn start(&self) -> u32 {
        self.short.start.min(self.long.start)
    }

    /// Whether the short form is the one before the parenthesis, as in
    /// "WHO (World Health Organization)".
    pub fn is_reversed(&self) -> bool {
        self.short.start < self.long.start
    }
}

/// Appends to `out` the definitions found in `sentence`, a span of `text`,
/// in text order.
///
/// Each parenthesis that follows whitespace is a candidate. The text inside
/// it, up to its closing parenthesis (or the next one, when it opens
/// another), and cut at the first ", " and then the first "; ", is the
/// short form candidate. The long form candidate is the text before it,
/// back to the closing parenthesis of the candidate before, or to the last
/// '.' or ',' followed by whitespace, whichever is later. When the text
/// inside has more than two words, or is longer than the text before, the
/// two swap: the word just before the parenthesis is the short form, and it
/// must hold an uppercase letter. A paren left open ends the sentence's
/// candidates.
pub fn definitions_in(text: &str, sentence: Span, out: &mut Vec<Definition>) {
    let hi = sentence.end as usize;
    let mut cursor = sentence.start as usize;
    while let Some(open) = next_open(text, cursor, hi) {
        let Some(mut close) = text[open + 1..hi].find(')').map(|i| open + 1 + i) else { break };
        if text[open + 1..close].contains('(')
            && let Some(next) = text[close + 1..hi].find(')')
        {
            close += 1 + next;
        }
        let mut inner_end = close;
        if let Some(i) = text[open + 1..inner_end].find(", ") {
            inner_end = open + 1 + i;
        }
        if let Some(i) = text[open + 1..inner_end].find("; ") {
            inner_end = open + 1 + i;
        }
        let inner = trim(text, open + 1, inner_end);
        let before = trim(text, window_start(text, cursor, open), open);
        cursor = close + 1;
        if inner.is_empty() || before.is_empty() {
            continue;
        }
        let inner_text = &text[inner.range()];
        let (short, long) = if inner_text.split_whitespace().count() > 2
            || inner_text.chars().count() > text[before.range()].chars().count()
        {
            let word = last_word(text, before);
            if !text[word.range()].chars().any(|c| general_category(c) == GeneralCategory::Lu) {
                continue;
            }
            (word, inner)
        } else {
            (inner, before)
        };
        if !is_valid_short_form(&text[short.range()]) {
            continue;
        }
        if let Some(long) = best_long_form(text, short, long).map(|l| tidy(text, l))
            && is_acceptable(&text[short.range()], &text[long.range()])
        {
            out.push(Definition { short, long, end: cursor as u32 });
        }
    }
}

/// The next '(' at or after `from` and before `hi` that follows a
/// whitespace character at or after `from`.
fn next_open(text: &str, from: usize, hi: usize) -> Option<usize> {
    let mut at = from;
    while let Some(i) = text[at..hi].find('(') {
        let open = at + i;
        if text[from..open].chars().next_back().is_some_and(char::is_whitespace) {
            return Some(open);
        }
        at = open + 1;
    }
    None
}

/// Where the long form candidate before `open` starts: after the last '.'
/// or ',' at or after `from` that a whitespace character follows, or at
/// `from`.
fn window_start(text: &str, from: usize, open: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = open;
    while i > from {
        i -= 1;
        if (bytes[i] == b'.' || bytes[i] == b',') && text[i + 1..open].chars().next().is_some_and(char::is_whitespace) {
            return i + 1;
        }
    }
    from
}

/// `[start, end)` without whitespace at either end.
fn trim(text: &str, start: usize, end: usize) -> Span {
    let s = &text[start..end];
    let lead = s.len() - s.trim_start().len();
    let trail = s.len() - s.trim_end().len();
    if lead == s.len() {
        return Span::new(end as u32, end as u32);
    }
    Span::new((start + lead) as u32, (end - trail) as u32)
}

/// The last whitespace-separated word of a trimmed span.
fn last_word(text: &str, span: Span) -> Span {
    let s = &text[span.range()];
    let start = s.rfind(char::is_whitespace).map_or(0, |i| i + s[i..].chars().next().map_or(1, char::len_utf8));
    Span::new(span.start + start as u32, span.end)
}

/// A short form holds a letter and starts with a letter, a digit or '('.
fn is_valid_short_form(short: &str) -> bool {
    short.chars().any(is_letter) && short.chars().next().is_some_and(|c| is_letter_or_digit(c) || c == '(')
}

/// The shortest end of `long` in which every letter and digit of `short`
/// is found, in order and case-insensitively, the first of them at the
/// start of a word, or `None`. The characters are matched from the right;
/// the long form found then runs from the start of the word, after the last
/// whitespace, that holds the first character's match.
pub fn best_long_form(text: &str, short: Span, long: Span) -> Option<Span> {
    let short_text = &text[short.range()];
    let long_text = &text[long.range()];
    let mut at = long_text.len();
    for (si, sc) in short_text.char_indices().rev() {
        let c = to_lower_char(sc);
        if !is_letter_or_digit(c) {
            continue;
        }
        loop {
            let lc = long_text[..at].chars().next_back()?;
            at -= lc.len_utf8();
            let starts_word = si != 0 || long_text[..at].chars().next_back().is_none_or(|p| !is_letter_or_digit(p));
            if to_lower_char(lc) == c && starts_word {
                break;
            }
        }
    }
    let start = long_text[..at]
        .rfind(char::is_whitespace)
        .map_or(0, |i| i + long_text[i..].chars().next().map_or(1, char::len_utf8));
    Some(Span::new(long.start + start as u32, long.end))
}

/// A long form without the punctuation its word boundaries leave at its
/// ends: leading characters that are neither letters nor digits, except an
/// opening bracket the long form closes ("(-)-[125I]-cyanopindolol" keeps
/// its brackets, "[genioglossus" and "/forced vital capacity" lose the
/// first character), and trailing quotation marks.
fn tidy(text: &str, long: Span) -> Span {
    let s = &text[long.range()];
    let mut start = 0;
    for (i, c) in s.char_indices() {
        let closed = match c {
            '(' => s[i..].contains(')'),
            '[' => s[i..].contains(']'),
            '{' => s[i..].contains('}'),
            _ => false,
        };
        if is_letter_or_digit(c) || closed {
            break;
        }
        start = i + c.len_utf8();
    }
    let rest = &s[start..];
    let end = rest
        .trim_end_matches(|c: char| {
            c == '"' || matches!(general_category(c), GeneralCategory::Pi | GeneralCategory::Pf)
        })
        .len();
    Span::new(long.start + start as u32, long.start + (start + end) as u32)
}

/// The reference implementation's last filters on a pair: a short form of
/// two characters or more with at most ten letters and digits, a long form
/// at least as long as it that neither ends with it nor holds it followed
/// by a space, and at most `min(n + 5, 2n)` words in the long form, for
/// `n` letters and digits in the short form, where words are split at
/// whitespace and hyphens.
fn is_acceptable(short: &str, long: &str) -> bool {
    let short_chars = short.chars().count();
    if short_chars < 2 {
        return false;
    }
    let n = short.chars().filter(|&c| is_letter_or_digit(c)).count();
    let words = long.split(|c: char| c.is_whitespace() || c == '-').filter(|w| !w.is_empty()).count();
    let holds_short = long.match_indices(short).any(|(i, _)| long[i + short.len()..].starts_with(' '));
    !(long.chars().count() < short_chars
        || holds_short
        || long.ends_with(short)
        || words > 2 * n
        || words > n + 5
        || n > 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(text: &str) -> Vec<(&str, &str)> {
        let mut out = Vec::new();
        definitions_in(text, Span::new(0, text.len() as u32), &mut out);
        out.iter().map(|d| (&text[d.short.range()], &text[d.long.range()])).collect()
    }

    fn long_of<'a>(long: &'a str, short: &str) -> Option<&'a str> {
        let text = format!("{long}{short}");
        let s = Span::new(long.len() as u32, text.len() as u32);
        best_long_form(&text, s, Span::new(0, long.len() as u32)).map(|l| &long[l.range()])
    }

    /// The cases of scispaCy's `test_find_abbreviation`.
    #[test]
    fn the_match_runs_right_to_left_and_anchors_the_first_letter() {
        assert_eq!(long_of("abbreviation", "abbrn"), Some("abbreviation"));
        assert_eq!(long_of("abbreviation", "ab-b9rn"), None, "no 9 in the long form");
        assert_eq!(long_of("abbreviation", "aeb-brn"), None, "the letters are out of order");
        assert_eq!(long_of("aaaabbreviation", "ab-brn"), Some("aaaabbreviation"));
        assert_eq!(long_of("more words are considered aaaabbreviation", "ab-brn"), Some("aaaabbreviation"));
        assert_eq!(long_of("Hyp7rtension", "H7"), Some("Hyp7rtension"));
    }

    #[test]
    fn the_first_letter_starts_a_word() {
        // The A of "AR" must start a word: "receptor" has an r after an
        // a in "androgen", but "a" inside "the" does not count.
        assert_eq!(long_of("within the androgen receptor", "AR"), Some("androgen receptor"));
        // A hyphen starts a word; the long form still widens to the space.
        assert_eq!(long_of("the pre-mRNA", "mR"), Some("pre-mRNA"));
    }

    #[test]
    fn both_orders_are_found() {
        assert_eq!(all("the World Health Organization (WHO) said"), [("WHO", "World Health Organization")]);
        assert_eq!(
            all("blah SBMA (Spinal and bulbar muscular atrophy)"),
            [("SBMA", "Spinal and bulbar muscular atrophy")]
        );
    }

    #[test]
    fn a_comma_or_period_bounds_the_long_form() {
        assert_eq!(all("cells, tumour necrosis factor (TNF) rose"), [("TNF", "tumour necrosis factor")]);
        assert_eq!(all("Done. Tumour necrosis factor (TNF) rose"), [("TNF", "Tumour necrosis factor")]);
        // A comma inside the parenthesis cuts the short form.
        assert_eq!(all("World Health Organization (WHO, 1999)"), [("WHO", "World Health Organization")]);
        assert_eq!(all("World Health Organization (WHO; Geneva)"), [("WHO", "World Health Organization")]);
    }

    #[test]
    fn a_closing_parenthesis_bounds_the_next_long_form() {
        // scispaCy issue 441.
        let text = "The thyroid hormone receptor (TR) inhibiting retinoic malate receptor (RMR) isoforms";
        assert_eq!(all(text), [("TR", "thyroid hormone receptor"), ("RMR", "retinoic malate receptor")]);
    }

    #[test]
    fn non_definitions_are_left_alone() {
        for text in [
            "the largest reductions in the number of ions transported antisunward occurred at the highest altitudes (Spenner et al., 1995).",
            "H2)]+(14)s.t. (1), (4).Similarly",
            ".(21)In (21), λ",
            "map expX (·) : R",
            "Φg(h),ThΦg(v) ) , (h, v)",
            "dimension;(S-iii) The optimal control problem obtained in (S-ii) is con-verted",
            "z), πut (z)) )",
            "(19, 9, 4) Hadamard Designs and Their Residual Designs",
            "a value (n = 12) was seen",
            "an unclosed (parenthesis",
        ] {
            assert_eq!(all(text), [], "{text}");
        }
    }

    #[test]
    fn the_word_count_limit_holds() {
        // Two letters allow min(2 + 5, 4) = 4 words.
        assert_eq!(all("alpha one two beta (AB)"), [("AB", "alpha one two beta")]);
        assert_eq!(all("alpha one two three beta (AB)"), []);
        // Hyphens split words for the count.
        assert_eq!(all("alpha-one-two-three beta (AB)"), []);
    }

    #[test]
    fn a_reversed_short_form_needs_a_capital() {
        assert_eq!(all("who (World Health Organization)"), []);
        assert_eq!(all("the WHO (World Health Organization) met"), [("WHO", "World Health Organization")]);
    }

    #[test]
    fn spacing_is_kept_inside_the_long_form() {
        // scispaCy's test_multiple_spaces.
        let text = "by      designing A     Lite BERT (ALBERT) architecture";
        assert_eq!(all(text), [("ALBERT", "A     Lite BERT")]);
    }

    #[test]
    fn punctuation_left_at_the_ends_of_a_long_form_is_dropped() {
        // An unclosed bracket, a slash after the parenthesis before, quotes.
        assert_eq!(all("muscles [genioglossus (GG)"), [("GG", "genioglossus")]);
        let text = "forced expiratory volume in 1 s (FEV1)/forced vital capacity (FVC)";
        assert_eq!(all(text), [("FEV1", "forced expiratory volume in 1 s"), ("FVC", "forced vital capacity")]);
        assert_eq!(all("an \"astigmatic intensity equation\" (AIE)"), [("AIE", "astigmatic intensity equation")]);
        // A bracket the long form closes stays.
        assert_eq!(all("of [K+]e times (KE)"), [("KE", "[K+]e times")]);
    }

    #[test]
    fn the_long_form_runs_to_the_parenthesis() {
        let text = "World Health Organization in Geneva (WHO)";
        assert_eq!(all(text), [("WHO", "World Health Organization in Geneva")]);
    }

    #[test]
    fn a_short_form_may_start_with_a_digit() {
        // scispaCy finds nothing here; the paper allows a digit first.
        assert_eq!(all("0,(3)with the following data: (3-i) (q"), [("3-i", "0,(3)with the following data:")]);
    }

    #[test]
    fn non_ascii_text_keeps_byte_offsets() {
        let text = "la Organización Mundial de la Salud (OMS) dijo";
        let mut out = Vec::new();
        definitions_in(text, Span::new(0, text.len() as u32), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(&text[out[0].long.range()], "Organización Mundial de la Salud");
        assert_eq!(out[0].short, Span::new(38, 41));
        assert_eq!(out[0].end, 42);
        assert!(!out[0].is_reversed());
        assert_eq!(out[0].start(), out[0].long.start);
    }

    #[test]
    fn only_the_sentence_is_read() {
        let text = "Tumour necrosis factor. Then (TNF) rose";
        let mut out = Vec::new();
        definitions_in(text, Span::new(24, text.len() as u32), &mut out);
        assert_eq!(out, []);
    }
}
