//! The text as a sequence of characters, each with the `Sentence_Break`
//! class the rules read, after a language's terminators and the line break
//! handling have changed it.

use turbo_unicode::SentenceBreak::{self, *};
use turbo_unicode::sentence_break::sentence_break;

use super::profile::LanguageRules;
use super::{LineBreaks, Rules};

/// One character: where it starts and its class.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Unit {
    pub start: u32,
    pub class: SentenceBreak,
}

/// U+2026 HORIZONTAL ELLIPSIS, `Other` in Unicode, ends a sentence here as
/// a full stop does.
pub(crate) const ELLIPSIS: char = '\u{2026}';

/// Whether a class is one of the paragraph separators of UAX #29.
#[inline]
pub(crate) fn is_para_sep(class: SentenceBreak) -> bool {
    matches!(class, Cr | Lf | Sep)
}

/// Whether a character ends a line: the line breaks of UAX #29 but
/// U+2029 PARAGRAPH SEPARATOR.
#[inline]
fn is_line_break(c: char) -> bool {
    matches!(c, '\r' | '\n' | '\u{85}' | '\u{2028}')
}

/// Fills `units` with one unit per character of `text`.
pub(crate) fn classify(
    text: &str,
    rules: Rules,
    language: &LanguageRules,
    line_breaks: LineBreaks,
    units: &mut Vec<Unit>,
) {
    units.clear();
    units.reserve(text.len() / 2);
    for (i, c) in text.char_indices() {
        let mut class = sentence_break(c as u32);
        if rules.terminators {
            if language.enders.contains(&c) {
                class = STerm;
            } else if language.not_enders.contains(&c) {
                class = SContinue;
            } else if c == ELLIPSIS {
                class = ATerm;
            }
        }
        units.push(Unit { start: i as u32, class });
    }
    if rules.terminators {
        bracketed_ellipses(text, units);
    }
    if line_breaks == LineBreaks::Paragraph {
        join_lines(text, units);
    }
}

/// An ellipsis in brackets, `[...]`, `[…]` or `(...)`, marks words left out
/// of a quotation and ends no sentence: its full stops become `Other`.
fn bracketed_ellipses(text: &str, units: &mut [Unit]) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < units.len() {
        let open = bytes[units[i].start as usize];
        if !(open == b'[' || open == b'(') {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while units.get(j).is_some_and(|u| u.class == ATerm) {
            j += 1;
        }
        let close = if open == b'[' { b']' } else { b')' };
        if j > i + 1 && units.get(j).is_some_and(|u| bytes[u.start as usize] == close) {
            for unit in &mut units[i + 1..j] {
                unit.class = Other;
            }
        }
        i = j.max(i + 1);
    }
}

/// Turns every line break that does not end a paragraph into a space. A
/// line break ends a paragraph when only spaces separate it from the next
/// line break or U+2029; a CR LF pair is one line break.
fn join_lines(text: &str, units: &mut [Unit]) {
    let bytes = text.as_bytes();
    let char_at = |u: &Unit| text[u.start as usize..].chars().next().unwrap_or(' ');
    let mut i = 0;
    while i < units.len() {
        let u = units[i];
        if !is_para_sep(u.class) || !is_line_break(char_at(&u)) {
            i += 1;
            continue;
        }
        let mut end = i + 1;
        if bytes[u.start as usize] == b'\r' && units.get(end).is_some_and(|n| n.class == Lf) {
            end += 1;
        }
        let mut next = end;
        while units.get(next).is_some_and(|n| matches!(n.class, Sp | Extend | Format)) {
            next += 1;
        }
        let ends_paragraph = match units.get(next) {
            Some(n) => is_para_sep(n.class),
            None => true,
        };
        if !ends_paragraph {
            for unit in &mut units[i..end] {
                unit.class = Sp;
            }
        }
        i = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::profile::{EL, HY, ROOT};

    fn classes(text: &str, rules: Rules, language: &LanguageRules, lines: LineBreaks) -> Vec<SentenceBreak> {
        let mut units = Vec::new();
        classify(text, rules, language, lines, &mut units);
        units.iter().map(|u| u.class).collect()
    }

    #[test]
    fn terminators_by_language() {
        let all = Rules::ALL;
        assert_eq!(classes("a;", all, &EL, LineBreaks::Sentence), [Lower, STerm]);
        assert_eq!(classes("a;", all, &ROOT, LineBreaks::Sentence), [Lower, SContinue]);
        assert_eq!(classes("a.:", all, &HY, LineBreaks::Sentence), [Lower, SContinue, STerm]);
        assert_eq!(classes("a\u{2026}", all, &ROOT, LineBreaks::Sentence), [Lower, ATerm]);
        assert_eq!(classes("a\u{2026}", Rules::NONE, &EL, LineBreaks::Sentence), [Lower, Other]);
        assert_eq!(classes("a;", Rules::NONE, &EL, LineBreaks::Sentence), [Lower, SContinue]);
        // A bracketed ellipsis ends nothing; an ellipsis alone still does.
        assert_eq!(classes("[...]", all, &ROOT, LineBreaks::Sentence), [Close, Other, Other, Other, Close]);
        assert_eq!(classes("(\u{2026})", all, &ROOT, LineBreaks::Sentence), [Close, Other, Close]);
        assert_eq!(classes("[..", all, &ROOT, LineBreaks::Sentence), [Close, ATerm, ATerm]);
        assert_eq!(classes("[...]", Rules::NONE, &ROOT, LineBreaks::Sentence), [Close, ATerm, ATerm, ATerm, Close]);
    }

    #[test]
    fn single_line_breaks_join() {
        let p = LineBreaks::Paragraph;
        assert_eq!(classes("a\nb", Rules::ALL, &ROOT, p), [Lower, Sp, Lower]);
        assert_eq!(classes("a\r\nb", Rules::ALL, &ROOT, p), [Lower, Sp, Sp, Lower]);
        assert_eq!(classes("a\n \nb", Rules::ALL, &ROOT, p), [Lower, Lf, Sp, Sp, Lower]);
        assert_eq!(classes("a\r\n\r\nb", Rules::ALL, &ROOT, p), [Lower, Cr, Lf, Sp, Sp, Lower]);
        assert_eq!(classes("a\u{2028}b\u{85}", Rules::ALL, &ROOT, p), [Lower, Sp, Lower, Sep]);
        assert_eq!(classes("a\u{2029}b", Rules::ALL, &ROOT, p), [Lower, Sep, Lower]);
        assert_eq!(classes("a\n\u{2029}b", Rules::ALL, &ROOT, p), [Lower, Lf, Sep, Lower]);
        assert_eq!(classes("a\nb", Rules::ALL, &ROOT, LineBreaks::Sentence), [Lower, Lf, Lower]);
    }
}
