//! The tailorings that look at the words around a boundary: numbered
//! references after a terminator, a lower case word after `!` or `?`,
//! abbreviations, initials and sentence starters.

use turbo_unicode::SentenceBreak::{self, *};

use super::profile::LanguageRules;
use super::uax29::{Break, NO_TERM};
use super::units::{Unit, is_para_sep};

/// The text and its units, with the helpers the tailorings share.
pub(crate) struct Text<'a> {
    pub text: &'a str,
    pub units: &'a [Unit],
}

impl Text<'_> {
    pub fn char_at(&self, i: usize) -> Option<char> {
        self.units.get(i).and_then(|u| self.text[u.start as usize..].chars().next())
    }

    pub fn class(&self, i: usize) -> Option<SentenceBreak> {
        self.units.get(i).map(|u| u.class)
    }

    /// The byte offset where unit `i` starts, or the text's length.
    pub fn offset(&self, i: usize) -> usize {
        self.units.get(i).map_or(self.text.len(), |u| u.start as usize)
    }

    /// The first unit at or after `i` that is not a space or an opening
    /// mark (a `Close` unit), where the words of a sentence begin.
    fn skip_open(&self, mut i: usize) -> usize {
        while matches!(self.class(i), Some(Sp | Close | Extend | Format)) {
            i += 1;
        }
        i
    }

    /// The word starting at unit `i`: letters, digits and marks.
    fn word_at(&self, i: usize) -> &str {
        let start = self.offset(i);
        let rest = &self.text[start..];
        let len = rest
            .char_indices()
            .find(|&(_, c)| !(c.is_alphanumeric() || matches!(self.class_of(c), Extend | Format)))
            .map_or(rest.len(), |(k, _)| k);
        &rest[..len]
    }

    fn class_of(&self, c: char) -> SentenceBreak {
        turbo_unicode::sentence_break::sentence_break(c as u32)
    }
}

/// Moves a boundary after a run of numbered references that follows the
/// terminator (`word.[1][2] Next`, `word. [3] Next`), so the references stay
/// with the sentence they belong to. Returns the boundary to keep, or
/// `None` when nothing but spaces follows the references.
pub(crate) fn after_references(t: &Text<'_>, b: Break) -> Option<Break> {
    if b.term == NO_TERM {
        return Some(b);
    }
    let mut i = b.term as usize + 1;
    let mut end = None;
    loop {
        let mut j = i;
        while t.class(j) == Some(Sp) {
            j += 1;
        }
        if t.char_at(j) != Some('[') {
            break;
        }
        let digits = j + 1;
        let mut k = digits;
        while t.char_at(k).is_some_and(|c| c.is_ascii_digit()) && k - digits < 4 {
            k += 1;
        }
        if k == digits || t.char_at(k) != Some(']') {
            break;
        }
        i = k + 1;
        end = Some(i);
    }
    let Some(end) = end else { return Some(b) };
    // References glued to a following word are not references.
    if !(t.class(end).is_none() || t.class(end) == Some(Sp) || t.class(end).is_some_and(is_para_sep)) {
        return Some(b);
    }
    let mut at = end;
    while t.class(at) == Some(Sp) {
        at += 1;
    }
    if at >= t.units.len() || t.class(at).is_some_and(is_para_sep) {
        return None;
    }
    Some(Break { at: at.max(b.at as usize) as u32, term: b.term })
}

/// Whether a boundary after `!` or `?` (an `STerm`) is followed by a lower
/// case word, which continues the sentence (`Yahoo! in`, `"Why?" she said`).
/// UAX #29 keeps such a sentence whole after a full stop (SB8) but not
/// after the other terminators.
pub(crate) fn continues_lower(t: &Text<'_>, b: Break) -> bool {
    b.term != NO_TERM && t.class(b.term as usize) == Some(STerm) && t.class(t.skip_open(b.at as usize)) == Some(Lower)
}

/// What the abbreviation and initial checks need to know about a full stop.
pub(crate) struct Options {
    pub abbreviations: bool,
    pub initials: bool,
}

/// Whether `token` is one or more initials: upper case letters, each
/// followed by a full stop (`E.`, `J.R.R.`, `Л.Н.`).
fn is_initials(token: &str) -> bool {
    let mut chars = token.chars();
    let mut any = false;
    loop {
        match (chars.next(), chars.next()) {
            (None, _) => return any,
            (Some(c), Some('.')) if c.is_uppercase() => any = true,
            _ => return false,
        }
    }
}

/// The most characters the word before a full stop may have for the
/// abbreviation and initial checks to look at it.
const MAX_TOKEN: usize = 64;

/// Whether a boundary after a full stop (an `ATerm`) is not one, because
/// the word it ends is an abbreviation or an initial, and the next word
/// is not one of the language's sentence starters.
pub(crate) fn after_abbreviation(
    t: &Text<'_>,
    b: Break,
    rules: &LanguageRules,
    o: &Options,
    lower: &mut String,
) -> bool {
    if b.term == NO_TERM || t.class(b.term as usize) != Some(ATerm) {
        return false;
    }
    let term = b.term as usize;
    // A run of terminators (`...`, `?.`) is not an abbreviation's stop.
    if term > 0 && matches!(t.class(term - 1), Some(ATerm | STerm)) {
        return false;
    }
    let mut start = term;
    while start > 0 && !matches!(t.class(start - 1), Some(Sp)) && !t.class(start - 1).is_some_and(is_para_sep) {
        // No abbreviation or run of initials is this long; stopping here
        // keeps a text without spaces (`a.Ba.Ba.B...`) linear.
        if term - start == MAX_TOKEN {
            return false;
        }
        start -= 1;
    }
    while start < term && t.class(start) == Some(Close) {
        start += 1;
    }
    if start == term {
        return false;
    }
    let token = &t.text[t.offset(start)..t.offset(term + 1)];
    let word = &t.text[t.offset(start)..t.offset(term)];
    let word = word.rsplit('/').next().unwrap_or(word);
    let mut suppressed = false;
    if o.abbreviations {
        if t.char_at(term) == Some('.') && rules.is_suppression(token) {
            suppressed = true;
        } else if !rules.abbreviations.is_empty() && !word.is_empty() {
            lower.clear();
            lower.extend(word.chars().flat_map(char::to_lowercase));
            suppressed = rules.is_abbreviation(lower);
        }
    }
    let next = t.word_at(t.skip_open(b.at as usize));
    let mut chars = word.chars();
    let single = chars.next().filter(|_| chars.next().is_none());
    let next_upper = next.chars().next().is_some_and(char::is_uppercase);
    if !suppressed && o.initials {
        suppressed = is_initials(token) && next_upper;
    }
    // The number may close a range (`1. -3. Jahrhundert`).
    let number = word.trim_start_matches(['-', '\u{2013}']);
    if !suppressed
        && rules.ordinal_stop
        && (1..=3).contains(&number.len())
        && number.bytes().all(|b| b.is_ascii_digit())
    {
        let after = t.char_at(t.skip_open(b.at as usize));
        suppressed = after.is_some_and(|c| c.is_uppercase() || c.is_ascii_digit() || matches!(c, '-' | '\u{2013}'));
    }
    if !suppressed {
        return false;
    }
    // A starter after it ends the sentence after all, unless the word is a
    // single lower case letter (`p. The`), which is rarely a sentence end.
    let single_lower = single.is_some_and(char::is_lowercase);
    !(rules.is_starter(next) && !single_lower)
}
