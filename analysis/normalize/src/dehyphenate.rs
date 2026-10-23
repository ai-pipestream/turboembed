//! Joins words that were hyphenated across a line break.

use std::sync::Arc;

use turbo_analysis::ViewBuilder;
use turbo_unicode::category::general_category;
use turbo_unicode::java::{is_letter, to_lowercase};
use turbo_unicode::props::uppercase;

use crate::noise::Dictionary;
use crate::substitute::{decode_at, decode_before, utf8_len};
use crate::tables::{is_horizontal_whitespace, is_line_break};
use crate::{Normalizer, hash_config};

/// What happens to a hyphen at a line break.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum DehyphenateMode {
    /// The hyphen, the line break and the indentation after it are all
    /// deleted. This is OpenNLP's behaviour, except that a letter may be
    /// followed by combining marks before the hyphen.
    #[default]
    Join,
    /// As `Join`, except that a U+002D HYPHEN-MINUS followed (after the
    /// break) by an uppercase letter is kept: only the break and the
    /// indentation are deleted.
    KeepCompounds,
    /// Decides whether to join or keep a U+002D HYPHEN-MINUS at a line
    /// break from a configured lexicon (`Dictionary`), such as a Hunspell
    /// or Morfologik dictionary.
    Lexicon,
}

/// Rejoins a word that a typesetter or a text extractor split at the end
/// of a line: "litiga-\ntion" becomes "litigation".
///
/// A join needs, in this order: a letter, optionally followed by
/// combining marks; a hyphen (U+002D HYPHEN-MINUS, U+00AD SOFT HYPHEN or
/// U+2010 HYPHEN); exactly one line break from
/// [`LINE_BREAKS`](crate::tables::LINE_BREAKS), where a carriage return
/// and line feed together count as one; any amount of whitespace that is
/// not a line break (the indentation of the continuation line); and a
/// letter. The hyphen, the break and the indentation are deleted as one
/// edit, so the joined word maps back to the whole source stretch, from
/// the first half through the break to the second half.
///
/// Letters are `java.lang.Character.isLetter` letters (general category
/// L) and combining marks are general category M. OpenNLP requires the
/// character right before the hyphen to be a letter, so it does not join
/// a word that ends in a combining mark: decomposed (NFD) text such as
/// "cafe\u{301}-\nteria", or Devanagari, Bengali and other Indic words
/// that end in a vowel sign. This normalizer joins them.
///
/// Anything else is left alone: a hyphen with no line break after it
/// ("well-known"), U+2011 NON-BREAKING HYPHEN (which says the word was not
/// broken there), a second line break after the first (a blank line or a
/// form feed page break), and a continuation that starts with a digit, punctuation or a
/// combining mark.
///
/// Without a dictionary the default mode cannot tell a broken word from a
/// compound that happened to be broken at its own hyphen, so
/// "well-\nknown" becomes "wellknown". [`Dehyphenate::keep_compounds`]
/// keeps the hyphen when it is U+002D and the next part starts with an
/// uppercase letter, which catches compounds of names and capitalised
/// words ("Wi-\nFi", "M\u{FC}ller-\nL\u{FC}denscheidt",
/// "Baden-\nW\u{FC}rttemberg"). [`Dehyphenate::with_lexicon`] (and the
/// [`Dehyphenate::with_hunspell`] and [`Dehyphenate::with_morfologik`]
/// constructors) consults a [`Dictionary`]: a U+002D hyphen-minus is
/// deleted when the joined word is in the lexicon ("litiga-\ntion",
/// "Mc-\nDonald"), kept when the joined word is not in the lexicon and
/// either the hyphenated form or both halves are in the lexicon
/// ("well-\nknown"), and otherwise falls back to the uppercase check.
///
/// OpenNLP counterpart: `DehyphenationCharSequenceNormalizer`.
#[derive(Clone, Default)]
pub struct Dehyphenate {
    mode: DehyphenateMode,
    lexicon: Option<Arc<dyn Dictionary>>,
}

impl std::fmt::Debug for Dehyphenate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dehyphenate").field("mode", &self.mode).field("lexicon", &self.lexicon.is_some()).finish()
    }
}

impl Dehyphenate {
    /// The OpenNLP behaviour: every qualifying hyphen is deleted.
    pub fn new() -> Dehyphenate {
        Dehyphenate { mode: DehyphenateMode::Join, lexicon: None }
    }

    /// Keeps a hyphen-minus that is followed by an uppercase letter.
    pub fn keep_compounds() -> Dehyphenate {
        Dehyphenate { mode: DehyphenateMode::KeepCompounds, lexicon: None }
    }

    /// Decides whether to keep a hyphen-minus at a line break using `lexicon`.
    pub fn with_lexicon(lexicon: impl Dictionary + 'static) -> Dehyphenate {
        Dehyphenate { mode: DehyphenateMode::Lexicon, lexicon: Some(Arc::new(lexicon)) }
    }

    /// Decides whether to keep a hyphen-minus at a line break using a
    /// Hunspell [`turbo_hunspell::Dictionary`].
    pub fn with_hunspell(dictionary: Arc<turbo_hunspell::Dictionary>) -> Dehyphenate {
        Self::with_lexicon(dictionary)
    }

    /// Decides whether to keep a hyphen-minus at a line break using a
    /// [`turbo_morfologik::MorfologikDictionary`].
    pub fn with_morfologik(dictionary: Arc<turbo_morfologik::MorfologikDictionary>) -> Dehyphenate {
        Self::with_lexicon(dictionary)
    }

    pub fn mode(&self) -> DehyphenateMode {
        self.mode
    }
}

impl Normalizer for Dehyphenate {
    const ID: &'static str = "turbo:normalize:dehyphenate";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let Some(hyphen) = hyphen_at(text, i) else {
                i += utf8_len(bytes[i]);
                continue;
            };
            let after_hyphen = i + hyphen.len_utf8();
            match join_end(text, i, after_hyphen) {
                Some(end) => {
                    let keep_hyphen = self.should_keep_hyphen(text, i, end, hyphen);
                    out.keep(if keep_hyphen { after_hyphen } else { i });
                    out.replace(end, "");
                    // The letter at `end` cannot be a hyphen, so the scan
                    // resumes there.
                    i = end;
                }
                None => i = after_hyphen,
            }
        }
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.mode)
    }
}

impl Dehyphenate {
    fn should_keep_hyphen(&self, text: &str, hyphen_idx: usize, cont_idx: usize, hyphen: char) -> bool {
        if hyphen != '-' {
            return false;
        }
        match self.mode {
            DehyphenateMode::Join => false,
            DehyphenateMode::KeepCompounds => uppercase(decode_at(text, cont_idx)),
            DehyphenateMode::Lexicon => {
                let Some(lexicon) = self.lexicon.as_deref() else {
                    return uppercase(decode_at(text, cont_idx));
                };
                let left = left_stem(text, hyphen_idx);
                let right = right_stem(text, cont_idx);
                let joined = format!("{left}{right}");
                if lexicon_knows(lexicon, &joined) {
                    return false;
                }
                let hyphenated = format!("{left}-{right}");
                if lexicon_knows(lexicon, &hyphenated)
                    || (lexicon_knows(lexicon, &left) && lexicon_knows(lexicon, &right))
                {
                    return true;
                }
                uppercase(decode_at(text, cont_idx))
            }
        }
    }
}

fn lexicon_knows(lexicon: &dyn Dictionary, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    if lexicon.contains(word) {
        return true;
    }
    let mut lower = String::with_capacity(word.len());
    to_lowercase(word, &mut lower);
    lower != word && lexicon.contains(&lower)
}

fn is_word_char(c: char) -> bool {
    is_letter(c) || general_category(c).is_mark()
}

/// Collects the letter-and-mark stem before `hyphen_idx`, walking across any
/// preceding qualifying line-break hyphens that are part of the same split word.
fn left_stem(text: &str, hyphen_idx: usize) -> String {
    let mut segments: Vec<&str> = Vec::new();
    let mut end = hyphen_idx;
    loop {
        let mut start = end;
        while start > 0 {
            let (prev, c) = decode_before(text, start);
            if !is_word_char(c) {
                break;
            }
            start = prev;
        }
        segments.push(&text[start..end]);
        if let Some(prev_hyphen) = preceding_join_hyphen(text, start) {
            end = prev_hyphen;
        } else {
            break;
        }
    }
    segments.reverse();
    segments.concat()
}

/// If `cont_idx` is the continuation of an earlier qualifying hyphen + line break,
/// returns the byte offset of that hyphen.
fn preceding_join_hyphen(text: &str, cont_idx: usize) -> Option<usize> {
    let mut at = cont_idx;
    while at > 0 {
        let (prev, c) = decode_before(text, at);
        if is_horizontal_whitespace(c) {
            at = prev;
        } else if is_line_break(c) {
            at = prev;
            if c == '\n' && at > 0 && text.as_bytes()[at - 1] == b'\r' {
                at -= 1;
            }
            break;
        } else {
            return None;
        }
    }
    if at == 0 {
        return None;
    }
    let (h_start, _) = decode_before(text, at);
    let h = hyphen_at(text, h_start)?;
    (h_start + h.len_utf8() == at && join_end(text, h_start, at) == Some(cont_idx)).then_some(h_start)
}

/// Collects the letter-and-mark stem starting at `cont_idx`, walking across any
/// subsequent qualifying soft or typeset line-break hyphens in the same word.
fn right_stem(text: &str, cont_idx: usize) -> String {
    let mut out = String::new();
    let mut start = cont_idx;
    loop {
        let mut end = start;
        while end < text.len() {
            let c = decode_at(text, end);
            if !is_word_char(c) {
                break;
            }
            end += c.len_utf8();
        }
        out.push_str(&text[start..end]);
        if end < text.len()
            && let Some(h) = hyphen_at(text, end)
            && let Some(next_cont) = join_end(text, end, end + h.len_utf8())
        {
            start = next_cont;
        } else {
            break;
        }
    }
    out
}

/// The hyphen that starts at byte `at`, if there is one. U+00AD starts
/// with the byte 0xC2 and U+2010 with 0xE2; no other lead byte needs
/// decoding.
#[inline]
fn hyphen_at(text: &str, at: usize) -> Option<char> {
    match text.as_bytes()[at] {
        b'-' => Some('-'),
        0xC2 | 0xE2 => {
            let c = decode_at(text, at);
            matches!(c, '\u{00AD}' | '\u{2010}').then_some(c)
        }
        _ => None,
    }
}

/// Whether the text before byte `at` ends with a letter, allowing combining
/// marks after it: "caf\u{E9}" written as "cafe" and U+0301, and Indic
/// words that end in a vowel sign, end with a letter and its marks.
fn ends_with_letter(text: &str, mut at: usize) -> bool {
    while at > 0 {
        let (start, c) = decode_before(text, at);
        if !general_category(c).is_mark() {
            return is_letter(c);
        }
        at = start;
    }
    false
}

/// Where the deletion for the hyphen at `hyphen..after_hyphen` ends (the
/// offset of the letter that continues the word), or `None` when the
/// hyphen does not qualify.
fn join_end(text: &str, hyphen: usize, after_hyphen: usize) -> Option<usize> {
    if !ends_with_letter(text, hyphen) || after_hyphen == text.len() {
        return None;
    }
    let first = decode_at(text, after_hyphen);
    if !is_line_break(first) {
        return None;
    }
    let mut end = after_hyphen + first.len_utf8();
    if first == '\r' && text.as_bytes().get(end) == Some(&b'\n') {
        end += 1;
    }
    while end < text.len() {
        let c = decode_at(text, end);
        if !is_horizontal_whitespace(c) {
            return is_letter(c).then_some(end);
        }
        end += c.len_utf8();
    }
    None
}
