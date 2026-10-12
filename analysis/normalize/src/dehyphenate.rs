//! Joins words that were hyphenated across a line break.

use turbo_analysis::ViewBuilder;
use turbo_unicode::category::general_category;
use turbo_unicode::java::is_letter;
use turbo_unicode::props::uppercase;

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
/// "Baden-\nW\u{FC}rttemberg"). It is a heuristic: it misses lowercase
/// compounds ("well-\nknown" is still joined) and wrongly keeps the
/// hyphen when a word is split before a capital it contains ("Mc-\nDonald"
/// becomes "Mc-Donald"). Deciding from a lexicon whether the joined form
/// is a word is not implemented.
///
/// OpenNLP counterpart: `DehyphenationCharSequenceNormalizer`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dehyphenate {
    mode: DehyphenateMode,
}

impl Dehyphenate {
    /// The OpenNLP behaviour: every qualifying hyphen is deleted.
    pub fn new() -> Dehyphenate {
        Dehyphenate { mode: DehyphenateMode::Join }
    }

    /// Keeps a hyphen-minus that is followed by an uppercase letter.
    pub fn keep_compounds() -> Dehyphenate {
        Dehyphenate { mode: DehyphenateMode::KeepCompounds }
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
                    let keep_hyphen =
                        self.mode == DehyphenateMode::KeepCompounds && hyphen == '-' && uppercase(decode_at(text, end));
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
