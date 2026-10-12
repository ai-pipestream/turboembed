//! Transliterates the German umlauts and sharp s to ASCII.

use turbo_analysis::ViewBuilder;
use turbo_unicode::category::general_category;
use turbo_unicode::props::uppercase;

use crate::substitute::{decode_at, decode_before, for_each_non_ascii, substitute};
use crate::{Normalizer, hash_config};

/// Writes the German umlauts and sharp s in ASCII, as DIN 5007-2 spells
/// them: ä ö ü become ae oe ue, Ä Ö Ü become Ae Oe Ue, ß becomes ss and
/// ẞ (U+1E9E) becomes SS. "Müller" becomes "Mueller", "Straße" becomes
/// "Strasse". Each character is one edit.
///
/// Only the precomposed letters are changed. A vowel followed by U+0308
/// COMBINING DIAERESIS is left as it is; composing the text (NFC) first
/// turns it into the precomposed letter.
///
/// [`GermanUmlauts::uppercase_aware`] writes a capital umlaut in an
/// uppercase word with two capitals, so "ÜBER" becomes "UEBER" rather than
/// "UeBER".
///
/// OpenNLP counterpart: `GermanUmlautCharSequenceNormalizer`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GermanUmlauts {
    uppercase_aware: bool,
}

impl GermanUmlauts {
    /// The fold of OpenNLP: a capital umlaut is always written with one
    /// capital (Ae, Oe, Ue).
    pub fn new() -> GermanUmlauts {
        GermanUmlauts { uppercase_aware: false }
    }

    /// A capital umlaut is written AE, OE or UE when it is part of an
    /// uppercase word: when the next letter is uppercase, or when there is
    /// no next letter in the word (the umlaut ends it) and the letter
    /// before it is uppercase. Combining marks between letters are passed
    /// over. "ÄRGER" becomes "AERGER", "MÜ" becomes "MUE", "Öl" stays
    /// "Oel" and a lone "Ö" stays "Oe".
    pub fn uppercase_aware() -> GermanUmlauts {
        GermanUmlauts { uppercase_aware: true }
    }
}

impl Normalizer for GermanUmlauts {
    const ID: &'static str = "turbo:normalize:german-umlauts";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        if !self.uppercase_aware {
            substitute(text, out, expansion);
            return;
        }
        for_each_non_ascii(text, |at, c| {
            let Some(with) = expansion(c) else { return };
            let end = at + c.len_utf8();
            let with = match all_caps(c) {
                Some(caps) if in_uppercase_word(text, at, end) => caps,
                _ => with,
            };
            out.keep(at);
            out.replace(end, with);
        });
    }

    fn config_hash(&self) -> u64 {
        hash_config(self)
    }
}

/// The DIN 5007-2 spelling of an umlaut or sharp s.
fn expansion(c: char) -> Option<&'static str> {
    Some(match c {
        '\u{E4}' => "ae",
        '\u{F6}' => "oe",
        '\u{FC}' => "ue",
        '\u{C4}' => "Ae",
        '\u{D6}' => "Oe",
        '\u{DC}' => "Ue",
        '\u{DF}' => "ss",
        '\u{1E9E}' => "SS",
        _ => return None,
    })
}

/// The spelling of a capital umlaut inside an uppercase word.
fn all_caps(c: char) -> Option<&'static str> {
    match c {
        '\u{C4}' => Some("AE"),
        '\u{D6}' => Some("OE"),
        '\u{DC}' => Some("UE"),
        _ => None,
    }
}

/// Whether the character at `start..end` of `text` is in an uppercase
/// word: the next letter is uppercase, or no letter follows and the
/// letter before is uppercase. Combining marks are passed over both ways.
fn in_uppercase_word(text: &str, start: usize, end: usize) -> bool {
    match next_non_mark(text, end) {
        Some(next) if general_category(next).is_letter() => uppercase(next),
        _ => previous_non_mark(text, start).is_some_and(|prev| general_category(prev).is_letter() && uppercase(prev)),
    }
}

fn next_non_mark(text: &str, mut at: usize) -> Option<char> {
    while at < text.len() {
        let c = decode_at(text, at);
        if !general_category(c).is_mark() {
            return Some(c);
        }
        at += c.len_utf8();
    }
    None
}

fn previous_non_mark(text: &str, mut at: usize) -> Option<char> {
    while at > 0 {
        let (start, c) = decode_before(text, at);
        if !general_category(c).is_mark() {
            return Some(c);
        }
        at = start;
    }
    None
}
