//! Lowercasing and Unicode case folding as a view.

use turbo_analysis::ViewBuilder;
use turbo_unicode::casefold;
use turbo_unicode::java::{self, Lowercase};

use crate::substitute::decode_at;
use crate::{Normalizer, hash_config};

/// Which case mapping [`CaseFold`] applies.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CaseFoldMode {
    /// `String.toLowerCase(Locale.ROOT)`: the simple lowercase mapping of
    /// each character, U+0130 to "i" U+0307, and U+03A3 to final sigma
    /// U+03C2 where the JDK's final sigma condition holds (which walks the
    /// word around the sigma, not only the case-ignorable characters;
    /// `turbo_unicode::java::to_lowercase`).
    Lowercase,
    /// Simple case folding: the CaseFolding.txt entries of status C and S.
    /// Every character folds to one character.
    Simple,
    /// Full case folding: the entries of status C and F. A character can
    /// fold to several ("ß" to "ss", U+FB00 to "ff").
    Full,
    /// Simple case folding with the Turkic entries (status T): "I" folds
    /// to dotless U+0131 and U+0130 to "i".
    TurkicSimple,
    /// Full case folding with the Turkic entries.
    TurkicFull,
    /// Turkish locale lowercase, including contextual dot removal.
    TurkishLowercase,
    /// Azerbaijani locale lowercase (the same SpecialCasing rules as Turkish).
    AzerbaijaniLowercase,
}

/// Lowercases or case folds the text, one character at a time.
///
/// Each character whose mapping differs from itself is replaced on its
/// own, so a character that expands ("ß" to "ss" under full folding) maps
/// back to exactly its source character. Every other character is kept.
/// Lowercasing uses the shared JDK sigma rule. Turkish/Azerbaijani modes
/// also inspect the original combining context of I and U+0307. Case
/// folds themselves are context free.
///
/// Case folding is meant for caseless matching, not for display: the full
/// fold maps "ß" to "ss", final sigma to "σ", and U+00B5 MICRO SIGN to
/// Greek mu, none of which lowercasing does. The folds work on the text as
/// it is. Canonical caseless matching (Unicode section 3.13, D145) puts the
/// text in NFD before and after folding, because folding does not keep a
/// normalization form: the full fold of U+01F0 is "j" U+030C, which is not
/// in NFC, and the combining mark U+0345 (class 240) folds to the letter
/// U+03B9 (class 0); chain
/// [`UnicodeForm::nfd`](crate::UnicodeForm::nfd) around this normalizer
/// for that.
///
/// OpenNLP counterparts: `CaseFoldCharSequenceNormalizer` with
/// `Locale.ROOT` ([`CaseFold::lowercase`]) and
/// `FullCaseFoldCharSequenceNormalizer` ([`CaseFold::full`]). OpenNLP's
/// `CaseFoldCharSequenceNormalizer` lowercases, it does not fold; the
/// simple fold of CaseFolding.txt has no OpenNLP counterpart.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CaseFold {
    mode: CaseFoldMode,
}

/// The lowercase ASCII letters, sliced to give a `&'static str` for each.
const ASCII_LOWER: &str = "abcdefghijklmnopqrstuvwxyz";

impl CaseFold {
    pub fn new(mode: CaseFoldMode) -> CaseFold {
        CaseFold { mode }
    }

    /// `String.toLowerCase(Locale.ROOT)`.
    pub fn lowercase() -> CaseFold {
        CaseFold::new(CaseFoldMode::Lowercase)
    }

    /// Simple case folding (CaseFolding.txt status C and S).
    pub fn simple() -> CaseFold {
        CaseFold::new(CaseFoldMode::Simple)
    }

    /// Full case folding (CaseFolding.txt status C and F).
    pub fn full() -> CaseFold {
        CaseFold::new(CaseFoldMode::Full)
    }

    /// Simple case folding with the Turkic mappings of "I" and U+0130.
    pub fn turkic_simple() -> CaseFold {
        CaseFold::new(CaseFoldMode::TurkicSimple)
    }

    /// Full case folding with the Turkic mappings of "I" and U+0130.
    pub fn turkic_full() -> CaseFold {
        CaseFold::new(CaseFoldMode::TurkicFull)
    }

    /// `String.toLowerCase(Locale.forLanguageTag("tr"))`.
    pub fn turkish_lowercase() -> CaseFold {
        CaseFold::new(CaseFoldMode::TurkishLowercase)
    }

    /// `String.toLowerCase(Locale.forLanguageTag("az"))`.
    pub fn azerbaijani_lowercase() -> CaseFold {
        CaseFold::new(CaseFoldMode::AzerbaijaniLowercase)
    }

    fn locale_lowercase(&self) -> bool {
        matches!(self.mode, CaseFoldMode::TurkishLowercase | CaseFoldMode::AzerbaijaniLowercase)
    }

    pub fn mode(&self) -> CaseFoldMode {
        self.mode
    }

    /// The replacement of the uppercase ASCII letter `b`. Every mode maps
    /// A to Z to a to z, except that the Turkic folds map "I" to U+0131.
    fn ascii_upper(&self, b: u8) -> &'static str {
        if b == b'I' && matches!(self.mode, CaseFoldMode::TurkicSimple | CaseFoldMode::TurkicFull) {
            return "\u{131}";
        }
        let k = usize::from(b - b'A');
        &ASCII_LOWER[k..k + 1]
    }

    /// Writes the mapping of the non-ASCII character `c` at byte `at` of
    /// `text` through `out` when it differs from `c`.
    fn fold_char(&self, text: &str, at: usize, c: char, out: &mut ViewBuilder<'_>) {
        let end = at + c.len_utf8();
        let mut buf = [0u8; 4];
        let with: &str = match self.mode {
            CaseFoldMode::Lowercase | CaseFoldMode::TurkishLowercase | CaseFoldMode::AzerbaijaniLowercase => {
                match if self.locale_lowercase() {
                    java::locale::turkic_lowercase_at(text, at, c)
                } else {
                    java::lowercase_at(text, at, c)
                } {
                    Lowercase::Char(lower) if lower == c => return,
                    Lowercase::Char(lower) => lower.encode_utf8(&mut buf),
                    Lowercase::Str(lower) => lower,
                }
            }
            CaseFoldMode::Simple => match casefold::simple(c) {
                folded if folded == c => return,
                folded => folded.encode_utf8(&mut buf),
            },
            CaseFoldMode::TurkicSimple => match casefold::turkic_simple(c) {
                folded if folded == c => return,
                folded => folded.encode_utf8(&mut buf),
            },
            CaseFoldMode::Full => match casefold::full(c) {
                Some(folded) => folded,
                None => return,
            },
            CaseFoldMode::TurkicFull => match casefold::turkic_full(c) {
                Some(folded) => folded,
                None => return,
            },
        };
        out.keep(at);
        out.replace(end, with);
    }
}

impl Normalizer for CaseFold {
    const ID: &'static str = "turbo:normalize:case-fold";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let bytes = text.as_bytes();
        let mut i = skip_unchanged_ascii(bytes, 0);
        while i < bytes.len() {
            let b = bytes[i];
            if b == b'I' && self.locale_lowercase() {
                self.fold_char(text, i, 'I', out);
                i += 1;
            } else if b.is_ascii_uppercase() {
                out.keep(i);
                out.replace(i + 1, self.ascii_upper(b));
                i += 1;
            } else if b.is_ascii() {
                i += 1;
            } else {
                let c = decode_at(text, i);
                self.fold_char(text, i, c, out);
                i += c.len_utf8();
            }
            i = skip_unchanged_ascii(bytes, i);
        }
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.mode)
    }
}

/// The first offset at or after `from` that holds an uppercase ASCII
/// letter or a non-ASCII byte, `bytes.len()` when there is none. Every
/// other byte is ASCII that no mode changes. Eight bytes are tested at a
/// time: for a word of ASCII bytes, adding 0x3F to each byte sets its high
/// bit when the byte is at least b'A' and adding 0x25 when it is at least
/// b'[' (no byte carries into the next, as each stays below 0x100), so a
/// byte is an uppercase letter when the first sum has the high bit and the
/// second does not.
#[inline]
fn skip_unchanged_ascii(bytes: &[u8], from: usize) -> usize {
    const HIGH: u64 = 0x8080_8080_8080_8080;
    const TO_A: u64 = 0x3F3F_3F3F_3F3F_3F3F;
    const PAST_Z: u64 = 0x2525_2525_2525_2525;
    let mut i = from;
    while let Some(chunk) = bytes.get(i..i + 8) {
        let word = u64::from_le_bytes(chunk.try_into().expect("a slice of eight bytes"));
        if word & HIGH != 0 || (word + TO_A) & !(word + PAST_Z) & HIGH != 0 {
            break;
        }
        i += 8;
    }
    while i < bytes.len() && !bytes[i].is_ascii_uppercase() && bytes[i].is_ascii() {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::skip_unchanged_ascii;

    #[test]
    fn skip_stops_at_every_uppercase_letter_and_non_ascii_byte() {
        // Each candidate byte at each position of a run of lowercase text,
        // so the word test and the byte loop are both exercised.
        for stop in [b'A', b'M', b'Z', 0x80, 0xC3, 0xFF] {
            for len in 0..20 {
                for at in 0..len {
                    let mut bytes = vec![b'a'; len];
                    bytes[at] = stop;
                    assert_eq!(skip_unchanged_ascii(&bytes, 0), at, "byte {stop:#x} at {at} of {len}");
                }
                assert_eq!(skip_unchanged_ascii(&vec![b'q'; len], 0), len);
            }
        }
        // The neighbours of the uppercase range are skipped.
        let edges = b"@[`{0 ~\t@[`{0 ~\t";
        assert_eq!(skip_unchanged_ascii(edges, 0), edges.len());
        assert_eq!(skip_unchanged_ascii(b"abcdefghijK", 3), 10);
    }
}
