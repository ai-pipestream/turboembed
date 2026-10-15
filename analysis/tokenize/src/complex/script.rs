//! Which characters the dictionaries segment.
//!
//! A character belongs to a complex script run when it has
//! `Line_Break=SA` (Complex_Context) and its script is Thai, Lao, Khmer or
//! Myanmar: the sets `[:sc=Thai:]&[:lb=SA:]` and so on that ICU4X's data
//! generator gives its word segmenter (ICU4X,
//! `provider/source/src/segmenter/mod.rs`). Digits, punctuation and
//! other characters of those scripts that are not `SA` are left to the
//! UAX #29 rules, as are the other `SA` scripts (Tai Tham, Tai Viet, Tai
//! Le, New Tai Lue, Ahom and others), for which there is no dictionary.

use turbo_unicode::Script;
use turbo_unicode::props::complex_context;
use turbo_unicode::script::script;

/// A script whose words a dictionary finds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ComplexScript {
    Thai,
    Lao,
    Khmer,
    /// Myanmar, the script of Burmese.
    Myanmar,
}

impl ComplexScript {
    /// Every value.
    pub const ALL: [ComplexScript; 4] =
        [ComplexScript::Thai, ComplexScript::Lao, ComplexScript::Khmer, ComplexScript::Myanmar];

    /// The script's name as `Scripts.txt` writes it.
    pub fn name(self) -> &'static str {
        match self {
            ComplexScript::Thai => "Thai",
            ComplexScript::Lao => "Lao",
            ComplexScript::Khmer => "Khmer",
            ComplexScript::Myanmar => "Myanmar",
        }
    }
}

/// The complex script run `c` belongs to, if any.
#[inline]
pub fn complex_script(c: char) -> Option<ComplexScript> {
    // The lowest character of the four scripts is U+0E01.
    if (c as u32) < 0x0E00 || !complex_context(c) {
        return None;
    }
    match script(c) {
        Script::Thai => Some(ComplexScript::Thai),
        Script::Lao => Some(ComplexScript::Lao),
        Script::Khmer => Some(ComplexScript::Khmer),
        Script::Myanmar => Some(ComplexScript::Myanmar),
        _ => None,
    }
}

/// Whether a UTF-8 byte can start a character of a complex script run:
/// every such character is in U+0E00 to U+0FFF (lead byte 0xE0), U+1000
/// to U+1FFF (0xE1) or U+A000 to U+AFFF (0xEA), which a test checks over
/// the whole code space.
#[inline]
pub(crate) fn may_start_complex(lead: u8) -> bool {
    matches!(lead, 0xE0 | 0xE1 | 0xEA)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lead_bytes_cover_every_complex_character() {
        let mut count = [0usize; 4];
        for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            if let Some(s) = complex_script(c) {
                let mut buf = [0u8; 4];
                assert!(may_start_complex(c.encode_utf8(&mut buf).as_bytes()[0]), "U+{:04X}", c as u32);
                count[ComplexScript::ALL.iter().position(|&x| x == s).unwrap()] += 1;
            }
        }
        // Thai, Lao, Khmer, Myanmar characters with Line_Break=SA in
        // Unicode 18.0. The ranges hardcoded in ICU4X's older (v1) word
        // segmenter give the same sets except for U+0EA3 LAO LETTER LO
        // LING, which its range `0x0E8C..0xEA3` leaves out.
        assert_eq!(count, [73, 73, 87, 187]);
    }

    #[test]
    fn examples() {
        assert_eq!(complex_script('\u{0E01}'), Some(ComplexScript::Thai));
        assert_eq!(complex_script('\u{0E31}'), Some(ComplexScript::Thai));
        // Thai digits and the Baht sign are not SA.
        assert_eq!(complex_script('\u{0E51}'), None);
        assert_eq!(complex_script('\u{0E3F}'), None);
        assert_eq!(complex_script('\u{0E81}'), Some(ComplexScript::Lao));
        assert_eq!(complex_script('\u{1780}'), Some(ComplexScript::Khmer));
        // The Khmer sign KHAN is punctuation.
        assert_eq!(complex_script('\u{17D4}'), None);
        assert_eq!(complex_script('\u{1000}'), Some(ComplexScript::Myanmar));
        assert_eq!(complex_script('\u{AA60}'), Some(ComplexScript::Myanmar));
        // Myanmar digits and section marks are not SA.
        assert_eq!(complex_script('\u{1041}'), None);
        assert_eq!(complex_script('\u{104A}'), None);
        // Tai Tham is SA but has no dictionary.
        assert_eq!(complex_script('\u{1A20}'), None);
        assert_eq!(complex_script('a'), None);
    }
}
