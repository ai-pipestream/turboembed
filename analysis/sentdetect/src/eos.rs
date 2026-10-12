//! The end-of-sentence scanner: OpenNLP's `DefaultEndOfSentenceScanner`.
//!
//! A model names the characters that may end a sentence (the manifest's
//! `eosCharacters`, a Java string, so a set of UTF-16 code units). Every
//! occurrence of one of them in the text is a candidate the model is
//! asked about. Every set OpenNLP ships is inside the Basic Multilingual
//! Plane, and a set holding a surrogate is refused: the Java scanner would
//! report each half of a character outside the BMP as a candidate of its
//! own, which is not a position a UTF-8 span can start or end at.

use std::fmt;

/// The characters `SentenceDetectorFactory` falls back to when a model's
/// manifest names none: `. ! ?`.
pub const DEFAULT_EOS: [u16; 3] = [b'.' as u16, b'!' as u16, b'?' as u16];

/// One candidate: the UTF-8 byte offset of an end-of-sentence character
/// and its UTF-16 index, the position Java reports.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Position {
    pub byte: u32,
    pub unit: u32,
}

/// A text with its length in UTF-16 code units, as
/// [`EndOfSentence::positions`] counted it while listing the candidates.
/// The context generator and the abbreviation veto index the text by code
/// unit and need that length; carrying it with the text keeps it from
/// being paired with another text.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Scanned<'t> {
    text: &'t str,
    len16: u32,
}

impl<'t> Scanned<'t> {
    /// The text.
    pub fn text(&self) -> &'t str {
        self.text
    }

    /// The text's length in UTF-16 code units.
    pub fn len16(&self) -> u32 {
        self.len16
    }

    /// Whether the text is all ASCII: one byte per code unit.
    pub(crate) fn is_ascii(&self) -> bool {
        self.len16 as usize == self.text.len()
    }

    /// Checks, in debug builds, what can be checked of a position without
    /// walking the text: a character boundary inside the text, and byte
    /// and unit indexes that agree in their bounds (each code unit before
    /// it is one to three bytes, and so is each after it).
    #[inline]
    pub(crate) fn debug_check(&self, at: Position) {
        let (byte, unit) = (at.byte as usize, at.unit as usize);
        let (len, len16) = (self.text.len(), self.len16 as usize);
        debug_assert!(byte < len && self.text.is_char_boundary(byte), "{at:?} is not a character of the text");
        debug_assert!(unit <= byte && byte <= 3 * unit, "{at:?}: byte and unit indexes disagree");
        debug_assert!(
            len16 - unit <= len - byte && len - byte <= 3 * (len16 - unit),
            "{at:?}: byte and unit indexes disagree with the text length"
        );
    }
}

/// A set of end-of-sentence characters held a surrogate code unit.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SurrogateInSet(pub u16);

impl fmt::Display for SurrogateInSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "end-of-sentence characters hold the surrogate U+{:04X}; only characters in the BMP can end a sentence",
            self.0
        )
    }
}

impl std::error::Error for SurrogateInSet {}

/// A set of end-of-sentence characters, all in the BMP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndOfSentence {
    /// Membership of each byte value; only ASCII entries can be set.
    ascii: [bool; 256],
    /// The members at or above U+0080, sorted, without repeats.
    wide: Vec<u16>,
}

impl EndOfSentence {
    /// The set of these code units. Repeats are allowed, as in the Java
    /// `HashSet` the scanner builds; a surrogate is refused.
    pub fn new(units: &[u16]) -> Result<EndOfSentence, SurrogateInSet> {
        let mut ascii = [false; 256];
        let mut wide = Vec::new();
        for &u in units {
            if (0xD800..=0xDFFF).contains(&u) {
                return Err(SurrogateInSet(u));
            }
            if u < 0x80 {
                ascii[usize::from(u)] = true;
            } else {
                wide.push(u);
            }
        }
        wide.sort_unstable();
        wide.dedup();
        Ok(EndOfSentence { ascii, wide })
    }

    /// The set of the UTF-16 code units of `chars`, as `String.toCharArray`
    /// gives them: a character outside the BMP is two surrogates, and is
    /// refused.
    pub fn from_chars(chars: &str) -> Result<EndOfSentence, SurrogateInSet> {
        let units: Vec<u16> = chars.encode_utf16().collect();
        EndOfSentence::new(&units)
    }

    /// Whether the code unit ends sentences.
    #[inline]
    pub fn contains(&self, unit: u16) -> bool {
        if unit < 0x80 { self.ascii[usize::from(unit)] } else { self.wide.binary_search(&unit).is_ok() }
    }

    /// Whether the code unit is in the set, for a unit that may be absent
    /// (before the start or past the end of a text).
    #[inline]
    pub(crate) fn contains_opt(&self, unit: Option<u16>) -> bool {
        unit.is_some_and(|u| self.contains(u))
    }

    /// Replaces the contents of `out` with every position of `text` that
    /// holds a member of the set, in text order, and returns the text with
    /// its length in UTF-16 code units. One pass over the bytes: the UTF-16
    /// index is counted as it goes (one unit per character, two for a
    /// four-byte one), and a character is decoded only when the set has
    /// members outside ASCII and the byte starts a two- or three-byte
    /// character.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes.
    pub fn positions<'t>(&self, text: &'t str, out: &mut Vec<Position>) -> Scanned<'t> {
        let len16 = self.scan(text, out);
        Scanned { text, len16 }
    }

    fn scan(&self, text: &str, out: &mut Vec<Position>) -> u32 {
        assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
        out.clear();
        let bytes = text.as_bytes();
        let mut unit = 0u32;
        if self.wide.is_empty() {
            for (i, &b) in bytes.iter().enumerate() {
                if self.ascii[usize::from(b)] {
                    out.push(Position { byte: i as u32, unit });
                }
                unit += u32::from(b & 0xC0 != 0x80) + u32::from(b >= 0xF0);
            }
            return unit;
        }
        for (i, &b) in bytes.iter().enumerate() {
            let hit = if b < 0x80 {
                self.ascii[usize::from(b)]
            } else if (0xC0..0xF0).contains(&b) {
                let cp = crate::scan::decode(bytes, i).0;
                self.wide.binary_search(&(cp as u16)).is_ok()
            } else {
                false
            };
            if hit {
                out.push(Position { byte: i as u32, unit });
            }
            unit += u32::from(b & 0xC0 != 0x80) + u32::from(b >= 0xF0);
        }
        unit
    }
}

impl Default for EndOfSentence {
    /// The default set, `. ! ?`.
    fn default() -> EndOfSentence {
        EndOfSentence::new(&DEFAULT_EOS).expect("no surrogates")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(set: &EndOfSentence, text: &str) -> (Vec<(u32, u32)>, u32) {
        let mut out = vec![Position::default(); 3];
        let scanned = set.positions(text, &mut out);
        assert_eq!(scanned.text(), text);
        (out.iter().map(|p| (p.byte, p.unit)).collect(), scanned.len16())
    }

    #[test]
    fn default_set() {
        let set = EndOfSentence::default();
        assert!(set.contains(u16::from(b'.')) && set.contains(u16::from(b'!')) && set.contains(u16::from(b'?')));
        assert!(!set.contains(u16::from(b';')) && !set.contains(0xAB));
        // "Hi." then U+1F600 (4 bytes, 2 units), "?" then "é" (2 bytes) and "!".
        assert_eq!(scan(&set, "Hi.\u{1F600}?\u{E9}!"), (vec![(2, 2), (7, 5), (10, 7)], 8));
        assert_eq!(scan(&set, ""), (vec![], 0));
        assert_eq!(scan(&set, "no end"), (vec![], 6));
    }

    #[test]
    fn portuguese_set_with_wide_members() {
        // The manifest value of the Portuguese model, after property decoding.
        let set = EndOfSentence::from_chars(".?!;:()\u{AB}\u{BB}'\"").unwrap();
        assert!(set.contains(0xAB) && set.contains(0xBB) && set.contains(u16::from(b'"')));
        // U+00AB at 0..2, x at 2, U+00BB at 3..5, U+20AC (not a member) at 5..8, ";" at 8.
        assert_eq!(scan(&set, "\u{AB}x\u{BB}\u{20AC};"), (vec![(0, 0), (3, 2), (8, 4)], 5));
        // A member never matches inside a four-byte character.
        assert_eq!(scan(&set, "\u{1F600}:"), (vec![(4, 2)], 3));
    }

    #[test]
    fn three_byte_members() {
        let set = EndOfSentence::from_chars("\u{3002}\u{FF01}\u{FF1F}").unwrap();
        assert_eq!(scan(&set, "\u{65E5}\u{3002}a\u{FF1F}"), (vec![(3, 1), (7, 3)], 4));
    }

    #[test]
    fn surrogates_are_refused() {
        assert_eq!(EndOfSentence::new(&[0x2E, 0xD83D]), Err(SurrogateInSet(0xD83D)));
        assert_eq!(EndOfSentence::from_chars(".\u{1F600}"), Err(SurrogateInSet(0xD83D)));
        assert!(SurrogateInSet(0xDC00).to_string().contains("U+DC00"));
        // Repeats are fine.
        assert_eq!(EndOfSentence::from_chars("..!?").unwrap(), EndOfSentence::default());
    }
}
