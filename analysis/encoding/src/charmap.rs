//! The single-byte code pages the mojibake repair encodes text back to
//! bytes with, as ftfy (by Robyn Speer, Apache-2.0; see NOTICE) names and
//! defines them.
//!
//! The "sloppy" Windows code pages are ftfy's: the code page with every
//! byte it leaves unassigned decoded as Latin-1 would, and byte 0x1A
//! standing for U+FFFD, so a text that went through a lossy decoder can
//! still be encoded. The WHATWG windows-125x tables already assign the
//! bytes Python's code pages leave undefined (to the C1 controls), so the
//! sloppy tables are the WHATWG ones with the Latin-1 fill for the few
//! holes left (0xAA, 0xD2 and 0xFF in windows-1253, for example).
//! ISO-8859-2 and Mac OS Roman are the WHATWG ones as they are; Latin-1
//! is the identity; code page 437 is the IBM PC character set. Each was
//! checked byte for byte against the Python codecs ftfy uses.

use crate::tables;
use turbo_encoding_layers::Encoding;

/// One code page ftfy tries, in the order it tries them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Charmap {
    Latin1,
    SloppyWindows1252,
    SloppyWindows1251,
    SloppyWindows1250,
    SloppyWindows1253,
    SloppyWindows1254,
    SloppyWindows1257,
    Iso8859_2,
    MacRoman,
    Cp437,
}

/// Bytes 0x80 to 0xFF of code page 437.
#[rustfmt::skip]
static CP437: [u16; 128] = [
    0x00C7, 0x00FC, 0x00E9, 0x00E2, 0x00E4, 0x00E0, 0x00E5, 0x00E7, 0x00EA, 0x00EB, 0x00E8, 0x00EF, 0x00EE, 0x00EC, 0x00C4, 0x00C5,
    0x00C9, 0x00E6, 0x00C6, 0x00F4, 0x00F6, 0x00F2, 0x00FB, 0x00F9, 0x00FF, 0x00D6, 0x00DC, 0x00A2, 0x00A3, 0x00A5, 0x20A7, 0x0192,
    0x00E1, 0x00ED, 0x00F3, 0x00FA, 0x00F1, 0x00D1, 0x00AA, 0x00BA, 0x00BF, 0x2310, 0x00AC, 0x00BD, 0x00BC, 0x00A1, 0x00AB, 0x00BB,
    0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556, 0x2555, 0x2563, 0x2551, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
    0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F, 0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
    0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B, 0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
    0x03B1, 0x00DF, 0x0393, 0x03C0, 0x03A3, 0x03C3, 0x00B5, 0x03C4, 0x03A6, 0x0398, 0x03A9, 0x03B4, 0x221E, 0x03C6, 0x03B5, 0x2229,
    0x2261, 0x00B1, 0x2265, 0x2264, 0x2320, 0x2321, 0x00F7, 0x2248, 0x00B0, 0x2219, 0x00B7, 0x221A, 0x207F, 0x00B2, 0x25A0, 0x00A0,
];

impl Charmap {
    /// ftfy's `CHARMAP_ENCODINGS`, in its order.
    pub const ALL: [Charmap; 10] = [
        Charmap::Latin1,
        Charmap::SloppyWindows1252,
        Charmap::SloppyWindows1251,
        Charmap::SloppyWindows1250,
        Charmap::SloppyWindows1253,
        Charmap::SloppyWindows1254,
        Charmap::SloppyWindows1257,
        Charmap::Iso8859_2,
        Charmap::MacRoman,
        Charmap::Cp437,
    ];

    /// ftfy's name for the code page.
    pub fn name(self) -> &'static str {
        match self {
            Charmap::Latin1 => "latin-1",
            Charmap::SloppyWindows1252 => "sloppy-windows-1252",
            Charmap::SloppyWindows1251 => "sloppy-windows-1251",
            Charmap::SloppyWindows1250 => "sloppy-windows-1250",
            Charmap::SloppyWindows1253 => "sloppy-windows-1253",
            Charmap::SloppyWindows1254 => "sloppy-windows-1254",
            Charmap::SloppyWindows1257 => "sloppy-windows-1257",
            Charmap::Iso8859_2 => "iso-8859-2",
            Charmap::MacRoman => "macroman",
            Charmap::Cp437 => "cp437",
        }
    }

    /// Whether byte 0x1A stands for U+FFFD.
    pub fn is_sloppy(self) -> bool {
        matches!(
            self,
            Charmap::SloppyWindows1252
                | Charmap::SloppyWindows1251
                | Charmap::SloppyWindows1250
                | Charmap::SloppyWindows1253
                | Charmap::SloppyWindows1254
                | Charmap::SloppyWindows1257
        )
    }

    /// The table for bytes 0x80 to 0xFF, 0 where the byte decodes as
    /// Latin-1 would. None for Latin-1 itself.
    fn high(self) -> Option<&'static [u16; 128]> {
        let encoding = match self {
            Charmap::Latin1 => return None,
            Charmap::Cp437 => return Some(&CP437),
            Charmap::SloppyWindows1252 => Encoding::Windows1252,
            Charmap::SloppyWindows1251 => Encoding::Windows1251,
            Charmap::SloppyWindows1250 => Encoding::Windows1250,
            Charmap::SloppyWindows1253 => Encoding::Windows1253,
            Charmap::SloppyWindows1254 => Encoding::Windows1254,
            Charmap::SloppyWindows1257 => Encoding::Windows1257,
            Charmap::Iso8859_2 => Encoding::Iso8859_2,
            Charmap::MacRoman => Encoding::Macintosh,
        };
        tables::single_byte_table(encoding)
    }

    /// The character a byte decodes to. Every byte decodes to something.
    pub fn decode_byte(self, byte: u8) -> char {
        if byte < 0x80 {
            if byte == 0x1A && self.is_sloppy() {
                return '\u{FFFD}';
            }
            return char::from(byte);
        }
        match self.high().map(|t| t[usize::from(byte - 0x80)]) {
            Some(cp) if cp != 0 => char::from_u32(u32::from(cp)).expect("tables hold BMP scalar values"),
            _ => char::from(byte),
        }
    }

    /// The byte a character encodes to, if the code page has it.
    pub fn encode_char(self, c: char) -> Option<u8> {
        let cp = u32::from(c);
        if cp < 0x80 {
            return (!(cp == 0x1A && self.is_sloppy())).then_some(cp as u8);
        }
        if c == '\u{FFFD}' && self.is_sloppy() {
            return Some(0x1A);
        }
        let Some(table) = self.high() else {
            return u8::try_from(cp).ok();
        };
        if let Some(i) = table.iter().position(|&t| u32::from(t) == cp) {
            return Some(0x80 + i as u8);
        }
        // A hole the table leaves to Latin-1.
        let byte = u8::try_from(cp).ok()?;
        (table[usize::from(byte - 0x80)] == 0).then_some(byte)
    }

    /// Whether every character of `text` encodes (ftfy's
    /// `possible_encoding`).
    pub fn can_encode(self, text: &str) -> bool {
        text.chars().all(|c| self.encode_char(c).is_some())
    }

    /// Decodes `bytes` in full.
    pub fn decode(self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.decode_byte(b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_round_trips() {
        for map in Charmap::ALL {
            for b in 0..=255u8 {
                let c = map.decode_byte(b);
                assert_eq!(map.encode_char(c), Some(b), "{} byte {b:#04X} ({c:?})", map.name());
            }
        }
    }

    #[test]
    fn sloppy_pages_swap_1a_for_the_replacement_character() {
        assert_eq!(Charmap::SloppyWindows1252.decode_byte(0x1A), '\u{FFFD}');
        assert_eq!(Charmap::SloppyWindows1252.encode_char('\u{1A}'), None);
        assert_eq!(Charmap::Latin1.decode_byte(0x1A), '\u{1A}');
        assert_eq!(Charmap::Latin1.encode_char('\u{FFFD}'), None);
    }

    #[test]
    fn holes_decode_as_latin1() {
        // windows-1252 leaves 0x81 to the C1 control, windows-1253 leaves
        // 0xAA unassigned.
        assert_eq!(Charmap::SloppyWindows1252.decode_byte(0x81), '\u{81}');
        assert_eq!(Charmap::SloppyWindows1253.decode_byte(0xAA), '\u{AA}');
        assert_eq!(Charmap::SloppyWindows1252.decode_byte(0x80), '\u{20AC}');
        assert_eq!(Charmap::MacRoman.decode_byte(0xF0), '\u{F8FF}');
        assert_eq!(Charmap::Cp437.decode_byte(0xB0), '\u{2591}');
    }

    #[test]
    fn can_encode_matches_per_character_encoding() {
        assert!(Charmap::SloppyWindows1252.can_encode("Ã©tÃ©"));
        assert!(!Charmap::Latin1.can_encode("â€œ"));
        assert!(Charmap::SloppyWindows1252.can_encode("â€œ"));
    }
}
