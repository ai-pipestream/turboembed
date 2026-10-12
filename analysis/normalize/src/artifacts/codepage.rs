//! The single-byte code pages whose decoding of UTF-8 bytes makes
//! mojibake, as reverse tables: for a character, the byte that encodes it.
//!
//! - Windows-1252: the table at 0x80..0x9F is CP1252.TXT from the Unicode
//!   mapping files (MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1252.TXT); 0xA0..0xFF
//!   are the Latin-1 characters of the same values. The five bytes it
//!   leaves undefined (0x81, 0x8D, 0x8F, 0x90, 0x9D), and every C1
//!   control besides, fall back to ISO-8859-1, where U+0080..U+009F are
//!   bytes 0x80..0x9F: a decoder that keeps undefined bytes as C1
//!   controls, or a plain Latin-1 decoder, produces those.
//! - MacRoman: ROMAN.TXT from the Unicode mapping files
//!   (MAPPINGS/VENDORS/APPLE/ROMAN.TXT), with 0xDB as U+20AC EURO SIGN as
//!   in Mac OS 8.5 and later.

/// The characters Windows-1252 assigns to 0x80..0x9F, indexed by byte
/// minus 0x80; `None` marks the undefined bytes.
const WINDOWS_1252_HIGH: [Option<char>; 32] = [
    Some('\u{20AC}'),
    None,
    Some('\u{201A}'),
    Some('\u{0192}'),
    Some('\u{201E}'),
    Some('\u{2026}'),
    Some('\u{2020}'),
    Some('\u{2021}'),
    Some('\u{02C6}'),
    Some('\u{2030}'),
    Some('\u{0160}'),
    Some('\u{2039}'),
    Some('\u{0152}'),
    None,
    Some('\u{017D}'),
    None,
    None,
    Some('\u{2018}'),
    Some('\u{2019}'),
    Some('\u{201C}'),
    Some('\u{201D}'),
    Some('\u{2022}'),
    Some('\u{2013}'),
    Some('\u{2014}'),
    Some('\u{02DC}'),
    Some('\u{2122}'),
    Some('\u{0161}'),
    Some('\u{203A}'),
    Some('\u{0153}'),
    None,
    Some('\u{017E}'),
    Some('\u{0178}'),
];

/// The characters MacRoman assigns to 0x80..0xFF, indexed by byte minus
/// 0x80.
pub(crate) const MAC_ROMAN_HIGH: [char; 128] = [
    '\u{00C4}', '\u{00C5}', '\u{00C7}', '\u{00C9}', '\u{00D1}', '\u{00D6}', '\u{00DC}', '\u{00E1}', // 0x80
    '\u{00E0}', '\u{00E2}', '\u{00E4}', '\u{00E3}', '\u{00E5}', '\u{00E7}', '\u{00E9}', '\u{00E8}', // 0x88
    '\u{00EA}', '\u{00EB}', '\u{00ED}', '\u{00EC}', '\u{00EE}', '\u{00EF}', '\u{00F1}', '\u{00F3}', // 0x90
    '\u{00F2}', '\u{00F4}', '\u{00F6}', '\u{00F5}', '\u{00FA}', '\u{00F9}', '\u{00FB}', '\u{00FC}', // 0x98
    '\u{2020}', '\u{00B0}', '\u{00A2}', '\u{00A3}', '\u{00A7}', '\u{2022}', '\u{00B6}', '\u{00DF}', // 0xA0
    '\u{00AE}', '\u{00A9}', '\u{2122}', '\u{00B4}', '\u{00A8}', '\u{2260}', '\u{00C6}', '\u{00D8}', // 0xA8
    '\u{221E}', '\u{00B1}', '\u{2264}', '\u{2265}', '\u{00A5}', '\u{00B5}', '\u{2202}', '\u{2211}', // 0xB0
    '\u{220F}', '\u{03C0}', '\u{222B}', '\u{00AA}', '\u{00BA}', '\u{03A9}', '\u{00E6}', '\u{00F8}', // 0xB8
    '\u{00BF}', '\u{00A1}', '\u{00AC}', '\u{221A}', '\u{0192}', '\u{2248}', '\u{2206}', '\u{00AB}', // 0xC0
    '\u{00BB}', '\u{2026}', '\u{00A0}', '\u{00C0}', '\u{00C3}', '\u{00D5}', '\u{0152}', '\u{0153}', // 0xC8
    '\u{2013}', '\u{2014}', '\u{201C}', '\u{201D}', '\u{2018}', '\u{2019}', '\u{00F7}', '\u{25CA}', // 0xD0
    '\u{00FF}', '\u{0178}', '\u{2044}', '\u{20AC}', '\u{2039}', '\u{203A}', '\u{FB01}', '\u{FB02}', // 0xD8
    '\u{2021}', '\u{00B7}', '\u{201A}', '\u{201E}', '\u{2030}', '\u{00C2}', '\u{00CA}', '\u{00C1}', // 0xE0
    '\u{00CB}', '\u{00C8}', '\u{00CD}', '\u{00CE}', '\u{00CF}', '\u{00CC}', '\u{00D3}', '\u{00D4}', // 0xE8
    '\u{F8FF}', '\u{00D2}', '\u{00DA}', '\u{00DB}', '\u{00D9}', '\u{0131}', '\u{02C6}', '\u{02DC}', // 0xF0
    '\u{00AF}', '\u{02D8}', '\u{02D9}', '\u{02DA}', '\u{00B8}', '\u{02DD}', '\u{02DB}', '\u{02C7}', // 0xF8
];

/// A code page a text may have been wrongly decoded with.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CodePage {
    /// Windows-1252, with ISO-8859-1 for the C1 controls.
    Windows1252,
    MacRoman,
}

impl CodePage {
    /// The byte that encodes `c`, or `None` when the code page has none.
    #[inline]
    pub(crate) fn byte(self, c: char) -> Option<u8> {
        if c.is_ascii() {
            return Some(c as u8);
        }
        match self {
            CodePage::Windows1252 => windows_1252(c),
            CodePage::MacRoman => mac_roman(c),
        }
    }
}

/// Windows-1252 with the ISO-8859-1 fallback, as OpenNLP's
/// `CursorArtifactDetector.singleByte` does it: Latin-1 letters and the
/// C1 controls map to themselves, the 27 other characters by the table.
fn windows_1252(c: char) -> Option<u8> {
    let cp = c as u32;
    if cp <= 0xFF {
        return Some(cp as u8);
    }
    WINDOWS_1252_HIGH.iter().position(|&h| h == Some(c)).map(|i| 0x80 + i as u8)
}

/// MacRoman by binary search in the reverse table.
fn mac_roman(c: char) -> Option<u8> {
    MAC_ROMAN_REVERSE.binary_search_by_key(&c, |&(ch, _)| ch).ok().map(|i| MAC_ROMAN_REVERSE[i].1)
}

/// `MAC_ROMAN_HIGH` inverted and sorted by character, at compile time.
static MAC_ROMAN_REVERSE: [(char, u8); 128] = {
    let mut t = [('\0', 0u8); 128];
    let mut i = 0;
    while i < 128 {
        t[i] = (MAC_ROMAN_HIGH[i], 0x80 + i as u8);
        i += 1;
    }
    // Insertion sort: 128 entries, run once by the compiler.
    let mut j = 1;
    while j < 128 {
        let mut k = j;
        while k > 0 && (t[k - 1].0 as u32) > (t[k].0 as u32) {
            let x = t[k - 1];
            t[k - 1] = t[k];
            t[k] = x;
            k -= 1;
        }
        j += 1;
    }
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mac_roman_is_a_bijection_on_the_high_half() {
        for (i, &c) in MAC_ROMAN_HIGH.iter().enumerate() {
            assert_eq!(CodePage::MacRoman.byte(c), Some(0x80 + i as u8), "{c:?}");
        }
        assert_eq!(CodePage::MacRoman.byte('\u{0080}'), None);
    }

    #[test]
    fn windows_1252_maps_its_table_and_falls_back_to_latin_1() {
        for (i, c) in WINDOWS_1252_HIGH.iter().enumerate() {
            if let Some(c) = *c {
                assert_eq!(CodePage::Windows1252.byte(c), Some(0x80 + i as u8), "{c:?}");
            }
        }
        for b in 0x80..=0xFFu32 {
            assert_eq!(CodePage::Windows1252.byte(char::from_u32(b).unwrap()), Some(b as u8));
        }
        assert_eq!(CodePage::Windows1252.byte('\u{2260}'), None);
    }
}
