//! Strict character encoding and decoding matching `java.nio.charset.Charset` for
//! Morfologik and PoliMorf dictionaries.

use crate::error::MorfologikError;

const UNMAPPED: u16 = 0xFFFF;

/// Java `windows-1250` upper half (`0x80..=0xFF`), with `0x81`, `0x83`, `0x88`, `0x90`, `0x98`
/// unmapped as in `sun.nio.cs.MS1250`.
#[rustfmt::skip]
static WINDOWS_1250: [u16; 128] = [
    0x20AC, UNMAPPED, 0x201A, UNMAPPED, 0x201E, 0x2026, 0x2020, 0x2021, UNMAPPED, 0x2030, 0x0160, 0x2039, 0x015A, 0x0164, 0x017D, 0x0179,
    UNMAPPED, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, UNMAPPED, 0x2122, 0x0161, 0x203A, 0x015B, 0x0165, 0x017E, 0x017A,
    0x00A0, 0x02C7, 0x02D8, 0x0141, 0x00A4, 0x0104, 0x00A6, 0x00A7, 0x00A8, 0x00A9, 0x015E, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x017B,
    0x00B0, 0x00B1, 0x02DB, 0x0142, 0x00B4, 0x00B5, 0x00B6, 0x00B7, 0x00B8, 0x0105, 0x015F, 0x00BB, 0x013D, 0x02DD, 0x013E, 0x017C,
    0x0154, 0x00C1, 0x00C2, 0x0102, 0x00C4, 0x0139, 0x0106, 0x00C7, 0x010C, 0x00C9, 0x0118, 0x00CB, 0x011A, 0x00CD, 0x00CE, 0x010E,
    0x0110, 0x0143, 0x0147, 0x00D3, 0x00D4, 0x0150, 0x00D6, 0x00D7, 0x0158, 0x016E, 0x00DA, 0x0170, 0x00DC, 0x00DD, 0x0162, 0x00DF,
    0x0155, 0x00E1, 0x00E2, 0x0103, 0x00E4, 0x013A, 0x0107, 0x00E7, 0x010D, 0x00E9, 0x0119, 0x00EB, 0x011B, 0x00ED, 0x00EE, 0x010F,
    0x0111, 0x0144, 0x0148, 0x00F3, 0x00F4, 0x0151, 0x00F6, 0x00F7, 0x0159, 0x016F, 0x00FA, 0x0171, 0x00FC, 0x00FD, 0x0163, 0x02D9,
];

/// Java `windows-1251` upper half (`0x80..=0xFF`), with `0x98` unmapped as in `sun.nio.cs.MS1251`.
#[rustfmt::skip]
static WINDOWS_1251: [u16; 128] = [
    0x0402, 0x0403, 0x201A, 0x0453, 0x201E, 0x2026, 0x2020, 0x2021, 0x20AC, 0x2030, 0x0409, 0x2039, 0x040A, 0x040C, 0x040B, 0x040F,
    0x0452, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, UNMAPPED, 0x2122, 0x0459, 0x203A, 0x045A, 0x045C, 0x045B, 0x045F,
    0x00A0, 0x040E, 0x045E, 0x0408, 0x00A4, 0x0490, 0x00A6, 0x00A7, 0x0401, 0x00A9, 0x0404, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x0407,
    0x00B0, 0x00B1, 0x0406, 0x0456, 0x0491, 0x00B5, 0x00B6, 0x00B7, 0x0451, 0x2116, 0x0454, 0x00BB, 0x0458, 0x0405, 0x0455, 0x0457,
    0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
    0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
    0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
    0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
];

/// Java `windows-1252` upper half (`0x80..=0xFF`), with `0x81`, `0x8D`, `0x8F`, `0x90`, `0x9D`
/// unmapped as in `sun.nio.cs.MS1252`.
#[rustfmt::skip]
static WINDOWS_1252: [u16; 128] = [
    0x20AC, UNMAPPED, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, UNMAPPED, 0x017D, UNMAPPED,
    UNMAPPED, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, UNMAPPED, 0x017E, 0x0178,
    0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7, 0x00A8, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
    0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7, 0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF,
    0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7, 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF,
    0x00D0, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x00D7, 0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF,
    0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7, 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF,
    0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7, 0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x00FF,
];

/// Java `ISO-8859-2` upper half (`0x80..=0xFF`).
#[rustfmt::skip]
static ISO_8859_2: [u16; 128] = [
    0x0080, 0x0081, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089, 0x008A, 0x008B, 0x008C, 0x008D, 0x008E, 0x008F,
    0x0090, 0x0091, 0x0092, 0x0093, 0x0094, 0x0095, 0x0096, 0x0097, 0x0098, 0x0099, 0x009A, 0x009B, 0x009C, 0x009D, 0x009E, 0x009F,
    0x00A0, 0x0104, 0x02D8, 0x0141, 0x00A4, 0x013D, 0x015A, 0x00A7, 0x00A8, 0x0160, 0x015E, 0x0164, 0x0179, 0x00AD, 0x017D, 0x017B,
    0x00B0, 0x0105, 0x02DB, 0x0142, 0x00B4, 0x013E, 0x015B, 0x02C7, 0x00B8, 0x0161, 0x015F, 0x0165, 0x017A, 0x02DD, 0x017E, 0x017C,
    0x0154, 0x00C1, 0x00C2, 0x0102, 0x00C4, 0x0139, 0x0106, 0x00C7, 0x010C, 0x00C9, 0x0118, 0x00CB, 0x011A, 0x00CD, 0x00CE, 0x010E,
    0x0110, 0x0143, 0x0147, 0x00D3, 0x00D4, 0x0150, 0x00D6, 0x00D7, 0x0158, 0x016E, 0x00DA, 0x0170, 0x00DC, 0x00DD, 0x0162, 0x00DF,
    0x0155, 0x00E1, 0x00E2, 0x0103, 0x00E4, 0x013A, 0x0107, 0x00E7, 0x010D, 0x00E9, 0x0119, 0x00EB, 0x011B, 0x00ED, 0x00EE, 0x010F,
    0x0111, 0x0144, 0x0148, 0x00F3, 0x00F4, 0x0151, 0x00F6, 0x00F7, 0x0159, 0x016F, 0x00FA, 0x0171, 0x00FC, 0x00FD, 0x0163, 0x02D9,
];

/// Supported character encodings for Morfologik and PoliMorf dictionaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DictionaryCharset {
    /// UTF-8 (`StandardCharsets.UTF_8`).
    #[default]
    Utf8,
    /// US-ASCII (`StandardCharsets.US_ASCII`).
    UsAscii,
    /// ISO-8859-1 (`StandardCharsets.ISO_8859_1`).
    Iso8859_1,
    /// ISO-8859-2 (`ISO-8859-2`).
    Iso8859_2,
    /// Windows-1250 (`windows-1250`).
    Windows1250,
    /// Windows-1251 (`windows-1251`).
    Windows1251,
    /// Windows-1252 (`windows-1252`).
    Windows1252,
    /// UTF-16 with byte-order mark (`StandardCharsets.UTF_16`).
    Utf16,
    /// UTF-16 big-endian (`StandardCharsets.UTF_16BE`).
    Utf16Be,
    /// UTF-16 little-endian (`StandardCharsets.UTF_16LE`).
    Utf16Le,
}

impl DictionaryCharset {
    /// Resolves a charset name as `Charset.forName` does.
    pub fn for_name(name: &str) -> Result<Self, MorfologikError> {
        let trimmed = name.trim();
        if trimmed.eq_ignore_ascii_case("UTF-8") || trimmed.eq_ignore_ascii_case("UTF8") {
            Ok(Self::Utf8)
        } else if trimmed.eq_ignore_ascii_case("US-ASCII") || trimmed.eq_ignore_ascii_case("ASCII") {
            Ok(Self::UsAscii)
        } else if trimmed.eq_ignore_ascii_case("ISO-8859-1")
            || trimmed.eq_ignore_ascii_case("ISO_8859_1")
            || trimmed.eq_ignore_ascii_case("LATIN1")
            || trimmed.eq_ignore_ascii_case("ISO-LATIN-1")
        {
            Ok(Self::Iso8859_1)
        } else if trimmed.eq_ignore_ascii_case("ISO-8859-2")
            || trimmed.eq_ignore_ascii_case("ISO_8859_2")
            || trimmed.eq_ignore_ascii_case("LATIN2")
            || trimmed.eq_ignore_ascii_case("ISO-LATIN-2")
        {
            Ok(Self::Iso8859_2)
        } else if trimmed.eq_ignore_ascii_case("windows-1250") || trimmed.eq_ignore_ascii_case("cp1250") {
            Ok(Self::Windows1250)
        } else if trimmed.eq_ignore_ascii_case("windows-1251") || trimmed.eq_ignore_ascii_case("cp1251") {
            Ok(Self::Windows1251)
        } else if trimmed.eq_ignore_ascii_case("windows-1252") || trimmed.eq_ignore_ascii_case("cp1252") {
            Ok(Self::Windows1252)
        } else if trimmed.eq_ignore_ascii_case("UTF-16") || trimmed.eq_ignore_ascii_case("UTF16") {
            Ok(Self::Utf16)
        } else if trimmed.eq_ignore_ascii_case("UTF-16BE") {
            Ok(Self::Utf16Be)
        } else if trimmed.eq_ignore_ascii_case("UTF-16LE") {
            Ok(Self::Utf16Le)
        } else {
            Err(MorfologikError::invalid_argument(format!("unsupported charset: {name}")))
        }
    }

    /// Canonical Java `Charset.name()`.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Utf8 => "UTF-8",
            Self::UsAscii => "US-ASCII",
            Self::Iso8859_1 => "ISO-8859-1",
            Self::Iso8859_2 => "ISO-8859-2",
            Self::Windows1250 => "windows-1250",
            Self::Windows1251 => "windows-1251",
            Self::Windows1252 => "windows-1252",
            Self::Utf16 => "UTF-16",
            Self::Utf16Be => "UTF-16BE",
            Self::Utf16Le => "UTF-16LE",
        }
    }

    /// Strictly decodes `bytes` in this charset, reporting malformed or unmappable
    /// sequences as `MorfologikError::Format`.
    pub fn decode(self, bytes: &[u8]) -> Result<String, MorfologikError> {
        match self {
            Self::Utf8 => std::str::from_utf8(bytes)
                .map(str::to_owned)
                .map_err(|_| MorfologikError::format("invalid morfologik character data")),
            Self::UsAscii => {
                if bytes.is_ascii() {
                    Ok(std::str::from_utf8(bytes).unwrap().to_owned())
                } else {
                    Err(MorfologikError::format("invalid morfologik character data"))
                }
            }
            Self::Iso8859_1 => {
                if bytes.is_ascii() {
                    Ok(std::str::from_utf8(bytes).unwrap().to_owned())
                } else {
                    Ok(bytes.iter().map(|&b| char::from(b)).collect())
                }
            }
            Self::Iso8859_2 => decode_table(bytes, &ISO_8859_2),
            Self::Windows1250 => decode_table(bytes, &WINDOWS_1250),
            Self::Windows1251 => decode_table(bytes, &WINDOWS_1251),
            Self::Windows1252 => decode_table(bytes, &WINDOWS_1252),
            Self::Utf16 => {
                if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
                    decode_utf16(rest, true)
                } else if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
                    decode_utf16(rest, false)
                } else {
                    decode_utf16(bytes, true)
                }
            }
            Self::Utf16Be => decode_utf16(bytes, true),
            Self::Utf16Le => decode_utf16(bytes, false),
        }
    }

    /// Strictly encodes `text` into bytes in this charset, reporting unmappable characters.
    pub fn encode(self, text: &str) -> Result<Vec<u8>, MorfologikError> {
        match self {
            Self::Utf8 => Ok(text.as_bytes().to_vec()),
            Self::UsAscii => {
                if text.is_ascii() {
                    Ok(text.as_bytes().to_vec())
                } else {
                    Err(MorfologikError::invalid_argument(format!("cannot encode in {}", self.canonical_name())))
                }
            }
            Self::Iso8859_1 => {
                let mut out = Vec::with_capacity(text.len());
                for ch in text.chars() {
                    let cp = ch as u32;
                    if cp <= 0xFF {
                        out.push(cp as u8);
                    } else {
                        return Err(MorfologikError::invalid_argument(format!(
                            "cannot encode in {}",
                            self.canonical_name()
                        )));
                    }
                }
                Ok(out)
            }
            Self::Iso8859_2 => encode_table(text, &ISO_8859_2, self.canonical_name()),
            Self::Windows1250 => encode_table(text, &WINDOWS_1250, self.canonical_name()),
            Self::Windows1251 => encode_table(text, &WINDOWS_1251, self.canonical_name()),
            Self::Windows1252 => encode_table(text, &WINDOWS_1252, self.canonical_name()),
            Self::Utf16 => {
                let mut out = vec![0xFE, 0xFF];
                for unit in text.encode_utf16() {
                    out.extend_from_slice(&unit.to_be_bytes());
                }
                Ok(out)
            }
            Self::Utf16Be => {
                let mut out = Vec::with_capacity(text.len() * 2);
                for unit in text.encode_utf16() {
                    out.extend_from_slice(&unit.to_be_bytes());
                }
                Ok(out)
            }
            Self::Utf16Le => {
                let mut out = Vec::with_capacity(text.len() * 2);
                for unit in text.encode_utf16() {
                    out.extend_from_slice(&unit.to_le_bytes());
                }
                Ok(out)
            }
        }
    }

    /// Encodes a single separator character in this charset, requiring that it produces
    /// exactly one byte.
    pub fn encode_single_byte(self, ch: char) -> Result<u8, MorfologikError> {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        let encoded = self.encode(s)?;
        if encoded.len() != 1 {
            return Err(MorfologikError::invalid_argument(format!(
                "fsa.dict.separator must encode as one byte in {}",
                self.canonical_name()
            )));
        }
        Ok(encoded[0])
    }
}

fn decode_table(bytes: &[u8], table: &[u16; 128]) -> Result<String, MorfologikError> {
    if bytes.is_ascii() {
        return Ok(std::str::from_utf8(bytes).unwrap().to_owned());
    }
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b < 0x80 {
            out.push(char::from(b));
        } else {
            let mapped = table[usize::from(b - 0x80)];
            if mapped == UNMAPPED {
                return Err(MorfologikError::format("invalid morfologik character data"));
            }
            out.push(char::from_u32(u32::from(mapped)).expect("valid BMP code point"));
        }
    }
    Ok(out)
}

fn encode_table(text: &str, table: &[u16; 128], name: &'static str) -> Result<Vec<u8>, MorfologikError> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let cp = ch as u32;
        if cp < 0x80 {
            out.push(cp as u8);
        } else if cp <= 0xFFFF && cp != u32::from(UNMAPPED) {
            let target = cp as u16;
            if let Some(idx) = table.iter().position(|&v| v == target) {
                out.push((idx as u8) + 0x80);
            } else {
                return Err(MorfologikError::invalid_argument(format!("cannot encode in {name}")));
            }
        } else {
            return Err(MorfologikError::invalid_argument(format!("cannot encode in {name}")));
        }
    }
    Ok(out)
}

fn decode_utf16(bytes: &[u8], big_endian: bool) -> Result<String, MorfologikError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(MorfologikError::format("invalid morfologik character data"));
    }
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| if big_endian { u16::from_be_bytes(pair) } else { u16::from_le_bytes(pair) })
        .collect();
    String::from_utf16(&units).map_err(|_| MorfologikError::format("invalid morfologik character data"))
}
