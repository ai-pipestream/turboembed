//! The encodings of the WHATWG Encoding Standard as a type, and the
//! `turbo:encoding` layer, kept apart from the decoders in
//! `turbo-encoding` so a step that only reads which encoding a document
//! was in does not depend on them.

use std::fmt;

use turbo_analysis::{Anchor, Layer};

mod labels;

/// One encoding of the WHATWG Encoding Standard, by its name there.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Encoding {
    Utf8,
    Ibm866,
    Iso8859_2,
    Iso8859_3,
    Iso8859_4,
    Iso8859_5,
    Iso8859_6,
    Iso8859_7,
    Iso8859_8,
    /// ISO-8859-8 in logical order: the same bytes and characters as
    /// `Iso8859_8`, a different label.
    Iso8859_8I,
    Iso8859_10,
    Iso8859_13,
    Iso8859_14,
    Iso8859_15,
    Iso8859_16,
    Koi8R,
    Koi8U,
    Macintosh,
    Windows874,
    Windows1250,
    Windows1251,
    Windows1252,
    Windows1253,
    Windows1254,
    Windows1255,
    Windows1256,
    Windows1257,
    Windows1258,
    XMacCyrillic,
    /// GBK decodes as gb18030 does.
    Gbk,
    Gb18030,
    Big5,
    EucJp,
    Iso2022Jp,
    ShiftJis,
    EucKr,
    /// The labels of encodings unsafe to decode (ISO-2022-KR, HZ-GB-2312
    /// and others): any input decodes to one replacement character.
    Replacement,
    Utf16Be,
    Utf16Le,
    XUserDefined,
}

impl Encoding {
    /// Every encoding, in the order the standard lists them.
    pub const ALL: [Encoding; 40] = [
        Encoding::Utf8,
        Encoding::Ibm866,
        Encoding::Iso8859_2,
        Encoding::Iso8859_3,
        Encoding::Iso8859_4,
        Encoding::Iso8859_5,
        Encoding::Iso8859_6,
        Encoding::Iso8859_7,
        Encoding::Iso8859_8,
        Encoding::Iso8859_8I,
        Encoding::Iso8859_10,
        Encoding::Iso8859_13,
        Encoding::Iso8859_14,
        Encoding::Iso8859_15,
        Encoding::Iso8859_16,
        Encoding::Koi8R,
        Encoding::Koi8U,
        Encoding::Macintosh,
        Encoding::Windows874,
        Encoding::Windows1250,
        Encoding::Windows1251,
        Encoding::Windows1252,
        Encoding::Windows1253,
        Encoding::Windows1254,
        Encoding::Windows1255,
        Encoding::Windows1256,
        Encoding::Windows1257,
        Encoding::Windows1258,
        Encoding::XMacCyrillic,
        Encoding::Gbk,
        Encoding::Gb18030,
        Encoding::Big5,
        Encoding::EucJp,
        Encoding::Iso2022Jp,
        Encoding::ShiftJis,
        Encoding::EucKr,
        Encoding::Replacement,
        Encoding::Utf16Be,
        Encoding::Utf16Le,
        Encoding::XUserDefined,
    ];

    /// The encoding a label names, as the standard's "get an encoding":
    /// ASCII whitespace around the label is ignored and ASCII letters
    /// match in either case. None for an unknown label.
    pub fn for_label(label: &str) -> Option<Encoding> {
        let label = label.trim_matches(|c| matches!(c, '\t' | '\n' | '\x0C' | '\r' | ' '));
        if label.len() > 32 {
            return None;
        }
        let lower = label.to_ascii_lowercase();
        labels::LABELS.binary_search_by(|(l, _)| (*l).cmp(lower.as_str())).ok().map(|i| labels::LABELS[i].1)
    }

    /// The standard's name for the encoding.
    pub fn name(self) -> &'static str {
        match self {
            Encoding::Utf8 => "UTF-8",
            Encoding::Ibm866 => "IBM866",
            Encoding::Iso8859_2 => "ISO-8859-2",
            Encoding::Iso8859_3 => "ISO-8859-3",
            Encoding::Iso8859_4 => "ISO-8859-4",
            Encoding::Iso8859_5 => "ISO-8859-5",
            Encoding::Iso8859_6 => "ISO-8859-6",
            Encoding::Iso8859_7 => "ISO-8859-7",
            Encoding::Iso8859_8 => "ISO-8859-8",
            Encoding::Iso8859_8I => "ISO-8859-8-I",
            Encoding::Iso8859_10 => "ISO-8859-10",
            Encoding::Iso8859_13 => "ISO-8859-13",
            Encoding::Iso8859_14 => "ISO-8859-14",
            Encoding::Iso8859_15 => "ISO-8859-15",
            Encoding::Iso8859_16 => "ISO-8859-16",
            Encoding::Koi8R => "KOI8-R",
            Encoding::Koi8U => "KOI8-U",
            Encoding::Macintosh => "macintosh",
            Encoding::Windows874 => "windows-874",
            Encoding::Windows1250 => "windows-1250",
            Encoding::Windows1251 => "windows-1251",
            Encoding::Windows1252 => "windows-1252",
            Encoding::Windows1253 => "windows-1253",
            Encoding::Windows1254 => "windows-1254",
            Encoding::Windows1255 => "windows-1255",
            Encoding::Windows1256 => "windows-1256",
            Encoding::Windows1257 => "windows-1257",
            Encoding::Windows1258 => "windows-1258",
            Encoding::XMacCyrillic => "x-mac-cyrillic",
            Encoding::Gbk => "GBK",
            Encoding::Gb18030 => "gb18030",
            Encoding::Big5 => "Big5",
            Encoding::EucJp => "EUC-JP",
            Encoding::Iso2022Jp => "ISO-2022-JP",
            Encoding::ShiftJis => "Shift_JIS",
            Encoding::EucKr => "EUC-KR",
            Encoding::Replacement => "replacement",
            Encoding::Utf16Be => "UTF-16BE",
            Encoding::Utf16Le => "UTF-16LE",
            Encoding::XUserDefined => "x-user-defined",
        }
    }

    /// Whether an ASCII byte outside a multi-byte sequence decodes to
    /// itself: true for all but UTF-16, ISO-2022-JP and the replacement
    /// encoding. In Shift_JIS, Big5, EUC-KR and gb18030 ASCII bytes can
    /// still be the trail bytes of a multi-byte character.
    pub fn ascii_compatible(self) -> bool {
        !matches!(self, Encoding::Utf16Be | Encoding::Utf16Le | Encoding::Iso2022Jp | Encoding::Replacement)
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How the encoding of a document was settled.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EncodingEvidence {
    /// A byte order mark at the start of the bytes.
    Bom,
    /// A label the caller supplied (an HTTP header, a `<meta>` tag, a
    /// file format's own field).
    Declared,
    /// The bytes are valid UTF-8 and contain non-ASCII.
    ValidUtf8,
    /// The bytes are all ASCII: any ASCII-compatible encoding reads them
    /// the same.
    Ascii,
    /// Scored by the detector against the legacy encodings.
    Detected,
}

/// The document's encoding (`turbo:encoding`): the encoding its bytes
/// were decoded with, how sure the choice is, and what settled it.
///
/// `confidence` is 1 for a byte order mark, a declared label, valid
/// UTF-8 and pure ASCII. For a detected encoding it is the detector's own
/// measure in [0, 1] (`turbo_encoding::Detection`): near 0 when another
/// encoding scored about as well, near 1 when the guess stood well clear
/// of the next one. It ranks guesses; it is not a calibrated probability.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SourceEncoding {
    pub encoding: Encoding,
    pub confidence: f32,
    pub evidence: EncodingEvidence,
}

impl Layer for SourceEncoding {
    const ID: &'static str = "turbo:encoding";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        1
    }

    fn check(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(format!("confidence {} is outside [0, 1]", self.confidence));
        }
        match self.evidence {
            EncodingEvidence::Bom => {
                if !matches!(self.encoding, Encoding::Utf8 | Encoding::Utf16Be | Encoding::Utf16Le) {
                    return Err(format!("encoding {} has no byte order mark", self.encoding));
                }
                if self.confidence != 1.0 {
                    return Err(format!("confidence {} is not 1 for {:?}", self.confidence, self.evidence));
                }
            }
            EncodingEvidence::Declared => {
                if self.confidence != 1.0 {
                    return Err(format!("confidence {} is not 1 for {:?}", self.confidence, self.evidence));
                }
            }
            EncodingEvidence::ValidUtf8 => {
                if self.encoding != Encoding::Utf8 {
                    return Err(format!("encoding {} is not UTF-8 for ValidUtf8 evidence", self.encoding));
                }
                if self.confidence != 1.0 {
                    return Err(format!("confidence {} is not 1 for {:?}", self.confidence, self.evidence));
                }
            }
            EncodingEvidence::Ascii => {
                if !self.encoding.ascii_compatible() {
                    return Err(format!("encoding {} is not ASCII-compatible for Ascii evidence", self.encoding));
                }
                if self.confidence != 1.0 {
                    return Err(format!("confidence {} is not 1 for {:?}", self.confidence, self.evidence));
                }
            }
            EncodingEvidence::Detected => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_resolve_as_the_standard_says() {
        assert_eq!(Encoding::for_label("utf-8"), Some(Encoding::Utf8));
        assert_eq!(Encoding::for_label("  UTF8\n"), Some(Encoding::Utf8));
        assert_eq!(Encoding::for_label("latin1"), Some(Encoding::Windows1252));
        assert_eq!(Encoding::for_label("ISO-8859-1"), Some(Encoding::Windows1252));
        assert_eq!(Encoding::for_label("ascii"), Some(Encoding::Windows1252));
        assert_eq!(Encoding::for_label("sjis"), Some(Encoding::ShiftJis));
        assert_eq!(Encoding::for_label("x-sjis"), Some(Encoding::ShiftJis));
        assert_eq!(Encoding::for_label("ks_c_5601-1987"), Some(Encoding::EucKr));
        assert_eq!(Encoding::for_label("gb2312"), Some(Encoding::Gbk));
        assert_eq!(Encoding::for_label("big5-hkscs"), Some(Encoding::Big5));
        assert_eq!(Encoding::for_label("utf-16"), Some(Encoding::Utf16Le));
        assert_eq!(Encoding::for_label("iso-2022-kr"), Some(Encoding::Replacement));
        assert_eq!(Encoding::for_label("utf-7"), None);
        assert_eq!(Encoding::for_label(""), None);
    }

    #[test]
    fn every_encoding_is_named_by_its_own_name() {
        for e in Encoding::ALL {
            assert_eq!(Encoding::for_label(e.name()), Some(e), "{e}");
        }
    }

    #[test]
    fn the_layer_refuses_a_confidence_outside_the_unit_interval() {
        let layer = |confidence| SourceEncoding {
            encoding: Encoding::Windows1252,
            confidence,
            evidence: EncodingEvidence::Detected,
        };
        assert!(layer(0.0).check().is_ok());
        assert!(layer(0.5).check().is_ok());
        assert!(layer(1.0).check().is_ok());
        assert!(layer(-0.1).check().is_err());
        assert!(layer(1.5).check().is_err());
        assert!(layer(f32::NAN).check().is_err());
    }

    #[test]
    fn the_layer_enforces_evidence_encoding_and_confidence_invariants() {
        for enc in [Encoding::Utf8, Encoding::Utf16Be, Encoding::Utf16Le] {
            assert!(SourceEncoding { encoding: enc, confidence: 1.0, evidence: EncodingEvidence::Bom }.check().is_ok());
            assert!(
                SourceEncoding { encoding: enc, confidence: 0.9, evidence: EncodingEvidence::Bom }.check().is_err()
            );
        }
        assert!(
            SourceEncoding { encoding: Encoding::Windows1252, confidence: 1.0, evidence: EncodingEvidence::Bom }
                .check()
                .is_err()
        );
        assert!(
            SourceEncoding { encoding: Encoding::Replacement, confidence: 1.0, evidence: EncodingEvidence::Declared }
                .check()
                .is_ok()
        );
        assert!(
            SourceEncoding { encoding: Encoding::Utf8, confidence: 0.5, evidence: EncodingEvidence::Declared }
                .check()
                .is_err()
        );
        assert!(
            SourceEncoding { encoding: Encoding::Utf8, confidence: 1.0, evidence: EncodingEvidence::ValidUtf8 }
                .check()
                .is_ok()
        );
        assert!(
            SourceEncoding { encoding: Encoding::Windows1252, confidence: 1.0, evidence: EncodingEvidence::ValidUtf8 }
                .check()
                .is_err()
        );
        assert!(
            SourceEncoding { encoding: Encoding::Utf8, confidence: 0.8, evidence: EncodingEvidence::ValidUtf8 }
                .check()
                .is_err()
        );
        assert!(
            SourceEncoding { encoding: Encoding::Windows1252, confidence: 1.0, evidence: EncodingEvidence::Ascii }
                .check()
                .is_ok()
        );
        for non_ascii in [Encoding::Utf16Be, Encoding::Utf16Le, Encoding::Iso2022Jp, Encoding::Replacement] {
            assert!(
                SourceEncoding { encoding: non_ascii, confidence: 1.0, evidence: EncodingEvidence::Ascii }
                    .check()
                    .is_err()
            );
        }
        assert!(
            SourceEncoding { encoding: Encoding::Utf8, confidence: 0.5, evidence: EncodingEvidence::Ascii }
                .check()
                .is_err()
        );
    }
}
