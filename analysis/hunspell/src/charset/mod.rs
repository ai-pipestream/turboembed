//! The character encoding `SET` declares, and decoding both files with it
//! the way Java's strict `CharsetDecoder` does.
//!
//! OpenNLP decodes with `Charset.forName(name)` and a decoder that reports
//! malformed and unmappable input. UTF-8 is decoded here as Java decodes it
//! (Rust's validation and Java's strict UTF-8 decoder refuse the same byte
//! sequences), and so are the single byte charsets of [`tables`], whose
//! tables the JDK wrote. A name the JDK knows that has no table here is
//! [`Lookup::NotPorted`]: Java would decode the files, the port refuses them.

mod tables;

use std::borrow::Cow;

/// A single byte charset: ASCII below 0x80, a table above.
pub(crate) struct SingleByte {
    pub(crate) name: &'static str,
    pub(crate) aliases: &'static [&'static str],
    /// The char of each byte from 0x80, or 0xFFFF where Java reports the
    /// byte as unmappable.
    pub(crate) high: [u16; 128],
}

/// An encoding the port decodes.
#[derive(Copy, Clone)]
pub(crate) enum Charset {
    Utf8,
    SingleByte(&'static SingleByte),
}

impl std::fmt::Debug for Charset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl PartialEq for Charset {
    fn eq(&self, other: &Charset) -> bool {
        self.name() == other.name()
    }
}

impl Eq for Charset {}

/// What `Charset.forName` makes of a name.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Lookup {
    Found(Charset),
    /// The JDK has the charset; this port has no table for it.
    NotPorted,
    /// The JDK does not know the name, or it is not a legal charset name.
    Unknown,
}

/// The names `Charset.forName` maps to UTF-8.
const UTF8_NAMES: [&str; 3] = ["utf-8", "utf8", "unicode-1-1-utf-8"];

impl Charset {
    /// The charset's canonical Java name, as `Charset.name()` gives it.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Charset::Utf8 => "UTF-8",
            Charset::SingleByte(s) => s.name,
        }
    }

    /// `Charset.forName(name)`: names are compared without regard to ASCII
    /// case, and a name with a character outside ASCII is illegal.
    pub(crate) fn for_name(name: &str) -> Lookup {
        if !name.is_ascii() {
            return Lookup::Unknown;
        }
        let lower = name.to_ascii_lowercase();
        if UTF8_NAMES.contains(&lower.as_str()) {
            return Lookup::Found(Charset::Utf8);
        }
        for s in &tables::SINGLE_BYTE {
            if s.aliases.iter().any(|a| a.eq_ignore_ascii_case(&lower)) {
                return Lookup::Found(Charset::SingleByte(s));
            }
        }
        if tables::JAVA_NAMES.binary_search(&lower.as_str()).is_ok() { Lookup::NotPorted } else { Lookup::Unknown }
    }

    /// The text of `bytes`, or `None` where Java's strict decoder reports
    /// a malformed or unmappable sequence.
    pub(crate) fn decode<'a>(&self, bytes: &'a [u8]) -> Option<Cow<'a, str>> {
        match self {
            Charset::Utf8 => std::str::from_utf8(bytes).ok().map(Cow::Borrowed),
            Charset::SingleByte(table) => {
                if bytes.is_ascii() {
                    return Some(Cow::Borrowed(std::str::from_utf8(bytes).expect("ASCII is UTF-8")));
                }
                let mut out = String::with_capacity(bytes.len() + bytes.len() / 2);
                for &b in bytes {
                    if b < 0x80 {
                        out.push(char::from(b));
                    } else {
                        let unit = table.high[usize::from(b - 0x80)];
                        if unit == 0xFFFF {
                            return None;
                        }
                        // Every table entry is a BMP character that is not a surrogate.
                        out.push(char::from_u32(u32::from(unit))?);
                    }
                }
                Some(Cow::Owned(out))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_resolve_as_java_resolves_them() {
        assert_eq!(Charset::for_name("UTF-8"), Lookup::Found(Charset::Utf8));
        assert_eq!(Charset::for_name("utf8"), Lookup::Found(Charset::Utf8));
        let Lookup::Found(latin1) = Charset::for_name("ISO8859-1") else { panic!() };
        assert_eq!(latin1.name(), "ISO-8859-1");
        let Lookup::Found(koi) = Charset::for_name("koi8-r") else { panic!() };
        assert_eq!(koi.name(), "KOI8-R");
        assert_eq!(Charset::for_name("Shift_JIS"), Lookup::NotPorted);
        assert_eq!(Charset::for_name("microsoft-cp1251"), Lookup::Unknown);
        assert_eq!(Charset::for_name("UTF-8\u{e9}"), Lookup::Unknown);
    }

    #[test]
    fn decodes_strictly() {
        assert_eq!(Charset::Utf8.decode(b"a\xc3\xa9").unwrap(), "a\u{e9}");
        assert!(Charset::Utf8.decode(b"a\xe9").is_none());
        assert!(Charset::Utf8.decode(b"\xed\xa0\x80").is_none());
        let Lookup::Found(latin1) = Charset::for_name("latin1") else { panic!() };
        assert_eq!(latin1.decode(b"a\xe9").unwrap(), "a\u{e9}");
        let Lookup::Found(latin2) = Charset::for_name("ISO8859-2") else { panic!() };
        assert_eq!(latin2.decode(b"\xb1").unwrap(), "\u{105}");
        let Lookup::Found(cp1252) = Charset::for_name("cp1252") else { panic!() };
        assert!(cp1252.decode(b"\x81").is_none());
        assert_eq!(cp1252.decode(b"\x80").unwrap(), "\u{20ac}");
    }
}
