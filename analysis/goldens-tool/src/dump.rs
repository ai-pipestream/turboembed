//! Reading the header of a decision point dump (`docs/analysis.md`,
//! "Decision point dumps"). A dump starts with the line `dump 1` and the
//! header records a golden has; the decision point blocks follow. This
//! module parses the header into typed fields and checks that what follows
//! it starts with a `point` record or is the end of the file. The blocks
//! themselves are read by the tests that compare against them.

use std::fmt;
use std::fs;
use std::path::Path;

/// The first line of every dump; the number is the format version.
pub const FIRST_LINE: &str = "dump 1";

/// Every header record of a dump, as typed fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DumpHeader {
    /// The `generator` record: the generator's name and version.
    pub generator: String,
    pub generator_version: u32,
    /// The `opennlp` record: the commit the runtime was built from and its Maven version.
    pub opennlp_commit: [u8; 20],
    pub opennlp_version: String,
    /// The `jdk` record: `java.version`, `java.vm.version` and `java.vendor`.
    pub java_version: String,
    pub vm_version: String,
    pub vendor: String,
    /// The `unicode` record: the Unicode version of the JDK's character tables.
    pub unicode: String,
    /// The `input` record.
    pub input: InputRecord,
    /// The `model` records, in the order the annotator loaded them.
    pub models: Vec<ModelRecord>,
    /// The `annotator` record: the OpenNLP class and the method called on it.
    pub annotator_class: String,
    pub method: String,
}

/// The `input` record: `<set>/<id>`, the lengths in UTF-8 bytes and UTF-16
/// units, and the SHA-256 of the input's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputRecord {
    pub id: String,
    pub utf8_len: u32,
    pub utf16_len: u32,
    pub sha256: [u8; 32],
}

/// A `model` record: the model's file name and the SHA-512 of its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRecord {
    pub file: String,
    pub sha512: [u8; 64],
}

/// A header that does not parse: the 1-based line number and what was
/// expected there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DumpError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for DumpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for DumpError {}

impl DumpHeader {
    /// Reads and parses the header of the dump at `path`. The error names
    /// the path.
    pub fn read(path: &Path) -> Result<DumpHeader, String> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let text = std::str::from_utf8(&bytes).map_err(|e| format!("{}: not UTF-8: {e}", path.display()))?;
        DumpHeader::parse(text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Parses the header of a whole dump file's text.
    pub fn parse(text: &str) -> Result<DumpHeader, DumpError> {
        let mut lines = Lines::new(text)?;
        let first = lines.next_line("the line `dump 1`")?;
        if first != FIRST_LINE {
            return Err(lines.error(format!("expected `{FIRST_LINE}`, found `{first}`")));
        }
        let f = lines.record("generator", 3)?;
        let (generator, generator_version) = (lines.str(f[1])?, lines.u32(f[2])?);
        let f = lines.record("opennlp", 3)?;
        let (opennlp_commit, opennlp_version) = (lines.hex::<20>(f[1])?, lines.str(f[2])?);
        let f = lines.record("jdk", 4)?;
        let (java_version, vm_version, vendor) = (lines.str(f[1])?, lines.str(f[2])?, lines.str(f[3])?);
        let f = lines.record("unicode", 2)?;
        let unicode = lines.str(f[1])?;
        let f = lines.record("input", 5)?;
        let input = InputRecord {
            id: lines.str(f[1])?,
            utf8_len: lines.u32(f[2])?,
            utf16_len: lines.u32(f[3])?,
            sha256: lines.hex::<32>(f[4])?,
        };
        let mut models = Vec::new();
        while lines.peek_tag() == Some("model") {
            let f = lines.record("model", 3)?;
            models.push(ModelRecord { file: lines.str(f[1])?, sha512: lines.hex::<64>(f[2])? });
        }
        let f = lines.record("annotator", 3)?;
        let (annotator_class, method) = (lines.str(f[1])?, lines.str(f[2])?);
        match lines.peek_tag() {
            None | Some("point") => {}
            Some(other) => {
                lines.next_line("")?;
                return Err(lines.error(format!("expected a `point` record or the end of the file, found `{other}`")));
            }
        }
        Ok(DumpHeader {
            generator,
            generator_version,
            opennlp_commit,
            opennlp_version,
            java_version,
            vm_version,
            vendor,
            unicode,
            input,
            models,
            annotator_class,
            method,
        })
    }
}

/// The lines of a dump, with the number of the line last taken.
struct Lines<'a> {
    rest: std::iter::Peekable<std::str::Split<'a, char>>,
    line: usize,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Result<Lines<'a>, DumpError> {
        let Some(body) = text.strip_suffix('\n') else {
            return Err(DumpError {
                line: text.split('\n').count(),
                message: "the file does not end with a line end".into(),
            });
        };
        Ok(Lines { rest: body.split('\n').peekable(), line: 0 })
    }

    fn error(&self, message: String) -> DumpError {
        DumpError { line: self.line, message }
    }

    fn next_line(&mut self, expected: &str) -> Result<&'a str, DumpError> {
        match self.rest.next() {
            Some(l) => {
                self.line += 1;
                Ok(l)
            }
            None => {
                self.line += 1;
                Err(self.error(format!("expected {expected}, found the end of the file")))
            }
        }
    }

    /// The first field of the next line, without taking it.
    fn peek_tag(&mut self) -> Option<&'a str> {
        self.rest.peek().map(|l| l.split('\t').next().unwrap_or(""))
    }

    /// Takes the next line, which must be the record `tag` with `fields`
    /// tab separated fields in all.
    fn record(&mut self, tag: &str, fields: usize) -> Result<Vec<&'a str>, DumpError> {
        let l = self.next_line(&format!("a `{tag}` record"))?;
        let f: Vec<&str> = l.split('\t').collect();
        if f[0] != tag {
            return Err(self.error(format!("expected a `{tag}` record, found `{}`", f[0])));
        }
        if f.len() != fields {
            return Err(self.error(format!("a `{tag}` record has {fields} fields, this one {}", f.len())));
        }
        Ok(f)
    }

    fn u32(&self, s: &str) -> Result<u32, DumpError> {
        parse_u32(s).ok_or_else(|| self.error(format!("expected a u32, found `{s}`")))
    }

    fn hex<const N: usize>(&self, s: &str) -> Result<[u8; N], DumpError> {
        parse_hex::<N>(s).ok_or_else(|| self.error(format!("expected {} lowercase hex digits, found `{s}`", 2 * N)))
    }

    fn str(&self, s: &str) -> Result<String, DumpError> {
        unescape(s).map_err(|e| self.error(e))
    }
}

/// A `u32` field: decimal digits, no sign, no leading zero except in `0`.
fn parse_u32(s: &str) -> Option<u32> {
    let digits = !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits || (s.len() > 1 && s.starts_with('0')) {
        return None;
    }
    s.parse().ok()
}

/// Exactly `2 * N` lowercase hex digits.
fn parse_hex<const N: usize>(s: &str) -> Option<[u8; N]> {
    let digits = s.as_bytes();
    if digits.len() != 2 * N {
        return None;
    }
    let value = |d: u8| match d {
        b'0'..=b'9' => Some(d - b'0'),
        b'a'..=b'f' => Some(d - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; N];
    for (byte, pair) in out.iter_mut().zip(digits.as_chunks::<2>().0) {
        *byte = (value(pair[0])? << 4) | value(pair[1])?;
    }
    Some(out)
}

/// Undoes the `str` escapes: `\t`, `\n`, `\r`, `\\` and `\uXXXX` (one
/// UTF-16 unit, four lowercase hex digits; a surrogate pair is two of
/// them). A lone surrogate has no place in a Rust string and is an error.
fn unescape(s: &str) -> Result<String, String> {
    if !s.contains('\\') {
        return Ok(s.to_owned());
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    let mut units: Vec<u16> = Vec::new();
    while let Some(c) = chars.next() {
        if c != '\\' {
            flush(&mut units, &mut out)?;
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                let unit = parse_hex::<2>(&hex)
                    .map(u16::from_be_bytes)
                    .ok_or_else(|| format!("`\\u{hex}` is not four lowercase hex digits"))?;
                units.push(unit);
            }
            Some(e) => {
                flush(&mut units, &mut out)?;
                out.push(match e {
                    't' => '\t',
                    'n' => '\n',
                    'r' => '\r',
                    '\\' => '\\',
                    _ => return Err(format!("unknown escape `\\{e}`")),
                });
            }
            None => return Err("a `\\` at the end of a field".into()),
        }
    }
    flush(&mut units, &mut out)?;
    Ok(out)
}

/// Appends the UTF-16 units of consecutive `\u` escapes as characters.
fn flush(units: &mut Vec<u16>, out: &mut String) -> Result<(), String> {
    for c in char::decode_utf16(units.drain(..)) {
        out.push(c.map_err(|e| format!("a lone surrogate {:04x}", e.unpaired_surrogate()))?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "dump 1\n\
        generator\tturbo-goldens-java\t1\n\
        opennlp\t92d486d05151ef00b6d145c405046aa3a33b5a48\t3.0.0-PRE-SNAPSHOT\n\
        jdk\t21.0.1\t21.0.1+12\tA\\tVendor\n\
        unicode\t15.0\n\
        input\ten/x\t9\t8\tdaae94d6ca8153be16e834da072d9f05a8437508fe55f4102378d0cfaae36036\n\
        model\tm.bin\t";

    fn dump(after_model: &str) -> String {
        format!("{HEADER}{}\n{after_model}", "ab".repeat(64))
    }

    #[test]
    fn a_header_parses_into_its_fields() {
        let text =
            dump("annotator\topennlp.tools.sentdetect.SentenceDetectorME\tsentPosDetect\npoint\t0\t2\t1\t2\t0\tnone\n");
        let h = DumpHeader::parse(&text).unwrap();
        assert_eq!(h.generator, "turbo-goldens-java");
        assert_eq!(h.generator_version, 1);
        assert_eq!(h.opennlp_commit[..3], [0x92, 0xd4, 0x86]);
        assert_eq!(h.opennlp_version, "3.0.0-PRE-SNAPSHOT");
        assert_eq!(h.vendor, "A\tVendor");
        assert_eq!(h.unicode, "15.0");
        assert_eq!(h.input.id, "en/x");
        assert_eq!((h.input.utf8_len, h.input.utf16_len), (9, 8));
        assert_eq!(h.input.sha256[0], 0xda);
        assert_eq!(h.models, vec![ModelRecord { file: "m.bin".into(), sha512: [0xab; 64] }]);
        assert_eq!(h.annotator_class, "opennlp.tools.sentdetect.SentenceDetectorME");
        assert_eq!(h.method, "sentPosDetect");
    }

    #[test]
    fn a_dump_without_points_is_its_header() {
        assert!(DumpHeader::parse(&dump("annotator\ta.B\trun\n")).is_ok());
    }

    #[test]
    fn a_golden_is_not_a_dump() {
        let text = dump("annotator\ta.B\trun\n").replacen("dump 1", "golden 1", 1);
        let e = DumpHeader::parse(&text).unwrap_err();
        assert_eq!(e.line, 1, "{e}");
    }

    #[test]
    fn errors_name_the_line_and_what_was_expected() {
        let e = DumpHeader::parse(&dump("annotator\ta.B\trun\nlayer\tx\n")).unwrap_err();
        assert_eq!(e.line, 9, "{e}");
        assert!(e.to_string().contains("`point`"), "{e}");

        let e = DumpHeader::parse(&dump("annotator\ta.B\n")).unwrap_err();
        assert_eq!(e.to_string(), "line 8: a `annotator` record has 3 fields, this one 2");

        let e = DumpHeader::parse(&dump("annotator\ta.B\trun")).unwrap_err();
        assert!(e.to_string().contains("line end"), "{e}");

        let text = dump("annotator\ta.B\trun\n").replace("\t9\t8\t", "\t09\t8\t");
        let e = DumpHeader::parse(&text).unwrap_err();
        assert_eq!((e.line, e.message.as_str()), (6, "expected a u32, found `09`"));

        let text = dump("annotator\ta.B\trun\n").replace("92d486d0", "92D486D0");
        assert_eq!(DumpHeader::parse(&text).unwrap_err().line, 3);

        let text = format!("{HEADER}{}\nannotator\ta.B\trun\n", "ab".repeat(63));
        assert_eq!(DumpHeader::parse(&text).unwrap_err().line, 7);
    }

    #[test]
    fn str_fields_undo_every_escape() {
        assert_eq!(unescape("a\\tb\\nc\\rd\\\\e").unwrap(), "a\tb\nc\rd\\e");
        assert_eq!(unescape("\\u0000\\u001f").unwrap(), "\u{0}\u{1f}");
        assert_eq!(unescape("x\\ud83d\\ude00y").unwrap(), "x😀y");
        assert!(unescape("x\\ud83dy").is_err());
        assert!(unescape("\\u00G0").is_err());
        assert!(unescape("\\q").is_err());
        assert!(unescape("x\\").is_err());
    }

    #[test]
    fn u32_fields_are_strict() {
        assert_eq!(parse_u32("0"), Some(0));
        assert_eq!(parse_u32("4294967295"), Some(u32::MAX));
        for bad in ["", "01", "+1", "-1", "4294967296", "1 "] {
            assert_eq!(parse_u32(bad), None, "{bad:?}");
        }
    }
}
