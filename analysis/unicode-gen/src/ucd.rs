//! Reading the Unicode Character Database text files.
//!
//! The UCD files share one format: one record per line, fields separated
//! by `;`, a `#` starting a comment, code points written in hexadecimal and
//! ranges written `XXXX..YYYY`. Anything that does not fit the format stops
//! the generator with the file name and line number, so a damaged or
//! changed input file cannot produce plausible but wrong tables.

use std::fs;
use std::ops::RangeInclusive;
use std::path::Path;

/// The number of Unicode code points, U+0000 to U+10FFFF.
pub const CODE_POINTS: usize = 0x11_0000;

/// One UCD file read into memory, kept with its name for error messages.
pub struct UcdFile {
    name: String,
    text: String,
}

/// A data line of a UCD file: its fields with surrounding spaces removed.
pub struct Record<'a> {
    file: &'a UcdFile,
    line: usize,
    fields: Vec<&'a str>,
}

impl UcdFile {
    /// Reads `name` from `dir`. A missing or unreadable file stops the generator.
    pub fn read(dir: &Path, name: &str) -> UcdFile {
        let path = dir.join(name);
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let text = text.strip_prefix('\u{FEFF}').map(str::to_owned).unwrap_or(text);
        UcdFile { name: name.to_owned(), text }
    }

    /// The file name, as given to [`UcdFile::read`].
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Every line with its 1-based line number, comments included.
    pub fn lines(&self) -> impl Iterator<Item = (usize, &str)> {
        self.text.lines().enumerate().map(|(i, line)| (i + 1, line))
    }

    /// The data lines: lines that hold something besides a comment.
    pub fn records(&self) -> impl Iterator<Item = Record<'_>> {
        self.lines().filter_map(move |(line, text)| {
            let data = text.split_once('#').map_or(text, |(data, _)| data);
            if data.trim().is_empty() {
                return None;
            }
            let fields = data.split(';').map(str::trim).collect();
            Some(Record { file: self, line, fields })
        })
    }

    /// Stops the generator with a message naming this file and `line`.
    pub fn fail(&self, line: usize, message: &str) -> ! {
        panic!("{}:{line}: {message}", self.name)
    }

    /// The Unicode version in the first line of a file whose header has the
    /// form `# Name-18.0.0.txt`.
    pub fn header_version(&self) -> (u8, u8, u8) {
        let first = self.text.lines().next().unwrap_or("");
        let stem = self.name.trim_end_matches(".txt");
        let version = first
            .strip_prefix("# ")
            .and_then(|rest| rest.strip_prefix(stem))
            .and_then(|rest| rest.strip_prefix('-'))
            .and_then(|rest| rest.strip_suffix(".txt"))
            .unwrap_or_else(|| self.fail(1, "expected a header of the form `# Name-X.Y.Z.txt`"));
        let parts: Vec<u8> = version
            .split('.')
            .map(|part| part.parse().unwrap_or_else(|_| self.fail(1, "malformed version number")))
            .collect();
        match parts[..] {
            [major, minor, update] => (major, minor, update),
            _ => self.fail(1, "a version has three parts"),
        }
    }
}

impl<'a> Record<'a> {
    /// The number of fields, counting an empty last field after a trailing `;`.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Field `i`, which must exist.
    pub fn field(&self, i: usize) -> &'a str {
        match self.fields.get(i) {
            Some(field) => field,
            None => self.fail(&format!("missing field {i}")),
        }
    }

    /// Field `i` read as one code point.
    pub fn code_point(&self, i: usize) -> u32 {
        self.parse_code_point(self.field(i))
    }

    /// Field `i` read as a code point or a `XXXX..YYYY` range.
    pub fn range(&self, i: usize) -> RangeInclusive<u32> {
        let field = self.field(i);
        match field.split_once("..") {
            Some((first, last)) => {
                let (first, last) = (self.parse_code_point(first), self.parse_code_point(last));
                if first > last {
                    self.fail(&format!("range {field} runs backwards"));
                }
                first..=last
            }
            None => {
                let cp = self.parse_code_point(field);
                cp..=cp
            }
        }
    }

    /// Field `i` read as a space-separated sequence of code points; an
    /// empty field gives an empty sequence.
    pub fn code_points(&self, i: usize) -> Vec<u32> {
        self.field(i).split_ascii_whitespace().map(|cp| self.parse_code_point(cp)).collect()
    }

    /// Stops the generator with a message naming the file and line of this record.
    pub fn fail(&self, message: &str) -> ! {
        self.file.fail(self.line, message)
    }

    fn parse_code_point(&self, text: &str) -> u32 {
        let valid = (4..=6).contains(&text.len()) && text.bytes().all(|b| b.is_ascii_hexdigit());
        match u32::from_str_radix(text, 16) {
            Ok(cp) if valid && cp < CODE_POINTS as u32 => cp,
            _ => self.fail(&format!("`{text}` is not a code point")),
        }
    }
}

/// Converts a code point to a `char`; the generator only calls this for
/// code points that are not surrogates.
pub fn to_char(cp: u32) -> char {
    char::from_u32(cp).unwrap_or_else(|| panic!("U+{cp:04X} is a surrogate and has no char"))
}

/// Converts a sequence of code points to a string.
pub fn to_string(cps: &[u32]) -> String {
    cps.iter().map(|&cp| to_char(cp)).collect()
}

#[cfg(test)]
mod tests {
    use super::UcdFile;

    fn file(text: &str) -> UcdFile {
        UcdFile { name: "Test.txt".to_owned(), text: text.to_owned() }
    }

    #[test]
    fn records_skip_comments_and_split_fields() {
        let f = file("# header\n\n0041..005A ; Upper # comment\n00DF; 00DF; 0053 0073; # sharp s\n");
        let records: Vec<_> = f.records().collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].range(0), 0x41..=0x5A);
        assert_eq!(records[0].field(1), "Upper");
        assert_eq!(records[1].field_count(), 4);
        assert_eq!(records[1].code_points(2), vec![0x53, 0x73]);
        assert_eq!(records[1].code_points(3), Vec::<u32>::new());
    }

    #[test]
    #[should_panic(expected = "Test.txt:1: `12G4` is not a code point")]
    fn a_malformed_code_point_names_the_file_and_line() {
        file("12G4 ; X\n").records().next().unwrap().code_point(0);
    }

    #[test]
    #[should_panic(expected = "Test.txt:2: range 0042..0041 runs backwards")]
    fn a_backwards_range_is_rejected() {
        file("0041 ; X\n0042..0041 ; X\n").records().nth(1).unwrap().range(0);
    }

    #[test]
    #[should_panic(expected = "is not a code point")]
    fn a_code_point_beyond_the_code_space_is_rejected() {
        file("110000 ; X\n").records().next().unwrap().code_point(0);
    }

    #[test]
    fn header_version() {
        assert_eq!(file("# Test-18.0.0.txt\n").header_version(), (18, 0, 0));
    }
}
