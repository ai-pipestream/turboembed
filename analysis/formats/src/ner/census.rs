//! The 1990 US Census name list reader.

use super::NerError;
use crate::error::FormatError;
use crate::source::{Field, Lines};

/// `s` with its first character upper-cased and the rest lower-cased, as
/// Java does it by UTF-16 code units (`s.substring(0, 1).toUpperCase()`
/// and `s.substring(1).toLowerCase()`): a first character outside the
/// Basic Multilingual Plane is two units, which Java cases apart and so
/// leaves as they are.
fn capitalize(s: &str, out: &mut String) {
    let Some(first) = s.chars().next() else { return };
    if first.len_utf16() == 1 {
        out.extend(first.to_uppercase());
    } else {
        out.push(first);
    }
    out.push_str(&s[first.len_utf8()..].to_lowercase());
}

/// The name a census line's first field becomes: capitalized, and a name
/// of more than two UTF-16 code units that starts with `MC` written
/// `Mc` and capitalized after it (`MCDONALD` is `McDonald`).
fn recase(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    match name.strip_prefix("MC") {
        Some(rest) if !rest.is_empty() => {
            out.push_str("Mc");
            capitalize(rest, &mut out);
        }
        _ => capitalize(name, &mut out),
    }
    out
}

/// Reads the names of a 1990 US Census name list
/// (`NameFinderCensus90NameStream`), which OpenNLP's
/// `CensusDictionaryCreatorTool` makes a name dictionary of. A line is a
/// name in capitals, a space and the name's frequencies; the name is the
/// text before the first space, recased (`SMITH` is `Smith`). Each name
/// is a [`Field`] with a text of its own and the span of the name in the
/// line.
///
/// ```text
/// SMITH          1.006  1.006      1
/// MCDONALD       0.140 32.149    101
/// ```
///
/// The list ends at the first empty line or line without a space, where
/// the stream returns null and the tool stops reading. A line that starts
/// with a space gives [`NerError::EmptyCensusName`], and the next read
/// goes on with the next line.
#[derive(Clone, Debug)]
pub struct CensusReader<'a> {
    lines: Lines<'a>,
    done: bool,
}

impl<'a> CensusReader<'a> {
    /// A reader of `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str) -> Result<CensusReader<'a>, FormatError> {
        Ok(CensusReader { lines: Lines::new(source)?, done: false })
    }
}

impl<'a> Iterator for CensusReader<'a> {
    type Item = Result<Field<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let line = self.lines.next().filter(|l| !l.text().is_empty());
        let Some((line, pos)) = line.and_then(|l| l.text().find(' ').map(|p| (l, p))) else {
            self.done = true;
            return None;
        };
        if pos == 0 {
            return Some(Err(NerError::EmptyCensusName { line: line.number }.into()));
        }
        let name = line.piece.slice(0, pos);
        Some(Ok(Field::made(recase(name.text), name.span)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(source: &str) -> Vec<Result<String, String>> {
        CensusReader::new(source)
            .unwrap()
            .map(|r| r.map(|f| f.as_str().to_owned()).map_err(|e| e.to_string()))
            .collect()
    }

    #[test]
    fn names_are_recased() {
        assert_eq!(
            read("SMITH 1.006\nMCDONALD 0.140\nMC 0.1\nMCX 0.1\nMcKAY 0.1\n"),
            [Ok("Smith"), Ok("McDonald"), Ok("Mc"), Ok("McX"), Ok("Mckay")].map(|r| r.map(str::to_owned))
        );
    }

    #[test]
    fn casing_follows_utf16_units() {
        assert_eq!(recase("ÉLAN"), "Élan");
        assert_eq!(recase("ßMITH"), "SSmith");
        assert_eq!(recase("𐐨𐐀B"), "𐐨𐐨b");
        assert_eq!(recase("MC𐐀𐐀"), "Mc𐐀𐐨");
    }

    #[test]
    fn the_list_ends_at_an_empty_line_or_one_without_a_space() {
        assert_eq!(read("A 1\n\nB 2\n"), [Ok("A".to_owned())]);
        assert_eq!(read("A 1\nB\nC 2\n"), [Ok("A".to_owned())]);
    }

    #[test]
    fn a_leading_space_is_an_error() {
        assert_eq!(
            read(" A 1\nB 2\n"),
            [Err("Range [0, 1) out of bounds for length 0".to_owned()), Ok("B".to_owned())]
        );
    }

    #[test]
    fn a_name_keeps_its_span() {
        let text = "x 1\nJONES 2\n";
        let names: Vec<_> = CensusReader::new(text).unwrap().map(Result::unwrap).collect();
        assert_eq!(&text[names[1].span.range()], "JONES");
        assert_eq!(names[1].as_str(), "Jones");
    }
}
