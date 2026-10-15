//! `PlaceProfiles.readTable`: the tab separated profile table, read line by
//! line as OpenNLP reads it.

use std::collections::HashSet;

use super::error::ProfileError;
use crate::text::{BYTE_ORDER_MARK, Lines, trim};

/// A byte range of the text a table keeps its names in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) start: u32,
    pub(crate) len: u32,
}

impl Span {
    pub(crate) fn get(self, text: &str) -> &str {
        &text[self.start as usize..(self.start + self.len) as usize]
    }
}

/// Where a place's row is in the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) id: Span,
    /// The one-based line number.
    pub(crate) line: u32,
    /// The byte offset of the line in the table.
    pub(crate) offset: usize,
}

/// A table as read, before standardizing.
pub(crate) struct RawTable {
    /// The metric names, then the place ids, without padding.
    pub(crate) names: String,
    pub(crate) metrics: Vec<Span>,
    pub(crate) rows: Vec<Row>,
    /// The values, one row of `metrics.len()` per place, in table order.
    pub(crate) values: Vec<f64>,
}

/// Whether OpenNLP skips the line: blank or, once trimmed, a comment.
fn is_ignored(line: &str) -> bool {
    let t = trim(line);
    t.is_empty() || t.starts_with('#')
}

/// The lines of the decodable start of `bytes`, and the line of the first
/// byte that is not UTF-8, if there is one. Java's reader hands out every
/// line that ends before that byte and fails on the next read, so the line
/// running into it is never seen.
struct Reader<'t> {
    lines: Lines<'t>,
    /// The length of the valid prefix, when the bytes stop being UTF-8.
    bad: Option<usize>,
    text: &'t str,
}

impl<'t> Reader<'t> {
    fn new(bytes: &'t [u8]) -> Reader<'t> {
        let (text, bad) = match std::str::from_utf8(bytes) {
            Ok(text) => (text, None),
            Err(e) => {
                let valid = e.valid_up_to();
                (std::str::from_utf8(&bytes[..valid]).expect("the valid prefix"), Some(valid))
            }
        };
        Reader { lines: Lines::new(text), bad, text }
    }

    /// The next line as `(number, offset, line)`, `Ok(None)` at the end, or
    /// the UTF-8 failure where Java's reader would report it.
    fn next(&mut self) -> Result<Option<(usize, usize, &'t str)>, ProfileError> {
        let next = self.lines.next();
        let Some(bad) = self.bad else { return Ok(next) };
        match next {
            Some((number, offset, line)) if offset + line.len() < self.text.len() => Ok(Some((number, offset, line))),
            Some((number, ..)) => Err(ProfileError::Utf8 { line: number }),
            None => Err(ProfileError::Utf8 { line: crate::load::line_of(self.text.as_bytes(), bad) }),
        }
    }
}

/// Converts a count of bytes to the 32-bit offsets the name text uses.
fn offset(n: usize) -> u32 {
    u32::try_from(n).expect("profile names longer than 4 GiB")
}

/// Reads a table: the header, then one row per place.
pub(crate) fn read(bytes: &[u8]) -> Result<RawTable, ProfileError> {
    let mut reader = Reader::new(bytes);
    let (header_line, header) = loop {
        let Some((number, _, mut line)) = reader.next()? else { return Err(ProfileError::NoHeader) };
        if number == 1 {
            line = line.strip_prefix(BYTE_ORDER_MARK).unwrap_or(line);
        }
        if !is_ignored(line) {
            break (number, line);
        }
    };
    let columns: Vec<&str> = header.split('\t').collect();
    if columns.len() < 2 || trim(columns[0]) != "id" {
        return Err(ProfileError::Header { line: header_line });
    }
    let width = columns.len() - 1;
    let mut names = String::new();
    let mut metrics = Vec::with_capacity(width);
    let mut seen = HashSet::with_capacity(width);
    for (m, column) in columns[1..].iter().enumerate() {
        let name = trim(column);
        let column = m + 2;
        if name.is_empty() {
            return Err(ProfileError::EmptyMetricName { line: header_line, column });
        }
        if !seen.insert(name) {
            return Err(ProfileError::DuplicateMetricName { line: header_line, column, name: name.to_owned() });
        }
        metrics.push(Span { start: offset(names.len()), len: offset(name.len()) });
        names.push_str(name);
    }
    let metric_names: Vec<&str> = columns[1..].iter().map(|c| trim(c)).collect();

    let mut rows = Vec::new();
    let mut values = Vec::new();
    let mut ids: HashSet<&str> = HashSet::new();
    let mut fields: Vec<&str> = Vec::with_capacity(columns.len());
    while let Some((number, at, line)) = reader.next()? {
        if is_ignored(line) {
            continue;
        }
        fields.clear();
        fields.extend(line.split('\t'));
        if fields.len() != columns.len() {
            return Err(ProfileError::FieldCount { line: number, found: fields.len(), expected: columns.len() });
        }
        let id = trim(fields[0]);
        if id.is_empty() {
            return Err(ProfileError::EmptyId { line: number });
        }
        if !ids.insert(id) {
            return Err(ProfileError::DuplicateId { line: number, id: id.to_owned() });
        }
        for (m, field) in fields[1..].iter().enumerate() {
            let value = trim(field);
            if value.is_empty() {
                return Err(ProfileError::EmptyValue {
                    line: number,
                    column: m + 2,
                    metric: metric_names[m].to_owned(),
                });
            }
            values.push(metric_value(value, number)?);
        }
        rows.push(Row {
            id: Span { start: offset(names.len()), len: offset(id.len()) },
            line: u32::try_from(number).expect("a table of fewer than 2^32 lines"),
            offset: at,
        });
        names.push_str(id);
    }
    if rows.is_empty() {
        return Err(ProfileError::NoPlaces);
    }
    Ok(RawTable { names, metrics, rows, values })
}

/// `PlaceProfiles.metricValue`: a finite double. The value is already
/// trimmed of Unicode `White_Space`. A last character `f`, `F`, `d` or `D`
/// or an `x` or `X` anywhere is refused before parsing; what remains is
/// read by `Double.parseDouble`, which trims the characters up to U+0020
/// itself. So a suffix followed by a control character such as U+0001 is
/// accepted, as it is by OpenNLP.
fn metric_value(value: &str, line: usize) -> Result<f64, ProfileError> {
    let malformed = || ProfileError::MalformedValue { line, value: value.to_owned() };
    if value.ends_with(['f', 'F', 'd', 'D']) || value.contains(['x', 'X']) {
        return Err(malformed());
    }
    let parsed = turbo_jvm::text::parse_double(value).map_err(|_| malformed())?;
    if !parsed.is_finite() {
        return Err(ProfileError::NonFiniteValue { line, value: value.to_owned() });
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error(text: &[u8]) -> String {
        read(text).err().expect("the table is refused").to_string()
    }

    #[test]
    fn a_table_reads_its_names_rows_and_values() {
        let t = read("# made up\n\nid\tarea\t pop \nb\t1\t2\r\n\u{A0}a\u{3000}\t3e1\t-4.5\n".as_bytes()).unwrap();
        let metrics: Vec<&str> = t.metrics.iter().map(|s| s.get(&t.names)).collect();
        assert_eq!(metrics, ["area", "pop"]);
        let ids: Vec<&str> = t.rows.iter().map(|r| r.id.get(&t.names)).collect();
        assert_eq!(ids, ["b", "a"]);
        assert_eq!(t.rows[0].line, 4);
        assert_eq!(t.rows[1].offset, 32);
        assert_eq!(t.values, [1.0, 2.0, 30.0, -4.5]);
    }

    #[test]
    fn the_byte_order_mark_is_dropped_only_from_the_first_line() {
        assert!(read("\u{FEFF}id\tm\na\t1\n".as_bytes()).is_ok());
        assert!(read("\u{FEFF}\nid\tm\na\t1\n".as_bytes()).is_ok());
        assert_eq!(error("\n\u{FEFF}id\tm\na\t1\n".as_bytes()), "the header must be: id, then at least one metric");
    }

    #[test]
    fn errors_carry_opennlp_messages() {
        assert_eq!(error(b""), "the profile table has no header");
        assert_eq!(error(b" # x\n\n"), "the profile table has no header");
        assert_eq!(error(b"id\n"), "the header must be: id, then at least one metric");
        assert_eq!(error(b"ID\tm\n"), "the header must be: id, then at least one metric");
        assert_eq!(error(b"id\tm\t \n"), "empty metric name in header column 3");
        assert_eq!(error(b"id\tm\t m\n"), "duplicate metric name in header column 3: m");
        assert_eq!(error(b"id\tm\n"), "the profile table lists no places");
        assert_eq!(error(b"id\tm\na\t1\t2\n"), "row 2 has 3 fields, expected 2");
        assert_eq!(error(b"id\tm\n \t1\n"), "empty id in row 2");
        assert_eq!(error(b"id\tm\na\t1\n\na \t2\n"), "duplicate id in row 4: a");
        assert_eq!(error(b"id\tm\tn\na\t1\t\n"), "empty value in row 2, column 3 (n)");
        assert_eq!(error(b"id\tm\na\t0x10\n"), "malformed value in row 2: 0x10");
        assert_eq!(error(b"id\tm\na\t1.5f\n"), "malformed value in row 2: 1.5f");
        assert_eq!(error(b"id\tm\na\t1,5\n"), "malformed value in row 2: 1,5");
        assert_eq!(error(b"id\tm\na\tNaN\n"), "non-finite value in row 2: NaN");
        assert_eq!(error(b"id\tm\na\t1e400\n"), "non-finite value in row 2: 1e400");
    }

    #[test]
    fn a_suffix_before_a_control_character_is_read() {
        let t = read("id\tm\na\t1.5d\u{1}\nb\t\u{1F}+.5e1\n".as_bytes()).unwrap();
        assert_eq!(t.values, [1.5, 5.0]);
    }

    #[test]
    fn bad_utf8_is_reported_where_the_reader_reaches_it() {
        // A bad row before the bad byte is reported first.
        assert_eq!(error(b"id\tm\na\tz\nb\t\xFF\n"), "malformed value in row 2: z");
        match read(b"id\tm\na\t1\nb\t\xFF\n") {
            Err(ProfileError::Utf8 { line }) => assert_eq!(line, 3),
            other => panic!("{:?}", other.err()),
        }
        // A line ended by a carriage return just before the byte is read.
        assert_eq!(error(b"id\tm\na\tz\r\xFF"), "malformed value in row 2: z");
        // The byte in the line itself stops the read before the line is seen.
        assert_eq!(error(b"id\tm\na\tz\xFF"), "profile table is not valid UTF-8");
        assert_eq!(error(b"\xFF"), "profile table is not valid UTF-8");
    }
}
