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

/// The lines of `bytes` as OpenNLP's reader hands them out when the bytes
/// are not all UTF-8. Java decodes the table a block of characters at a
/// time and fails as soon as its decoder reaches the bad bytes, before
/// any line of that block is returned. So only the lines whose ends lie in
/// earlier blocks are seen, and a table of less than 8 KiB fails before
/// its first line, unless it ends inside a character: then the decoder
/// waits for the rest, and the failure comes on the read after the last
/// whole character.
struct Reader<'t> {
    lines: Lines<'t>,
    /// The byte offset from which no line end is seen, and the one-based
    /// line of the first bad byte, when the bytes are not all UTF-8.
    bad: Option<(usize, usize)>,
}

impl<'t> Reader<'t> {
    fn new(bytes: &'t [u8]) -> Reader<'t> {
        let (text, bad) = match std::str::from_utf8(bytes) {
            Ok(text) => (text, None),
            Err(e) => {
                let valid = e.valid_up_to();
                let text = std::str::from_utf8(&bytes[..valid]).expect("the valid prefix");
                let line = crate::load::line_of(bytes, valid);
                (text, Some((decoded_before_failure(text, &bytes[valid..]), line)))
            }
        };
        Reader { lines: Lines::new(text), bad }
    }

    /// The next line as `(number, offset, line)`, `Ok(None)` at the end, or
    /// the UTF-8 failure where Java's reader would report it.
    fn next(&mut self) -> Result<Option<(usize, usize, &'t str)>, ProfileError> {
        let next = self.lines.next();
        let Some((seen, line)) = self.bad else { return Ok(next) };
        match next {
            // The line is returned once its end, at offset + len, is decoded.
            Some((number, offset, text)) if offset + text.len() < seen => Ok(Some((number, offset, text))),
            _ => Err(ProfileError::Utf8 { line }),
        }
    }
}

/// The UTF-16 units `BufferedReader` asks its decoder for at a time.
const CHAR_BLOCK: usize = 8192;
/// The bytes `StreamDecoder` reads from the stream at a time.
const BYTE_BLOCK: usize = 8192;

/// What JDK 21's UTF-8 decoder (`decodeArrayLoop` of `sun.nio.cs.UTF_8`)
/// makes of a bad sequence at the start of `bytes`, the part of it read
/// so far, with `room` UTF-16 units left in the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtBadBytes {
    /// It reports the bytes, failing the read.
    Malformed,
    /// It ends the block without looking further.
    Full,
    /// It waits for more bytes.
    Short,
}

fn at_bad_bytes(bytes: &[u8], room: usize) -> AtBadBytes {
    use AtBadBytes::{Full, Malformed, Short};
    let continuation = |b: u8| b & 0xC0 == 0x80;
    let Some(&b1) = bytes.first() else { return Short };
    let avail = bytes.len();
    match b1 {
        0xC2..=0xDF if avail < 2 => Short,
        0xC2..=0xDF if room < 1 => Full,
        0xE0..=0xEF if avail < 3 || room < 1 => {
            let b2 = bytes.get(1).copied();
            match b2 {
                Some(b2) if (b1 == 0xE0 && b2 & 0xE0 == 0x80) || !continuation(b2) => Malformed,
                _ if avail < 3 => Short,
                _ => Full,
            }
        }
        0xF0..=0xF7 if avail < 4 || room < 2 => {
            let malformed_2 = |b2: u8| {
                (b1 == 0xF0 && !(0x90..=0xBF).contains(&b2)) || (b1 == 0xF4 && b2 & 0xF0 != 0x80) || !continuation(b2)
            };
            if b1 > 0xF4
                || bytes.get(1).is_some_and(|&b2| malformed_2(b2))
                || bytes.get(2).is_some_and(|&b3| !continuation(b3))
            {
                Malformed
            } else if avail < 4 {
                Short
            } else {
                Full
            }
        }
        // A lead byte that cannot start a character, or a whole sequence
        // the decoder has room for: either way it is refused.
        _ => Malformed,
    }
}

/// The byte offset where Java's reader starts the block of characters
/// whose read fails, for a table that is `text` up to the first bad byte
/// and `bad` from there. It follows `BufferedReader`, which reads 8192
/// UTF-16 units at a time, `StreamDecoder`, which reads 8192 bytes at a
/// time and reads again while its block is not full, and the decoder of
/// [`at_bad_bytes`]; a stream whose reads return all the bytes asked for,
/// as `ByteArrayInputStream` does, is assumed.
fn decoded_before_failure(text: &str, bad: &[u8]) -> usize {
    let total = text.len() + bad.len();
    let mut read = 0;
    let mut block = 0;
    let mut units = 0;
    let mut chars = text.char_indices().peekable();
    loop {
        let (at, need, width) = match chars.peek() {
            Some(&(at, c)) => (at, c.len_utf8(), c.len_utf16()),
            None => (text.len(), 0, 0),
        };
        let room = CHAR_BLOCK - units;
        let short = if at == text.len() {
            match at_bad_bytes(&bad[..read - at], room) {
                AtBadBytes::Malformed => return block,
                AtBadBytes::Full => false,
                // Nothing more to read: the decoder fails at the end of
                // the stream, on the read after this one if this block
                // holds characters.
                AtBadBytes::Short if read == total => return at,
                AtBadBytes::Short => true,
            }
        } else if read - at < need {
            true
        } else if room >= width {
            chars.next();
            units += width;
            continue;
        } else {
            false
        };
        if short && room > 0 {
            read = (at + BYTE_BLOCK).min(total);
        } else {
            block = at;
            units = 0;
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
        // Java decodes the first 8192 characters before returning a line,
        // so a short table with a bad byte fails before its bad row.
        match read(b"id\tm\na\tz\nb\t\xFF\n") {
            Err(ProfileError::Utf8 { line }) => assert_eq!(line, 3),
            other => panic!("{:?}", other.err()),
        }
        assert_eq!(error(b"id\tm\na\tz\r\xFF"), "profile table is not valid UTF-8");
        assert_eq!(error(b"\xFF"), "profile table is not valid UTF-8");
        // A table that ends inside a character is decoded up to it first.
        assert_eq!(error(b"id\tm\na\tz\n\xE2\x82"), "malformed value in row 2: z");
        assert_eq!(error(b"id\tm\na\tz\n\xED\xA0"), "malformed value in row 2: z");
        assert_eq!(error(b"id\tm\na\tz\xE2"), "profile table is not valid UTF-8");
        // The bad byte after the first block: the rows of that block are read.
        let mut long = b"id\tm\na\tz\n".to_vec();
        for _ in 0..90 {
            long.extend_from_slice(b"#");
            long.extend_from_slice(&[b'y'; 100]);
            long.extend_from_slice(b"\n");
        }
        long.extend_from_slice(b"\xFF\n");
        assert_eq!(error(&long), "malformed value in row 2: z");
    }

    /// The table `id m / a z / # ...` padded with a comment to exactly
    /// `units` UTF-16 units, with an `é` in the comment when `wide`, then
    /// `tail`.
    fn padded(units: usize, wide: bool, tail: &[u8]) -> Vec<u8> {
        let mut t = b"id\tm\na\tz\n#".to_vec();
        if wide {
            t.extend_from_slice("\u{e9}".as_bytes());
        }
        t.resize(units - 1 + usize::from(wide), b'x');
        t.push(b'\n');
        t.extend_from_slice(tail);
        t
    }

    #[test]
    fn bad_bytes_at_the_end_of_a_block_follow_the_decoder() {
        // Right after 8192 units of ASCII the decoder has not read the bad
        // byte when the block is full, so the block's rows are read.
        assert_eq!(error(&padded(8192, false, b"\x80\n")), "malformed value in row 2: z");
        // With a two-byte character in the block it has, and fails at once.
        assert_eq!(error(&padded(8192, true, b"\x80\n")), "profile table is not valid UTF-8");
        // One unit left and a four-byte sequence whose last byte is bad: the
        // decoder needs two units, ends the block, and fails in the next.
        assert_eq!(error(&padded(8191, false, b"\xF0\x9F\x98\x41")), "malformed value in row 2: z");
        assert_eq!(error(&padded(8190, false, b"\xF0\x9F\x98\x41")), "profile table is not valid UTF-8");
    }
}
