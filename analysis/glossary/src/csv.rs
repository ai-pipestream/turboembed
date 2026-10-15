//! RFC 4180 CSV and TSV glossary reader and writer (`CsvGlossaryReader`).

use std::io::{self, Read, Write};

use crate::{GlossaryEntry, GlossaryError, is_blank};

/// Reads [`GlossaryEntry`] records from RFC 4180 delimited text (CSV, TSV, or
/// custom delimiter).
///
/// The first two columns of each non-blank record are interpreted as
/// `(id, term)`. Additional columns are parsed (including quote validation)
/// and ignored. Leading UTF-8 BOM (`U+FEFF`) is stripped, and blank lines
/// outside quotes are skipped without consuming `skip_header`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CsvGlossaryReader {
    delimiter: char,
    skip_header: bool,
}

impl CsvGlossaryReader {
    /// Creates a reader for the given `delimiter` and `skip_header` flag.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::InvalidDelimiter`] if `delimiter` is `'"'`,
    /// `'\n'`, or `'\r'`.
    pub fn new(delimiter: char, skip_header: bool) -> Result<Self, GlossaryError> {
        if delimiter == '"' || delimiter == '\n' || delimiter == '\r' {
            return Err(GlossaryError::InvalidDelimiter(delimiter));
        }
        Ok(Self { delimiter, skip_header })
    }

    /// Creates a comma-separated (`,`) glossary reader.
    pub fn csv(skip_header: bool) -> Self {
        Self { delimiter: ',', skip_header }
    }

    /// Creates a tab-separated (`\t`) glossary reader.
    pub fn tsv(skip_header: bool) -> Self {
        Self { delimiter: '\t', skip_header }
    }

    /// Returns the configured field delimiter.
    pub fn delimiter(&self) -> char {
        self.delimiter
    }

    /// Returns whether the first non-blank record is skipped as a header row.
    pub fn skip_header(&self) -> bool {
        self.skip_header
    }

    /// Reads all glossary entries from `reader` (UTF-8 encoded).
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::Io`] on I/O or UTF-8 decoding errors, or
    /// [`GlossaryError::Format`] if a record has malformed quoting, fewer than
    /// two columns, or a blank `id` or `term`.
    pub fn read<R: Read>(&self, mut reader: R) -> Result<Vec<GlossaryEntry>, GlossaryError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| GlossaryError::Io(io::Error::new(io::ErrorKind::InvalidData, e)))?;
        self.read_str(text)
    }

    /// Parses all glossary entries from a `&str`.
    pub fn read_str(&self, text: &str) -> Result<Vec<GlossaryEntry>, GlossaryError> {
        let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
        let mut chars = text.chars().peekable();
        let mut entries = Vec::new();
        let mut physical_line: usize = 1;
        let mut header_skipped = !self.skip_header;

        while chars.peek().is_some() {
            let record_start_line = physical_line;
            let (fields, quoted_any) = self.read_record(&mut chars, &mut physical_line, record_start_line)?;

            if fields.len() == 1 && !quoted_any && is_blank(&fields[0]) {
                continue;
            }
            if !header_skipped {
                header_skipped = true;
                continue;
            }
            if fields.len() < 2 {
                return Err(GlossaryError::Format(format!(
                    "Glossary CSV line {record_start_line} must have at least 2 columns (id, term), found {}",
                    fields.len()
                )));
            }
            let entry = GlossaryEntry::new(&fields[0], &fields[1]).map_err(|e| {
                GlossaryError::Format(format!("Invalid glossary entry at line {record_start_line}: {e}"))
            })?;
            entries.push(entry);
        }

        Ok(entries)
    }

    fn read_record(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
        physical_line: &mut usize,
        record_start_line: usize,
    ) -> Result<(Vec<String>, bool), GlossaryError> {
        let mut fields = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut after_closing_quote = false;
        let mut quoted_any = false;

        loop {
            let Some(ch) = chars.next() else {
                if in_quotes {
                    return Err(GlossaryError::Format(format!(
                        "Unterminated quoted field starting at line {record_start_line}"
                    )));
                }
                fields.push(current);
                return Ok((fields, quoted_any));
            };

            if in_quotes {
                if ch == '"' {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        current.push('"');
                    } else {
                        in_quotes = false;
                        after_closing_quote = true;
                    }
                } else if ch == '\r' {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    current.push('\n');
                    *physical_line += 1;
                } else if ch == '\n' {
                    current.push('\n');
                    *physical_line += 1;
                } else {
                    current.push(ch);
                }
            } else if after_closing_quote {
                if ch == self.delimiter {
                    fields.push(std::mem::take(&mut current));
                    after_closing_quote = false;
                } else if ch == '\r' {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    *physical_line += 1;
                    fields.push(current);
                    return Ok((fields, quoted_any));
                } else if ch == '\n' {
                    *physical_line += 1;
                    fields.push(current);
                    return Ok((fields, quoted_any));
                } else {
                    return Err(GlossaryError::Format(format!(
                        "Unexpected character after closing quote at line {}",
                        *physical_line
                    )));
                }
            } else if ch == '"' {
                if !current.is_empty() {
                    return Err(GlossaryError::Format(format!(
                        "Unexpected quote in unquoted field at line {}",
                        *physical_line
                    )));
                }
                in_quotes = true;
                quoted_any = true;
            } else if ch == self.delimiter {
                fields.push(std::mem::take(&mut current));
            } else if ch == '\r' {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                *physical_line += 1;
                fields.push(current);
                return Ok((fields, quoted_any));
            } else if ch == '\n' {
                *physical_line += 1;
                fields.push(current);
                return Ok((fields, quoted_any));
            } else {
                current.push(ch);
            }
        }
    }

    /// Writes `entries` in RFC 4180 delimited format using this reader's
    /// delimiter (and an optional `id<delim>term` header when
    /// `self.skip_header()` is true).
    pub fn write<W: Write>(&self, entries: &[GlossaryEntry], mut writer: W) -> Result<(), GlossaryError> {
        if self.skip_header {
            write_field(&mut writer, "id", self.delimiter)?;
            write!(writer, "{}", self.delimiter)?;
            write_field(&mut writer, "term", self.delimiter)?;
            writeln!(writer)?;
        }
        for entry in entries {
            write_field(&mut writer, entry.id(), self.delimiter)?;
            write!(writer, "{}", self.delimiter)?;
            write_field(&mut writer, entry.term(), self.delimiter)?;
            writeln!(writer)?;
        }
        Ok(())
    }
}

fn write_field<W: Write>(writer: &mut W, value: &str, delimiter: char) -> io::Result<()> {
    let needs_quotes = value.contains(delimiter)
        || value.contains('"')
        || value.contains('\n')
        || value.contains('\r')
        || is_blank(value);
    if needs_quotes {
        write!(writer, "\"")?;
        for ch in value.chars() {
            if ch == '"' {
                write!(writer, "\"\"")?;
            } else {
                write!(writer, "{ch}")?;
            }
        }
        write!(writer, "\"")
    } else {
        write!(writer, "{value}")
    }
}
