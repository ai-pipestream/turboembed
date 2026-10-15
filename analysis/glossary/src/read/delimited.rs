//! Comma and tab separated term lists, OpenNLP's `CsvGlossaryReader`.

use turbo_analysis::Span;
use turbo_glossary_layers::Glossary;

use super::{GlossaryFile, decode};
use crate::error::ReadError;

const QUOTE: char = '"';

/// The character between fields: not the quote and not a line break.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Delimiter(char);

impl Delimiter {
    pub const COMMA: Delimiter = Delimiter(',');
    pub const TAB: Delimiter = Delimiter('\t');

    /// `c` as a delimiter, or `None` for the quote, a line feed or a
    /// carriage return.
    pub const fn new(c: char) -> Option<Delimiter> {
        match c {
            QUOTE | '\n' | '\r' => None,
            _ => Some(Delimiter(c)),
        }
    }

    pub const fn char(self) -> char {
        self.0
    }
}

/// What the first record of a file is.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum FirstRecord {
    /// An entry like every other.
    #[default]
    Data,
    /// A header, dropped. Only its quoting is checked.
    Header,
}

/// Reads a delimited term list: each record an identifier, a term and any
/// further fields, which are checked for quoting and otherwise ignored.
///
/// Quoting follows RFC 4180: a quoted field may hold delimiters, doubled
/// quotes and line breaks; an unquoted field holds no quote; only a
/// delimiter, a line end or the end of the input may follow a closing
/// quote. Records end at a line feed, a carriage return or the pair.
/// Lines that are empty are skipped (a line of spaces is a record of one
/// field, and refused), and a byte order mark at the start is dropped.
/// Fields are taken as they are, spaces included.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DelimitedReader {
    delimiter: Delimiter,
    first: FirstRecord,
}

impl DelimitedReader {
    pub const fn new(delimiter: Delimiter, first: FirstRecord) -> DelimitedReader {
        DelimitedReader { delimiter, first }
    }

    /// Comma separated, every record an entry: OpenNLP's default.
    pub const fn csv() -> DelimitedReader {
        DelimitedReader::new(Delimiter::COMMA, FirstRecord::Data)
    }

    /// Tab separated.
    pub const fn tsv(first: FirstRecord) -> DelimitedReader {
        DelimitedReader::new(Delimiter::TAB, first)
    }

    /// The entries of `bytes` in file order, each term with its span in
    /// `bytes`: the field's bytes, without the quotes around a quoted one.
    pub fn read(&self, bytes: &[u8]) -> Result<GlossaryFile, ReadError> {
        let (text, base) = decode(bytes)?;
        let mut parser = Parser {
            delimiter: self.delimiter.0,
            header_pending: self.first == FirstRecord::Header,
            glossary: Glossary::new(),
            term_at: Vec::new(),
            fields: Vec::new(),
            buf: String::new(),
            field_start: 0,
            quoted: None,
            base,
        };
        parser.run(text)?;
        Ok(GlossaryFile { glossary: parser.glossary, term_at: parser.term_at })
    }
}

/// One record's fields: where each one's value lies in the buffer, and
/// its span in the input.
struct Parser {
    delimiter: char,
    header_pending: bool,
    glossary: Glossary,
    term_at: Vec<Span>,
    fields: Vec<(usize, usize, Span)>,
    buf: String,
    /// Where the field being read began in the text, and the content of
    /// its quotes once they have closed.
    field_start: usize,
    quoted: Option<Span>,
    /// The bytes before the text (a byte order mark), added to every span.
    base: u32,
}

impl Parser {
    fn run(&mut self, text: &str) -> Result<(), ReadError> {
        let mut line = 1u32;
        let mut record_line = 1u32;
        let mut previous_cr = false;
        let mut in_quotes = false;
        let mut quote_closed = false;
        let mut has_content = false;
        let mut field_from = self.buf.len();
        let mut quote_open = 0usize;
        let mut chars = text.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            if c == '\r' || (c == '\n' && !previous_cr) {
                line += 1;
            }
            previous_cr = c == '\r';
            if in_quotes {
                if c == QUOTE {
                    if chars.peek().is_some_and(|&(_, n)| n == QUOTE) {
                        chars.next();
                        self.buf.push(QUOTE);
                    } else {
                        in_quotes = false;
                        quote_closed = true;
                        self.quoted = Some(self.span(quote_open, i));
                    }
                } else {
                    self.buf.push(c);
                }
            } else if quote_closed && c != self.delimiter && c != '\n' && c != '\r' {
                return Err(ReadError::AfterClosingQuote { line });
            } else if c == QUOTE {
                if self.buf.len() > field_from {
                    return Err(ReadError::QuoteInUnquotedField { line });
                }
                in_quotes = true;
                has_content = true;
                quote_open = i + 1;
            } else if c == self.delimiter {
                self.end_field(field_from, i);
                field_from = self.buf.len();
                self.field_start = i + c.len_utf8();
                quote_closed = false;
                has_content = true;
            } else if c == '\n' || c == '\r' {
                if has_content || self.buf.len() > field_from {
                    self.end_field(field_from, i);
                    self.end_record(record_line)?;
                }
                field_from = self.buf.len();
                self.field_start = i + 1;
                has_content = false;
                quote_closed = false;
                record_line = line;
            } else {
                self.buf.push(c);
            }
        }
        if in_quotes {
            return Err(ReadError::UnclosedQuote { line: record_line });
        }
        if has_content || self.buf.len() > field_from {
            self.end_field(field_from, text.len());
            self.end_record(record_line)?;
        }
        Ok(())
    }

    fn span(&self, start: usize, end: usize) -> Span {
        Span::new(self.base + start as u32, self.base + end as u32)
    }

    /// Ends the field that began at `from` in the buffer and ends at byte
    /// `at` of the text.
    fn end_field(&mut self, from: usize, at: usize) {
        let span = self.quoted.take().unwrap_or_else(|| self.span(self.field_start, at));
        self.fields.push((from, self.buf.len(), span));
    }

    fn end_record(&mut self, line: u32) -> Result<(), ReadError> {
        let result = if self.header_pending {
            self.header_pending = false;
            Ok(())
        } else if self.fields.len() < 2 {
            Err(ReadError::MissingColumns { line, found: self.fields.len() })
        } else {
            let (id, term) = (self.fields[0], self.fields[1]);
            match self.glossary.push(&self.buf[id.0..id.1], &self.buf[term.0..term.1]) {
                Ok(_) => {
                    self.term_at.push(term.2);
                    Ok(())
                }
                Err(error) => Err(ReadError::Entry { line, error }),
            }
        };
        self.fields.clear();
        self.buf.clear();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_glossary_layers::{EntryError, EntryIx};

    fn entries(f: &GlossaryFile) -> Vec<(String, String)> {
        f.glossary.iter().map(|(_, e)| (e.id.to_string(), e.term.to_string())).collect()
    }

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn quoting_line_ends_and_extra_columns() {
        let input = "\u{feff}c1,hot dog,x\r\n\r\nc2,\"New, \"\"York\"\"\nCity\"\rc3,naïve,,\n";
        let f = DelimitedReader::csv().read(input.as_bytes()).unwrap();
        assert_eq!(entries(&f), pairs(&[("c1", "hot dog"), ("c2", "New, \"York\"\nCity"), ("c3", "naïve")]));
        let spans: Vec<&str> = f.term_at.iter().map(|s| &input[s.range()]).collect();
        assert_eq!(spans, ["hot dog", "New, \"\"York\"\"\nCity", "naïve"]);
        assert_eq!(f.glossary.get(EntryIx(0)).id, "c1");
    }

    #[test]
    fn header_and_tabs() {
        let f = DelimitedReader::tsv(FirstRecord::Header).read(b"id\tterm\nA\ta b\n").unwrap();
        assert_eq!(entries(&f), pairs(&[("A", "a b")]));
        let f = DelimitedReader::tsv(FirstRecord::Header).read(b"").unwrap();
        assert!(f.glossary.is_empty());
        // The header's quoting is checked.
        let err = DelimitedReader::tsv(FirstRecord::Header).read(b"i\"d\tterm\n").unwrap_err();
        assert_eq!(err, ReadError::QuoteInUnquotedField { line: 1 });
    }

    #[test]
    fn errors_with_their_lines() {
        let r = DelimitedReader::csv();
        assert_eq!(r.read(b"a,b\nc\n").unwrap_err(), ReadError::MissingColumns { line: 2, found: 1 });
        assert_eq!(r.read(b"a,b\n\"c\"d,e").unwrap_err(), ReadError::AfterClosingQuote { line: 2 });
        assert_eq!(r.read(b"a,b\n\n\"c,\nd").unwrap_err(), ReadError::UnclosedQuote { line: 3 });
        assert_eq!(r.read(b" ,b").unwrap_err(), ReadError::Entry { line: 1, error: EntryError::BlankId });
        assert_eq!(r.read(b"a,\xc2\xa0").unwrap_err(), ReadError::Entry { line: 1, error: EntryError::BlankTerm });
        assert_eq!(r.read(b"a,\xff").unwrap_err(), ReadError::Utf8 { offset: 2 });
        assert_eq!(r.read(b"   \n").unwrap_err(), ReadError::MissingColumns { line: 1, found: 1 });
        assert_eq!(Delimiter::new('"'), None);
        assert_eq!(Delimiter::new(';').map(Delimiter::char), Some(';'));
    }
}
