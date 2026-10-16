//! RFC 4180 CSV record reader and Who's On First meta CSV column helpers for
//! [`super::ContainmentSpineBuilder`].

use crate::text::is_blank;

use super::HierarchyError;

pub(super) const WOF_ID_HEADER: &str = "id";
pub(super) const WOF_PARENT_HEADER: &str = "parent_id";
pub(super) const WOF_NAME_HEADER: &str = "name";
pub(super) const WOF_TYPE_HEADER: &str = "placetype";

pub(super) fn column_index(header: &[String], name: &str, source: &str) -> Result<usize, HierarchyError> {
    header
        .iter()
        .position(|h| h == name)
        .ok_or_else(|| HierarchyError::Format(format!("WOF meta CSV at {source} is missing required column '{name}'")))
}

pub(super) fn is_wof_root_parent(value: &str, source: &str, line_no: usize) -> Result<bool, HierarchyError> {
    if is_blank(value) {
        return Ok(true);
    }
    let bytes = value.as_bytes();
    let start = usize::from(bytes[0] == b'-');
    if start == bytes.len() || !bytes[start..].iter().all(u8::is_ascii_digit) {
        return Err(HierarchyError::Format(format!(
            "WOF meta row at {source} line {line_no} has non-numeric parent_id '{value}'"
        )));
    }
    if start == 1 {
        return Ok(true);
    }
    Ok(bytes.iter().all(|&b| b == b'0'))
}

/// RFC 4180 CSV record reader matching `ContainmentSpine.CsvRecordReader`.
pub(super) struct CsvRecordReader<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    source: &'a str,
    line_number: usize,
}

impl<'a> CsvRecordReader<'a> {
    pub(super) fn new(text: &'a str, source: &'a str) -> CsvRecordReader<'a> {
        CsvRecordReader { chars: text.chars().peekable(), source, line_number: 1 }
    }

    pub(super) fn next_record(&mut self) -> Result<Option<(Vec<String>, usize)>, HierarchyError> {
        if self.chars.peek().is_none() {
            return Ok(None);
        }
        let record_line = self.line_number;
        let mut fields = Vec::new();
        let mut field = String::new();
        loop {
            match self.chars.peek().copied() {
                Some('"') => {
                    let _ = self.chars.next();
                    let finished = self.read_quoted_field(&mut field, record_line)?;
                    fields.push(std::mem::take(&mut field));
                    if finished {
                        return Ok(Some((fields, record_line)));
                    }
                }
                _ => {
                    let finished = self.read_unquoted_field(&mut field)?;
                    fields.push(std::mem::take(&mut field));
                    if finished {
                        return Ok(Some((fields, record_line)));
                    }
                }
            }
        }
    }

    fn read_unquoted_field(&mut self, field: &mut String) -> Result<bool, HierarchyError> {
        while let Some(c) = self.chars.next() {
            match c {
                ',' => return Ok(false),
                '\r' => {
                    if self.chars.peek() == Some(&'\n') {
                        let _ = self.chars.next();
                    }
                    self.line_number += 1;
                    return Ok(true);
                }
                '\n' => {
                    self.line_number += 1;
                    return Ok(true);
                }
                '"' => {
                    return Err(HierarchyError::Format(format!(
                        "Malformed CSV at {} line {}: stray quote in unquoted field",
                        self.source, self.line_number
                    )));
                }
                other => field.push(other),
            }
        }
        Ok(true)
    }

    fn read_quoted_field(&mut self, field: &mut String, record_line: usize) -> Result<bool, HierarchyError> {
        while let Some(c) = self.chars.next() {
            match c {
                '"' => {
                    if self.chars.peek() == Some(&'"') {
                        let _ = self.chars.next();
                        field.push('"');
                        continue;
                    }
                    return match self.chars.next() {
                        None => Ok(true),
                        Some(',') => Ok(false),
                        Some('\r') => {
                            if self.chars.peek() == Some(&'\n') {
                                let _ = self.chars.next();
                            }
                            self.line_number += 1;
                            Ok(true)
                        }
                        Some('\n') => {
                            self.line_number += 1;
                            Ok(true)
                        }
                        Some(_) => Err(HierarchyError::Format(format!(
                            "Malformed CSV at {} line {}: unexpected character after a closing quote",
                            self.source, self.line_number
                        ))),
                    };
                }
                '\r' => {
                    if self.chars.peek() == Some(&'\n') {
                        let _ = self.chars.next();
                    }
                    self.line_number += 1;
                    field.push('\n');
                }
                '\n' => {
                    self.line_number += 1;
                    field.push('\n');
                }
                other => field.push(other),
            }
        }
        Err(HierarchyError::Format(format!(
            "Malformed CSV at {} line {}: unterminated quoted field",
            self.source, record_line
        )))
    }
}
