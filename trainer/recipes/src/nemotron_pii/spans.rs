//! Python-literal `spans` parser and `SimpleTokenizer` BIO aligner for
//! `nvidia/Nemotron-PII` (#1888).
//!
//! Each Parquet row's `spans` column is a Python `repr` list of dictionaries
//! (`[{'start': 3, 'end': 8, 'text': 'Jason', 'label': 'first_name'}, ...]`)
//! whose `start` and `end` offsets are Unicode scalar (`char`) indices into
//! the row's `text` column.

use std::cmp::Reverse;

use turbo_analysis::Span;
use turbo_tokenize::{TokenizerOptions, simple_spans};

use crate::corpus::Sentence;
use crate::{Error, Result};

/// One parsed character-offset span from a `Nemotron-PII` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawSpan {
    /// Start index in Unicode scalar values (`char`s).
    pub start: usize,
    /// Exclusive end index in Unicode scalar values (`char`s).
    pub end: usize,
    /// Fine-grained `Nemotron-PII` entity type name.
    pub label: &'static str,
}

/// Tokenizes `text` with `SimpleTokenizer` and projects `spans` (char offsets)
/// onto exact token boundaries as `O`, `B-<label>` and `I-<label>`.
///
/// Overlapping gold spans (24 in `train`, 34 in `test`) resolve
/// deterministically by earliest byte start, then longest byte end. Synthetic
/// character-substring collisions that cut inside an alphanumeric or repeated
/// punctuation token are left `O` rather than corrupting surrounding tokens.
pub fn to_sentence(file: &str, row: usize, text: &str, spans_repr: &str) -> Result<Option<Sentence>> {
    let raw = parse_spans_repr(file, row, spans_repr)?;
    let mut tok_spans = Vec::new();
    simple_spans(text, TokenizerOptions::default(), &mut tok_spans);
    if tok_spans.is_empty() {
        return Ok(None);
    }

    let char_bytes = (!text.is_ascii() && !raw.is_empty()).then(|| char_byte_table(text));
    let char_len = char_bytes.as_ref().map_or(text.len(), |t| t.len() - 1);

    let mut byte_spans: Vec<(u32, u32, &'static str)> = Vec::with_capacity(raw.len());
    for s in raw {
        if s.start >= s.end || s.end > char_len {
            return Err(Error(format!(
                "{file}: row {row}: span {}..{} is outside {char_len} characters",
                s.start, s.end
            )));
        }
        let (sb, eb) = match &char_bytes {
            Some(t) => (t[s.start], t[s.end]),
            None => (s.start as u32, s.end as u32),
        };
        byte_spans.push((sb, eb, s.label));
    }
    byte_spans.sort_by_key(|&(sb, eb, _)| (sb, Reverse(eb)));

    let words: Vec<String> = tok_spans.iter().map(|s| text[s.range()].to_owned()).collect();
    let mut labels = vec!["O".to_owned(); words.len()];
    let mut accepted_end = 0u32;
    for (sb, eb, kind) in byte_spans {
        if sb < accepted_end {
            continue;
        }
        let first = tok_spans.partition_point(|t: &Span| t.end <= sb);
        let last = tok_spans.partition_point(|t: &Span| t.start < eb);
        if first < last && tok_spans[first].start == sb && tok_spans[last - 1].end == eb {
            labels[first] = format!("B-{kind}");
            for slot in &mut labels[first + 1..last] {
                *slot = format!("I-{kind}");
            }
            accepted_end = eb;
        }
    }
    Ok(Some(Sentence { words, labels }))
}

fn char_byte_table(text: &str) -> Vec<u32> {
    let mut table: Vec<u32> = text.char_indices().map(|(b, _)| b as u32).collect();
    table.push(text.len() as u32);
    table
}

/// Parses a Python `repr` list of span dicts from `Nemotron-PII`.
pub fn parse_spans_repr(file: &str, row: usize, input: &str) -> Result<Vec<RawSpan>> {
    let mut p = Parser { file, row, bytes: input.as_bytes(), pos: 0 };
    p.ws();
    p.expect(b'[')?;
    p.ws();
    let mut out = Vec::new();
    if p.eat(b']') {
        p.ws();
        p.eof()?;
        return Ok(out);
    }
    loop {
        p.ws();
        out.push(p.dict()?);
        p.ws();
        if p.eat(b',') {
            p.ws();
            if p.eat(b']') {
                break;
            }
        } else {
            p.expect(b']')?;
            break;
        }
    }
    p.ws();
    p.eof()?;
    Ok(out)
}

struct Parser<'a> {
    file: &'a str,
    row: usize,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> Error {
        Error(format!("{}: row {} byte {}: {msg}", self.file, self.row, self.pos))
    }

    fn ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, b: u8) -> Result<()> {
        if self.eat(b) { Ok(()) } else { Err(self.err(&format!("expected '{}'", b as char))) }
    }

    fn eof(&self) -> Result<()> {
        if self.pos == self.bytes.len() { Ok(()) } else { Err(self.err("trailing bytes after list")) }
    }

    fn dict(&mut self) -> Result<RawSpan> {
        self.expect(b'{')?;
        let mut start: Option<usize> = None;
        let mut end: Option<usize> = None;
        let mut label: Option<&'static str> = None;
        let mut saw_text = false;
        self.ws();
        if self.eat(b'}') {
            return Err(self.err("empty span dict"));
        }
        loop {
            self.ws();
            let key = self.py_string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            match key.as_str() {
                "start" => start = Some(self.uint()?),
                "end" => end = Some(self.uint()?),
                "label" => {
                    let raw = self.py_string()?;
                    let interned = super::intern_type(&raw).ok_or_else(|| {
                        Error(format!("{}: row {}: unknown Nemotron-PII label {raw:?}", self.file, self.row))
                    })?;
                    label = Some(interned);
                }
                "text" => {
                    self.skip_scalar()?;
                    saw_text = true;
                }
                other => return Err(self.err(&format!("unexpected span key {other:?}"))),
            }
            self.ws();
            if self.eat(b',') {
                self.ws();
                if self.eat(b'}') {
                    break;
                }
            } else {
                self.expect(b'}')?;
                break;
            }
        }
        let _ = saw_text;
        Ok(RawSpan {
            start: start.ok_or_else(|| self.err("missing 'start'"))?,
            end: end.ok_or_else(|| self.err("missing 'end'"))?,
            label: label.ok_or_else(|| self.err("missing 'label'"))?,
        })
    }

    fn uint(&mut self) -> Result<usize> {
        let from = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == from {
            return Err(self.err("expected non-negative integer"));
        }
        std::str::from_utf8(&self.bytes[from..self.pos])
            .expect("ASCII digits")
            .parse::<usize>()
            .map_err(|_| self.err("integer overflow"))
    }

    fn skip_scalar(&mut self) -> Result<()> {
        match self.peek() {
            Some(b'\'' | b'"') => self.py_string().map(|_| ()),
            Some(_) => {
                let from = self.pos;
                while let Some(b) = self.peek() {
                    if matches!(b, b',' | b'}' | b']') || b.is_ascii_whitespace() {
                        break;
                    }
                    self.pos += 1;
                }
                if self.pos == from { Err(self.err("expected scalar value")) } else { Ok(()) }
            }
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn py_string(&mut self) -> Result<String> {
        let quote = match self.peek() {
            Some(q @ (b'\'' | b'"')) => {
                self.pos += 1;
                q
            }
            _ => return Err(self.err("expected quoted string")),
        };
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == quote {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                self.pos += 1;
                match esc {
                    b'\\' => out.push('\\'),
                    b'\'' => out.push('\''),
                    b'"' => out.push('"'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'v' => out.push('\u{000B}'),
                    b'a' => out.push('\u{0007}'),
                    b'0' => out.push('\0'),
                    b'x' => out.push(self.hex_char(2)?),
                    b'u' => out.push(self.hex_char(4)?),
                    b'U' => out.push(self.hex_char(8)?),
                    other => {
                        out.push('\\');
                        out.push(other as char);
                    }
                }
                continue;
            }
            let rest = std::str::from_utf8(&self.bytes[self.pos..]).map_err(|_| self.err("invalid UTF-8 in string"))?;
            let ch = rest.chars().next().expect("non-empty slice");
            out.push(ch);
            self.pos += ch.len_utf8();
        }
        Err(self.err("unterminated string literal"))
    }

    fn hex_char(&mut self, digits: usize) -> Result<char> {
        let end = self
            .pos
            .checked_add(digits)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| self.err("short hex escape"))?;
        let hex = std::str::from_utf8(&self.bytes[self.pos..end]).map_err(|_| self.err("invalid hex escape"))?;
        let cp = u32::from_str_radix(hex, 16).map_err(|_| self.err("invalid hex digits"))?;
        self.pos = end;
        char::from_u32(cp).ok_or_else(|| self.err("escape is not a Unicode scalar value"))
    }
}
