use turbo_analysis::Span;

use crate::source::Field;

/// Decoded text and the raw byte boundary corresponding to every decoded
/// boundary. Escapes and markup can therefore never lose source offsets.
#[derive(Clone, Debug)]
pub struct Decoded {
    text: String,
    raw: Vec<u32>,
}

impl Decoded {
    pub fn new(start: usize) -> Decoded {
        Decoded { text: String::new(), raw: vec![start as u32] }
    }

    pub fn push(&mut self, c: char, start: usize, end: usize) {
        let decoded_start = self.text.len();
        self.raw.resize(decoded_start + 1, start as u32);
        self.text.push(c);
        self.raw.resize(self.text.len() + 1, end as u32);
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn span(&self, start: usize, end: usize) -> Span {
        Span::new(self.raw[start], self.raw[end])
    }

    pub fn field(&self) -> Field<'static> {
        Field::made(self.text.clone(), self.span(0, self.text.len()))
    }

    pub fn tokens(&self) -> Vec<Field<'static>> {
        let mut out = Vec::new();
        let mut start = None;
        for (i, c) in self.text.char_indices() {
            if turbo_unicode::whitespace::is_whitespace(c as u32) {
                if let Some(start) = start.take() {
                    out.push(self.token(start, i));
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(start) = start {
            out.push(self.token(start, self.text.len()));
        }
        out
    }

    fn token(&self, start: usize, end: usize) -> Field<'static> {
        Field::made(self.text[start..end].to_owned(), self.span(start, end))
    }
}
