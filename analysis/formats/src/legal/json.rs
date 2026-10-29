use turbo_analysis::Span;

use super::LegalError;
use super::text::Decoded;

#[derive(Clone, Debug)]
pub enum Json {
    Object(Span, Vec<(String, Json)>),
    Array(Vec<Json>),
    String(Decoded),
    Number,
    Bool,
    Null,
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(_, fields) => fields.iter().find(|(name, _)| name == key).map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(values) => Some(values),
            _ => None,
        }
    }

    pub fn string(&self) -> Option<&Decoded> {
        match self {
            Json::String(text) => Some(text),
            _ => None,
        }
    }
}

pub fn parse_many(source: &str) -> Result<Vec<Json>, LegalError> {
    if source.len() > u32::MAX as usize {
        return Err(LegalError::Json { offset: source.len(), expected: "a document within the offset range" });
    }
    let mut parser = Parser { source, pos: 0 };
    let mut out = Vec::new();
    parser.ws();
    while parser.pos < source.len() {
        out.push(parser.value()?);
        parser.ws();
    }
    Ok(out)
}

struct Parser<'a> {
    source: &'a str,
    pos: usize,
}

impl Parser<'_> {
    fn error<T>(&self, expected: &'static str) -> Result<T, LegalError> {
        Err(LegalError::Json { offset: self.pos, expected })
    }

    fn ws(&mut self) {
        while self.source[self.pos..].chars().next().is_some_and(|c| matches!(c, ' ' | '\n' | '\r' | '\t')) {
            self.pos += 1;
        }
    }

    fn value(&mut self) -> Result<Json, LegalError> {
        self.ws();
        match self.source.as_bytes().get(self.pos) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Json::String),
            Some(b't') => {
                self.literal("true")?;
                Ok(Json::Bool)
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Json::Bool)
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => self.error("a JSON value"),
        }
    }

    fn object(&mut self) -> Result<Json, LegalError> {
        let start = self.pos;
        self.pos += 1;
        let mut fields = Vec::new();
        self.ws();
        if self.source.as_bytes().get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Json::Object(Span::new(start as u32, self.pos as u32), fields));
        }
        loop {
            self.ws();
            if self.source.as_bytes().get(self.pos) != Some(&b'"') {
                return self.error("an object key");
            }
            let key = self.string()?.text().to_owned();
            self.ws();
            if self.source.as_bytes().get(self.pos) != Some(&b':') {
                return self.error("':'");
            }
            self.pos += 1;
            let value = self.value()?;
            fields.push((key, value));
            self.ws();
            match self.source.as_bytes().get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Object(Span::new(start as u32, self.pos as u32), fields));
                }
                _ => return self.error("',' or '}'"),
            }
        }
    }

    fn array(&mut self) -> Result<Json, LegalError> {
        self.pos += 1;
        let mut values = Vec::new();
        self.ws();
        if self.source.as_bytes().get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Array(values));
        }
        loop {
            values.push(self.value()?);
            self.ws();
            match self.source.as_bytes().get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Array(values));
                }
                _ => return self.error("',' or ']'"),
            }
        }
    }

    fn string(&mut self) -> Result<Decoded, LegalError> {
        self.pos += 1;
        let mut out = Decoded::new(self.pos);
        loop {
            let start = self.pos;
            let c = self.source[self.pos..]
                .chars()
                .next()
                .ok_or(LegalError::Json { offset: self.pos, expected: "a closing quote" })?;
            self.pos += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => self.escape(&mut out, start)?,
                c if c <= '\u{1f}' => return self.error("an unescaped string character"),
                c => out.push(c, start, self.pos),
            }
        }
    }

    fn escape(&mut self, out: &mut Decoded, start: usize) -> Result<(), LegalError> {
        let c = self.source[self.pos..]
            .chars()
            .next()
            .ok_or(LegalError::Json { offset: self.pos, expected: "an escape" })?;
        self.pos += c.len_utf8();
        let plain = match c {
            '"' => Some('"'),
            '\\' => Some('\\'),
            '/' => Some('/'),
            'b' => Some('\u{8}'),
            'f' => Some('\u{c}'),
            'n' => Some('\n'),
            'r' => Some('\r'),
            't' => Some('\t'),
            'u' => None,
            _ => return self.error("a valid JSON escape"),
        };
        if let Some(c) = plain {
            out.push(c, start, self.pos);
            return Ok(());
        }
        let high = self.hex4()?;
        let scalar = if (0xd800..0xdc00).contains(&high) {
            if !self.source[self.pos..].starts_with("\\u") {
                return self.error("a low surrogate");
            }
            self.pos += 2;
            let low = self.hex4()?;
            if !(0xdc00..0xe000).contains(&low) {
                return self.error("a low surrogate");
            }
            0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
        } else if (0xdc00..0xe000).contains(&high) {
            return self.error("a high surrogate");
        } else {
            high
        };
        let c = char::from_u32(scalar).ok_or(LegalError::Json { offset: start, expected: "a Unicode scalar" })?;
        out.push(c, start, self.pos);
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32, LegalError> {
        let start = self.pos;
        let mut value = 0u32;
        for _ in 0..4 {
            let b = *self
                .source
                .as_bytes()
                .get(self.pos)
                .ok_or(LegalError::Json { offset: self.pos, expected: "four hexadecimal digits" })?;
            let digit = (b as char)
                .to_digit(16)
                .ok_or(LegalError::Json { offset: self.pos, expected: "four hexadecimal digits" })?;
            value = value * 16 + digit;
            self.pos += 1;
        }
        if self.pos != start + 4 {
            return self.error("four hexadecimal digits");
        }
        Ok(value)
    }

    fn literal(&mut self, text: &str) -> Result<(), LegalError> {
        if self.source[self.pos..].starts_with(text) {
            self.pos += text.len();
            Ok(())
        } else {
            self.error("a JSON literal")
        }
    }

    fn number(&mut self) -> Result<Json, LegalError> {
        if self.source.as_bytes().get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        match self.source.as_bytes().get(self.pos) {
            Some(b'0') => {
                self.pos += 1;
                if self.source.as_bytes().get(self.pos).is_some_and(u8::is_ascii_digit) {
                    return self.error("no leading zero in a number");
                }
            }
            Some(b'1'..=b'9') => self.digits(),
            _ => return self.error("a number"),
        }
        if self.source.as_bytes().get(self.pos) == Some(&b'.') {
            self.pos += 1;
            if !self.source.as_bytes().get(self.pos).is_some_and(u8::is_ascii_digit) {
                return self.error("a digit after the decimal point");
            }
            self.digits();
        }
        if self.source.as_bytes().get(self.pos).is_some_and(|b| matches!(b, b'e' | b'E')) {
            self.pos += 1;
            if self.source.as_bytes().get(self.pos).is_some_and(|b| matches!(b, b'+' | b'-')) {
                self.pos += 1;
            }
            if !self.source.as_bytes().get(self.pos).is_some_and(u8::is_ascii_digit) {
                return self.error("an exponent digit");
            }
            self.digits();
        }
        Ok(Json::Number)
    }

    fn digits(&mut self) {
        while self.source.as_bytes().get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
    }
}
