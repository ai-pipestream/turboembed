//! A bounded JSON reader for pattern files. No permissive recovery: a
//! duplicate key, bad escape, or trailing content refuses the file.
use super::ImportError;

#[derive(Debug)]
pub(super) enum Value {
    String(String),
    Number(String),
    Bool(bool),
    Null,
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

pub(super) fn parse(text: &str) -> Result<Value, ImportError> {
    let mut r = Reader { bytes: text.as_bytes(), at: 0 };
    let value = r.value(0)?;
    r.space();
    if r.at != r.bytes.len() {
        return Err(r.error());
    }
    Ok(value)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl Reader<'_> {
    fn error(&self) -> ImportError {
        ImportError::Json { byte: self.at }
    }
    fn space(&mut self) {
        // JSON syntax permits exactly these four ASCII whitespace bytes.
        while self.bytes.get(self.at).is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t')) {
            self.at += 1;
        }
    }
    fn take(&mut self, b: u8) -> bool {
        if self.bytes.get(self.at) == Some(&b) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, ImportError> {
        self.space();
        if depth > 32 {
            return Err(self.error());
        }
        match self.bytes.get(self.at).copied() {
            Some(b'"') => self.string().map(Value::String),
            Some(b'[') => {
                self.at += 1;
                self.space();
                let mut a = Vec::new();
                if self.take(b']') {
                    return Ok(Value::Array(a));
                }
                loop {
                    a.push(self.value(depth + 1)?);
                    self.space();
                    if self.take(b']') {
                        break;
                    }
                    if !self.take(b',') {
                        return Err(self.error());
                    }
                }
                Ok(Value::Array(a))
            }
            Some(b'{') => {
                self.at += 1;
                self.space();
                let mut o: Vec<(String, Value)> = Vec::new();
                if self.take(b'}') {
                    return Ok(Value::Object(o));
                }
                loop {
                    self.space();
                    if o.len() >= 64 {
                        return Err(self.error());
                    }
                    let k = self.string()?;
                    if o.iter().any(|(old, _)| *old == k) {
                        return Err(self.error());
                    }
                    self.space();
                    if !self.take(b':') {
                        return Err(self.error());
                    }
                    o.push((k, self.value(depth + 1)?));
                    self.space();
                    if self.take(b'}') {
                        break;
                    }
                    if !self.take(b',') {
                        return Err(self.error());
                    }
                }
                Ok(Value::Object(o))
            }
            Some(b't') if self.literal(b"true") => Ok(Value::Bool(true)),
            Some(b'f') if self.literal(b"false") => Ok(Value::Bool(false)),
            Some(b'n') if self.literal(b"null") => Ok(Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error()),
        }
    }
    fn literal(&mut self, s: &[u8]) -> bool {
        if self.bytes[self.at..].starts_with(s) {
            self.at += s.len();
            true
        } else {
            false
        }
    }
    fn number(&mut self) -> Result<Value, ImportError> {
        let start = self.at;
        self.take(b'-');
        if !self.take(b'0') && !self.digits() {
            return Err(self.error());
        }
        if self.take(b'.') && !self.digits() {
            return Err(self.error());
        }
        if self.take(b'e') || self.take(b'E') {
            if !self.take(b'+') {
                self.take(b'-');
            }
            if !self.digits() {
                return Err(self.error());
            }
        }
        Ok(Value::Number(std::str::from_utf8(&self.bytes[start..self.at]).expect("ASCII").into()))
    }
    fn digits(&mut self) -> bool {
        let start = self.at;
        while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
            self.at += 1;
        }
        self.at > start
    }
    fn string(&mut self) -> Result<String, ImportError> {
        if !self.take(b'"') {
            return Err(self.error());
        }
        let mut out = Vec::new();
        loop {
            let b = *self.bytes.get(self.at).ok_or_else(|| self.error())?;
            self.at += 1;
            match b {
                b'"' => return String::from_utf8(out).map_err(|_| self.error()),
                0..=31 => return Err(self.error()),
                b'\\' => {
                    let e = *self.bytes.get(self.at).ok_or_else(|| self.error())?;
                    self.at += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(10),
                        b'r' => out.push(13),
                        b't' => out.push(9),
                        b'u' => {
                            let mut cp = self.hex()?;
                            if (0xd800..=0xdbff).contains(&cp) {
                                if !self.literal(b"\\u") {
                                    return Err(self.error());
                                }
                                let low = self.hex()?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return Err(self.error());
                                }
                                cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
                            }
                            let c = char::from_u32(cp).ok_or_else(|| self.error())?;
                            out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                        }
                        _ => return Err(self.error()),
                    }
                }
                _ => out.push(b),
            }
        }
    }
    fn hex(&mut self) -> Result<u32, ImportError> {
        let mut cp = 0;
        for _ in 0..4 {
            let b = *self.bytes.get(self.at).ok_or_else(|| self.error())?;
            let h = (b as char).to_digit(16).ok_or_else(|| self.error())?;
            cp = cp * 16 + h;
            self.at += 1;
        }
        Ok(cp)
    }
}
