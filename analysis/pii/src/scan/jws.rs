//! The header check of a JWT, OpenNLP's `JwsHeader`: JSON syntax with an
//! explicit stack, and one top-level `alg` member whose value is a
//! non-empty ASCII string. Values are checked for syntax only.
//!
//! The Java reads the decoded header as UTF-16; this reads its UTF-8
//! bytes. Every byte of a multibyte character is non-ASCII, as every unit
//! of a non-ASCII character is, so the syntax and the flags come out the
//! same.

#[derive(Copy, Clone)]
enum State {
    ObjectFirstKey,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectEnd,
    ArrayFirstValue,
    ArrayValue,
    ArrayEnd,
}

/// Flags of a string read: whether it is `alg`, all ASCII, not empty.
const ALGORITHM_NAME: u8 = 1;
const ASCII: u8 = 2;
const NON_EMPTY: u8 = 4;

struct Reader<'a> {
    text: &'a [u8],
    at: usize,
    states: Vec<State>,
    algorithm_next: bool,
    algorithm_seen: bool,
}

/// Whether the decoded header passes.
pub(crate) fn valid_header(header: &str) -> bool {
    let mut r =
        Reader { text: header.as_bytes(), at: 0, states: Vec::new(), algorithm_next: false, algorithm_seen: false };
    r.valid().unwrap_or(false)
}

impl Reader<'_> {
    /// `None` where the Java returns false early.
    fn valid(&mut self) -> Option<bool> {
        self.skip_whitespace();
        if !self.consume(b'{') {
            return None;
        }
        self.states.push(State::ObjectFirstKey);
        while let Some(state) = {
            self.skip_whitespace();
            self.states.pop()
        } {
            match state {
                State::ObjectFirstKey | State::ObjectKey => {
                    if matches!(state, State::ObjectFirstKey) && self.consume(b'}') {
                        continue;
                    }
                    let flags = self.read_string()?;
                    if self.states.is_empty() {
                        self.algorithm_next = flags & ALGORITHM_NAME != 0;
                        if self.algorithm_next && self.algorithm_seen {
                            return None;
                        }
                    }
                    self.states.push(State::ObjectColon);
                }
                State::ObjectColon => {
                    if !self.consume(b':') {
                        return None;
                    }
                    self.states.push(State::ObjectValue);
                }
                State::ObjectValue => {
                    let algorithm = self.states.is_empty() && self.algorithm_next;
                    self.states.push(State::ObjectEnd);
                    if algorithm {
                        let flags = self.read_string()?;
                        if flags & (ASCII | NON_EMPTY) != ASCII | NON_EMPTY {
                            return None;
                        }
                        self.algorithm_seen = true;
                    } else if !self.read_value() {
                        return None;
                    }
                }
                State::ObjectEnd => {
                    if self.consume(b',') {
                        self.states.push(State::ObjectKey);
                    } else if !self.consume(b'}') {
                        return None;
                    }
                }
                State::ArrayFirstValue | State::ArrayValue => {
                    if matches!(state, State::ArrayFirstValue) && self.consume(b']') {
                        continue;
                    }
                    self.states.push(State::ArrayEnd);
                    if !self.read_value() {
                        return None;
                    }
                }
                State::ArrayEnd => {
                    if self.consume(b',') {
                        self.states.push(State::ArrayValue);
                    } else if !self.consume(b']') {
                        return None;
                    }
                }
            }
        }
        self.skip_whitespace();
        Some(self.at == self.text.len() && self.algorithm_seen)
    }

    fn read_value(&mut self) -> bool {
        match self.text.get(self.at) {
            None => false,
            Some(b'"') => self.read_string().is_some(),
            Some(b'{') => {
                self.at += 1;
                self.states.push(State::ObjectFirstKey);
                true
            }
            Some(b'[') => {
                self.at += 1;
                self.states.push(State::ArrayFirstValue);
                true
            }
            Some(b't') => self.literal(b"true"),
            Some(b'f') => self.literal(b"false"),
            Some(b'n') => self.literal(b"null"),
            Some(_) => self.read_number(),
        }
    }

    /// A string's flags, or `None` for bad syntax.
    fn read_string(&mut self) -> Option<u8> {
        if !self.consume(b'"') {
            return None;
        }
        let mut algorithm = true;
        let mut ascii = true;
        let mut length = 0;
        while let Some(&c) = self.text.get(self.at) {
            self.at += 1;
            if c == b'"' {
                let name = if algorithm && length == 3 { ALGORITHM_NAME } else { 0 };
                return Some(name | if ascii { ASCII } else { 0 } | if length > 0 { NON_EMPTY } else { 0 });
            }
            if c < 0x20 {
                return None;
            }
            let unit = if c == b'\\' { self.read_escape()? } else { u32::from(c) };
            algorithm &= length < 3 && unit == u32::from(b"alg"[length]);
            ascii &= unit < 0x80;
            length += 1;
        }
        None
    }

    /// The UTF-16 unit an escape stands for.
    fn read_escape(&mut self) -> Option<u32> {
        let c = *self.text.get(self.at)?;
        self.at += 1;
        Some(match c {
            b'"' | b'\\' | b'/' => u32::from(c),
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => 0x0a,
            b'r' => 0x0d,
            b't' => 0x09,
            b'u' => {
                let digits = self.text.get(self.at..self.at + 4)?;
                if !digits.iter().all(u8::is_ascii_hexdigit) {
                    return None;
                }
                self.at += 4;
                digits.iter().fold(0, |v, &d| v * 16 + crate::text::hex_value(d))
            }
            _ => return None,
        })
    }

    fn read_number(&mut self) -> bool {
        self.consume(b'-');
        if self.consume(b'0') {
            if self.text.get(self.at).is_some_and(u8::is_ascii_digit) {
                return false;
            }
        } else if !self.read_digits() {
            return false;
        }
        if self.consume(b'.') && !self.read_digits() {
            return false;
        }
        if self.consume(b'e') || self.consume(b'E') {
            if !self.consume(b'+') {
                self.consume(b'-');
            }
            return self.read_digits();
        }
        true
    }

    fn read_digits(&mut self) -> bool {
        let start = self.at;
        while self.text.get(self.at).is_some_and(u8::is_ascii_digit) {
            self.at += 1;
        }
        self.at > start
    }

    fn literal(&mut self, word: &[u8]) -> bool {
        word.iter().all(|&c| self.consume(c))
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.text.get(self.at), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.at += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.text.get(self.at) == Some(&expected) {
            self.at += 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::valid_header;

    #[test]
    fn headers() {
        assert!(valid_header(r#"{"alg":"HS256","typ":"JWT"}"#));
        assert!(valid_header(" {\n \"typ\" : [1, {\"a\": null}, -0.5e+3], \"alg\": \"none\" } "));
        assert!(valid_header(r#"{"alg":"RS256"}"#));
        assert!(valid_header(r#"{"x":{"alg":1},"alg":"ES256"}"#));
        assert!(!valid_header(r#"{"alg":"HS256","alg":"RS256"}"#));
        assert!(!valid_header(r#"{"alg":""}"#));
        assert!(!valid_header(r#"{"alg":"é"}"#));
        assert!(!valid_header(r#"{"alg":1}"#));
        assert!(!valid_header(r#"{"typ":"JWT"}"#));
        assert!(!valid_header(r#"{"alg":"HS256"} x"#));
        assert!(!valid_header(r#"{"alg":"HS256",}"#));
        assert!(!valid_header(r#"{"a":01,"alg":"x"}"#));
        assert!(!valid_header(r#"["alg"]"#));
        assert!(!valid_header("{\"alg\":\"a\u{1}\"}"));
        assert!(!valid_header(r#"{"alg":"é"}"#));
        assert!(!valid_header(r#"{"alg":"\x"}"#));
    }
}
