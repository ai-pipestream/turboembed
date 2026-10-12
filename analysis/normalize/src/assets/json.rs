//! A bounded JSON syntax scanner for the two objects of a JSON Web Token,
//! OpenNLP's `JsonObjectParser`: it checks that a text is exactly one
//! RFC 8259 object and can capture one string member of that outermost
//! object (the `alg` of a token's header). Nothing else is built.

/// The deepest nesting of objects and arrays accepted (`MAX_DEPTH`).
const MAX_DEPTH: usize = 64;

/// The outcome of [`parse`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    /// Whether the text is one complete object, with nothing but white
    /// space around it and the captured member present at most once.
    pub valid: bool,
    /// The decoded value of the captured member, when it is a string.
    pub member: Option<String>,
}

/// Checks that `json` is one JSON object and, when `member` is given,
/// captures the string value of that top-level member.
///
/// Member names are compared after their escapes are decoded, so
/// `"\u0061lg"` is `alg`. A second top-level member of that name makes the
/// text invalid. Nested objects are not searched.
pub fn parse(json: &str, member: Option<&str>) -> Parsed {
    let mut s =
        Scanner { json: json.as_bytes(), at: 0, depth: 0, member, captured: None, seen: false, duplicate: false };
    s.skip_whitespace();
    let valid = s.object(true);
    s.skip_whitespace();
    Parsed { valid: valid && s.at == s.json.len() && !s.duplicate, member: s.captured }
}

struct Scanner<'j> {
    json: &'j [u8],
    at: usize,
    depth: usize,
    member: Option<&'j str>,
    captured: Option<String>,
    seen: bool,
    duplicate: bool,
}

impl Scanner<'_> {
    fn object(&mut self, outer: bool) -> bool {
        if !self.enter(b'{') {
            return false;
        }
        self.skip_whitespace();
        if self.take(b'}') {
            self.depth -= 1;
            return true;
        }
        while self.at < self.json.len() {
            let mut key = Vec::new();
            if !self.string(Some(&mut key)) {
                return false;
            }
            self.skip_whitespace();
            if !self.take(b':') {
                return false;
            }
            self.skip_whitespace();
            let captures = outer && self.member.is_some_and(|m| m.as_bytes() == key.as_slice());
            if captures {
                self.duplicate |= self.seen;
                self.seen = true;
                if self.peek() == Some(b'"') {
                    let mut value = Vec::new();
                    if !self.string(Some(&mut value)) {
                        return false;
                    }
                    self.captured = Some(String::from_utf8_lossy(&value).into_owned());
                } else if !self.value() {
                    return false;
                }
            } else if !self.value() {
                return false;
            }
            self.skip_whitespace();
            if self.take(b'}') {
                self.depth -= 1;
                return true;
            }
            if !self.take(b',') {
                return false;
            }
            self.skip_whitespace();
        }
        false
    }

    fn value(&mut self) -> bool {
        match self.peek() {
            None => false,
            Some(b'"') => self.string(None),
            Some(b'{') => self.object(false),
            Some(b'[') => self.array(),
            Some(b't') => self.literal(b"true"),
            Some(b'f') => self.literal(b"false"),
            Some(b'n') => self.literal(b"null"),
            Some(_) => self.number(),
        }
    }

    fn array(&mut self) -> bool {
        if !self.enter(b'[') {
            return false;
        }
        self.skip_whitespace();
        if self.take(b']') {
            self.depth -= 1;
            return true;
        }
        while self.at < self.json.len() {
            if !self.value() {
                return false;
            }
            self.skip_whitespace();
            if self.take(b']') {
                self.depth -= 1;
                return true;
            }
            if !self.take(b',') {
                return false;
            }
            self.skip_whitespace();
        }
        false
    }

    /// Scans one string, writing its decoded UTF-8 into `out` when given.
    /// A `\u` escape of an unpaired surrogate decodes to U+FFFD; OpenNLP
    /// keeps the UTF-16 unit, which changes neither the validity nor
    /// whether the value is empty or `none`, the only uses of the value.
    fn string(&mut self, mut out: Option<&mut Vec<u8>>) -> bool {
        if !self.take(b'"') {
            return false;
        }
        while let Some(c) = self.bump() {
            match c {
                b'"' => return true,
                0x00..=0x1f => return false,
                b'\\' => {
                    let Some(decoded) = self.escape() else { return false };
                    if let Some(out) = out.as_deref_mut() {
                        out.extend_from_slice(decoded.encode_utf8(&mut [0; 4]).as_bytes());
                    }
                }
                _ => {
                    if let Some(out) = out.as_deref_mut() {
                        out.push(c);
                    }
                }
            }
        }
        false
    }

    /// The character an escape after its backslash stands for.
    fn escape(&mut self) -> Option<char> {
        let c = match self.bump()? {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let unit = self.hex_quad()?;
                if !(0xD800..0xDC00).contains(&unit) || !self.json[self.at..].starts_with(b"\\u") {
                    return Some(char::from_u32(unit).unwrap_or('\u{FFFD}'));
                }
                // A high surrogate escape: combine it with a low one that
                // follows at once, otherwise leave the next escape alone.
                let mark = self.at;
                self.at += 2;
                match self.hex_quad() {
                    Some(low @ 0xDC00..0xE000) => {
                        char::from_u32(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)).unwrap_or('\u{FFFD}')
                    }
                    _ => {
                        self.at = mark;
                        '\u{FFFD}'
                    }
                }
            }
            _ => return None,
        };
        Some(c)
    }

    fn hex_quad(&mut self) -> Option<u32> {
        let quad = self.json.get(self.at..self.at + 4)?;
        let mut value = 0;
        for &b in quad {
            value = (value << 4) | super::payload::hex_digit(b)? as u32;
        }
        self.at += 4;
        Some(value)
    }

    fn number(&mut self) -> bool {
        let start = self.at;
        self.take(b'-');
        if self.take(b'0') {
            if self.peek().is_some_and(|b| b.is_ascii_digit()) {
                return false;
            }
        } else if !self.digits() {
            return false;
        }
        if self.take(b'.') && !self.digits() {
            return false;
        }
        if self.take(b'e') || self.take(b'E') {
            if !self.take(b'+') {
                self.take(b'-');
            }
            if !self.digits() {
                return false;
            }
        }
        self.at > start
    }

    fn digits(&mut self) -> bool {
        let start = self.at;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.at += 1;
        }
        self.at > start
    }

    fn literal(&mut self, literal: &[u8]) -> bool {
        if self.json[self.at..].starts_with(literal) {
            self.at += literal.len();
            true
        } else {
            false
        }
    }

    /// Opens an object or array, refusing one nested deeper than
    /// `MAX_DEPTH`.
    fn enter(&mut self, opening: u8) -> bool {
        if !self.take(opening) || self.depth == MAX_DEPTH {
            return false;
        }
        self.depth += 1;
        true
    }

    /// Skips the four white space characters of RFC 8259.
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.at += 1;
        }
    }

    fn take(&mut self, expected: u8) -> bool {
        let found = self.peek() == Some(expected);
        if found {
            self.at += 1;
        }
        found
    }

    fn peek(&self) -> Option<u8> {
        self.json.get(self.at).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.at += 1;
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    // JsonObjectParserTest.validObjects
    #[test]
    fn every_value_production_is_accepted() {
        for json in [
            "{}",
            " { \"n\" : null, \"b\" : true, \"f\" : false } ",
            "{\"array\":[0,-1,1.25,2e3,{\"nested\":\"yes\"}]}",
            "{\"escaped\":\"quote \\\" slash \\\\ unicode \\u0041\"}",
        ] {
            assert!(parse(json, None).valid, "{json}");
        }
    }

    // JsonObjectParserTest.invalidObjects
    #[test]
    fn malformed_lookalikes_fail() {
        for json in [
            "",
            "[]",
            "{\"trailing\":true,}",
            "{\"leading\":01}",
            "{\"fraction\":1.}",
            "{\"escape\":\"\\x\"}",
            "{\"unicode\":\"\\u\u{FF26}\u{FF26}\u{FF26}\u{FF26}\"}",
            "{\"unterminated\":\"value}",
            "{\"array\":[true,]}",
            "{\"missing\" true}",
        ] {
            assert!(!parse(json, None).valid, "{json}");
        }
    }

    // JsonObjectParserTest.testCapturesOnlyOneTopLevelStringMember
    #[test]
    fn captures_one_decoded_top_level_member() {
        let parsed = parse("{\"nested\":{\"alg\":\"wrong\"},\"\\u0061lg\":\"HS256\"}", Some("alg"));
        assert!(parsed.valid);
        assert_eq!(parsed.member.as_deref(), Some("HS256"));
        assert_eq!(parse("{\"nested\":{\"alg\":\"HS256\"}}", Some("alg")).member, None);
        assert!(!parse("{\"alg\":\"HS256\",\"alg\":\"RS256\"}", Some("alg")).valid);
    }

    // JsonObjectParserTest.testNestingDepthBoundary
    #[test]
    fn nesting_stops_at_64() {
        let permitted = format!("{{\"v\":{}0{}}}", "[".repeat(63), "]".repeat(63));
        let excessive = format!("{{\"v\":{}0{}}}", "[".repeat(64), "]".repeat(64));
        assert!(parse(&permitted, None).valid);
        assert!(!parse(&excessive, None).valid);
    }

    #[test]
    fn a_non_string_member_is_not_captured() {
        let parsed = parse("{\"alg\":1}", Some("alg"));
        assert!(parsed.valid);
        assert_eq!(parsed.member, None);
    }

    #[test]
    fn surrogate_escapes_decode_to_their_character() {
        let parsed = parse("{\"alg\":\"\\ud83d\\ude00 \\ud800x\"}", Some("alg"));
        assert!(parsed.valid);
        assert_eq!(parsed.member.as_deref(), Some("\u{1F600} \u{FFFD}x"));
        let lone_low = parse("{\"alg\":\"\\udc00\"}", Some("alg"));
        assert_eq!(lone_low.member.as_deref(), Some("\u{FFFD}"));
    }
}
