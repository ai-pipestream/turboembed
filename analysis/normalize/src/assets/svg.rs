//! Whether a payload is an SVG image: its first element is `svg` in the
//! SVG namespace. OpenNLP's `SvgHeader` asks the JDK's StAX parser for the
//! first start element, with DTDs and external entities disabled. This is
//! a hand-written scan of the same well-formedness rules up to the end of
//! that start tag; it resolves nothing and builds no tree.
//!
//! What is read: an optional byte-order mark and the encoding it or the
//! first four bytes imply (XML 1.0 appendix F); the XML declaration; then
//! white space, comments, processing instructions and one document type
//! declaration, in any order the XML grammar allows; then the root start
//! tag, its attributes and the namespace declarations on it. Any
//! well-formedness error before the end of that tag, or the end of the
//! 64 KiB decoded prefix, means the payload is not SVG.

use super::payload::{Decoder, Payload};

/// The decoded prefix the check reads (`SvgHeader.MAX_HEADER_BYTES`).
pub(crate) const MAX_HEADER_BYTES: usize = 64 * 1024;

const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";
const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
const XMLNS_NAMESPACE: &str = "http://www.w3.org/2000/xmlns/";

/// Whether a decoded header can start an XML document: `<`, XML white
/// space, `<` in UTF-16BE without a byte-order mark, or a UTF-8 or UTF-16
/// byte-order mark (`SvgHeader.isCandidate`).
pub(crate) fn is_candidate(header: &[u8]) -> bool {
    matches!(
        header,
        [b'<' | b' ' | b'\t' | b'\r' | b'\n', ..]
            | [0, b'<', ..]
            | [0xEF, 0xBB, 0xBF, ..]
            | [0xFE, 0xFF, ..]
            | [0xFF, 0xFE, ..]
    )
}

/// Whether the payload's first element is the SVG root.
pub(crate) fn is_svg(text: &[u8], payload: &Payload) -> bool {
    Xml::new(payload.decoder(text, MAX_HEADER_BYTES)).svg_root() == Some(true)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Encoding {
    Utf8,
    Utf16Be,
    Utf16Le,
    Latin1,
    Ascii,
}

/// The characters of the decoded bytes: decoded in the document's
/// encoding, with line ends normalized to LF (XML 1.0 section 2.11).
/// A malformed sequence or a code point outside the XML `Char` production
/// ends the stream, which makes every unfinished construct fail.
struct Chars<'t> {
    bytes: Decoder<'t>,
    pending: [u8; 4],
    pending_len: usize,
    pending_at: usize,
    encoding: Encoding,
    held: Option<char>,
    ended: bool,
}

impl<'t> Chars<'t> {
    /// Detects the encoding from the first four bytes and skips a
    /// byte-order mark.
    fn new(mut bytes: Decoder<'t>) -> Chars<'t> {
        let mut pending = [0u8; 4];
        let mut pending_len = 0;
        for b in bytes.by_ref().take(4) {
            pending[pending_len] = b;
            pending_len += 1;
        }
        let (encoding, skip) = match &pending[..pending_len] {
            [0xEF, 0xBB, 0xBF, ..] => (Encoding::Utf8, 3),
            [0xFE, 0xFF, ..] => (Encoding::Utf16Be, 2),
            [0xFF, 0xFE, ..] => (Encoding::Utf16Le, 2),
            [0x00, b'<', 0x00, b'?'] => (Encoding::Utf16Be, 0),
            [b'<', 0x00, b'?', 0x00] => (Encoding::Utf16Le, 0),
            _ => (Encoding::Utf8, 0),
        };
        Chars { bytes, pending, pending_len, pending_at: skip, encoding, held: None, ended: false }
    }

    fn byte(&mut self) -> Option<u8> {
        if self.pending_at < self.pending_len {
            self.pending_at += 1;
            return Some(self.pending[self.pending_at - 1]);
        }
        self.bytes.next()
    }

    fn decode(&mut self) -> Option<char> {
        let first = self.byte()?;
        match self.encoding {
            Encoding::Ascii => first.is_ascii().then_some(first as char),
            Encoding::Latin1 => Some(first as char),
            Encoding::Utf8 => self.utf8(first),
            Encoding::Utf16Be | Encoding::Utf16Le => {
                let unit = self.unit(first)?;
                match unit {
                    0xD800..=0xDBFF => {
                        let next = self.byte()?;
                        let low = self.unit(next)?;
                        if !(0xDC00..=0xDFFF).contains(&low) {
                            return None;
                        }
                        char::from_u32(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00))
                    }
                    _ => char::from_u32(unit),
                }
            }
        }
    }

    /// A UTF-16 code unit whose first byte is `first`.
    fn unit(&mut self, first: u8) -> Option<u32> {
        let second = self.byte()?;
        let (high, low) = if self.encoding == Encoding::Utf16Be { (first, second) } else { (second, first) };
        Some(((high as u32) << 8) | low as u32)
    }

    /// Strict UTF-8: no overlong forms, surrogates or values past U+10FFFF.
    fn utf8(&mut self, first: u8) -> Option<char> {
        let (len, init, min) = match first {
            0x00..=0x7F => return Some(first as char),
            0xC2..=0xDF => (2, (first & 0x1F) as u32, 0x80),
            0xE0..=0xEF => (3, (first & 0x0F) as u32, 0x800),
            0xF0..=0xF4 => (4, (first & 0x07) as u32, 0x10000),
            _ => return None,
        };
        let mut value = init;
        for _ in 1..len {
            let b = self.byte()?;
            if b & 0xC0 != 0x80 {
                return None;
            }
            value = (value << 6) | (b & 0x3F) as u32;
        }
        if value < min {
            return None;
        }
        char::from_u32(value)
    }

    fn next(&mut self) -> Option<char> {
        if self.ended {
            return None;
        }
        let c = match self.held.take() {
            Some(c) => Some(c),
            None => self.decode().filter(|&c| is_xml_char(c)),
        };
        let Some(c) = c else {
            self.ended = true;
            return None;
        };
        if c != '\r' {
            return Some(c);
        }
        match self.decode().filter(|&c| is_xml_char(c)) {
            Some('\n') => {}
            Some(other) => self.held = Some(other),
            None => self.ended = true,
        }
        Some('\n')
    }
}

/// The XML `Char` production.
fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..)
}

/// XML white space: space, tab, LF (CR has been normalized away).
fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// `NameStartChar` of XML 1.0, fifth edition.
/// A `PubidChar` of XML 1.0.
fn is_pubid_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c,
            ' ' | '\r'
                | '\n'
                | '-'
                | '\''
                | '('
                | ')'
                | '+'
                | ','
                | '.'
                | '/'
                | ':'
                | '='
                | '?'
                | ';'
                | '!'
                | '*'
                | '#'
                | '@'
                | '$'
                | '_'
                | '%'
        )
}

fn is_name_start(c: char) -> bool {
    matches!(c,
        ':' | 'A'..='Z' | '_' | 'a'..='z' | '\u{C0}'..='\u{D6}' | '\u{D8}'..='\u{F6}' | '\u{F8}'..='\u{2FF}'
        | '\u{370}'..='\u{37D}' | '\u{37F}'..='\u{1FFF}' | '\u{200C}'..='\u{200D}' | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}' | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}')
}

/// `NameChar` of XML 1.0, fifth edition.
fn is_name_char(c: char) -> bool {
    is_name_start(c) || matches!(c, '-' | '.' | '0'..='9' | '\u{B7}' | '\u{300}'..='\u{36F}' | '\u{203F}'..='\u{2040}')
}

/// How far ahead the scan looks: the longest literal it compares,
/// `standalone`, fits.
const LOOKAHEAD: usize = 16;

struct Xml<'t> {
    chars: Chars<'t>,
    look: [char; LOOKAHEAD],
    look_len: usize,
}

/// One attribute of the root start tag: its qualified name and its
/// normalized value.
struct Attribute {
    name: String,
    value: String,
}

impl<'t> Xml<'t> {
    fn new(bytes: Decoder<'t>) -> Xml<'t> {
        Xml { chars: Chars::new(bytes), look: ['\0'; LOOKAHEAD], look_len: 0 }
    }

    fn peek_at(&mut self, k: usize) -> Option<char> {
        while self.look_len <= k {
            self.look[self.look_len] = self.chars.next()?;
            self.look_len += 1;
        }
        Some(self.look[k])
    }

    fn peek(&mut self) -> Option<char> {
        self.peek_at(0)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.look.copy_within(1..self.look_len, 0);
        self.look_len -= 1;
        Some(c)
    }

    /// Whether the next characters are `literal`. Characters are decoded
    /// only as far as they match, so a mismatch never reads past the point
    /// where an encoding declaration takes effect.
    fn starts_with(&mut self, literal: &str) -> bool {
        literal.chars().enumerate().all(|(k, c)| self.peek_at(k) == Some(c))
    }

    fn expect(&mut self, literal: &str) -> Option<()> {
        if !self.starts_with(literal) {
            return None;
        }
        for _ in literal.chars() {
            self.bump();
        }
        Some(())
    }

    /// Skips white space and says whether there was any.
    fn skip_space(&mut self) -> bool {
        let mut any = false;
        while self.peek().is_some_and(is_space) {
            self.bump();
            any = true;
        }
        any
    }

    /// `true` when the root element is `svg` in the SVG namespace, `false`
    /// when it is another element, `None` when the prolog or the start tag
    /// is not well-formed or does not end within the decoded prefix.
    fn svg_root(&mut self) -> Option<bool> {
        if self.starts_with("<?xml") && self.peek_at(5).is_some_and(is_space) {
            self.xml_declaration()?;
        }
        let mut doctype = false;
        loop {
            self.skip_space();
            if self.starts_with("<!--") {
                self.comment()?;
            } else if self.starts_with("<?") {
                self.processing_instruction()?;
            } else if self.starts_with("<!DOCTYPE") && !doctype {
                doctype = true;
                self.doctype()?;
            } else if self.peek() == Some('<') && self.peek_at(1).is_some_and(is_name_start) {
                return self.root_start_tag();
            } else {
                return None;
            }
        }
    }

    /// `<?xml version="1.x" encoding="..." standalone="..."?>`, and the
    /// switch to the declared encoding.
    fn xml_declaration(&mut self) -> Option<()> {
        self.expect("<?xml")?;
        self.skip_space();
        self.expect("version")?;
        let version = self.pseudo_attribute()?;
        if version != "1.0" && version != "1.1" {
            return None;
        }
        let mut space = self.skip_space();
        let mut declared = None;
        if space && self.starts_with("encoding") {
            self.expect("encoding")?;
            let name = self.pseudo_attribute()?;
            let mut chars = name.chars();
            if !chars.next().is_some_and(|c| c.is_ascii_alphabetic())
                || !chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            {
                return None;
            }
            declared = Some(name);
            space = self.skip_space();
        }
        if space && self.starts_with("standalone") {
            self.expect("standalone")?;
            if !matches!(self.pseudo_attribute()?.as_str(), "yes" | "no") {
                return None;
            }
            self.skip_space();
        }
        self.expect("?>")?;
        if let Some(name) = declared {
            self.chars.encoding = declared_encoding(self.chars.encoding, &name)?;
        }
        Some(())
    }

    /// `S? = S? 'value'` of the XML declaration, with the value. The values
    /// it accepts are short ASCII names, so the length is capped.
    fn pseudo_attribute(&mut self) -> Option<String> {
        self.skip_space();
        self.expect("=")?;
        self.skip_space();
        let quote = self.bump().filter(|&q| q == '"' || q == '\'')?;
        let mut value = String::new();
        loop {
            let c = self.bump()?;
            if c == quote {
                return Some(value);
            }
            if value.len() >= 64 {
                return None;
            }
            value.push(c);
        }
    }

    /// `<!-- ... -->`, where the text may not hold `--`.
    fn comment(&mut self) -> Option<()> {
        self.expect("<!--")?;
        loop {
            if self.bump()? == '-' && self.peek()? == '-' {
                self.bump();
                return (self.bump()? == '>').then_some(());
            }
        }
    }

    /// `<?target ...?>`: the target is a name other than `xml` in any
    /// letter case. A colon in it is accepted, as the JDK parser accepts it.
    fn processing_instruction(&mut self) -> Option<()> {
        self.expect("<?")?;
        let mut target = String::new();
        self.name(&mut target)?;
        if target.eq_ignore_ascii_case("xml") {
            return None;
        }
        if self.starts_with("?>") {
            return self.expect("?>");
        }
        if !self.skip_space() {
            return None;
        }
        loop {
            if self.bump()? == '?' && self.peek()? == '>' {
                self.bump();
                return Some(());
            }
        }
    }

    /// `<!DOCTYPE name ExternalID? [internal subset]? >`, read as the JDK
    /// parser reads it with DTD support off: the external subset is never
    /// fetched, and the internal subset is skipped unread up to its first
    /// `]`, so its declarations are neither checked nor applied.
    fn doctype(&mut self) -> Option<()> {
        self.expect("<!DOCTYPE")?;
        if !self.skip_space() {
            return None;
        }
        self.name(&mut String::new())?;
        let space = self.skip_space();
        if space && self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.literal_after_space(false)?;
            self.skip_space();
        } else if space && self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.literal_after_space(true)?;
            self.literal_after_space(false)?;
            self.skip_space();
        }
        if self.peek()? == '[' {
            while self.bump()? != ']' {}
            self.skip_space();
        }
        (self.bump()? == '>').then_some(())
    }

    /// White space and then a quoted literal, of public identifier
    /// characters only when `public`.
    fn literal_after_space(&mut self, public: bool) -> Option<()> {
        if !self.skip_space() {
            return None;
        }
        let quote = self.bump().filter(|&q| q == '"' || q == '\'')?;
        loop {
            let c = self.bump()?;
            if c == quote {
                return Some(());
            }
            if public && !is_pubid_char(c) {
                return None;
            }
        }
    }

    /// Reads a `Name` into `out`.
    fn name(&mut self, out: &mut String) -> Option<()> {
        let first = self.bump().filter(|&c| is_name_start(c))?;
        out.push(first);
        while let Some(c) = self.peek().filter(|&c| is_name_char(c)) {
            self.bump();
            out.push(c);
        }
        Some(())
    }

    /// The root start tag, and whether it is the SVG root.
    fn root_start_tag(&mut self) -> Option<bool> {
        self.expect("<")?;
        let mut element = String::new();
        self.name(&mut element)?;
        let mut attributes: Vec<Attribute> = Vec::new();
        loop {
            let space = self.skip_space();
            if self.starts_with("/>") || self.starts_with(">") {
                break;
            }
            if !space {
                return None;
            }
            let mut name = String::new();
            self.name(&mut name)?;
            self.skip_space();
            self.expect("=")?;
            self.skip_space();
            let mut value = String::new();
            self.attribute_value(&mut value)?;
            if attributes.iter().any(|a| a.name == name) {
                return None;
            }
            attributes.push(Attribute { name, value });
        }
        let (prefix, local) = qualified(&element)?;
        let scope = Scope::declared(&attributes)?;
        let namespace = scope.resolve(prefix, true)?;
        scope.check_attributes(&attributes)?;
        Some(local == "svg" && namespace == Some(SVG_NAMESPACE))
    }

    /// A quoted attribute value with its references replaced and its
    /// white space characters normalized to spaces (XML 1.0 section 3.3.3,
    /// for an attribute no DTD declares).
    fn attribute_value(&mut self, out: &mut String) -> Option<()> {
        let quote = self.bump().filter(|&q| q == '"' || q == '\'')?;
        loop {
            match self.bump()? {
                c if c == quote => return Some(()),
                '<' => return None,
                '&' => out.push(self.reference()?),
                '\n' | '\t' => out.push(' '),
                c => out.push(c),
            }
        }
    }

    /// The character a reference after its `&` stands for: a character
    /// reference or one of the five predefined entities. Any other entity
    /// is undeclared, since no DTD is read.
    fn reference(&mut self) -> Option<char> {
        let c = if self.peek()? == '#' {
            self.bump();
            let radix = if self.peek()? == 'x' {
                self.bump();
                16
            } else {
                10
            };
            let mut value: u32 = 0;
            let mut digits = 0;
            while let Some(d) = self.peek().and_then(|c| c.to_digit(radix)) {
                self.bump();
                value = value.checked_mul(radix)?.checked_add(d)?;
                digits += 1;
            }
            if digits == 0 {
                return None;
            }
            char::from_u32(value).filter(|&c| is_xml_char(c))?
        } else {
            let mut name = String::new();
            self.name(&mut name)?;
            match name.as_str() {
                "lt" => '<',
                "gt" => '>',
                "amp" => '&',
                "apos" => '\'',
                "quot" => '"',
                _ => return None,
            }
        };
        (self.bump()? == ';').then_some(c)
    }
}

/// The encoding after a declaration names `name`, given the one the first
/// bytes implied. A declaration may not move between 8-bit and 16-bit
/// encodings. Beyond UTF-8, UTF-16, US-ASCII and ISO-8859-1 the payload is
/// taken not to be SVG.
fn declared_encoding(detected: Encoding, name: &str) -> Option<Encoding> {
    let name = name.to_ascii_uppercase();
    let sixteen = matches!(detected, Encoding::Utf16Be | Encoding::Utf16Le);
    match name.as_str() {
        "UTF-16" | "UTF-16BE" | "UTF-16LE" if sixteen => Some(detected),
        "UTF-8" if !sixteen => Some(Encoding::Utf8),
        "US-ASCII" | "ASCII" | "ANSI_X3.4-1968" | "ANSI_X3.4-1986" | "ISO-IR-6" | "ISO646-US" | "US" | "IBM367"
        | "CP367" | "CSASCII"
            if !sixteen =>
        {
            Some(Encoding::Ascii)
        }
        "ISO-8859-1" | "ISO_8859-1" | "LATIN1" | "ISO-IR-100" | "L1" | "IBM819" | "CP819" | "CSISOLATIN1"
            if !sixteen =>
        {
            Some(Encoding::Latin1)
        }
        _ => None,
    }
}

/// A qualified name split into prefix and local part, or `None` when it
/// is not a `QName`. As in the JDK parser, a colon in first place is part of
/// the name, the first colon after it splits the prefix from the local part,
/// and the local part must start with a name start character and may not
/// hold another colon.
fn qualified(name: &str) -> Option<(Option<&str>, &str)> {
    let first = name.chars().next().map_or(0, char::len_utf8);
    let Some(colon) = name[first..].find(':').map(|i| i + first) else {
        return Some((None, name));
    };
    let local = &name[colon + 1..];
    (local.starts_with(is_name_start) && !local.contains(':')).then_some((Some(&name[..colon]), local))
}

/// The namespace declarations of the root element (the only element in
/// scope).
struct Scope<'a> {
    default: Option<&'a str>,
    prefixes: Vec<(&'a str, &'a str)>,
}

impl<'a> Scope<'a> {
    /// Reads `xmlns` and `xmlns:p` attributes, enforcing the constraints
    /// of Namespaces in XML 1.0: `xmlns` is never declared, `xml` only to
    /// its own namespace, no other prefix to either reserved namespace,
    /// and no prefix to the empty name.
    fn declared(attributes: &'a [Attribute]) -> Option<Scope<'a>> {
        let mut scope = Scope { default: None, prefixes: Vec::new() };
        for a in attributes {
            let value = a.value.as_str();
            if a.name == "xmlns" {
                if value == XML_NAMESPACE || value == XMLNS_NAMESPACE {
                    return None;
                }
                scope.default = (!value.is_empty()).then_some(value);
            } else if let Some(prefix) = a.name.strip_prefix("xmlns:") {
                let reserved = value == XML_NAMESPACE || value == XMLNS_NAMESPACE;
                let allowed = match prefix {
                    "xmlns" => false,
                    "xml" => value == XML_NAMESPACE,
                    _ => !value.is_empty() && !reserved,
                };
                if !allowed || qualified(&a.name).is_none() {
                    return None;
                }
                scope.prefixes.push((prefix, value));
            }
        }
        Some(scope)
    }

    /// The namespace of a prefix; an unprefixed element takes the default
    /// namespace, an unprefixed attribute none. `None` for an unbound
    /// prefix.
    fn resolve(&self, prefix: Option<&str>, element: bool) -> Option<Option<&'a str>> {
        match prefix {
            None => Some(if element { self.default } else { None }),
            Some("xml") => Some(Some(XML_NAMESPACE)),
            Some("xmlns") => None,
            Some(p) => self.prefixes.iter().find(|(q, _)| *q == p).map(|&(_, uri)| Some(uri)),
        }
    }

    /// Every attribute's prefix is bound, and no two attributes have the
    /// same namespace and local name.
    fn check_attributes(&self, attributes: &[Attribute]) -> Option<()> {
        let mut expanded: Vec<(Option<&str>, &str)> = Vec::new();
        for a in attributes {
            if a.name == "xmlns" || a.name.starts_with("xmlns:") {
                continue;
            }
            let (prefix, local) = qualified(&a.name)?;
            let name = (self.resolve(prefix, false)?, local);
            if expanded.contains(&name) {
                return None;
            }
            expanded.push(name);
        }
        Some(())
    }
}
