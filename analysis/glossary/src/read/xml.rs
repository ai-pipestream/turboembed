//! A small XML 1.0 pull parser with namespaces, enough to read a TBX
//! termbase and to refuse what is not well-formed.
//!
//! It reads UTF-8 only, and an XML declaration naming another encoding is
//! refused. A document type declaration is read for its internal general
//! entities, which are expanded in text and attribute values; external
//! entities are never loaded, and declaring one is an error. Parameter
//! entity references in the internal subset are skipped, not expanded,
//! and attribute defaults declared there are not applied. An entity whose
//! replacement text holds markup is refused rather than parsed. Entity
//! expansion is limited in count, nesting and the bytes it adds.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::error::ReadError;

/// The namespace `xml:` is bound to.
const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";
/// The namespace `xmlns` stands for, which no prefix may be bound to.
const XMLNS_NS: &str = "http://www.w3.org/2000/xmlns/";
/// The most entity expansions one document may make, as the JDK's
/// `jdk.xml.entityExpansionLimit` default.
const MAX_EXPANSIONS: u32 = 64_000;
/// The deepest one entity may nest others.
const MAX_ENTITY_DEPTH: usize = 64;
/// The most bytes entity expansions may add to one document, the number
/// of the JDK 21 `jdk.xml.totalEntitySizeLimit` default (which counts
/// UTF-16 units).
const MAX_EXPANDED_BYTES: usize = 50_000_000;

/// A start tag: its local name, whether its namespace is none or the one
/// the parser was given, its qualified name, its unqualified `id` and its
/// `xml:lang`, the line it is on and the byte its content starts at.
pub(super) struct Start<'a> {
    pub qname: &'a str,
    pub local: &'a str,
    pub known: bool,
    pub id: Option<String>,
    pub lang: Option<String>,
    pub line: u32,
    pub content_start: usize,
}

pub(super) enum Event<'a> {
    Start(Start<'a>),
    /// An end tag, or the end of an empty element; `at` is the byte its
    /// element's content ends at.
    End {
        at: usize,
    },
    /// Character data, with references expanded and line ends normalized.
    Text(Cow<'a, str>),
    Eof,
}

pub(super) struct Xml<'a> {
    src: &'a str,
    pos: usize,
    /// Bytes before `src` in the caller's input, for spans.
    base: usize,
    known_ns: &'static str,
    /// The namespaces bound to each prefix in scope, innermost last; the
    /// prefixes the open elements bound, in order; and how many of those
    /// there were when each open element started.
    bindings: HashMap<&'a str, Vec<String>>,
    bound: Vec<&'a str>,
    marks: Vec<usize>,
    open: Vec<&'a str>,
    entities: HashMap<String, String>,
    doctype_seen: bool,
    started: bool,
    finished: bool,
    pending_end: bool,
    expansions: u32,
    expanded_bytes: usize,
    /// A byte of `src` and how many line ends lie before it, so that
    /// [`Xml::line_at`] counts on from there for a later byte.
    counted: Cell<(usize, u32)>,
}

fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..)
}

fn is_name_start(c: char) -> bool {
    matches!(c,
        ':' | 'A'..='Z' | '_' | 'a'..='z' | '\u{C0}'..='\u{D6}' | '\u{D8}'..='\u{F6}' | '\u{F8}'..='\u{2FF}'
        | '\u{370}'..='\u{37D}' | '\u{37F}'..='\u{1FFF}' | '\u{200C}'..='\u{200D}' | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}' | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}')
}

fn is_name_char(c: char) -> bool {
    is_name_start(c) || matches!(c, '-' | '.' | '0'..='9' | '\u{B7}' | '\u{300}'..='\u{36F}' | '\u{203F}'..='\u{2040}')
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

impl<'a> Xml<'a> {
    /// A parser over `src`, which starts `base` bytes into the caller's
    /// input. Elements in `known_ns` or in no namespace are reported as
    /// known.
    pub fn new(src: &'a str, base: usize, known_ns: &'static str) -> Xml<'a> {
        Xml {
            src,
            pos: 0,
            base,
            known_ns,
            bindings: HashMap::new(),
            bound: Vec::new(),
            marks: Vec::new(),
            open: Vec::new(),
            entities: HashMap::new(),
            doctype_seen: false,
            started: false,
            finished: false,
            pending_end: false,
            expansions: 0,
            expanded_bytes: 0,
            counted: Cell::new((0, 0)),
        }
    }

    /// The line byte `at` is on, counting from 1; a carriage return just
    /// before `at` ends a line. The parser asks about later and later
    /// bytes, so the count goes on from the byte asked about before and
    /// reading a document counts each line end once.
    pub fn line_at(&self, at: usize) -> u32 {
        let at = at.min(self.src.len());
        let b = self.src.as_bytes();
        let (mut from, mut ends) = self.counted.get();
        if at < from {
            (from, ends) = (0, 0);
        }
        // A carriage return followed by a line feed is counted at the line feed.
        for i in from..at {
            if b[i] == b'\n' || (b[i] == b'\r' && b.get(i + 1) != Some(&b'\n')) {
                ends += 1;
            }
        }
        self.counted.set((at, ends));
        let split_pair = at > 0 && b[at - 1] == b'\r' && b.get(at) == Some(&b'\n');
        ends + 1 + u32::from(split_pair)
    }

    pub fn line(&self) -> u32 {
        self.line_at(self.pos)
    }

    fn err<T>(&self, detail: impl Into<String>) -> Result<T, ReadError> {
        Err(ReadError::Xml { line: self.line(), detail: detail.into() })
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn starts(&self, s: &str) -> bool {
        self.rest().starts_with(s)
    }

    fn skip_space(&mut self) -> bool {
        let b = self.src.as_bytes();
        let from = self.pos;
        while self.pos < b.len() && is_space(b[self.pos]) {
            self.pos += 1;
        }
        self.pos > from
    }

    fn expect(&mut self, s: &str) -> Result<(), ReadError> {
        if self.starts(s) {
            self.pos += s.len();
            Ok(())
        } else {
            self.err(format!("expected {s:?}"))
        }
    }

    fn name(&mut self) -> Result<&'a str, ReadError> {
        let rest = self.rest();
        let mut chars = rest.char_indices();
        match chars.next() {
            Some((_, c)) if is_name_start(c) => {}
            _ => return self.err("expected a name"),
        }
        let end = chars.find(|&(_, c)| !is_name_char(c)).map_or(rest.len(), |(i, _)| i);
        self.pos += end;
        Ok(&rest[..end])
    }

    /// The next event.
    pub fn next(&mut self) -> Result<Event<'a>, ReadError> {
        if self.pending_end {
            self.pending_end = false;
            return Ok(self.close(self.pos));
        }
        loop {
            if !self.started || self.finished {
                // Prolog or epilog: markup and whitespace only.
                self.skip_space();
                if self.pos == self.src.len() {
                    return if self.finished { Ok(Event::Eof) } else { self.err("no root element") };
                }
                if self.starts("<?") {
                    self.pi()?;
                } else if self.starts("<!--") {
                    self.comment()?;
                } else if self.starts("<!DOCTYPE") {
                    if self.started || self.doctype_seen {
                        return self.err("a document type declaration out of place");
                    }
                    self.doctype()?;
                } else if self.starts("<") && !self.finished {
                    self.started = true;
                    return self.start_tag();
                } else if self.finished {
                    return self.err("content after the root element");
                } else {
                    return self.err("text before the root element");
                }
                continue;
            }
            if self.pos == self.src.len() {
                return self.err(format!("element {} is not closed", self.open.last().unwrap_or(&"")));
            }
            if self.starts("</") {
                return self.end_tag();
            } else if self.starts("<!--") {
                self.comment()?;
            } else if self.starts("<?") {
                self.pi()?;
            } else if self.starts("<![CDATA[") {
                self.pos += "<![CDATA[".len();
                let Some(end) = self.rest().find("]]>") else { return self.err("unclosed CDATA section") };
                let raw = &self.rest()[..end];
                self.pos += end + 3;
                self.check_chars(raw)?;
                return Ok(Event::Text(normalize_line_ends(raw)));
            } else if self.starts("<!") {
                return self.err("unexpected markup declaration in content");
            } else if self.starts("<") {
                return self.start_tag();
            } else {
                return self.text();
            }
        }
    }

    fn check_chars(&self, s: &str) -> Result<(), ReadError> {
        match s.chars().find(|&c| !is_xml_char(c)) {
            Some(c) => self.err(format!("character U+{:04X} is not allowed", c as u32)),
            None => Ok(()),
        }
    }

    fn pi(&mut self) -> Result<(), ReadError> {
        let at = self.pos;
        self.pos += 2;
        let target = self.name()?;
        let Some(end) = self.rest().find("?>") else { return self.err("unclosed processing instruction") };
        let body = &self.rest()[..end];
        self.check_chars(body)?;
        if !body.is_empty() && !is_space(body.as_bytes()[0]) {
            return self.err("expected whitespace after a processing instruction's target");
        }
        if target.eq_ignore_ascii_case("xml") {
            if at != 0 || target != "xml" {
                return self.err("an XML declaration out of place");
            }
            self.xml_declaration(body)?;
        }
        self.pos += end + 2;
        Ok(())
    }

    /// Checks the pseudo attributes of the XML declaration: a version, and
    /// an encoding only if it is UTF-8.
    fn xml_declaration(&self, body: &str) -> Result<(), ReadError> {
        let mut rest = body.trim_start_matches([' ', '\t', '\n', '\r']);
        let mut names = Vec::new();
        while !rest.is_empty() {
            let Some(eq) = rest.find('=') else { return self.err("a malformed XML declaration") };
            let name = rest[..eq].trim_end_matches([' ', '\t', '\n', '\r']);
            let after = rest[eq + 1..].trim_start_matches([' ', '\t', '\n', '\r']);
            let Some(q) = after.chars().next().filter(|&q| q == '"' || q == '\'') else {
                return self.err("a malformed XML declaration");
            };
            let Some(close) = after[1..].find(q) else { return self.err("a malformed XML declaration") };
            let value = &after[1..1 + close];
            match name {
                "version" if names.is_empty() && value.starts_with("1.") => {}
                "encoding" if names == ["version"] => {
                    if !value.eq_ignore_ascii_case("UTF-8") && !value.eq_ignore_ascii_case("UTF8") {
                        return self.err(format!("encoding {value} is not read; only UTF-8 is"));
                    }
                }
                "standalone"
                    if (names == ["version"] || names == ["version", "encoding"])
                        && (value == "yes" || value == "no") => {}
                _ => return self.err("a malformed XML declaration"),
            }
            names.push(name);
            rest = after[1 + close + 1..].trim_start_matches([' ', '\t', '\n', '\r']);
        }
        if names.first() != Some(&"version") {
            return self.err("an XML declaration without a version");
        }
        Ok(())
    }

    fn comment(&mut self) -> Result<(), ReadError> {
        self.pos += 4;
        let Some(end) = self.rest().find("--") else { return self.err("unclosed comment") };
        if !self.rest()[end..].starts_with("-->") {
            return self.err("\"--\" inside a comment");
        }
        let body = &self.rest()[..end];
        self.check_chars(body)?;
        self.pos += end + 3;
        Ok(())
    }

    fn quoted(&mut self) -> Result<&'a str, ReadError> {
        let q = match self.src.as_bytes().get(self.pos) {
            Some(&b'"') => '"',
            Some(&b'\'') => '\'',
            _ => return self.err("expected a quoted literal"),
        };
        self.pos += 1;
        let Some(end) = self.rest().find(q) else { return self.err("unclosed literal") };
        let s = &self.rest()[..end];
        self.pos += end + 1;
        Ok(s)
    }

    fn doctype(&mut self) -> Result<(), ReadError> {
        self.doctype_seen = true;
        self.pos += "<!DOCTYPE".len();
        if !self.skip_space() {
            return self.err("expected whitespace after DOCTYPE");
        }
        self.name()?;
        self.skip_space();
        self.external_id(false)?;
        self.skip_space();
        if self.starts("[") {
            self.pos += 1;
            self.internal_subset()?;
            self.skip_space();
        }
        self.expect(">")
    }

    /// An optional `SYSTEM` or `PUBLIC` identifier; whether there was one.
    /// The literals are read and never resolved.
    fn external_id(&mut self, required: bool) -> Result<bool, ReadError> {
        if self.starts("SYSTEM") {
            self.pos += 6;
            if !self.skip_space() {
                return self.err("expected whitespace after SYSTEM");
            }
            self.quoted()?;
        } else if self.starts("PUBLIC") {
            self.pos += 6;
            if !self.skip_space() {
                return self.err("expected whitespace after PUBLIC");
            }
            self.quoted()?;
            if !self.skip_space() {
                return self.err("expected a system literal");
            }
            self.quoted()?;
        } else if required {
            return self.err("expected an entity value or an external identifier");
        } else {
            return Ok(false);
        }
        Ok(true)
    }

    fn internal_subset(&mut self) -> Result<(), ReadError> {
        loop {
            self.skip_space();
            if self.pos == self.src.len() {
                return self.err("unclosed internal subset");
            }
            if self.starts("]") {
                self.pos += 1;
                return Ok(());
            } else if self.starts("%") {
                self.pos += 1;
                self.name()?;
                self.expect(";")?;
            } else if self.starts("<!--") {
                self.comment()?;
            } else if self.starts("<?") {
                self.pi()?;
            } else if self.starts("<!ENTITY") {
                self.entity_declaration()?;
            } else if self.starts("<!ELEMENT") || self.starts("<!ATTLIST") || self.starts("<!NOTATION") {
                self.skip_declaration()?;
            } else {
                return self.err("unexpected content in the internal subset");
            }
        }
    }

    /// Skips a declaration up to its closing `>`, past quoted literals.
    fn skip_declaration(&mut self) -> Result<(), ReadError> {
        let b = self.src.as_bytes();
        let mut quote = None;
        while self.pos < b.len() {
            let c = b[self.pos];
            self.pos += 1;
            match (quote, c) {
                (None, b'"' | b'\'') => quote = Some(c),
                (Some(q), _) if q == c => quote = None,
                (None, b'>') => return Ok(()),
                _ => {}
            }
        }
        self.err("unclosed markup declaration")
    }

    fn entity_declaration(&mut self) -> Result<(), ReadError> {
        self.pos += "<!ENTITY".len();
        if !self.skip_space() {
            return self.err("expected whitespace after ENTITY");
        }
        let parameter = self.starts("%");
        if parameter {
            self.pos += 1;
            if !self.skip_space() {
                return self.err("expected whitespace after %");
            }
        }
        let line = self.line();
        let name = self.name()?;
        if !self.skip_space() {
            return self.err("expected whitespace after the entity name");
        }
        if matches!(self.src.as_bytes().get(self.pos), Some(&b'"') | Some(&b'\'')) {
            let raw = self.quoted()?;
            let value = self.entity_value(raw)?;
            if !parameter {
                self.entities.entry(name.to_string()).or_insert(value);
            }
        } else {
            self.external_id(true)?;
            return Err(ReadError::ExternalEntity { line, name: name.to_string() });
        }
        self.skip_space();
        self.expect(">")
    }

    /// An entity value literal with its character references replaced;
    /// general entity references stay, to be expanded where it is used.
    fn entity_value(&self, raw: &str) -> Result<String, ReadError> {
        self.check_chars(raw)?;
        let mut out = String::with_capacity(raw.len());
        let mut rest = raw;
        while let Some(i) = rest.find(['&', '%']) {
            out.push_str(&normalize_line_ends(&rest[..i]));
            rest = &rest[i..];
            if rest.starts_with('%') {
                return self.err("a parameter entity reference in an entity value");
            }
            let Some(semi) = rest.find(';') else { return self.err("an unterminated reference") };
            let body = &rest[1..semi];
            if let Some(num) = body.strip_prefix('#') {
                out.push(self.char_ref(num)?);
            } else {
                if !valid_name(body) {
                    return self.err("a malformed entity reference");
                }
                out.push_str(&rest[..=semi]);
            }
            rest = &rest[semi + 1..];
        }
        out.push_str(&normalize_line_ends(rest));
        Ok(out)
    }

    fn char_ref(&self, num: &str) -> Result<char, ReadError> {
        let value = match num.strip_prefix('x') {
            Some(hex) if !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()) => {
                u32::from_str_radix(hex, 16).ok()
            }
            None if !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit()) => num.parse::<u32>().ok(),
            _ => None,
        };
        match value.and_then(char::from_u32).filter(|&c| is_xml_char(c)) {
            Some(c) => Ok(c),
            None => self.err(format!("character reference &#{num}; is not a character XML allows")),
        }
    }

    /// Appends the replacement of the reference `&body;` to `out`.
    fn reference(&mut self, body: &str, out: &mut String, depth: usize, in_attribute: bool) -> Result<(), ReadError> {
        if let Some(num) = body.strip_prefix('#') {
            out.push(self.char_ref(num)?);
            return Ok(());
        }
        let predefined = match body {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "apos" => Some('\''),
            "quot" => Some('"'),
            _ => None,
        };
        if let Some(c) = predefined {
            out.push(c);
            return Ok(());
        }
        if !valid_name(body) {
            return self.err("a malformed entity reference");
        }
        let Some(value) = self.entities.get(body).cloned() else {
            return self.err(format!("entity {body} is not declared"));
        };
        self.expansions += 1;
        if self.expansions > MAX_EXPANSIONS || depth >= MAX_ENTITY_DEPTH {
            return self.err("too many entity expansions");
        }
        if value.contains('<') {
            return self.err(format!("entity {body} holds markup, which this reader does not parse"));
        }
        self.expand(&value, out, depth + 1, in_attribute)
    }

    /// Appends `s` with its references expanded: text as written when
    /// `depth` is 0, else an entity's replacement text. In an attribute
    /// value, its whitespace becomes spaces.
    fn expand(&mut self, s: &str, out: &mut String, depth: usize, in_attribute: bool) -> Result<(), ReadError> {
        let mut rest = s;
        while let Some(i) = rest.find('&') {
            self.push_text(out, &rest[..i], depth, in_attribute)?;
            rest = &rest[i..];
            let Some(semi) = rest.find(';') else { return self.err("an unterminated reference") };
            let body = &rest[1..semi];
            self.reference(body, out, depth, in_attribute)?;
            rest = &rest[semi + 1..];
        }
        self.push_text(out, rest, depth, in_attribute)
    }

    /// Appends `s`, text between references. Line ends are normalized in
    /// text as written; an entity's replacement text had its line ends
    /// normalized when it was declared, so a carriage return in it came
    /// from a character reference and stays. In an attribute value each
    /// tab, line feed or carriage return becomes a space. The text that
    /// entities add counts against [`MAX_EXPANDED_BYTES`].
    fn push_text(&mut self, out: &mut String, s: &str, depth: usize, in_attribute: bool) -> Result<(), ReadError> {
        if depth > 0 {
            self.expanded_bytes += s.len();
            if self.expanded_bytes > MAX_EXPANDED_BYTES {
                return self.err(format!("entity expansions add more than {MAX_EXPANDED_BYTES} bytes"));
            }
        }
        let s = if depth == 0 { normalize_line_ends(s) } else { Cow::Borrowed(s) };
        if in_attribute {
            out.extend(s.chars().map(|c| if matches!(c, '\t' | '\n' | '\r') { ' ' } else { c }));
        } else {
            out.push_str(&s);
        }
        Ok(())
    }

    fn text(&mut self) -> Result<Event<'a>, ReadError> {
        let rest = self.rest();
        let end = rest.find('<').unwrap_or(rest.len());
        let raw = &rest[..end];
        if raw.contains("]]>") {
            return self.err("\"]]>\" in text");
        }
        self.check_chars(raw)?;
        let out = if raw.contains('&') {
            let mut out = String::with_capacity(raw.len());
            self.expand(raw, &mut out, 0, false)?;
            Cow::Owned(out)
        } else {
            normalize_line_ends(raw)
        };
        self.pos += end;
        Ok(Event::Text(out))
    }

    fn attribute_value(&mut self) -> Result<String, ReadError> {
        let raw = self.quoted()?;
        if raw.contains('<') {
            return self.err("\"<\" in an attribute value");
        }
        self.check_chars(raw)?;
        let mut out = String::with_capacity(raw.len());
        self.expand(raw, &mut out, 0, true)?;
        Ok(out)
    }

    fn start_tag(&mut self) -> Result<Event<'a>, ReadError> {
        let line = self.line();
        self.pos += 1;
        let qname = self.name()?;
        let mut attributes: Vec<(&'a str, String)> = Vec::new();
        let mut names = Distinct::default();
        let empty = loop {
            let spaced = self.skip_space();
            if self.starts("/>") {
                self.pos += 2;
                break true;
            }
            if self.starts(">") {
                self.pos += 1;
                break false;
            }
            if !spaced {
                return self.err("expected whitespace between attributes");
            }
            let name = self.name()?;
            self.skip_space();
            self.expect("=")?;
            self.skip_space();
            let value = self.attribute_value()?;
            if !names.insert(name) {
                return self.err(format!("attribute {name} appears twice"));
            }
            attributes.push((name, value));
        };
        self.marks.push(self.bound.len());
        for (name, value) in &attributes {
            let prefix = if *name == "xmlns" {
                ""
            } else if let Some(prefix) = name.strip_prefix("xmlns:") {
                if value.is_empty() {
                    return self.err(format!("prefix {prefix} is bound to no namespace"));
                }
                prefix
            } else {
                continue;
            };
            // Namespaces in XML: `xml` is bound to its namespace only and
            // only it is, `xmlns` and its namespace are bound to nothing.
            if prefix == "xmlns" || value == XMLNS_NS || (prefix == "xml") != (value == XML_NS) {
                return self.err(format!("{name} binds a reserved prefix or namespace"));
            }
            self.bindings.entry(prefix).or_default().push(value.clone());
            self.bound.push(prefix);
        }
        let (prefix, local) = self.split(qname)?;
        if prefix == Some("xmlns") {
            return self.err("an element may not have the xmlns prefix");
        }
        let ns = match prefix {
            None => self.resolve("").map(str::to_string),
            Some(p) => match self.resolve(p) {
                Some(uri) => Some(uri.to_string()),
                None => return self.err(format!("the prefix {p} of element {qname} is not bound")),
            },
        };
        let known = ns.as_deref().is_none_or(|n| n.is_empty() || n == self.known_ns);
        let mut id = None;
        let mut lang = None;
        let mut expanded = Distinct::default();
        for (name, value) in attributes {
            if name == "xmlns" || name.starts_with("xmlns:") {
                continue;
            }
            let (p, l) = self.split(name)?;
            let uri = match p {
                None => None,
                Some(p) => match self.resolve(p) {
                    Some(u) => Some(u),
                    None => return self.err(format!("the prefix {p} of attribute {name} is not bound")),
                },
            };
            if !expanded.insert((uri, l)) {
                return self.err(format!("attribute {name} appears twice"));
            }
            match (uri, l) {
                (None, "id") => id = Some(value),
                (Some(XML_NS), "lang") => lang = Some(value),
                _ => {}
            }
        }
        self.open.push(qname);
        self.pending_end = empty;
        Ok(Event::Start(Start { qname, local, known, id, lang, line, content_start: self.base + self.pos }))
    }

    fn split(&self, qname: &'a str) -> Result<(Option<&'a str>, &'a str), ReadError> {
        match qname.split_once(':') {
            None => Ok((None, qname)),
            Some((p, l)) if !p.is_empty() && !l.is_empty() && !l.contains(':') && valid_name(l) => Ok((Some(p), l)),
            _ => self.err(format!("{qname} is not a qualified name")),
        }
    }

    fn resolve(&self, prefix: &str) -> Option<&str> {
        if prefix == "xml" {
            return Some(XML_NS);
        }
        self.bindings.get(prefix).and_then(|uris| uris.last()).map(String::as_str)
    }

    fn end_tag(&mut self) -> Result<Event<'a>, ReadError> {
        let at = self.pos;
        self.pos += 2;
        let name = self.name()?;
        self.skip_space();
        self.expect(">")?;
        if self.open.last() != Some(&name) {
            return self.err(format!("end tag {name} does not match the open element"));
        }
        Ok(self.close(at))
    }

    fn close(&mut self, at: usize) -> Event<'a> {
        self.open.pop();
        if let Some(mark) = self.marks.pop() {
            for prefix in self.bound.drain(mark..) {
                if let Some(uris) = self.bindings.get_mut(prefix) {
                    uris.pop();
                }
            }
        }
        if self.open.is_empty() {
            self.finished = true;
        }
        Event::End { at: self.base + at }
    }
}

/// The names of one tag's attributes, to find one given twice: a scan
/// while there are few, a hash set once there are many, so a tag with
/// thousands of attributes is not checked in quadratic time.
struct Distinct<T> {
    list: Vec<T>,
    set: HashSet<T>,
}

impl<T> Default for Distinct<T> {
    fn default() -> Distinct<T> {
        Distinct { list: Vec::new(), set: HashSet::new() }
    }
}

impl<T: Copy + Eq + Hash> Distinct<T> {
    /// The most names checked by a scan.
    const SCAN: usize = 16;

    /// Adds `name`; false when it was there already.
    fn insert(&mut self, name: T) -> bool {
        if self.list.len() < Self::SCAN {
            if self.list.contains(&name) {
                return false;
            }
            self.list.push(name);
            return true;
        }
        if self.set.is_empty() {
            self.set.extend(self.list.iter().copied());
        }
        self.set.insert(name)
    }
}

fn valid_name(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(is_name_start) && chars.all(is_name_char)
}

/// `s` with each carriage return, alone or before a line feed, as one
/// line feed.
fn normalize_line_ends(s: &str) -> Cow<'_, str> {
    if !s.contains('\r') {
        return Cow::Borrowed(s);
    }
    Cow::Owned(s.replace("\r\n", "\n").replace('\r', "\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS: &str = "urn:test";

    /// Every event as text: `<local known id lang>`, `</>`, the text.
    fn events(src: &str) -> Result<Vec<String>, ReadError> {
        let mut x = Xml::new(src, 0, NS);
        let mut out = Vec::new();
        loop {
            match x.next()? {
                Event::Start(s) => out.push(format!("<{} {} {:?} {:?}>", s.local, s.known, s.id, s.lang)),
                Event::End { .. } => out.push("</>".into()),
                Event::Text(t) => out.push(t.into_owned()),
                Event::Eof => return Ok(out),
            }
        }
    }

    #[test]
    fn namespaces_attributes_and_references() {
        let src = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<!DOCTYPE r [<!ENTITY e \"x&#38;#38;y\"><!ELEMENT r ANY>]>\
                   <r xmlns=\"urn:test\" xmlns:o=\"urn:other\"><o:a id=\"1\" xml:lang=\"en\t \"/>\
                   <b>a&amp;b&e;<![CDATA[<c>]]>\r\nd</b></r>";
        let got = events(src).unwrap();
        assert_eq!(
            got,
            [
                "<r true None None>",
                "<a false Some(\"1\") Some(\"en  \")>",
                "</>",
                "<b true None None>",
                "a&bx&y",
                "<c>",
                "\nd",
                "</>",
                "</>"
            ]
        );
    }

    #[test]
    fn refuses_what_is_not_well_formed() {
        let bad = [
            "",
            "<a>",
            "<a></b>",
            "<a/><b/>",
            "<a x='1' x='2'/>",
            "<a x='<'/>",
            "<p:a/>",
            "<a>&nope;</a>",
            "<a>&#0;</a>",
            "<a>]]></a>",
            "<a><!-- a -- b --></a>",
            "<?xml version='1.0' encoding='ISO-8859-1'?><a/>",
            "<a/><?xml version='1.0'?>",
            "<!DOCTYPE a [<!ENTITY e '<b/>'>]><a>&e;</a>",
            "<!DOCTYPE a [<!ENTITY e '&e;'>]><a>&e;</a>",
            "<a xmlns:p='u' p:x='1' xmlns:q='u' q:x='2'/>",
            "<a xmlns:xml='urn:x'/>",
            "<a xmlns:p='http://www.w3.org/XML/1998/namespace'/>",
            "<a xmlns='http://www.w3.org/XML/1998/namespace'/>",
            "<a xmlns:xmlns='urn:x'/>",
            "<a xmlns:p='http://www.w3.org/2000/xmlns/'/>",
            "text<a/>",
            "<a/>text",
        ];
        for src in bad {
            assert!(events(src).is_err(), "{src:?}");
        }
        assert!(events("<a xmlns:xml='http://www.w3.org/XML/1998/namespace'/>").is_ok());
        let err = events("<!DOCTYPE a [\n<!ENTITY e SYSTEM 'file:///etc/passwd'>]><a/>").unwrap_err();
        assert_eq!(err, ReadError::ExternalEntity { line: 2, name: "e".into() });
    }

    #[test]
    fn lines_count_on_and_back() {
        let src = "<a>\r\n<b/>\r<c/>\n\n<d/></a>";
        let x = Xml::new(src, 0, NS);
        let want = |at: usize| {
            let b = &src.as_bytes()[..at];
            1 + (0..at).filter(|&i| b[i] == b'\n' || (b[i] == b'\r' && b.get(i + 1) != Some(&b'\n'))).count() as u32
        };
        for at in (0..=src.len()).chain([4, 0, src.len(), 3, 5]) {
            assert_eq!(x.line_at(at), want(at), "{at}");
        }
        let mut x = Xml::new(src, 0, NS);
        let mut lines = Vec::new();
        loop {
            match x.next().unwrap() {
                Event::Start(s) => lines.push((s.local, s.line)),
                Event::Eof => break,
                _ => {}
            }
        }
        assert_eq!(lines, [("a", 1), ("b", 2), ("c", 3), ("d", 5)]);
    }

    #[test]
    fn many_attributes_and_nested_bindings() {
        let mut src = String::from("<a");
        for i in 0..40 {
            src.push_str(&format!(" x{i}='{i}'"));
        }
        assert!(events(&format!("{src}/>")).is_ok());
        let err = events(&format!("{src} x17='again'/>")).unwrap_err();
        assert!(err.to_string().contains("x17 appears twice"), "{err}");
        let mut src = String::from("<a");
        for i in 0..40 {
            src.push_str(&format!(" xmlns:p{i}='u' p{i}:x='{i}'"));
        }
        let err = events(&format!("{src}/>")).unwrap_err();
        assert!(err.to_string().contains("p1:x appears twice"), "{err}");
        // A prefix bound again inside, then out of scope again.
        let src = "<a xmlns:p='urn:test'><p:b xmlns:p='urn:other'><p:c/></p:b><p:d/></a>";
        assert_eq!(
            events(src).unwrap(),
            [
                "<a true None None>",
                "<b false None None>",
                "<c false None None>",
                "</>",
                "</>",
                "<d true None None>",
                "</>",
                "</>"
            ]
        );
        assert!(events("<a><p:b xmlns:p='u'/><p:c/></a>").is_err());
    }

    #[test]
    fn carriage_returns_from_references_stay() {
        let src = "<!DOCTYPE a [<!ENTITY e 'x&#13;y\r\nz'>]><a v='&e;'>&e;&#13;\r\n</a>";
        let mut x = Xml::new(src, 0, NS);
        let Event::Start(start) = x.next().unwrap() else { panic!() };
        assert!(start.id.is_none());
        let mut text = String::new();
        while let Event::Text(t) = x.next().unwrap() {
            text.push_str(&t);
        }
        assert_eq!(text, "x\ry\nz\r\n");
        let mut x = Xml::new("<!DOCTYPE a [<!ENTITY e 'x&#13;&#10;y'>]><a id='&e;&#13;'/>", 0, NS);
        let Event::Start(start) = x.next().unwrap() else { panic!() };
        assert_eq!(start.id.as_deref(), Some("x  y\r"));
    }

    #[test]
    fn expansions_stop_at_a_total_size() {
        let big = "x".repeat(1 << 20);
        let mut src = format!("<!DOCTYPE a [<!ENTITY b '{big}'>]><a>");
        for _ in 0..60 {
            src.push_str("&b;");
        }
        src.push_str("</a>");
        let err = events(&src).unwrap_err();
        assert!(err.to_string().contains("entity expansions add more than"), "{err}");
        src.replace_range(src.len() - 4 - 3 * 20.., "</a>");
        assert!(events(&src).is_ok());
    }

    #[test]
    fn the_billion_laughs_stop() {
        let mut src = String::from("<!DOCTYPE a [<!ENTITY l0 'ha'>");
        for i in 1..10 {
            let prev = format!("&l{};", i - 1);
            src.push_str(&format!("<!ENTITY l{i} '{}'>", prev.repeat(10)));
        }
        src.push_str("]><a>&l9;</a>");
        let err = events(&src).unwrap_err();
        assert!(err.to_string().contains("too many entity expansions"), "{err}");
    }
}
