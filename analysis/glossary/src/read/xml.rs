//! A small XML 1.0 pull parser with namespaces, enough to read a TBX
//! termbase and to refuse what is not well-formed.
//!
//! It reads UTF-8 only, and an XML declaration naming another encoding is
//! refused. A document type declaration is read for its internal general
//! entities, which are expanded in text and attribute values; external
//! entities are never loaded, and declaring one is an error. Parameter
//! entity references in the internal subset are skipped, not expanded,
//! and attribute defaults declared there are not applied. An entity whose
//! replacement text holds markup is refused rather than parsed.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::error::ReadError;

/// The namespace `xml:` is bound to.
const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";
/// The most entity expansions one document may make, as the JDK's
/// `jdk.xml.entityExpansionLimit` default.
const MAX_EXPANSIONS: u32 = 64_000;
/// The deepest one entity may nest others.
const MAX_ENTITY_DEPTH: usize = 64;

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
    /// Namespace bindings in scope, innermost last, and how many there
    /// were when each open element started.
    ns: Vec<(&'a str, String)>,
    marks: Vec<usize>,
    open: Vec<&'a str>,
    entities: HashMap<String, String>,
    doctype_seen: bool,
    started: bool,
    finished: bool,
    pending_end: bool,
    expansions: u32,
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
            ns: Vec::new(),
            marks: Vec::new(),
            open: Vec::new(),
            entities: HashMap::new(),
            doctype_seen: false,
            started: false,
            finished: false,
            pending_end: false,
            expansions: 0,
        }
    }

    /// The line byte `at` is on, counting from 1.
    pub fn line_at(&self, at: usize) -> u32 {
        let at = at.min(self.src.len());
        let b = &self.src.as_bytes()[..at];
        let mut line = 1u32;
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'\n' || (b[i] == b'\r' && b.get(i + 1) != Some(&b'\n')) {
                line += 1;
            }
            i += 1;
        }
        line
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
            out.push_str(&rest[..i]);
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
        out.push_str(rest);
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

    /// Appends `s`, an entity's replacement text, with its references
    /// expanded; in an attribute value, its whitespace becomes spaces.
    fn expand(&mut self, s: &str, out: &mut String, depth: usize, in_attribute: bool) -> Result<(), ReadError> {
        let mut rest = s;
        while let Some(i) = rest.find('&') {
            push_text(out, &rest[..i], in_attribute);
            rest = &rest[i..];
            let Some(semi) = rest.find(';') else { return self.err("an unterminated reference") };
            let body = &rest[1..semi];
            self.reference(body, out, depth, in_attribute)?;
            rest = &rest[semi + 1..];
        }
        push_text(out, rest, in_attribute);
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
            if attributes.iter().any(|(n, _)| *n == name) {
                return self.err(format!("attribute {name} appears twice"));
            }
            attributes.push((name, value));
        };
        self.marks.push(self.ns.len());
        for (name, value) in &attributes {
            if *name == "xmlns" {
                self.ns.push(("", value.clone()));
            } else if let Some(prefix) = name.strip_prefix("xmlns:") {
                if value.is_empty() {
                    return self.err(format!("prefix {prefix} is bound to no namespace"));
                }
                self.ns.push((prefix, value.clone()));
            }
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
        let mut expanded: Vec<(Option<String>, &str)> = Vec::new();
        for (name, value) in attributes {
            if name == "xmlns" || name.starts_with("xmlns:") {
                continue;
            }
            let (p, l) = self.split(name)?;
            let uri = match p {
                None => None,
                Some(p) => match self.resolve(p) {
                    Some(u) => Some(u.to_string()),
                    None => return self.err(format!("the prefix {p} of attribute {name} is not bound")),
                },
            };
            if expanded.iter().any(|(u, n)| *u == uri && *n == l) {
                return self.err(format!("attribute {name} appears twice"));
            }
            match (uri.as_deref(), l) {
                (None, "id") => id = Some(value),
                (Some(XML_NS), "lang") => lang = Some(value),
                _ => {}
            }
            expanded.push((uri, l));
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
        self.ns.iter().rev().find(|(p, _)| *p == prefix).map(|(_, u)| u.as_str())
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
            self.ns.truncate(mark);
        }
        if self.open.is_empty() {
            self.finished = true;
        }
        Event::End { at: self.base + at }
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

/// Appends `s` with line ends normalized; in an attribute value, each
/// tab, line feed or carriage return (a pair as one) becomes a space.
fn push_text(out: &mut String, s: &str, in_attribute: bool) {
    let s = normalize_line_ends(s);
    if in_attribute {
        out.extend(s.chars().map(|c| if matches!(c, '\t' | '\n') { ' ' } else { c }));
    } else {
        out.push_str(&s);
    }
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
            "text<a/>",
            "<a/>text",
        ];
        for src in bad {
            assert!(events(src).is_err(), "{src:?}");
        }
        let err = events("<!DOCTYPE a [\n<!ENTITY e SYSTEM 'file:///etc/passwd'>]><a/>").unwrap_err();
        assert_eq!(err, ReadError::ExternalEntity { line: 2, name: "e".into() });
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
