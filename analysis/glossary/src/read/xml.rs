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

mod dtd;

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
mod tests;
