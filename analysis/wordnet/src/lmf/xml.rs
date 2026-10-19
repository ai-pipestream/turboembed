//! Self-contained namespace-aware XML pull parser with line numbers.
//!
//! It accepts the subset of XML 1.0 that WN-LMF documents use. A `DOCTYPE` is skipped without
//! being read, so no external entity is ever fetched and no internal entity is ever expanded:
//! only the five predefined entities and numeric character references are recognized.

mod escape;
#[cfg(test)]
mod tests;

use std::collections::HashSet;

pub(super) use self::escape::unescape_xml;
use self::escape::{is_xml_char, split_qname};
use super::malformed;
use crate::error::WordNetError;

const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
const XMLNS_NAMESPACE: &str = "http://www.w3.org/2000/xmlns/";

#[derive(Debug, Clone)]
pub(super) struct XmlAttr {
    pub(super) namespace_uri: String,
    pub(super) local_name: String,
    pub(super) value: String,
}

#[derive(Debug, Clone)]
pub(super) enum XmlEvent {
    Start { namespace_uri: String, local_name: String, attrs: Vec<XmlAttr>, line: usize },
    End { local_name: String, line: usize },
    Text(String),
    Eof { line: usize },
}

/// `(raw_qname, local_name, namespace_bindings)` of an open element.
type XmlElementFrame = (String, String, Vec<(String, String)>);

pub(super) struct XmlPullParser<'a> {
    input: &'a str,
    resource_name: &'a str,
    pos: usize,
    line: usize,
    pending_end: Option<(String, usize)>,
    element_stack: Vec<XmlElementFrame>,
    root_seen: bool,
    doctype_seen: bool,
    chars_checked: bool,
}

impl<'a> XmlPullParser<'a> {
    pub(super) fn new(input: &'a str, resource_name: &'a str) -> Self {
        let mut pos = 0;
        if input.starts_with('\u{feff}') {
            pos = '\u{feff}'.len_utf8();
        }
        Self {
            input,
            resource_name,
            pos,
            line: 1,
            pending_end: None,
            element_stack: Vec::new(),
            root_seen: false,
            doctype_seen: false,
            chars_checked: false,
        }
    }

    fn err(&self, line: usize, msg: impl AsRef<str>) -> WordNetError {
        malformed(self.resource_name, Some(line), msg)
    }

    fn advance(&mut self, n: usize) {
        let slice = &self.input.as_bytes()[self.pos..self.pos + n];
        for &b in slice {
            if b == b'\n' {
                self.line += 1;
            }
        }
        self.pos += n;
    }

    fn resolve_prefix(&self, prefix: &str) -> Option<String> {
        if prefix == "xml" {
            return Some(XML_NAMESPACE.to_string());
        }
        for (_, _, bindings) in self.element_stack.iter().rev() {
            for (k, v) in bindings.iter().rev() {
                if k == prefix {
                    return Some(v.clone());
                }
            }
        }
        if prefix.is_empty() { Some(String::new()) } else { None }
    }

    pub(super) fn next_event(&mut self) -> Result<XmlEvent, WordNetError> {
        if !self.chars_checked {
            self.check_chars()?;
        }
        if let Some((local, line)) = self.pending_end.take() {
            return Ok(XmlEvent::End { local_name: local, line });
        }

        let bytes = self.input.as_bytes();
        loop {
            if self.pos >= bytes.len() {
                if let Some((raw_qname, _, _)) = self.element_stack.last() {
                    return Err(self.err(self.line, format!("Unclosed XML element <{raw_qname}> at end of stream")));
                }
                return Ok(XmlEvent::Eof { line: self.line });
            }

            if bytes[self.pos] != b'<' {
                let start = self.pos;
                let start_line = self.line;
                while self.pos < bytes.len() && bytes[self.pos] != b'<' {
                    if bytes[self.pos] == b'>' && self.pos >= 2 && &bytes[self.pos - 2..=self.pos] == b"]]>" {
                        return Err(self.err(self.line, "Unexpected ']]>' in XML text"));
                    }
                    if bytes[self.pos] == b'\n' {
                        self.line += 1;
                    }
                    self.pos += 1;
                }
                let raw_text = &self.input[start..self.pos];
                if self.element_stack.is_empty() {
                    if !raw_text.trim_ascii().is_empty() {
                        return Err(self.err(start_line, "Unexpected text outside root XML element"));
                    }
                    continue;
                }
                let decoded = unescape_xml(raw_text, self.resource_name, start_line, false)?;
                return Ok(XmlEvent::Text(decoded));
            }

            let rem = &self.input[self.pos..];
            let tag_line = self.line;
            if let Some(after_comment) = rem.strip_prefix("<!--") {
                let end = after_comment.find("-->").ok_or_else(|| self.err(tag_line, "Unclosed XML comment '<!--'"))?;
                let body = &after_comment[..end];
                if body.contains("--") || body.ends_with('-') {
                    return Err(self.err(tag_line, "Illegal '--' inside XML comment"));
                }
                self.advance(4 + end + 3);
                continue;
            }
            if let Some(after_pi) = rem.strip_prefix("<?") {
                let end = after_pi
                    .find("?>")
                    .ok_or_else(|| self.err(tag_line, "Unclosed XML processing instruction '<?'"))?;
                self.advance(2 + end + 2);
                continue;
            }
            if let Some(after_cdata) = rem.strip_prefix("<![CDATA[") {
                if self.element_stack.is_empty() {
                    return Err(self.err(tag_line, "CDATA section outside root element"));
                }
                let end =
                    after_cdata.find("]]>").ok_or_else(|| self.err(tag_line, "Unclosed CDATA section '<![CDATA['"))?;
                let cdata = after_cdata[..end].to_string();
                self.advance(9 + end + 3);
                return Ok(XmlEvent::Text(cdata));
            }
            if rem.starts_with("<!DOCTYPE") {
                self.skip_doctype(tag_line)?;
                continue;
            }
            if rem.starts_with("<!") {
                return Err(self.err(tag_line, format!("Unexpected markup declaration '{}'", declaration_keyword(rem))));
            }
            if rem.starts_with("</") {
                return self.parse_end_tag(tag_line);
            }
            return self.parse_start_tag(tag_line);
        }
    }

    /// XML 1.0 section 2.2: every literal character of the document must match `Char`, the same
    /// production `character_reference` applies to `&#...;`.
    fn check_chars(&mut self) -> Result<(), WordNetError> {
        self.chars_checked = true;
        let rest = &self.input[self.pos..];
        if let Some((at, ch)) = rest.char_indices().find(|&(_, ch)| !is_xml_char(ch)) {
            let line = self.line + rest[..at].bytes().filter(|&b| b == b'\n').count();
            return Err(self.err(line, format!("Illegal XML character U+{:04X}", u32::from(ch))));
        }
        Ok(())
    }

    /// Skips a `<!DOCTYPE ...>` without reading it. Only one is legal, before the root element.
    /// Inside the internal subset, quoted literals, comments, and processing instructions are
    /// opaque, so a `]`, `>`, or quote inside them does not end the declaration.
    fn skip_doctype(&mut self, start_line: usize) -> Result<(), WordNetError> {
        if self.root_seen {
            return Err(self.err(start_line, "DOCTYPE after the root element"));
        }
        if self.doctype_seen {
            return Err(self.err(start_line, "Repeated DOCTYPE"));
        }
        self.doctype_seen = true;
        let bytes = self.input.as_bytes();
        self.advance("<!DOCTYPE".len());
        let mut in_subset = false;
        let mut quote: Option<u8> = None;
        while self.pos < bytes.len() {
            let b = bytes[self.pos];
            if let Some(q) = quote {
                if b == q {
                    quote = None;
                }
                self.advance(1);
                continue;
            }
            // `pos` may sit inside a multibyte character here; only slice at an ASCII `<`.
            let rem = if in_subset && b == b'<' { &self.input[self.pos..] } else { "" };
            if let Some(comment) = rem.strip_prefix("<!--") {
                let line = self.line;
                let end = comment.find("-->").ok_or_else(|| self.err(line, "Unclosed XML comment '<!--'"))?;
                let body = &comment[..end];
                if body.contains("--") || body.ends_with('-') {
                    return Err(self.err(line, "Illegal '--' inside XML comment"));
                }
                self.advance(4 + end + 3);
                continue;
            }
            if let Some(pi) = rem.strip_prefix("<?") {
                let line = self.line;
                let end = pi.find("?>").ok_or_else(|| self.err(line, "Unclosed XML processing instruction '<?'"))?;
                self.advance(2 + end + 2);
                continue;
            }
            match b {
                b'"' | b'\'' => quote = Some(b),
                b'[' if !in_subset => in_subset = true,
                b']' if in_subset => in_subset = false,
                b'>' if !in_subset => {
                    self.advance(1);
                    return Ok(());
                }
                _ => {}
            }
            self.advance(1);
        }
        Err(self.err(start_line, "Unclosed XML declaration or DOCTYPE"))
    }

    fn parse_end_tag(&mut self, tag_line: usize) -> Result<XmlEvent, WordNetError> {
        let bytes = self.input.as_bytes();
        self.advance(2); // skip "</"
        let name_start = self.pos;
        while self.pos < bytes.len() && bytes[self.pos] != b'>' && !bytes[self.pos].is_ascii_whitespace() {
            self.advance(1);
        }
        let raw_qname = &self.input[name_start..self.pos];
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
            self.advance(1);
        }
        if self.pos >= bytes.len() || bytes[self.pos] != b'>' {
            return Err(self.err(tag_line, format!("Malformed closing tag </{raw_qname}>")));
        }
        self.advance(1); // skip '>'

        let Some((expected_qname, local, _)) = self.element_stack.pop() else {
            return Err(self.err(tag_line, format!("Unexpected closing tag </{raw_qname}> with empty element stack")));
        };
        if expected_qname != raw_qname {
            return Err(
                self.err(tag_line, format!("Mismatched closing tag </{raw_qname}>, expected </{expected_qname}>"))
            );
        }
        Ok(XmlEvent::End { local_name: local, line: tag_line })
    }

    fn parse_start_tag(&mut self, tag_line: usize) -> Result<XmlEvent, WordNetError> {
        let bytes = self.input.as_bytes();
        self.advance(1); // skip '<'
        let name_start = self.pos;
        while self.pos < bytes.len()
            && !bytes[self.pos].is_ascii_whitespace()
            && bytes[self.pos] != b'>'
            && bytes[self.pos] != b'/'
        {
            if bytes[self.pos] == b'<' {
                return Err(self.err(self.line, "Unexpected '<' inside element tag"));
            }
            self.advance(1);
        }
        let raw_qname = self.input[name_start..self.pos].to_string();
        if raw_qname.is_empty() {
            return Err(self.err(tag_line, "Empty XML element tag '<>'"));
        }
        if self.element_stack.is_empty() {
            if self.root_seen {
                return Err(self.err(tag_line, format!("Unexpected second root element <{raw_qname}>")));
            }
            self.root_seen = true;
        }

        let mut raw_attrs: Vec<(String, String)> = Vec::new();
        let mut seen_raw_names: HashSet<String> = HashSet::new();
        let mut self_closing = false;
        let mut need_space = false;

        loop {
            let mut had_space = false;
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                had_space = true;
                self.advance(1);
            }
            if self.pos >= bytes.len() {
                return Err(self.err(tag_line, format!("Unclosed start tag <{raw_qname}>")));
            }
            if bytes[self.pos] == b'>' {
                self.advance(1);
                break;
            }
            if self.pos + 1 < bytes.len() && &bytes[self.pos..self.pos + 2] == b"/>" {
                self.advance(2);
                self_closing = true;
                break;
            }
            if need_space && !had_space {
                return Err(self.err(self.line, format!("Expected whitespace between attributes in <{raw_qname}>")));
            }

            let attr_line = self.line;
            let attr_start = self.pos;
            while self.pos < bytes.len()
                && !bytes[self.pos].is_ascii_whitespace()
                && bytes[self.pos] != b'='
                && bytes[self.pos] != b'>'
                && bytes[self.pos] != b'/'
            {
                self.advance(1);
            }
            let attr_name = self.input[attr_start..self.pos].to_string();
            if attr_name.is_empty() {
                return Err(self.err(attr_line, format!("Malformed attribute in <{raw_qname}>")));
            }
            if !seen_raw_names.insert(attr_name.clone()) {
                return Err(self.err(attr_line, format!("Duplicate attribute '{attr_name}' on <{raw_qname}>")));
            }

            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.advance(1);
            }
            if self.pos >= bytes.len() || bytes[self.pos] != b'=' {
                return Err(self.err(attr_line, format!("Expected '=' after attribute '{attr_name}' in <{raw_qname}>")));
            }
            self.advance(1); // skip '='

            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.advance(1);
            }
            if self.pos >= bytes.len() || (bytes[self.pos] != b'"' && bytes[self.pos] != b'\'') {
                return Err(
                    self.err(attr_line, format!("Expected quoted value for attribute '{attr_name}' in <{raw_qname}>"))
                );
            }
            let quote = bytes[self.pos];
            self.advance(1);
            let val_start = self.pos;
            while self.pos < bytes.len() && bytes[self.pos] != quote {
                if bytes[self.pos] == b'<' {
                    return Err(
                        self.err(self.line, format!("Unescaped '<' inside attribute '{attr_name}' in <{raw_qname}>"))
                    );
                }
                self.advance(1);
            }
            if self.pos >= bytes.len() {
                return Err(self.err(attr_line, format!("Unclosed attribute value for '{attr_name}' in <{raw_qname}>")));
            }
            let raw_val = &self.input[val_start..self.pos];
            self.advance(1); // skip closing quote
            let val = unescape_xml(raw_val, self.resource_name, attr_line, true)?;
            raw_attrs.push((attr_name, val));
            need_space = true;
        }

        let mut ns_bindings: Vec<(String, String)> = Vec::new();
        for (k, v) in &raw_attrs {
            if k == "xmlns" {
                if v == XML_NAMESPACE || v == XMLNS_NAMESPACE {
                    return Err(
                        self.err(tag_line, format!("Illegal default namespace binding to '{v}' on <{raw_qname}>"))
                    );
                }
                ns_bindings.push((String::new(), v.clone()));
            } else if let Some(prefix) = k.strip_prefix("xmlns:") {
                if prefix.is_empty() {
                    return Err(self.err(tag_line, format!("Empty namespace prefix in '{k}' on <{raw_qname}>")));
                }
                if prefix.contains(':')
                    || prefix == "xmlns"
                    || v.is_empty()
                    || (prefix == "xml" && v != XML_NAMESPACE)
                    || (prefix != "xml" && (v == XML_NAMESPACE || v == XMLNS_NAMESPACE))
                {
                    return Err(
                        self.err(tag_line, format!("Illegal namespace declaration '{k}=\"{v}\"' on <{raw_qname}>"))
                    );
                }
                ns_bindings.push((prefix.to_string(), v.clone()));
            }
        }

        self.element_stack.push((raw_qname.clone(), String::new(), ns_bindings));

        let (elem_prefix, elem_local) = split_qname(&raw_qname, self.resource_name, tag_line)?;
        let elem_ns = self.resolve_prefix(elem_prefix).ok_or_else(|| {
            self.err(tag_line, format!("Unbound XML namespace prefix '{elem_prefix}' on <{raw_qname}>"))
        })?;
        let elem_local_owned = elem_local.to_string();

        if let Some(top) = self.element_stack.last_mut() {
            top.1 = elem_local_owned.clone();
        }

        let mut resolved_attrs = Vec::with_capacity(raw_attrs.len());
        let mut seen_qnames: HashSet<(String, String)> = HashSet::new();
        for (k, v) in raw_attrs {
            if k == "xmlns" || k.starts_with("xmlns:") {
                continue;
            }
            let (attr_prefix, attr_local) = split_qname(&k, self.resource_name, tag_line)?;
            let attr_ns = if attr_prefix.is_empty() {
                String::new()
            } else {
                self.resolve_prefix(attr_prefix).ok_or_else(|| {
                    self.err(tag_line, format!("Unbound XML namespace prefix '{attr_prefix}' on attribute '{k}'"))
                })?
            };
            if !seen_qnames.insert((attr_ns.clone(), attr_local.to_string())) {
                return Err(self.err(
                    tag_line,
                    format!("Duplicate expanded attribute '{{{attr_ns}}}{attr_local}' on <{raw_qname}>"),
                ));
            }
            resolved_attrs.push(XmlAttr { namespace_uri: attr_ns, local_name: attr_local.to_string(), value: v });
        }

        if self_closing {
            self.element_stack.pop();
            self.pending_end = Some((elem_local_owned.clone(), tag_line));
        }

        Ok(XmlEvent::Start {
            namespace_uri: elem_ns,
            local_name: elem_local_owned,
            attrs: resolved_attrs,
            line: tag_line,
        })
    }
}

/// Names a stray `<!` construct for diagnostics: `<!ELEMENT`, `<![IGNORE`, and so on.
fn declaration_keyword(rem: &str) -> &str {
    let rest = &rem[2..];
    let lead = usize::from(rest.starts_with('['));
    let len = rest[lead..].bytes().take_while(u8::is_ascii_alphanumeric).count();
    &rem[..2 + lead + len]
}
