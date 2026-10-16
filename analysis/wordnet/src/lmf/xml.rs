//! Self-contained namespace-aware XML pull parser with line numbers.
//!
//! It accepts the subset of XML 1.0 that WN-LMF documents use. A `DOCTYPE` is skipped without
//! being read, so no external entity is ever fetched and no internal entity is ever expanded:
//! only the five predefined entities and numeric character references are recognized.

use std::collections::HashSet;

use super::malformed;
use crate::error::WordNetError;

const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";

#[derive(Debug, Clone)]
pub(super) struct XmlAttr {
    pub(super) namespace_uri: String,
    pub(super) local_name: String,
    pub(super) value: String,
}

#[derive(Debug, Clone)]
pub(super) enum XmlEvent {
    Start { namespace_uri: String, local_name: String, attrs: Vec<XmlAttr>, line: usize },
    End { namespace_uri: String, local_name: String, line: usize },
    Text(String),
    Eof { line: usize },
}

type XmlElementFrame = (String, String, String, Vec<(String, String)>);

pub(super) struct XmlPullParser<'a> {
    input: &'a str,
    resource_name: &'a str,
    pos: usize,
    line: usize,
    pending_end: Option<(String, String, usize)>,
    element_stack: Vec<XmlElementFrame>,
}

impl<'a> XmlPullParser<'a> {
    pub(super) fn new(input: &'a str, resource_name: &'a str) -> Self {
        let mut pos = 0;
        if input.starts_with('\u{feff}') {
            pos = '\u{feff}'.len_utf8();
        }
        Self { input, resource_name, pos, line: 1, pending_end: None, element_stack: Vec::new() }
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
        for (_, _, _, bindings) in self.element_stack.iter().rev() {
            for (k, v) in bindings.iter().rev() {
                if k == prefix {
                    return Some(v.clone());
                }
            }
        }
        if prefix.is_empty() { Some(String::new()) } else { None }
    }

    pub(super) fn next_event(&mut self) -> Result<XmlEvent, WordNetError> {
        if let Some((ns, local, line)) = self.pending_end.take() {
            return Ok(XmlEvent::End { namespace_uri: ns, local_name: local, line });
        }

        let bytes = self.input.as_bytes();
        loop {
            if self.pos >= bytes.len() {
                if let Some((raw_qname, _, _, _)) = self.element_stack.last() {
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
                    if !raw_text.trim().is_empty() {
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
            if rem.starts_with("<!DOCTYPE") || rem.starts_with("<!") {
                self.skip_doctype_or_decl(tag_line)?;
                continue;
            }
            if rem.starts_with("</") {
                return self.parse_end_tag(tag_line);
            }
            return self.parse_start_tag(tag_line);
        }
    }

    fn skip_doctype_or_decl(&mut self, start_line: usize) -> Result<(), WordNetError> {
        let bytes = self.input.as_bytes();
        self.advance(2); // skip "<!"
        let mut bracket_depth = 0usize;
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
            match b {
                b'"' | b'\'' => {
                    quote = Some(b);
                    self.advance(1);
                }
                b'[' => {
                    bracket_depth += 1;
                    self.advance(1);
                }
                b']' => {
                    bracket_depth = bracket_depth.saturating_sub(1);
                    self.advance(1);
                }
                b'>' if bracket_depth == 0 => {
                    self.advance(1);
                    return Ok(());
                }
                _ => {
                    self.advance(1);
                }
            }
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

        let Some((expected_qname, ns, local, _)) = self.element_stack.pop() else {
            return Err(self.err(tag_line, format!("Unexpected closing tag </{raw_qname}> with empty element stack")));
        };
        if expected_qname != raw_qname {
            return Err(
                self.err(tag_line, format!("Mismatched closing tag </{raw_qname}>, expected </{expected_qname}>"))
            );
        }
        Ok(XmlEvent::End { namespace_uri: ns, local_name: local, line: tag_line })
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

        let mut raw_attrs: Vec<(String, String)> = Vec::new();
        let mut seen_raw_names: HashSet<String> = HashSet::new();
        let mut self_closing = false;

        loop {
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
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
        }

        let mut ns_bindings: Vec<(String, String)> = Vec::new();
        for (k, v) in &raw_attrs {
            if k == "xmlns" {
                ns_bindings.push((String::new(), v.clone()));
            } else if let Some(prefix) = k.strip_prefix("xmlns:") {
                if prefix.is_empty() {
                    return Err(self.err(tag_line, format!("Empty namespace prefix in '{k}' on <{raw_qname}>")));
                }
                ns_bindings.push((prefix.to_string(), v.clone()));
            }
        }

        self.element_stack.push((raw_qname.clone(), String::new(), String::new(), ns_bindings));

        let (elem_prefix, elem_local) = split_qname(&raw_qname, self.resource_name, tag_line)?;
        let elem_ns = self.resolve_prefix(elem_prefix).ok_or_else(|| {
            self.err(tag_line, format!("Unbound XML namespace prefix '{elem_prefix}' on <{raw_qname}>"))
        })?;
        let elem_local_owned = elem_local.to_string();

        if let Some(top) = self.element_stack.last_mut() {
            top.1 = elem_ns.clone();
            top.2 = elem_local_owned.clone();
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
            self.pending_end = Some((elem_ns.clone(), elem_local_owned.clone(), tag_line));
        }

        Ok(XmlEvent::Start {
            namespace_uri: elem_ns,
            local_name: elem_local_owned,
            attrs: resolved_attrs,
            line: tag_line,
        })
    }
}

fn split_qname<'a>(qname: &'a str, resource_name: &str, line: usize) -> Result<(&'a str, &'a str), WordNetError> {
    if let Some((prefix, local)) = qname.split_once(':') {
        if prefix.is_empty() || local.is_empty() || local.contains(':') {
            return Err(malformed(resource_name, Some(line), format!("Invalid XML QName '{qname}'")));
        }
        Ok((prefix, local))
    } else {
        Ok(("", qname))
    }
}

/// Unescapes XML character/entity references.
/// When `normalize_attr_whitespace` is `true` (per W3C XML 1.0 section 3.3.3), literal
/// `\r\n`, `\r`, `\n`, and `\t` outside character references are normalized to `' '`,
/// whereas numeric character references like `&#9;` expand to their exact character (`\t`).
pub(super) fn unescape_xml(
    raw: &str,
    resource_name: &str,
    line: usize,
    normalize_attr_whitespace: bool,
) -> Result<String, WordNetError> {
    let mut out = String::with_capacity(raw.len());
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if normalize_attr_whitespace && matches!(b, b'\r' | b'\n' | b'\t') {
            if b == b'\r' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            out.push(' ');
            continue;
        }
        if b == b'&' {
            let after_amp = &raw[i + 1..];
            let semi = after_amp.find(';').ok_or_else(|| {
                malformed(resource_name, Some(line), format!("Unescaped '&' or unclosed entity in '{raw}'"))
            })?;
            let entity = &after_amp[..semi];
            match entity {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "apos" => out.push('\''),
                "quot" => out.push('"'),
                _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                    let code = u32::from_str_radix(&entity[2..], 16).map_err(|_| {
                        malformed(resource_name, Some(line), format!("Invalid hex character reference '&{entity};'"))
                    })?;
                    let ch = char::from_u32(code).ok_or_else(|| {
                        malformed(resource_name, Some(line), format!("Invalid Unicode scalar '&{entity};'"))
                    })?;
                    out.push(ch);
                }
                _ if entity.starts_with('#') => {
                    let code = entity[1..].parse::<u32>().map_err(|_| {
                        malformed(
                            resource_name,
                            Some(line),
                            format!("Invalid decimal character reference '&{entity};'"),
                        )
                    })?;
                    let ch = char::from_u32(code).ok_or_else(|| {
                        malformed(resource_name, Some(line), format!("Invalid Unicode scalar '&{entity};'"))
                    })?;
                    out.push(ch);
                }
                _ => {
                    return Err(malformed(
                        resource_name,
                        Some(line),
                        format!("Unknown XML entity reference '&{entity};'"),
                    ));
                }
            }
            i += 1 + semi + 1;
            continue;
        }
        let ch = raw[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}
