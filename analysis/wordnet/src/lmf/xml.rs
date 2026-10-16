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

    /// Skips a `<!DOCTYPE ...>` without reading it. Only one is legal, before the root element.
    /// Inside the internal subset, quoted literals, comments, and processing instructions are
    /// opaque, so a `]`, `>`, or quote inside them does not end the declaration.
    /// XML 1.0 section 2.2: every literal character of the document must match `Char`, the same
    /// production [`character_reference`] applies to `&#...;`.
    fn check_chars(&mut self) -> Result<(), WordNetError> {
        self.chars_checked = true;
        let rest = &self.input[self.pos..];
        if let Some((at, ch)) = rest.char_indices().find(|&(_, ch)| !is_xml_char(ch)) {
            let line = self.line + rest[..at].bytes().filter(|&b| b == b'\n').count();
            return Err(self.err(line, format!("Illegal XML character U+{:04X}", u32::from(ch))));
        }
        Ok(())
    }

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

/// Decodes the body of a numeric character reference (`#65` or `#x41`, without `&` and `;`).
///
/// XML 1.0 section 4.1 allows only ASCII digits (no sign, no blanks) after `#` or a lowercase `#x`,
/// and the result must match the `Char` production of section 2.2.
fn character_reference(entity: &str, resource_name: &str, line: usize) -> Result<char, WordNetError> {
    let (digits, radix) = match entity.strip_prefix("#x") {
        Some(hex) => (hex, 16),
        None => (&entity[1..], 10),
    };
    let well_formed = !digits.is_empty() && digits.chars().all(|c| c.is_digit(radix));
    let code = u32::from_str_radix(digits, radix)
        .ok()
        .filter(|_| well_formed)
        .ok_or_else(|| malformed(resource_name, Some(line), format!("Invalid character reference '&{entity};'")))?;
    char::from_u32(code).filter(|&ch| is_xml_char(ch)).ok_or_else(|| {
        malformed(resource_name, Some(line), format!("Character reference '&{entity};' is not a legal XML character"))
    })
}

/// XML 1.0 section 2.2 `Char`: tab, newline, carriage return, and every scalar value from
/// U+0020 except the surrogates (already excluded by [`char`]) and U+FFFE and U+FFFF.
fn is_xml_char(ch: char) -> bool {
    matches!(ch, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..)
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
                _ if entity.starts_with('#') => out.push(character_reference(entity, resource_name, line)?),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn events(input: &str) -> Result<Vec<XmlEvent>, WordNetError> {
        let mut parser = XmlPullParser::new(input, "test.xml");
        let mut out = Vec::new();
        loop {
            let event = parser.next_event()?;
            if let XmlEvent::Eof { .. } = event {
                return Ok(out);
            }
            out.push(event);
        }
    }

    fn start_names(events: &[XmlEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                XmlEvent::Start { local_name, .. } => Some(local_name.as_str()),
                _ => None,
            })
            .collect()
    }

    fn text(input: &str) -> Result<String, WordNetError> {
        unescape_xml(input, "test.xml", 1, false)
    }

    fn assert_err_contains(result: Result<impl std::fmt::Debug, WordNetError>, fragment: &str) {
        let err = result.expect_err("expected a parse error");
        assert!(err.to_string().contains(fragment), "expected '{fragment}' in '{err}'");
    }

    #[test]
    fn numeric_character_references_decode_decimal_and_hex() {
        assert_eq!("A", text("&#65;").unwrap());
        assert_eq!("A", text("&#x41;").unwrap());
        assert_eq!("\u{e9}", text("&#xe9;").unwrap());
        assert_eq!("\u{1F600}", text("&#x1F600;").unwrap());
        assert_eq!("\u{1F600}", text("&#128512;").unwrap());
        assert_eq!("\t\n\r", text("&#9;&#xA;&#13;").unwrap());
    }

    #[test]
    fn hex_character_references_need_a_lowercase_x() {
        assert_err_contains(text("&#X41;"), "Invalid character reference '&#X41;'");
        assert_eq!("\u{AB}", text("&#xAb;").unwrap(), "hex digits themselves may be either case");
    }

    #[test]
    fn literal_characters_outside_the_xml_char_production_are_rejected() {
        for (doc, line) in [
            ("<r>a\u{0}b</r>", 1),
            ("<r>\n\n\u{1}</r>", 3),
            ("<r a=\"\u{8}\"/>", 1),
            ("<r><!-- \u{1F} --></r>", 1),
            ("<r><![CDATA[\u{B}]]></r>", 1),
            ("<r>\u{FFFE}</r>", 1),
            ("\u{C}<r/>", 1),
        ] {
            assert_err_contains(events(doc), &format!("line {line}: Illegal XML character"));
        }
        assert!(events("\u{feff}<r a=\"\t\">\t\n\u{D7FF}\u{E000}\u{FFFD}\u{10FFFF}</r>").is_ok());
    }

    #[test]
    fn numeric_character_references_reject_signs_and_empty_digits() {
        for reference in ["&#+65;", "&#x+41;", "&#-65;", "&#x-41;", "&#;", "&#x;", "&# 65;", "&#x 41;", "&#65 ;"] {
            assert!(text(reference).is_err(), "{reference} must be rejected");
        }
    }

    #[test]
    fn numeric_character_references_reject_code_points_outside_the_xml_char_production() {
        for reference in ["&#0;", "&#x0;", "&#x1;", "&#8;", "&#xB;", "&#xC;", "&#x1F;", "&#xFFFE;", "&#xFFFF;"] {
            assert_err_contains(text(reference), "not a legal XML character");
        }
        for reference in ["&#xD800;", "&#xDFFF;", "&#x110000;", "&#99999999999;"] {
            assert!(text(reference).is_err(), "{reference} must be rejected");
        }
        for (reference, expected) in [
            ("&#x20;", '\u{20}'),
            ("&#xD7FF;", '\u{D7FF}'),
            ("&#xE000;", '\u{E000}'),
            ("&#xFFFD;", '\u{FFFD}'),
            ("&#x10000;", '\u{10000}'),
            ("&#x10FFFF;", '\u{10FFFF}'),
        ] {
            assert_eq!(expected.to_string(), text(reference).unwrap(), "{reference}");
        }
    }

    #[test]
    fn unknown_and_unterminated_entity_references_are_rejected() {
        assert_err_contains(text("&nbsp;"), "Unknown XML entity reference '&nbsp;'");
        assert_err_contains(text("fish & chips"), "Unescaped '&'");
        assert_err_contains(text("&amp"), "Unescaped '&'");
    }

    #[test]
    fn multibyte_text_survives_next_to_references_and_attribute_normalization() {
        assert_eq!("caf\u{e9}\u{e9}&\u{fc}", text("caf&#xE9;\u{e9}&amp;\u{fc}").unwrap());
        let attr = unescape_xml("\u{65e5}\t\u{672c}\n&#9;\u{8a9e}", "test.xml", 1, true).unwrap();
        assert_eq!("\u{65e5} \u{672c} \t\u{8a9e}", attr);
        let events = events("<r a=\"\u{1F600}&lt;\u{e9}\">\u{4e2d}&gt;\u{6587}</r>").unwrap();
        let XmlEvent::Start { attrs, .. } = &events[0] else { panic!("expected start") };
        assert_eq!("\u{1F600}<\u{e9}", attrs[0].value);
        let XmlEvent::Text(body) = &events[1] else { panic!("expected text") };
        assert_eq!("\u{4e2d}>\u{6587}", body);
    }

    #[test]
    fn internal_entities_are_never_expanded() {
        let mut doctype = String::from("<!DOCTYPE r [\n<!ENTITY lol0 \"lol\">\n");
        for i in 1..10 {
            let prev = format!("&lol{};", i - 1);
            doctype.push_str(&format!("<!ENTITY lol{i} \"{}\">\n", prev.repeat(10)));
        }
        doctype.push_str("]>\n");
        let doc = format!("{doctype}<r>&lol9;</r>");
        assert_err_contains(events(&doc), "Unknown XML entity reference '&lol9;'");
        let attr_doc = format!("{doctype}<r a=\"&lol9;\"/>");
        assert_err_contains(events(&attr_doc), "Unknown XML entity reference '&lol9;'");
    }

    #[test]
    fn doctype_internal_subset_may_hold_comments_and_processing_instructions_with_quotes_and_brackets() {
        for subset in [
            "<!-- don't panic -->",
            "<!-- a ] in a comment -->",
            "<!-- a ]> in a comment -->",
            "<?pi it's \"odd\" ]> ?>",
            "<!ENTITY e \"]>\"> <!-- ' -->",
            "<!ATTLIST r a CDATA '\"]>'>",
            "<!ENTITY \u{e9} \"\u{65e5}\u{1F600}\"> <!-- \u{e9}' --> <?\u{e9} ?>",
        ] {
            let doc = format!("<!DOCTYPE r [ {subset} ]>\n<r a='x'>it's</r>");
            let events = events(&doc).unwrap_or_else(|e| panic!("{subset}: {e}"));
            assert_eq!(vec!["r"], start_names(&events), "{subset}");
            let XmlEvent::Text(body) = &events[1] else { panic!("expected text for {subset}") };
            assert_eq!("it's", body);
        }
    }

    #[test]
    fn doctype_is_only_legal_once_before_the_root_element() {
        assert_err_contains(events("<r><!DOCTYPE r></r>"), "DOCTYPE");
        assert_err_contains(events("<r/><!DOCTYPE r>"), "DOCTYPE");
        assert_err_contains(events("<!DOCTYPE r><!DOCTYPE r><r/>"), "DOCTYPE");
        assert!(events("<?xml version=\"1.0\"?>\n<!-- c -->\n<!DOCTYPE r SYSTEM \"r.dtd\">\n<r/>").is_ok());
    }

    #[test]
    fn markup_declarations_outside_a_doctype_are_rejected() {
        assert_err_contains(events("<!ELEMENT r ANY><r/>"), "<!ELEMENT");
        assert_err_contains(events("<r><!ENTITY e \"x\"></r>"), "<!ENTITY");
        assert_err_contains(events("<r><![IGNORE[ x ]]></r>"), "<![IGNORE");
    }

    #[test]
    fn a_second_root_element_is_rejected() {
        assert_err_contains(events("<r/><s/>"), "second root element <s>");
        assert_err_contains(events("<r></r>\n<r></r>"), "second root element <r>");
        assert!(events("<r/>\n<!-- trailing comment -->\n<?pi trailing?>\n").is_ok());
    }

    #[test]
    fn unclosed_constructs_are_reported_with_their_opening_line() {
        assert_err_contains(events("<r>\n<!-- never closed"), "line 2: Unclosed XML comment");
        assert_err_contains(events("<r>\n\n<?pi never closed"), "line 3: Unclosed XML processing instruction");
        assert_err_contains(events("<r><![CDATA[ never closed"), "Unclosed CDATA section");
        assert_err_contains(events("<!DOCTYPE r [ <!ENTITY e \"x\">"), "Unclosed XML declaration or DOCTYPE");
        assert_err_contains(events("<!DOCTYPE r [ <!-- ]> never closed"), "Unclosed XML comment");
        assert_err_contains(events("<r a=\"open"), "Unclosed attribute value for 'a'");
        assert_err_contains(events("<r a=\"x\""), "Unclosed start tag <r>");
        assert_err_contains(events("<r>\n<s>\n</s>"), "line 3: Unclosed XML element <r>");
        assert_err_contains(events("<r></s>"), "Mismatched closing tag </s>, expected </r>");
        assert_err_contains(events("<r>]]></r>"), "Unexpected ']]>'");
    }

    #[test]
    fn line_numbers_count_newlines_inside_multibyte_text_comments_and_cdata() {
        let doc = "<r>\n\u{e9}\u{1F600}\n<!-- \n\u{4e2d} -->\n<![CDATA[\n]]>\n<bad attr></r>";
        assert_err_contains(events(doc), "line 7: Expected '=' after attribute 'attr'");
    }

    #[test]
    fn namespaces_resolve_through_ancestors_and_reject_unbound_prefixes() {
        let events = events("<r xmlns=\"urn:d\" xmlns:p=\"urn:p\"><p:c p:a=\"1\" b=\"2\"/><d/></r>").unwrap();
        let resolved: Vec<(&str, &str)> = events
            .iter()
            .filter_map(|e| match e {
                XmlEvent::Start { namespace_uri, local_name, .. } => {
                    Some((namespace_uri.as_str(), local_name.as_str()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(vec![("urn:d", "r"), ("urn:p", "c"), ("urn:d", "d")], resolved);
        let XmlEvent::Start { attrs, .. } = &events[1] else { panic!("expected start") };
        assert_eq!(("urn:p", "a"), (attrs[0].namespace_uri.as_str(), attrs[0].local_name.as_str()));
        assert_eq!(("", "b"), (attrs[1].namespace_uri.as_str(), attrs[1].local_name.as_str()));

        assert_err_contains(self::events("<q:r/>"), "Unbound XML namespace prefix 'q'");
        assert_err_contains(
            self::events("<r xmlns:a=\"urn:x\" xmlns:b=\"urn:x\" a:k=\"1\" b:k=\"2\"/>"),
            "Duplicate expanded attribute",
        );
    }
}
