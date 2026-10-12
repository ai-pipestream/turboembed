//! The XML reader the dictionary format needs, in the style of a SAX
//! parser: it walks the document once and calls a [`Handler`] for each
//! element start, element end and run of character data.
//!
//! It supports elements, attributes in single or double quotes, the five
//! predefined entities, decimal and hexadecimal character references,
//! CDATA sections, comments, processing instructions and the XML
//! declaration (the last three are skipped). Line ends are normalized to
//! `\n` and attribute values have their whitespace normalized, as XML 1.0
//! requires. A namespace prefix is dropped from element and attribute
//! names, and `xmlns` attributes are not reported, which is what a
//! namespace-aware SAX parser hands OpenNLP's dictionary reader.
//!
//! A document type declaration is an error, as it is for OpenNLP, which
//! sets `disallow-doctype-decl`. Other well-formedness errors (mismatched
//! tags, a second root, text outside the root, a character XML does not
//! allow, an undefined entity) are errors naming the byte offset in the
//! document's UTF-8 text.

use std::borrow::Cow;

use crate::error::{Error, Result};

/// One attribute of an element: its local name and its value with
/// references resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Attribute<'a> {
    pub name: &'a str,
    pub value: Cow<'a, str>,
}

/// Receives the document's content in order.
pub(crate) trait Handler {
    /// An element starts. `name` is its local name.
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], offset: usize) -> Result<()>;
    /// An element ends (also called right after `start_element` for an
    /// empty-element tag).
    fn end_element(&mut self, name: &str, offset: usize) -> Result<()>;
    /// Character data inside the root element. One run of text may arrive
    /// in several calls.
    fn characters(&mut self, text: &str);
}

/// Decodes the document's bytes to text: UTF-8 (with or without a byte
/// order mark) unless the XML declaration names US-ASCII or ISO-8859-1.
pub(crate) fn decode(bytes: &[u8]) -> Result<Cow<'_, str>> {
    if bytes.starts_with(&[0xFE, 0xFF]) || bytes.starts_with(&[0xFF, 0xFE]) {
        return Err(Error::XmlEncoding("UTF-16".into()));
    }
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    let encoding = declared_encoding(bytes);
    let utf8 = |bytes| {
        std::str::from_utf8(bytes)
            .map_err(|e| Error::Xml { offset: e.valid_up_to(), message: "the document is not valid UTF-8" })
    };
    match encoding.map(|e| e.to_ascii_uppercase()) {
        None => Ok(Cow::Borrowed(utf8(bytes)?)),
        Some(e) if e == b"UTF-8" || e == b"UTF8" => Ok(Cow::Borrowed(utf8(bytes)?)),
        Some(e) if e == b"US-ASCII" || e == b"ASCII" => match bytes.iter().position(|b| !b.is_ascii()) {
            Some(offset) => Err(Error::Xml { offset, message: "a byte above 0x7F in a US-ASCII document" }),
            None => Ok(Cow::Borrowed(utf8(bytes)?)),
        },
        Some(e) if e == b"ISO-8859-1" || e == b"LATIN1" || e == b"ISO8859-1" || e == b"ISO_8859-1" => {
            Ok(Cow::Owned(bytes.iter().map(|&b| char::from(b)).collect()))
        }
        Some(_) => {
            let name = encoding.expect("matched Some");
            Err(Error::XmlEncoding(String::from_utf8_lossy(name).into()))
        }
    }
}

/// The `encoding` pseudo-attribute of the XML declaration, if the
/// document starts with one that has it.
fn declared_encoding(bytes: &[u8]) -> Option<&[u8]> {
    let rest = bytes.strip_prefix(b"<?xml")?;
    if !rest.first().is_some_and(|b| b.is_ascii_whitespace()) {
        return None;
    }
    let end = rest.windows(2).position(|w| w == b"?>")?;
    let decl = &rest[..end];
    let at = decl.windows(8).position(|w| w == b"encoding")?;
    let mut i = at + 8;
    while decl.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        i += 1;
    }
    if decl.get(i) != Some(&b'=') {
        return None;
    }
    i += 1;
    while decl.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        i += 1;
    }
    let quote = *decl.get(i)?;
    if quote != b'"' && quote != b'\'' {
        return None;
    }
    let value = &decl[i + 1..];
    let close = value.iter().position(|&b| b == quote)?;
    Some(&value[..close])
}

/// Parses a decoded document, calling `handler` for its content.
pub(crate) fn parse(text: &str, handler: &mut impl Handler) -> Result<()> {
    if let Some((offset, _)) = text.char_indices().find(|&(_, c)| !is_xml_char(c)) {
        return Err(Error::Xml { offset, message: "a character XML does not allow" });
    }
    let mut p =
        Parser { text, pos: 0, stack: Vec::new(), attributes: Vec::new(), qualified: Vec::new(), root_seen: false };
    p.document(handler)
}

/// XML 1.0's `Char` production. Rust strings cannot hold surrogates, so
/// only control characters and U+FFFE and U+FFFF are left to refuse.
fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{FFFD}' | '\u{10000}'..)
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

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

/// The local part of a qualified name.
fn local(name: &str) -> &str {
    name.split_once(':').map_or(name, |(_, local)| local)
}

struct Parser<'a> {
    text: &'a str,
    pos: usize,
    /// Qualified names of the open elements.
    stack: Vec<&'a str>,
    /// The attributes of the current start tag that are reported (not
    /// `xmlns` declarations), reused for each start tag.
    attributes: Vec<Attribute<'a>>,
    /// The qualified names of all the current start tag's attributes,
    /// for refusing one given twice.
    qualified: Vec<&'a str>,
    root_seen: bool,
}

impl<'a> Parser<'a> {
    fn err<T>(&self, offset: usize, message: &'static str) -> Result<T> {
        Err(Error::Xml { offset, message })
    }

    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }

    fn document(&mut self, handler: &mut impl Handler) -> Result<()> {
        if self.rest().starts_with("<?xml") && self.rest()[5..].starts_with(|c: char| c.is_ascii_whitespace()) {
            self.skip_past("?>", "the XML declaration is not closed")?;
        }
        while self.pos < self.text.len() {
            let rest = self.rest();
            if rest.starts_with("<!--") {
                self.comment()?;
            } else if rest.starts_with("<?") {
                self.processing_instruction()?;
            } else if rest.starts_with("<!DOCTYPE") {
                return self.err(self.pos, "document type declarations are not allowed");
            } else if rest.starts_with("<![CDATA[") {
                if self.stack.is_empty() {
                    return self.err(self.pos, "a CDATA section outside the root element");
                }
                self.cdata(handler)?;
            } else if rest.starts_with("</") {
                self.end_tag(handler)?;
            } else if rest.starts_with('<') {
                self.start_tag(handler)?;
            } else {
                self.text(handler)?;
            }
        }
        if !self.stack.is_empty() {
            return self.err(self.text.len(), "the document ends inside an element");
        }
        if !self.root_seen {
            return self.err(self.text.len(), "the document has no root element");
        }
        Ok(())
    }

    /// Moves past the next `end`, or fails with `message`.
    fn skip_past(&mut self, end: &str, message: &'static str) -> Result<&'a str> {
        let rest = self.rest();
        match rest.find(end) {
            Some(i) => {
                self.pos += i + end.len();
                Ok(&rest[..i])
            }
            None => self.err(self.pos, message),
        }
    }

    fn comment(&mut self) -> Result<()> {
        let start = self.pos;
        self.pos += 4;
        let body = self.skip_past("-->", "a comment is not closed")?;
        if body.contains("--") || body.ends_with('-') {
            return self.err(start, "a comment contains \"--\"");
        }
        Ok(())
    }

    fn processing_instruction(&mut self) -> Result<()> {
        let start = self.pos;
        self.pos += 2;
        let body = self.skip_past("?>", "a processing instruction is not closed")?;
        let target = body.split(|c: char| (c as u32) < 0x80 && is_space(c as u8)).next().unwrap_or("");
        if target.is_empty() || !target.starts_with(is_name_start) || !target.chars().all(is_name_char) {
            return self.err(start, "a processing instruction has no valid target");
        }
        if target.eq_ignore_ascii_case("xml") {
            return self.err(start, "an XML declaration that is not at the start of the document");
        }
        Ok(())
    }

    fn cdata(&mut self, handler: &mut impl Handler) -> Result<()> {
        self.pos += 9;
        let body = self.skip_past("]]>", "a CDATA section is not closed")?;
        emit_normalized(body, handler);
        Ok(())
    }

    fn name(&mut self) -> Result<&'a str> {
        let rest = self.rest();
        let mut chars = rest.char_indices();
        match chars.next() {
            Some((_, c)) if is_name_start(c) => {}
            _ => return self.err(self.pos, "a name is expected"),
        }
        let end = chars.find(|&(_, c)| !is_name_char(c)).map_or(rest.len(), |(i, _)| i);
        self.pos += end;
        Ok(&rest[..end])
    }

    fn skip_spaces(&mut self) -> bool {
        let start = self.pos;
        while self.text.as_bytes().get(self.pos).is_some_and(|&b| is_space(b)) {
            self.pos += 1;
        }
        self.pos > start
    }

    fn start_tag(&mut self, handler: &mut impl Handler) -> Result<()> {
        let start = self.pos;
        if self.stack.is_empty() && self.root_seen {
            return self.err(start, "a second root element");
        }
        self.pos += 1;
        let name = self.name()?;
        self.attributes.clear();
        self.qualified.clear();
        loop {
            let spaced = self.skip_spaces();
            match self.text.as_bytes().get(self.pos) {
                Some(b'>') => {
                    self.pos += 1;
                    self.open(name, start, handler)?;
                    return Ok(());
                }
                Some(b'/') if self.rest().starts_with("/>") => {
                    self.pos += 2;
                    self.open(name, start, handler)?;
                    self.stack.pop();
                    if self.stack.is_empty() {
                        self.root_seen = true;
                    }
                    return handler.end_element(local(name), start);
                }
                Some(_) if spaced => self.attribute()?,
                Some(_) => return self.err(self.pos, "a start tag needs a space before each attribute"),
                None => return self.err(start, "a start tag is not closed"),
            }
        }
    }

    fn open(&mut self, name: &'a str, start: usize, handler: &mut impl Handler) -> Result<()> {
        self.stack.push(name);
        self.root_seen = true;
        handler.start_element(local(name), &self.attributes, start)
    }

    fn attribute(&mut self) -> Result<()> {
        let at = self.pos;
        let qualified = self.name()?;
        self.skip_spaces();
        if self.text.as_bytes().get(self.pos) != Some(&b'=') {
            return self.err(self.pos, "an attribute name must be followed by '='");
        }
        self.pos += 1;
        self.skip_spaces();
        let quote = match self.text.as_bytes().get(self.pos) {
            Some(&q @ (b'"' | b'\'')) => q,
            _ => return self.err(self.pos, "an attribute value must be quoted"),
        };
        self.pos += 1;
        let value_start = self.pos;
        let Some(len) = self.rest().bytes().position(|b| b == quote) else {
            return self.err(value_start, "an attribute value is not closed");
        };
        self.pos += len + 1;
        let raw = &self.text[value_start..value_start + len];
        if let Some(i) = raw.find('<') {
            return self.err(value_start + i, "'<' in an attribute value");
        }
        if self.qualified.contains(&qualified) {
            return self.err(at, "an attribute is given twice");
        }
        let value = attribute_value(raw, value_start)?;
        self.qualified.push(qualified);
        if qualified != "xmlns" && !qualified.starts_with("xmlns:") {
            self.attributes.push(Attribute { name: local(qualified), value });
        }
        Ok(())
    }

    fn end_tag(&mut self, handler: &mut impl Handler) -> Result<()> {
        let start = self.pos;
        self.pos += 2;
        let name = self.name()?;
        self.skip_spaces();
        if self.text.as_bytes().get(self.pos) != Some(&b'>') {
            return self.err(self.pos, "an end tag is not closed");
        }
        self.pos += 1;
        match self.stack.pop() {
            Some(open) if open == name => handler.end_element(local(name), start),
            Some(_) => self.err(start, "an end tag does not match its start tag"),
            None => self.err(start, "an end tag with no open element"),
        }
    }

    fn text(&mut self, handler: &mut impl Handler) -> Result<()> {
        let start = self.pos;
        let len = self.rest().find('<').unwrap_or(self.rest().len());
        self.pos += len;
        let raw = &self.text[start..start + len];
        if self.stack.is_empty() {
            if let Some(i) = raw.bytes().position(|b| !is_space(b)) {
                return self.err(start + i, "text outside the root element");
            }
            return Ok(());
        }
        if let Some(i) = raw.find("]]>") {
            return self.err(start + i, "\"]]>\" in text");
        }
        let mut rest = raw;
        let mut offset = start;
        while let Some(amp) = rest.find('&') {
            emit_normalized(&rest[..amp], handler);
            let (c, used) = reference(&rest[amp..], offset + amp)?;
            let mut buf = [0u8; 4];
            handler.characters(c.encode_utf8(&mut buf));
            rest = &rest[amp + used..];
            offset += amp + used;
        }
        emit_normalized(rest, handler);
        Ok(())
    }
}

/// Passes text to the handler with `\r\n` and lone `\r` turned into `\n`.
fn emit_normalized(mut text: &str, handler: &mut impl Handler) {
    while let Some(cr) = text.find('\r') {
        if cr > 0 {
            handler.characters(&text[..cr]);
        }
        handler.characters("\n");
        let skip = if text[cr + 1..].starts_with('\n') { 2 } else { 1 };
        text = &text[cr + skip..];
    }
    if !text.is_empty() {
        handler.characters(text);
    }
}

/// Resolves the entity or character reference at the start of `text`,
/// giving the character and the length of the reference.
fn reference(text: &str, offset: usize) -> Result<(char, usize)> {
    let Some(semi) = text.find(';') else {
        return Err(Error::Xml { offset, message: "a reference is not closed by ';'" });
    };
    let body = &text[1..semi];
    let c = match body {
        "lt" => '<',
        "gt" => '>',
        "amp" => '&',
        "apos" => '\'',
        "quot" => '"',
        _ => {
            let value = if let Some(hex) = body.strip_prefix("#x") {
                (!hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()))
                    .then(|| u32::from_str_radix(hex, 16).ok())
                    .flatten()
            } else if let Some(dec) = body.strip_prefix('#') {
                (!dec.is_empty() && dec.bytes().all(|b| b.is_ascii_digit())).then(|| dec.parse::<u32>().ok()).flatten()
            } else {
                return Err(Error::Xml { offset, message: "an entity other than the five XML defines" });
            };
            match value.and_then(char::from_u32).filter(|&c| is_xml_char(c)) {
                Some(c) => c,
                None => {
                    return Err(Error::Xml {
                        offset,
                        message: "a character reference to a character XML does not allow",
                    });
                }
            }
        }
    };
    Ok((c, semi + 1))
}

/// An attribute value with references resolved and each tab, line feed
/// and carriage return (a `\r\n` pair counting as one) replaced by a
/// space. A character reference to whitespace is kept as it is.
fn attribute_value(raw: &str, offset: usize) -> Result<Cow<'_, str>> {
    if !raw.bytes().any(|b| matches!(b, b'&' | b'\t' | b'\n' | b'\r')) {
        return Ok(Cow::Borrowed(raw));
    }
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    let bytes = raw.as_bytes();
    while i < raw.len() {
        match bytes[i] {
            b'&' => {
                let (c, used) = reference(&raw[i..], offset + i)?;
                out.push(c);
                i += used;
            }
            b'\r' => {
                out.push(' ');
                i += if bytes.get(i + 1) == Some(&b'\n') { 2 } else { 1 };
            }
            b'\t' | b'\n' => {
                out.push(' ');
                i += 1;
            }
            _ => {
                let next = raw[i..].find(['&', '\t', '\n', '\r']).map_or(raw.len(), |n| i + n);
                out.push_str(&raw[i..next]);
                i = next;
            }
        }
    }
    Ok(Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the events as text: `<name a=v>`, `</name>`, and text.
    #[derive(Default)]
    struct Log(String);

    impl Handler for Log {
        fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], _: usize) -> Result<()> {
            self.0.push('<');
            self.0.push_str(name);
            for a in attributes {
                self.0.push_str(&format!(" {}={}", a.name, a.value));
            }
            self.0.push('>');
            Ok(())
        }
        fn end_element(&mut self, name: &str, _: usize) -> Result<()> {
            self.0.push_str(&format!("</{name}>"));
            Ok(())
        }
        fn characters(&mut self, text: &str) {
            self.0.push_str(text);
        }
    }

    fn log(xml: &str) -> String {
        let mut l = Log::default();
        parse(xml, &mut l).unwrap();
        l.0
    }

    fn error(xml: &str) -> (usize, &'static str) {
        match parse(xml, &mut Log::default()) {
            Err(Error::Xml { offset, message }) => (offset, message),
            other => panic!("expected an XML error, got {other:?}"),
        }
    }

    #[test]
    fn elements_attributes_and_text() {
        assert_eq!(
            log("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- c --><d a='1' b=\"x y\"><e/>t<?pi data?></d>\n"),
            "<d a=1 b=x y><e></e>t</d>"
        );
    }

    #[test]
    fn references_and_cdata() {
        assert_eq!(
            log("<d>&lt;&gt;&amp;&apos;&quot;&#65;&#x42;&#x1F600;<![CDATA[<&>]]></d>"),
            "<d><>&'\"AB\u{1F600}<&></d>"
        );
    }

    #[test]
    fn line_ends_and_attribute_whitespace_are_normalized() {
        assert_eq!(log("<d a=\"1\r\n2\t3&#10;4\">x\r\ny\rz</d>"), "<d a=1 2 3\n4>x\ny\nz</d>");
    }

    #[test]
    fn prefixes_are_dropped_and_xmlns_is_not_reported() {
        assert_eq!(log("<n:d xmlns:n=\"urn:x\" n:a=\"1\"></n:d>"), "<d a=1></d>");
    }

    #[test]
    fn malformed_documents_name_the_offset() {
        assert_eq!(error("<!DOCTYPE d><d/>"), (0, "document type declarations are not allowed"));
        assert_eq!(error("<d></e>"), (3, "an end tag does not match its start tag"));
        assert_eq!(error("<d/><e/>"), (4, "a second root element"));
        assert_eq!(error("<d/>x"), (4, "text outside the root element"));
        assert_eq!(error("<d>&nbsp;</d>"), (3, "an entity other than the five XML defines"));
        assert_eq!(error("<d>&#0;</d>"), (3, "a character reference to a character XML does not allow"));
        assert_eq!(error("<d>\u{1}</d>"), (3, "a character XML does not allow"));
        assert_eq!(error("<d a='1' a='2'/>"), (9, "an attribute is given twice"));
        assert_eq!(error("<d a='<'/>"), (6, "'<' in an attribute value"));
        assert_eq!(error("<d a=1/>"), (5, "an attribute value must be quoted"));
        assert_eq!(error("<d>"), (3, "the document ends inside an element"));
        assert_eq!(error(""), (0, "the document has no root element"));
        assert_eq!(error("<d><!-- a -- b --></d>"), (3, "a comment contains \"--\""));
        assert_eq!(error("<d>]]></d>"), (3, "\"]]>\" in text"));
        assert_eq!(
            error("<d/><?xml version='1.0'?>"),
            (4, "an XML declaration that is not at the start of the document")
        );
    }

    #[test]
    fn encodings() {
        assert_eq!(decode(b"\xEF\xBB\xBF<d/>").unwrap(), "<d/>");
        assert_eq!(
            decode(b"<?xml version='1.0' encoding='ISO-8859-1'?><d>\xe9</d>").unwrap(),
            "<?xml version='1.0' encoding='ISO-8859-1'?><d>\u{e9}</d>"
        );
        assert!(
            matches!(decode(b"<?xml version='1.0' encoding='Shift_JIS'?><d/>"), Err(Error::XmlEncoding(e)) if &*e == "Shift_JIS")
        );
        assert!(matches!(decode(b"<d>\xe9</d>"), Err(Error::Xml { offset: 3, .. })));
        assert!(matches!(decode(b"\xFF\xFE<\x00"), Err(Error::XmlEncoding(_))));
    }
}
