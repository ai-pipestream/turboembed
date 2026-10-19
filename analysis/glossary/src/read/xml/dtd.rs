//! Prolog, DOCTYPE, internal subset, character/entity reference expansion,
//! and character data scanning for [`Xml`].

use std::borrow::Cow;

use crate::error::ReadError;

use super::{
    Event, MAX_ENTITY_DEPTH, MAX_EXPANDED_BYTES, MAX_EXPANSIONS, Xml, is_space, is_xml_char, normalize_line_ends,
    valid_name,
};

impl<'a> Xml<'a> {
    pub(super) fn check_chars(&self, s: &str) -> Result<(), ReadError> {
        match s.chars().find(|&c| !is_xml_char(c)) {
            Some(c) => self.err(format!("character U+{:04X} is not allowed", c as u32)),
            None => Ok(()),
        }
    }

    pub(super) fn pi(&mut self) -> Result<(), ReadError> {
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

    pub(super) fn comment(&mut self) -> Result<(), ReadError> {
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

    pub(super) fn quoted(&mut self) -> Result<&'a str, ReadError> {
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

    pub(super) fn doctype(&mut self) -> Result<(), ReadError> {
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

    pub(super) fn text(&mut self) -> Result<Event<'a>, ReadError> {
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

    pub(super) fn attribute_value(&mut self) -> Result<String, ReadError> {
        let raw = self.quoted()?;
        if raw.contains('<') {
            return self.err("\"<\" in an attribute value");
        }
        self.check_chars(raw)?;
        let mut out = String::with_capacity(raw.len());
        self.expand(raw, &mut out, 0, true)?;
        Ok(out)
    }
}
