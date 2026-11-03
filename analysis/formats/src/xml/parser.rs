use super::*;
/// Parses a decoded document, calling `handler` for its content with the
/// `names` asked for. `xmlns` attributes are not reported either way.
pub fn parse_with(text: &str, names: Names, handler: &mut impl Handler) -> Result<()> {
    if let Some((offset, _)) = text.char_indices().find(|&(_, c)| !is_xml_char(c)) {
        return Err(Error::Xml { offset, message: "a character XML does not allow" });
    }
    let mut p = Parser {
        text,
        pos: 0,
        stack: Vec::new(),
        attributes: Vec::new(),
        qualified: Vec::new(),
        seen: HashSet::new(),
        root_seen: false,
        names,
    };
    p.document(handler)
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
    /// The same names, kept once a start tag has more than
    /// [`SCANNED_ATTRIBUTES`], so a tag with many attributes is checked in
    /// time linear in their number. Empty otherwise.
    seen: HashSet<&'a str>,
    root_seen: bool,
    names: Names,
}

impl<'a> Parser<'a> {
    /// The name the handler is given for `name` as written.
    fn reported(&self, name: &'a str) -> &'a str {
        match self.names {
            Names::Local => local(name),
            Names::Qualified => name,
        }
    }

    fn err<T>(&self, offset: usize, message: &'static str) -> Result<T> {
        Err(Error::Xml { offset, message })
    }

    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }

    fn document(&mut self, handler: &mut impl Handler) -> Result<()> {
        if self.rest().starts_with("<?xml") && self.rest()[5..].starts_with(|c: char| c.is_ascii_whitespace()) {
            let declaration = self.skip_past("?>", "the XML declaration is not closed")?;
            super::decode::validate_declaration(declaration)?;
        }
        while self.pos < self.text.len() {
            let rest = self.rest();
            if rest.starts_with("<!--") {
                self.comment(handler)?;
            } else if rest.starts_with("<?") {
                self.processing_instruction(handler)?;
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

    fn comment(&mut self, handler: &mut impl Handler) -> Result<()> {
        let start = self.pos;
        self.pos += 4;
        let body = self.skip_past("-->", "a comment is not closed")?;
        if body.contains("--") || body.ends_with('-') {
            return self.err(start, "a comment contains \"--\"");
        }
        handler.comment(start, self.pos);
        Ok(())
    }

    fn processing_instruction(&mut self, handler: &mut impl Handler) -> Result<()> {
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
        handler.processing_instruction(target, start, self.pos);
        Ok(())
    }

    fn cdata(&mut self, handler: &mut impl Handler) -> Result<()> {
        let start = self.pos;
        self.pos += 9;
        let body = self.skip_past("]]>", "a CDATA section is not closed")?;
        emit_normalized(body, handler);
        handler.cdata_section(start, self.pos);
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
        if !self.seen.is_empty() {
            // Dropped rather than cleared: clearing walks the whole table,
            // as large as the largest tag's.
            self.seen = HashSet::new();
        }
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
                    handler.end_element(self.reported(name), start)?;
                    handler.element_closed(self.pos);
                    return Ok(());
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
        handler.start_element(self.reported(name), &self.attributes, start)
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
        let twice = if self.qualified.len() < SCANNED_ATTRIBUTES {
            self.qualified.contains(&qualified)
        } else {
            if self.seen.is_empty() {
                self.seen.extend(self.qualified.iter().copied());
            }
            !self.seen.insert(qualified)
        };
        if twice {
            return self.err(at, "an attribute is given twice");
        }
        let value = attribute_value(raw, value_start)?;
        self.qualified.push(qualified);
        if qualified != "xmlns" && !qualified.starts_with("xmlns:") {
            self.attributes.push(Attribute { name: self.reported(qualified), value });
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
            Some(open) if open == name => {
                handler.end_element(self.reported(name), start)?;
                handler.element_closed(self.pos);
                Ok(())
            }
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
        handler.text_run(start, start + len);
        Ok(())
    }
}
