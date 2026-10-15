//! ISO 30042 TermBase eXchange (TBX v2 and TBX v3) glossary reader and writer.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read, Write};

use crate::{GlossaryEntry, GlossaryError, is_blank};

const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
const TBX_V3_NAMESPACE: &str = "urn:iso:std:iso:30042:ed-2";
const MAX_ENTITY_DEPTH: usize = 16;
const MAX_ENTITY_EXPANSION_BYTES: usize = 64 * 1024;

/// Reads [`GlossaryEntry`] items for a target language from ISO 30042 TermBase
/// eXchange (TBX) XML streams (both TBX v2 `<martif>` and TBX v3 `<tbx>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TbxGlossaryReader {
    language: String,
}

impl TbxGlossaryReader {
    /// Creates a TBX glossary reader that extracts terms for `language` (matched
    /// case-insensitively by exact BCP 47 tag or primary subtag prefix, so `"en"`
    /// matches `"en"` and `"en-US"`).
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::BlankLanguage`] if `language` is empty or blank.
    pub fn new(language: &str) -> Result<Self, GlossaryError> {
        if is_blank(language) {
            return Err(GlossaryError::BlankLanguage);
        }
        Ok(Self { language: trim_java(language).to_owned() })
    }

    /// Returns the target BCP 47 language tag.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Reads all glossary entries for the configured language from `reader`.
    pub fn read<R: Read>(&self, mut reader: R) -> Result<Vec<GlossaryEntry>, GlossaryError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| GlossaryError::Io(io::Error::new(io::ErrorKind::InvalidData, e)))?;
        self.read_str(text)
    }

    /// Parses all glossary entries for the configured language from a `&str`.
    pub fn read_str(&self, xml: &str) -> Result<Vec<GlossaryEntry>, GlossaryError> {
        let xml = xml.strip_prefix('\u{FEFF}').unwrap_or(xml);
        let mut parser = TbxXmlParser::new(xml, &self.language);
        parser.parse()
    }

    /// Writes `entries` as a TBX v3 (`<tbx xmlns="urn:iso:std:iso:30042:ed-2">`)
    /// XML document for this reader's language.
    pub fn write_tbx_v3<W: Write>(&self, entries: &[GlossaryEntry], mut writer: W) -> Result<(), GlossaryError> {
        writeln!(writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        write!(writer, "<tbx xmlns=\"{TBX_V3_NAMESPACE}\" type=\"TBX-Basic\" style=\"dca\" xml:lang=\"")?;
        write_xml_escaped(&mut writer, &self.language, true)?;
        writeln!(writer, "\">")?;
        writeln!(writer, "  <text>")?;
        writeln!(writer, "    <body>")?;
        for entry in entries {
            write!(writer, "      <conceptEntry id=\"")?;
            write_xml_escaped(&mut writer, entry.id(), true)?;
            writeln!(writer, "\">")?;
            write!(writer, "        <langSec xml:lang=\"")?;
            write_xml_escaped(&mut writer, &self.language, true)?;
            writeln!(writer, "\">")?;
            writeln!(writer, "          <termSec>")?;
            write!(writer, "            <term>")?;
            write_xml_escaped(&mut writer, entry.term(), false)?;
            writeln!(writer, "</term>")?;
            writeln!(writer, "          </termSec>")?;
            writeln!(writer, "        </langSec>")?;
            writeln!(writer, "      </conceptEntry>")?;
        }
        writeln!(writer, "    </body>")?;
        writeln!(writer, "  </text>")?;
        writeln!(writer, "</tbx>")?;
        Ok(())
    }

    /// Writes `entries` as a TBX v2 (`<martif type="TBX">`) XML document for
    /// this reader's language.
    pub fn write_tbx_v2<W: Write>(&self, entries: &[GlossaryEntry], mut writer: W) -> Result<(), GlossaryError> {
        writeln!(writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        write!(writer, "<martif type=\"TBX\" xml:lang=\"")?;
        write_xml_escaped(&mut writer, &self.language, true)?;
        writeln!(writer, "\">")?;
        writeln!(writer, "  <text>")?;
        writeln!(writer, "    <body>")?;
        for entry in entries {
            write!(writer, "      <termEntry id=\"")?;
            write_xml_escaped(&mut writer, entry.id(), true)?;
            writeln!(writer, "\">")?;
            write!(writer, "        <langSet xml:lang=\"")?;
            write_xml_escaped(&mut writer, &self.language, true)?;
            writeln!(writer, "\">")?;
            writeln!(writer, "          <tig>")?;
            write!(writer, "            <term>")?;
            write_xml_escaped(&mut writer, entry.term(), false)?;
            writeln!(writer, "</term>")?;
            writeln!(writer, "          </tig>")?;
            writeln!(writer, "        </langSet>")?;
            writeln!(writer, "      </termEntry>")?;
        }
        writeln!(writer, "    </body>")?;
        writeln!(writer, "  </text>")?;
        writeln!(writer, "</martif>")?;
        Ok(())
    }
}

fn write_xml_escaped<W: Write>(writer: &mut W, text: &str, is_attr: bool) -> io::Result<()> {
    for ch in text.chars() {
        match ch {
            '&' => write!(writer, "&amp;")?,
            '<' => write!(writer, "&lt;")?,
            '>' => write!(writer, "&gt;")?,
            '"' if is_attr => write!(writer, "&quot;")?,
            '\'' if is_attr => write!(writer, "&apos;")?,
            '\t' if is_attr => write!(writer, "&#9;")?,
            '\n' if is_attr => write!(writer, "&#10;")?,
            '\r' if is_attr => write!(writer, "&#13;")?,
            _ => write!(writer, "{ch}")?,
        }
    }
    Ok(())
}

/// Trims leading and trailing code points `<= U+0020`, matching Java's
/// `String.trim()`.
fn trim_java(s: &str) -> &str {
    s.trim_matches(|c: char| (c as u32) <= 0x20)
}

fn matches_language(lang_attr: Option<&str>, target_lang: &str) -> bool {
    let Some(lang) = lang_attr else {
        return false;
    };
    if lang.eq_ignore_ascii_case(target_lang) {
        return true;
    }
    if lang.len() > target_lang.len()
        && lang.is_char_boundary(target_lang.len())
        && lang[..target_lang.len()].eq_ignore_ascii_case(target_lang)
        && lang.as_bytes()[target_lang.len()] == b'-'
    {
        return true;
    }
    false
}

fn is_supported_tbx_namespace(ns: &str) -> bool {
    ns.is_empty() || ns == TBX_V3_NAMESPACE
}

fn is_entry_element(local: &str, root: &str) -> bool {
    match root {
        "martif" => local == "termEntry",
        "tbx" => local == "conceptEntry",
        _ => false,
    }
}

fn is_lang_element(local: &str, root: &str) -> bool {
    match root {
        "martif" => local == "langSet",
        "tbx" => local == "langSec",
        _ => false,
    }
}

fn is_term_container_element(local: &str, root: &str, in_ntig: bool) -> bool {
    match root {
        "martif" => {
            if in_ntig {
                local == "termGrp"
            } else {
                local == "tig"
            }
        }
        "tbx" => local == "termSec",
        _ => false,
    }
}

struct NsFrame {
    default_ns: String,
    prefixes: BTreeMap<String, String>,
}

struct RawAttr<'a> {
    name: &'a str,
    raw_value: &'a str,
}

struct TbxXmlParser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    target_lang: &'a str,
    entities: BTreeMap<String, String>,
    ns_stack: Vec<NsFrame>,
    open_elements: Vec<String>,
}

impl<'a> TbxXmlParser<'a> {
    fn new(src: &'a str, target_lang: &'a str) -> Self {
        let mut root_prefixes = BTreeMap::new();
        root_prefixes.insert("xml".to_owned(), XML_NAMESPACE.to_owned());
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            target_lang,
            entities: BTreeMap::new(),
            ns_stack: vec![NsFrame { default_ns: String::new(), prefixes: root_prefixes }],
            open_elements: Vec::new(),
        }
    }

    fn parse(&mut self) -> Result<Vec<GlossaryEntry>, GlossaryError> {
        let mut entries = Vec::new();
        let mut root_element: Option<String> = None;
        let mut depth: usize = 0;
        let mut text_depth: usize = 0;
        let mut body_depth: usize = 0;
        let mut entry_depth: usize = 0;
        let mut lang_depth: usize = 0;
        let mut ntig_depth: usize = 0;
        let mut term_container_depth: usize = 0;
        let mut term_depth: usize = 0;
        let mut foreign_subtree_depth: usize = 0;

        let mut current_entry_id: Option<String> = None;
        let mut in_matching_lang = false;
        let mut current_term: Option<String> = None;

        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] == b'<' {
                if self.starts_with(b"<!--") {
                    self.skip_comment()?;
                } else if self.starts_with(b"<![CDATA[") {
                    let cdata = self.read_cdata()?;
                    if root_element.is_none() {
                        return Err(GlossaryError::Format("Unexpected CDATA section before root element".to_owned()));
                    }
                    if foreign_subtree_depth == 0
                        && let Some(buf) = &mut current_term
                    {
                        buf.push_str(cdata);
                    }
                } else if self.starts_with(b"<!DOCTYPE") {
                    if root_element.is_some() {
                        return Err(GlossaryError::Format("Unexpected DOCTYPE declaration inside document".to_owned()));
                    }
                    self.parse_doctype()?;
                } else if self.starts_with(b"<?") {
                    self.skip_pi()?;
                } else if self.starts_with(b"</") {
                    let qname = self.read_end_tag()?;
                    let Some(expected) = self.open_elements.pop() else {
                        return Err(GlossaryError::Format(format!("Unexpected closing tag </{qname}>")));
                    };
                    if expected != qname {
                        return Err(GlossaryError::Format(format!(
                            "Mismatched closing tag </{qname}>, expected </{expected}>"
                        )));
                    }
                    let (_, local_name) = split_qname(&qname);
                    self.handle_end_element(
                        local_name,
                        &root_element,
                        depth,
                        &mut text_depth,
                        &mut body_depth,
                        &mut entry_depth,
                        &mut lang_depth,
                        &mut ntig_depth,
                        &mut term_container_depth,
                        &mut term_depth,
                        &mut foreign_subtree_depth,
                        &mut current_entry_id,
                        &mut in_matching_lang,
                        &mut current_term,
                        &mut entries,
                    )?;
                    self.ns_stack.pop();
                    depth -= 1;
                } else {
                    let (qname, attrs, self_closing) = self.read_start_tag()?;
                    depth += 1;
                    self.push_ns_frame(&attrs)?;

                    let (prefix, local_name) = split_qname(&qname);
                    let elem_ns = self.resolve_element_ns(prefix)?;

                    if foreign_subtree_depth > 0 {
                        if !self_closing {
                            self.open_elements.push(qname);
                        } else {
                            self.ns_stack.pop();
                            depth -= 1;
                        }
                        continue;
                    }

                    if depth == 1 {
                        if (local_name != "martif" && local_name != "tbx") || !is_supported_tbx_namespace(&elem_ns) {
                            return Err(GlossaryError::Format(format!(
                                "Unsupported TBX root element <{local_name}> (expected <martif> or <tbx> in TBX namespace)"
                            )));
                        }
                        root_element = Some(local_name.to_owned());
                    } else if !is_supported_tbx_namespace(&elem_ns) {
                        foreign_subtree_depth = depth;
                        if !self_closing {
                            self.open_elements.push(qname);
                        } else {
                            foreign_subtree_depth = 0;
                            self.ns_stack.pop();
                            depth -= 1;
                        }
                        continue;
                    } else {
                        let root = root_element.as_deref().unwrap_or("");
                        if depth == 2 && text_depth == 0 && local_name == "text" {
                            text_depth = depth;
                        } else if text_depth > 0 && depth == text_depth + 1 && body_depth == 0 && local_name == "body" {
                            body_depth = depth;
                        } else if body_depth > 0
                            && depth == body_depth + 1
                            && entry_depth == 0
                            && is_entry_element(local_name, root)
                        {
                            let id_val = self.parse_id_attribute(&attrs)?;
                            let Some(id_val) = id_val else {
                                return Err(GlossaryError::Format(format!(
                                    "TBX entry element <{local_name}> is missing a required non-blank 'id' attribute"
                                )));
                            };
                            entry_depth = depth;
                            current_entry_id = Some(id_val);
                        } else if entry_depth > 0
                            && depth == entry_depth + 1
                            && lang_depth == 0
                            && is_lang_element(local_name, root)
                        {
                            lang_depth = depth;
                            let lang = self.parse_lang_attribute(&attrs)?;
                            in_matching_lang = matches_language(lang.as_deref(), self.target_lang);
                        } else if lang_depth > 0
                            && depth == lang_depth + 1
                            && ntig_depth == 0
                            && term_container_depth == 0
                            && root == "martif"
                            && local_name == "ntig"
                        {
                            ntig_depth = depth;
                        } else if lang_depth > 0
                            && term_container_depth == 0
                            && is_term_container_element(local_name, root, ntig_depth > 0)
                            && depth == if ntig_depth > 0 { ntig_depth + 1 } else { lang_depth + 1 }
                        {
                            term_container_depth = depth;
                        } else if term_container_depth > 0
                            && depth == term_container_depth + 1
                            && term_depth == 0
                            && local_name == "term"
                        {
                            term_depth = depth;
                            if in_matching_lang && current_entry_id.is_some() {
                                current_term = Some(String::new());
                            }
                        }
                    }

                    if self_closing {
                        self.handle_end_element(
                            local_name,
                            &root_element,
                            depth,
                            &mut text_depth,
                            &mut body_depth,
                            &mut entry_depth,
                            &mut lang_depth,
                            &mut ntig_depth,
                            &mut term_container_depth,
                            &mut term_depth,
                            &mut foreign_subtree_depth,
                            &mut current_entry_id,
                            &mut in_matching_lang,
                            &mut current_term,
                            &mut entries,
                        )?;
                        self.ns_stack.pop();
                        depth -= 1;
                    } else {
                        self.open_elements.push(qname);
                    }
                }
            } else {
                let raw_text = self.read_text_until_lt();
                if root_element.is_none() || depth == 0 {
                    if !raw_text.chars().all(|c| matches!(c, ' ' | '\t' | '\r' | '\n')) {
                        return Err(GlossaryError::Format("Unexpected character data outside root element".to_owned()));
                    }
                } else if foreign_subtree_depth == 0 {
                    let expanded = self.expand_text(raw_text, false, 0)?;
                    if let Some(buf) = &mut current_term {
                        buf.push_str(&expanded);
                    }
                } else {
                    // Still validate entity syntax in foreign subtrees so malformed XML fails.
                    let _ = self.expand_text(raw_text, false, 0)?;
                }
            }
        }

        if !self.open_elements.is_empty() {
            return Err(GlossaryError::Format("Malformed TBX XML: unclosed elements at end of input".to_owned()));
        }
        if root_element.is_none() {
            return Err(GlossaryError::Format("Malformed TBX XML: empty document with no root element".to_owned()));
        }

        Ok(entries)
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_end_element(
        &self,
        local_name: &str,
        root_element: &Option<String>,
        depth: usize,
        text_depth: &mut usize,
        body_depth: &mut usize,
        entry_depth: &mut usize,
        lang_depth: &mut usize,
        ntig_depth: &mut usize,
        term_container_depth: &mut usize,
        term_depth: &mut usize,
        foreign_subtree_depth: &mut usize,
        current_entry_id: &mut Option<String>,
        in_matching_lang: &mut bool,
        current_term: &mut Option<String>,
        entries: &mut Vec<GlossaryEntry>,
    ) -> Result<(), GlossaryError> {
        if *foreign_subtree_depth > 0 {
            if depth == *foreign_subtree_depth {
                *foreign_subtree_depth = 0;
            }
            return Ok(());
        }
        let root = root_element.as_deref().unwrap_or("");
        if local_name == "term" && depth == *term_depth {
            if let Some(raw_term) = current_term.take()
                && *in_matching_lang
                && let Some(entry_id) = current_entry_id.as_deref()
            {
                let term = trim_java(&raw_term);
                if is_blank(term) {
                    return Err(GlossaryError::Format("TBX <term> text must not be blank".to_owned()));
                }
                entries.push(GlossaryEntry::new(entry_id, term)?);
            }
            *term_depth = 0;
        } else if depth == *term_container_depth && is_term_container_element(local_name, root, *ntig_depth > 0) {
            *term_container_depth = 0;
        } else if depth == *ntig_depth && local_name == "ntig" {
            *ntig_depth = 0;
        } else if depth == *lang_depth && is_lang_element(local_name, root) {
            *lang_depth = 0;
            *in_matching_lang = false;
        } else if depth == *entry_depth && is_entry_element(local_name, root) {
            *entry_depth = 0;
            *current_entry_id = None;
        } else if depth == *body_depth && local_name == "body" {
            *body_depth = 0;
        } else if depth == *text_depth && local_name == "text" {
            *text_depth = 0;
        }
        Ok(())
    }

    fn push_ns_frame(&mut self, attrs: &[RawAttr<'a>]) -> Result<(), GlossaryError> {
        let parent = self.ns_stack.last().unwrap();
        let mut frame = NsFrame { default_ns: parent.default_ns.clone(), prefixes: parent.prefixes.clone() };
        for attr in attrs {
            if attr.name == "xmlns" {
                frame.default_ns = self.expand_text(attr.raw_value, true, 0)?;
            } else if let Some(prefix) = attr.name.strip_prefix("xmlns:") {
                if prefix.is_empty() {
                    return Err(GlossaryError::Format("Malformed xmlns attribute with empty prefix".to_owned()));
                }
                let uri = self.expand_text(attr.raw_value, true, 0)?;
                frame.prefixes.insert(prefix.to_owned(), uri);
            }
        }
        self.ns_stack.push(frame);
        Ok(())
    }

    fn resolve_element_ns(&self, prefix: &str) -> Result<String, GlossaryError> {
        let frame = self.ns_stack.last().unwrap();
        if prefix.is_empty() {
            Ok(frame.default_ns.clone())
        } else if let Some(uri) = frame.prefixes.get(prefix) {
            Ok(uri.clone())
        } else {
            Err(GlossaryError::Format(format!("Unbound XML namespace prefix '{prefix}'")))
        }
    }

    fn resolve_attr_ns(&self, prefix: &str) -> Result<String, GlossaryError> {
        if prefix.is_empty() {
            return Ok(String::new());
        }
        let frame = self.ns_stack.last().unwrap();
        frame
            .prefixes
            .get(prefix)
            .cloned()
            .ok_or_else(|| GlossaryError::Format(format!("Unbound XML attribute namespace prefix '{prefix}'")))
    }

    fn parse_id_attribute(&self, attrs: &[RawAttr<'a>]) -> Result<Option<String>, GlossaryError> {
        for attr in attrs {
            if attr.name == "xmlns" || attr.name.starts_with("xmlns:") {
                continue;
            }
            let (prefix, local) = split_qname(attr.name);
            let ns = self.resolve_attr_ns(prefix)?;
            if local == "id" && ns.is_empty() {
                let value = self.expand_text(attr.raw_value, true, 0)?;
                if !is_blank(&value) {
                    return Ok(Some(value));
                }
            }
        }
        Ok(None)
    }

    fn parse_lang_attribute(&self, attrs: &[RawAttr<'a>]) -> Result<Option<String>, GlossaryError> {
        for attr in attrs {
            if attr.name == "xmlns" || attr.name.starts_with("xmlns:") {
                continue;
            }
            let (prefix, local) = split_qname(attr.name);
            let ns = self.resolve_attr_ns(prefix)?;
            if local == "lang" && (ns.is_empty() || ns == XML_NAMESPACE) {
                let value = self.expand_text(attr.raw_value, true, 0)?;
                if !is_blank(&value) {
                    return Ok(Some(trim_java(&value).to_owned()));
                }
            }
        }
        Ok(None)
    }

    fn starts_with(&self, prefix: &[u8]) -> bool {
        self.bytes[self.pos..].starts_with(prefix)
    }

    fn skip_comment(&mut self) -> Result<(), GlossaryError> {
        self.pos += 4; // "<!--"
        if let Some(rel) = self.src[self.pos..].find("-->") {
            let body = &self.src[self.pos..self.pos + rel];
            if body.contains("--") {
                return Err(GlossaryError::Format("Invalid '--' inside XML comment".to_owned()));
            }
            self.pos += rel + 3;
            Ok(())
        } else {
            Err(GlossaryError::Format("Unterminated XML comment".to_owned()))
        }
    }

    fn skip_pi(&mut self) -> Result<(), GlossaryError> {
        self.pos += 2; // "<?"
        if let Some(rel) = self.src[self.pos..].find("?>") {
            self.pos += rel + 2;
            Ok(())
        } else {
            Err(GlossaryError::Format("Unterminated XML processing instruction".to_owned()))
        }
    }

    fn read_cdata(&mut self) -> Result<&'a str, GlossaryError> {
        self.pos += 9; // "<![CDATA["
        if let Some(rel) = self.src[self.pos..].find("]]>") {
            let content = &self.src[self.pos..self.pos + rel];
            self.pos += rel + 3;
            Ok(content)
        } else {
            Err(GlossaryError::Format("Unterminated CDATA section".to_owned()))
        }
    }

    fn parse_doctype(&mut self) -> Result<(), GlossaryError> {
        self.pos += 9; // "<!DOCTYPE"
        self.skip_xml_ws();
        let root_name = self.read_name()?;
        if root_name.is_empty() {
            return Err(GlossaryError::Format("Malformed DOCTYPE declaration".to_owned()));
        }

        // Advance to either '[' (internal subset) or '>' (end of DOCTYPE).
        let mut in_quote: Option<u8> = None;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if let Some(q) = in_quote {
                self.pos += 1;
                if b == q {
                    in_quote = None;
                }
            } else if b == b'"' || b == b'\'' {
                in_quote = Some(b);
                self.pos += 1;
            } else if b == b'[' {
                self.pos += 1;
                self.parse_internal_subset()?;
                self.skip_xml_ws();
                if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'>' {
                    return Err(GlossaryError::Format("Expected '>' after DOCTYPE internal subset".to_owned()));
                }
                self.pos += 1;
                return Ok(());
            } else if b == b'>' {
                self.pos += 1;
                return Ok(());
            } else {
                self.pos += 1;
            }
        }
        Err(GlossaryError::Format("Unterminated DOCTYPE declaration".to_owned()))
    }

    fn parse_internal_subset(&mut self) -> Result<(), GlossaryError> {
        while self.pos < self.bytes.len() {
            self.skip_xml_ws();
            if self.pos >= self.bytes.len() {
                break;
            }
            if self.bytes[self.pos] == b']' {
                self.pos += 1;
                return Ok(());
            }
            if self.starts_with(b"<!--") {
                self.skip_comment()?;
            } else if self.starts_with(b"<?") {
                self.skip_pi()?;
            } else if self.starts_with(b"<!ENTITY") {
                self.parse_entity_decl()?;
            } else if self.starts_with(b"<!") {
                // Other DTD declarations (ELEMENT, ATTLIST, NOTATION): skip up to matching '>'.
                self.pos += 2;
                let mut in_quote: Option<u8> = None;
                while self.pos < self.bytes.len() {
                    let b = self.bytes[self.pos];
                    self.pos += 1;
                    if let Some(q) = in_quote {
                        if b == q {
                            in_quote = None;
                        }
                    } else if b == b'"' || b == b'\'' {
                        in_quote = Some(b);
                    } else if b == b'>' {
                        break;
                    }
                }
            } else {
                return Err(GlossaryError::Format("Unexpected token in DOCTYPE internal subset".to_owned()));
            }
        }
        Err(GlossaryError::Format("Unterminated DOCTYPE internal subset".to_owned()))
    }

    fn parse_entity_decl(&mut self) -> Result<(), GlossaryError> {
        self.pos += 8; // "<!ENTITY"
        self.skip_xml_ws();
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'%' {
            // Parameter entity declaration (`<!ENTITY % name ...>`).
            self.pos += 1;
            self.skip_xml_ws();
        }
        let name = self.read_name()?;
        if name.is_empty() {
            return Err(GlossaryError::Format("Malformed <!ENTITY> declaration without name".to_owned()));
        }
        self.skip_xml_ws();
        if self.pos >= self.bytes.len() {
            return Err(GlossaryError::Format("Unterminated <!ENTITY> declaration".to_owned()));
        }
        let b = self.bytes[self.pos];
        if b == b'"' || b == b'\'' {
            let val = self.read_quoted_literal()?;
            self.skip_xml_ws();
            if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'>' {
                return Err(GlossaryError::Format("Expected '>' at end of internal <!ENTITY> declaration".to_owned()));
            }
            self.pos += 1;
            // First declaration wins per XML 1.0 §4.2.
            self.entities.entry(name.to_owned()).or_insert_with(|| val.to_owned());
            Ok(())
        } else {
            // External entity (SYSTEM / PUBLIC) or NDATA: strictly forbidden!
            Err(GlossaryError::Format(
                "External XML entity declarations (SYSTEM/PUBLIC) are forbidden in TBX".to_owned(),
            ))
        }
    }

    fn read_start_tag(&mut self) -> Result<(String, Vec<RawAttr<'a>>, bool), GlossaryError> {
        self.pos += 1; // '<'
        let qname = self.read_name()?;
        if qname.is_empty() {
            return Err(GlossaryError::Format("Empty XML element name".to_owned()));
        }
        let mut attrs = Vec::new();
        let mut seen_names = BTreeSet::new();

        loop {
            let had_ws = self.skip_xml_ws();
            if self.pos >= self.bytes.len() {
                return Err(GlossaryError::Format(format!("Unterminated start tag <{qname}>")));
            }
            if self.starts_with(b"/>") {
                self.pos += 2;
                return Ok((qname.to_owned(), attrs, true));
            }
            if self.bytes[self.pos] == b'>' {
                self.pos += 1;
                return Ok((qname.to_owned(), attrs, false));
            }
            if !had_ws {
                return Err(GlossaryError::Format(format!("Missing whitespace before attribute in <{qname}>")));
            }
            let attr_name = self.read_name()?;
            if attr_name.is_empty() {
                return Err(GlossaryError::Format(format!("Malformed attribute in <{qname}>")));
            }
            if !seen_names.insert(attr_name) {
                return Err(GlossaryError::Format(format!("Duplicate attribute '{attr_name}' in <{qname}>")));
            }
            self.skip_xml_ws();
            if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'=' {
                return Err(GlossaryError::Format(format!("Expected '=' after attribute '{attr_name}' in <{qname}>")));
            }
            self.pos += 1;
            self.skip_xml_ws();
            let raw_value = self.read_quoted_literal()?;
            if raw_value.contains('<') {
                return Err(GlossaryError::Format(format!("Unescaped '<' in attribute '{attr_name}' value")));
            }
            attrs.push(RawAttr { name: attr_name, raw_value });
        }
    }

    fn read_end_tag(&mut self) -> Result<String, GlossaryError> {
        self.pos += 2; // "</"
        let qname = self.read_name()?;
        if qname.is_empty() {
            return Err(GlossaryError::Format("Empty XML closing tag".to_owned()));
        }
        self.skip_xml_ws();
        if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'>' {
            return Err(GlossaryError::Format(format!("Expected '>' at end of closing tag </{qname}>")));
        }
        self.pos += 1;
        Ok(qname.to_owned())
    }

    fn read_quoted_literal(&mut self) -> Result<&'a str, GlossaryError> {
        if self.pos >= self.bytes.len() {
            return Err(GlossaryError::Format("Unexpected end of input before quoted value".to_owned()));
        }
        let quote = self.bytes[self.pos];
        if quote != b'"' && quote != b'\'' {
            return Err(GlossaryError::Format("Expected quoted literal in XML".to_owned()));
        }
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] == quote {
                let content = &self.src[start..self.pos];
                self.pos += 1;
                return Ok(content);
            }
            self.pos += 1;
        }
        Err(GlossaryError::Format("Unterminated quoted literal in XML".to_owned()))
    }

    fn read_name(&mut self) -> Result<&'a str, GlossaryError> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':') || b >= 0x80 {
                let ch_len = self.src[self.pos..].chars().next().map_or(1, char::len_utf8);
                self.pos += ch_len;
            } else {
                break;
            }
        }
        Ok(&self.src[start..self.pos])
    }

    fn skip_xml_ws(&mut self) -> bool {
        let start = self.pos;
        while self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
            self.pos += 1;
        }
        self.pos > start
    }

    fn read_text_until_lt(&mut self) -> &'a str {
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'<' {
            self.pos += 1;
        }
        &self.src[start..self.pos]
    }

    fn expand_text(&self, raw: &str, normalize_attr_ws: bool, depth: usize) -> Result<String, GlossaryError> {
        if depth > MAX_ENTITY_DEPTH {
            return Err(GlossaryError::Format("XML entity expansion exceeded maximum nesting depth".to_owned()));
        }
        let mut out = String::with_capacity(raw.len());
        let mut chars = raw.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '&' {
                let mut ref_name = String::new();
                let mut found_semi = false;
                for c in chars.by_ref() {
                    if c == ';' {
                        found_semi = true;
                        break;
                    }
                    ref_name.push(c);
                }
                if !found_semi || ref_name.is_empty() {
                    return Err(GlossaryError::Format("Malformed XML entity reference".to_owned()));
                }
                match ref_name.as_str() {
                    "lt" => out.push('<'),
                    "gt" => out.push('>'),
                    "amp" => out.push('&'),
                    "quot" => out.push('"'),
                    "apos" => out.push('\''),
                    _ if ref_name.starts_with("#x") || ref_name.starts_with("#X") => {
                        let hex = &ref_name[2..];
                        let cp = u32::from_str_radix(hex, 16).map_err(|_| {
                            GlossaryError::Format(format!("Invalid hexadecimal character reference '&{ref_name};'"))
                        })?;
                        let decoded = decode_xml_char_ref(cp)?;
                        out.push(decoded);
                    }
                    _ if let Some(dec) = ref_name.strip_prefix('#') => {
                        let cp = dec.parse::<u32>().map_err(|_| {
                            GlossaryError::Format(format!("Invalid decimal character reference '&{ref_name};'"))
                        })?;
                        let decoded = decode_xml_char_ref(cp)?;
                        out.push(decoded);
                    }
                    _ => {
                        let Some(ent_val) = self.entities.get(&ref_name) else {
                            return Err(GlossaryError::Format(format!(
                                "Undeclared XML entity reference '&{ref_name};'"
                            )));
                        };
                        let expanded = self.expand_text(ent_val, normalize_attr_ws, depth + 1)?;
                        if out.len() + expanded.len() > MAX_ENTITY_EXPANSION_BYTES {
                            return Err(GlossaryError::Format(
                                "XML entity expansion exceeded maximum allowed size".to_owned(),
                            ));
                        }
                        out.push_str(&expanded);
                    }
                }
            } else if normalize_attr_ws && matches!(ch, '\r' | '\n' | '\t') {
                if ch == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push(' ');
            } else if ch == '\r' {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push('\n');
            } else {
                out.push(ch);
            }
            if out.len() > MAX_ENTITY_EXPANSION_BYTES {
                return Err(GlossaryError::Format("XML text expansion exceeded maximum allowed size".to_owned()));
            }
        }
        Ok(out)
    }
}

fn decode_xml_char_ref(cp: u32) -> Result<char, GlossaryError> {
    let valid_xml = cp == 0x09
        || cp == 0x0A
        || cp == 0x0D
        || (0x20..=0xD7FF).contains(&cp)
        || (0xE000..=0xFFFD).contains(&cp)
        || (0x10000..=0x10FFFF).contains(&cp);
    if !valid_xml {
        return Err(GlossaryError::Format(format!("Invalid XML character reference U+{cp:04X}")));
    }
    char::from_u32(cp).ok_or_else(|| GlossaryError::Format(format!("Invalid Unicode scalar value U+{cp:04X}")))
}

fn split_qname(qname: &str) -> (&str, &str) {
    if let Some((prefix, local)) = qname.split_once(':') { (prefix, local) } else { ("", qname) }
}
