//! Turns XML events into unresolved per-lexicon tables ([`RawLexicon`]).
//!
//! This stage checks document structure (element nesting, required attributes, document-wide id
//! uniqueness) but not references between ids; [`super::merge`] and [`super::materialize`] resolve those.

mod parser;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;

use self::parser::RawParser;
use super::xml::{XmlAttr, XmlEvent, XmlPullParser};
use super::{QName, WnLmfDependency, malformed};
use crate::error::WordNetError;
use crate::pos::WordNetPos;

#[derive(Debug, Clone)]
pub(super) struct RawRelation {
    pub(super) rel_type: String,
    pub(super) target: String,
    pub(super) line: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RawSenseRelation {
    pub(super) rel_type: String,
    pub(super) target: String,
    pub(super) line: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RawEntry {
    pub(super) id: String,
    pub(super) lemma: String,
    pub(super) pos: WordNetPos,
    pub(super) line: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RawSense {
    pub(super) id: String,
    pub(super) entry_id: String,
    pub(super) synset_id: String,
    pub(super) relations: Vec<RawSenseRelation>,
    pub(super) line: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RawExternalSense {
    pub(super) id: String,
    pub(super) relations: Vec<RawSenseRelation>,
    pub(super) line: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RawExternalEntry {
    pub(super) id: String,
    pub(super) external_senses: Vec<RawExternalSense>,
    pub(super) line: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RawSynset {
    pub(super) id: String,
    pub(super) pos: WordNetPos,
    pub(super) members: Option<String>,
    pub(super) extra_members: Vec<String>,
    pub(super) gloss: String,
    pub(super) relations: Vec<RawRelation>,
    pub(super) line: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RawExternalSynset {
    pub(super) id: String,
    pub(super) definition: Option<String>,
    pub(super) relations: Vec<RawRelation>,
    pub(super) line: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RawLexicon {
    pub(super) extension: bool,
    pub(super) id: String,
    pub(super) label: String,
    pub(super) language: String,
    pub(super) version: String,
    pub(super) metadata: BTreeMap<QName, String>,
    pub(super) requires: Vec<WnLmfDependency>,
    pub(super) extends_ref: Option<WnLmfDependency>,
    pub(super) entries: HashMap<String, RawEntry>,
    pub(super) entry_order: Vec<String>,
    pub(super) senses: HashMap<String, RawSense>,
    pub(super) sense_order: Vec<String>,
    pub(super) synsets: HashMap<String, RawSynset>,
    pub(super) synset_order: Vec<String>,
    pub(super) external_entries: Vec<RawExternalEntry>,
    pub(super) external_synsets: Vec<RawExternalSynset>,
    pub(super) line: usize,
}

pub(super) fn get_attr<'a>(attrs: &'a [XmlAttr], name: &str) -> Option<&'a str> {
    attrs.iter().find(|a| a.namespace_uri.is_empty() && a.local_name == name).map(|a| a.value.as_str())
}

pub(super) fn require_attr(
    attrs: &[XmlAttr],
    element: &str,
    attr: &str,
    resource_name: &str,
    line: usize,
) -> Result<String, WordNetError> {
    match get_attr(attrs, attr) {
        Some(v) if !v.is_empty() => Ok(v.to_string()),
        _ => {
            Err(malformed(resource_name, Some(line), format!("Element {element} is missing required attribute {attr}")))
        }
    }
}

pub(super) fn read_lexicon_metadata(attrs: &[XmlAttr]) -> BTreeMap<QName, String> {
    let mut out = BTreeMap::new();
    for attr in attrs {
        if attr.namespace_uri.is_empty() && matches!(attr.local_name.as_str(), "id" | "label" | "language" | "version")
        {
            continue;
        }
        out.insert(QName::new(attr.namespace_uri.clone(), attr.local_name.clone()), attr.value.clone());
    }
    out
}

pub(super) fn read_dependency(
    attrs: &[XmlAttr],
    element: &str,
    resource_name: &str,
    line: usize,
) -> Result<WnLmfDependency, WordNetError> {
    let id = require_attr(attrs, element, "ref", resource_name, line)?;
    let version = require_attr(attrs, element, "version", resource_name, line)?;
    let url = get_attr(attrs, "url").filter(|s| !s.is_empty()).map(|s| s.to_string());
    WnLmfDependency::new(id, version, url)
}

pub(super) fn parse_pos(code: &str, resource_name: &str, line: usize) -> Result<WordNetPos, WordNetError> {
    match code {
        "n" => Ok(WordNetPos::Noun),
        "v" => Ok(WordNetPos::Verb),
        "a" | "s" => Ok(WordNetPos::Adjective),
        "r" => Ok(WordNetPos::Adverb),
        _ => Err(malformed(resource_name, Some(line), format!("Unknown part-of-speech code: {code}"))),
    }
}

pub(super) fn claim_document_id(
    document_ids: &mut HashSet<String>,
    id: &str,
    kind: &str,
    resource_name: &str,
    line: usize,
) -> Result<(), WordNetError> {
    if !document_ids.insert(id.to_string()) {
        return Err(malformed(resource_name, Some(line), format!("Duplicate {kind} id {id}")));
    }
    Ok(())
}

pub(super) fn require_parent(
    elements: &[String],
    child: &str,
    expected: &str,
    alternative: Option<&str>,
    resource_name: &str,
    line: usize,
) -> Result<(), WordNetError> {
    let actual = elements.last().map(String::as_str);
    let matches_expected = actual == Some(expected);
    let matches_alt = alternative.is_some_and(|alt| actual == Some(alt));
    if !matches_expected && !matches_alt {
        let alt_clause = alternative.map_or(String::new(), |alt| format!(" or {alt}"));
        let found = actual.unwrap_or("document root");
        return Err(malformed(
            resource_name,
            Some(line),
            format!("{child} must be a direct child of {expected}{alt_clause}; found {found}"),
        ));
    }
    Ok(())
}

pub(super) struct EntryBuilder {
    pub(super) id: String,
    pub(super) lemma: Option<(String, WordNetPos)>,
    pub(super) line: usize,
}

pub(super) struct SenseBuilder {
    pub(super) id: String,
    pub(super) entry_id: String,
    pub(super) synset_id: String,
    pub(super) line: usize,
    pub(super) relations: Vec<RawSenseRelation>,
}

pub(super) struct SynsetBuilder {
    pub(super) id: String,
    pub(super) pos: WordNetPos,
    pub(super) members: Option<String>,
    pub(super) line: usize,
    pub(super) relations: Vec<RawRelation>,
    pub(super) definitions: Vec<String>,
}

pub(super) struct ExternalEntryBuilder {
    pub(super) id: String,
    pub(super) line: usize,
    pub(super) external_senses: Vec<RawExternalSense>,
}

pub(super) struct ExternalSenseBuilder {
    pub(super) id: String,
    pub(super) line: usize,
    pub(super) relations: Vec<RawSenseRelation>,
}

pub(super) struct ExternalSynsetBuilder {
    pub(super) id: String,
    pub(super) line: usize,
    pub(super) relations: Vec<RawRelation>,
    pub(super) definitions: Vec<String>,
}

/// Parses a WN-LMF document into one [`RawLexicon`] per `<Lexicon>` or `<LexiconExtension>`.
pub(super) fn parse_raw<R: Read>(
    mut reader: R,
    resource_name: &str,
    allow_extensions: bool,
) -> Result<Vec<RawLexicon>, WordNetError> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    let mut xml_string = String::from_utf8(bytes).map_err(|e| {
        let valid = &e.as_bytes()[..e.utf8_error().valid_up_to()];
        let line = 1 + valid.iter().filter(|&&b| b == b'\n').count();
        malformed(resource_name, Some(line), "Document is not valid UTF-8")
    })?;
    // XML 1.0 section 2.11: a parser hands `\r\n` and a lone `\r` to the application as `\n`.
    if xml_string.contains('\r') {
        xml_string = xml_string.replace("\r\n", "\n").replace('\r', "\n");
    }
    let mut parser = XmlPullParser::new(&xml_string, resource_name);
    let mut state = RawParser::new(resource_name, allow_extensions);
    loop {
        match parser.next_event()? {
            XmlEvent::Start { namespace_uri, local_name, attrs, line } => {
                state.start(&namespace_uri, local_name, &attrs, line)?
            }
            XmlEvent::Text(text) => state.text(&text),
            XmlEvent::End { local_name, line } => state.end(&local_name, line)?,
            XmlEvent::Eof { line } => return state.finish(line),
        }
    }
}
