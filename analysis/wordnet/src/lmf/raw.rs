//! Turns XML events into unresolved per-lexicon tables ([`RawLexicon`]).
//!
//! This stage checks document structure (element nesting, required attributes, document-wide id
//! uniqueness) but not references between ids; [`super::merge`] and [`super::materialize`] resolve those.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;

use super::xml::{XmlAttr, XmlEvent, XmlPullParser};
use super::{EXT_NAMESPACE, OTHER_RELATION, QName, WnLmfDependency, malformed};
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

fn get_attr<'a>(attrs: &'a [XmlAttr], name: &str) -> Option<&'a str> {
    attrs.iter().find(|a| a.namespace_uri.is_empty() && a.local_name == name).map(|a| a.value.as_str())
}

fn require_attr(
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

fn read_lexicon_metadata(attrs: &[XmlAttr]) -> BTreeMap<QName, String> {
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

fn read_dependency(
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

fn parse_pos(code: &str, resource_name: &str, line: usize) -> Result<WordNetPos, WordNetError> {
    match code {
        "n" => Ok(WordNetPos::Noun),
        "v" => Ok(WordNetPos::Verb),
        "a" | "s" => Ok(WordNetPos::Adjective),
        "r" => Ok(WordNetPos::Adverb),
        _ => Err(malformed(resource_name, Some(line), format!("Unknown part-of-speech code: {code}"))),
    }
}

fn claim_document_id(
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

fn require_parent(
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

struct EntryBuilder {
    id: String,
    lemma: Option<(String, WordNetPos)>,
    line: usize,
}

struct SenseBuilder {
    id: String,
    entry_id: String,
    synset_id: String,
    line: usize,
    relations: Vec<RawSenseRelation>,
}

struct SynsetBuilder {
    id: String,
    pos: WordNetPos,
    members: Option<String>,
    line: usize,
    relations: Vec<RawRelation>,
    definitions: Vec<String>,
}

struct ExternalEntryBuilder {
    id: String,
    line: usize,
    external_senses: Vec<RawExternalSense>,
}

struct ExternalSenseBuilder {
    id: String,
    line: usize,
    relations: Vec<RawSenseRelation>,
}

struct ExternalSynsetBuilder {
    id: String,
    line: usize,
    relations: Vec<RawRelation>,
    definition: Option<String>,
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

/// Event-driven state for [`parse_raw`]: one `handle_*` method per opening element and one `end_*`
/// method per closing element that carries state.
struct RawParser<'a> {
    resource_name: &'a str,
    allow_extensions: bool,
    lexicons: Vec<RawLexicon>,
    document_ids: HashSet<String>,
    elements: Vec<String>,
    lexicon: Option<RawLexicon>,
    entry: Option<EntryBuilder>,
    sense: Option<SenseBuilder>,
    synset: Option<SynsetBuilder>,
    external_entry: Option<ExternalEntryBuilder>,
    external_sense: Option<ExternalSenseBuilder>,
    external_synset: Option<ExternalSynsetBuilder>,
    definition: Option<String>,
    skip_depth: usize,
}

impl<'a> RawParser<'a> {
    fn new(resource_name: &'a str, allow_extensions: bool) -> Self {
        Self {
            resource_name,
            allow_extensions,
            lexicons: Vec::new(),
            document_ids: HashSet::new(),
            elements: Vec::new(),
            lexicon: None,
            entry: None,
            sense: None,
            synset: None,
            external_entry: None,
            external_sense: None,
            external_synset: None,
            definition: None,
            skip_depth: 0,
        }
    }

    fn fail(&self, line: usize, message: impl AsRef<str>) -> WordNetError {
        malformed(self.resource_name, Some(line), message)
    }

    fn require_attr(&self, attrs: &[XmlAttr], element: &str, attr: &str, line: usize) -> Result<String, WordNetError> {
        require_attr(attrs, element, attr, self.resource_name, line)
    }

    fn require_parent(
        &self,
        child: &str,
        expected: &str,
        alternative: Option<&str>,
        line: usize,
    ) -> Result<(), WordNetError> {
        require_parent(&self.elements, child, expected, alternative, self.resource_name, line)
    }

    fn claim_id(&mut self, id: &str, kind: &str, line: usize) -> Result<(), WordNetError> {
        claim_document_id(&mut self.document_ids, id, kind, self.resource_name, line)
    }

    fn in_extension(&self) -> bool {
        self.lexicon.as_ref().is_some_and(|lexicon| lexicon.extension)
    }

    /// The open `<Lexicon>`; every caller sits below one because [`Self::require_parent`] succeeded.
    fn lexicon_mut(&mut self) -> &mut RawLexicon {
        self.lexicon.as_mut().expect("element handled only inside an open Lexicon")
    }

    fn start(&mut self, namespace_uri: &str, name: String, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if self.skip_depth > 0 {
            self.skip_depth += 1;
            return Ok(());
        }
        if self.definition.is_some() {
            return Err(self.fail(line, format!("Unexpected child element <{name}> inside Definition")));
        }
        // WN-LMF elements carry no namespace; the composition extension may also qualify them with
        // its own. An element from any other namespace is foreign content, whatever its local name.
        if !namespace_uri.is_empty() && namespace_uri != EXT_NAMESPACE {
            self.skip_depth = 1;
            return Ok(());
        }
        match name.as_str() {
            "LexicalResource" => self.handle_lexical_resource(line)?,
            "Lexicon" => self.handle_lexicon("Lexicon", false, attrs, line)?,
            "LexiconExtension" => self.handle_lexicon("LexiconExtension", true, attrs, line)?,
            "Extends" => self.handle_extends(attrs, line)?,
            "Requires" => self.handle_requires(attrs, line)?,
            "LexicalEntry" => self.handle_lexical_entry(attrs, line)?,
            "ExternalLexicalEntry" => self.handle_external_lexical_entry(attrs, line)?,
            "Lemma" => self.handle_lemma(attrs, line)?,
            "Sense" => self.handle_sense(attrs, line)?,
            "ExternalSense" => self.handle_external_sense(attrs, line)?,
            "SenseRelation" => self.handle_sense_relation(attrs, line)?,
            "Synset" => self.handle_synset(attrs, line)?,
            "ExternalSynset" => self.handle_external_synset(attrs, line)?,
            "Definition" => self.handle_definition(line)?,
            "SynsetRelation" => self.handle_synset_relation(attrs, line)?,
            // Elements the knowledge base does not model (Form, Pronunciation, Tag, Count, Example,
            // ILIDefinition, SyntacticBehaviour, ...) are skipped with their whole subtree.
            _ => {
                self.skip_depth = 1;
                return Ok(());
            }
        }
        self.elements.push(name);
        Ok(())
    }

    fn text(&mut self, text: &str) {
        if self.skip_depth == 0
            && let Some(ref mut definition) = self.definition
        {
            definition.push_str(text);
        }
    }

    fn end(&mut self, name: &str, line: usize) -> Result<(), WordNetError> {
        if self.skip_depth > 0 {
            self.skip_depth -= 1;
            return Ok(());
        }
        self.elements.pop();
        match name {
            "Definition" => self.end_definition(),
            "LexicalEntry" => self.end_lexical_entry()?,
            "Sense" => self.end_sense(),
            "ExternalSense" => self.end_external_sense(),
            "ExternalLexicalEntry" => self.end_external_lexical_entry(),
            "Synset" => self.end_synset(),
            "ExternalSynset" => self.end_external_synset(),
            "Lexicon" | "LexiconExtension" => self.end_lexicon(line)?,
            _ => {}
        }
        Ok(())
    }

    fn finish(self, line: usize) -> Result<Vec<RawLexicon>, WordNetError> {
        if let Some(ref lexicon) = self.lexicon {
            return Err(self.fail(line, format!("Unclosed Lexicon {}", lexicon.id)));
        }
        if self.lexicons.is_empty() {
            return Err(self.fail(line, "Document contains no Lexicon"));
        }
        Ok(self.lexicons)
    }

    fn handle_lexical_resource(&mut self, line: usize) -> Result<(), WordNetError> {
        if !self.elements.is_empty() {
            return Err(self.fail(line, "Nested LexicalResource"));
        }
        Ok(())
    }

    fn handle_lexicon(
        &mut self,
        name: &str,
        extension: bool,
        attrs: &[XmlAttr],
        line: usize,
    ) -> Result<(), WordNetError> {
        if extension && !self.allow_extensions {
            return Err(self.fail(
                line,
                "LexiconExtension is not supported without a resolver; use a read or read_resource overload that takes a WnLmfResolver",
            ));
        }
        if let Some(ref outer) = self.lexicon {
            return Err(self.fail(line, format!("Nested {name} inside {}", outer.id)));
        }
        self.require_parent(name, "LexicalResource", None, line)?;
        let id = self.require_attr(attrs, name, "id", line)?;
        self.claim_id(&id, "lexicon", line)?;
        let label = self.require_attr(attrs, name, "label", line)?;
        let language = self.require_attr(attrs, name, "language", line)?;
        let version = self.require_attr(attrs, name, "version", line)?;
        self.lexicon = Some(RawLexicon {
            extension,
            id,
            label,
            language,
            version,
            metadata: read_lexicon_metadata(attrs),
            requires: Vec::new(),
            extends_ref: None,
            entries: HashMap::new(),
            entry_order: Vec::new(),
            senses: HashMap::new(),
            sense_order: Vec::new(),
            synsets: HashMap::new(),
            synset_order: Vec::new(),
            external_entries: Vec::new(),
            external_synsets: Vec::new(),
            line,
        });
        Ok(())
    }

    fn handle_extends(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if !self.in_extension() {
            return Err(self.fail(line, "Extends outside a LexiconExtension"));
        }
        self.require_parent("Extends", "LexiconExtension", None, line)?;
        if let Some(ref lexicon) = self.lexicon
            && lexicon.extends_ref.is_some()
        {
            return Err(self.fail(line, format!("Repeated Extends in LexiconExtension {}", lexicon.id)));
        }
        let dependency = read_dependency(attrs, "Extends", self.resource_name, line)?;
        self.lexicon_mut().extends_ref = Some(dependency);
        Ok(())
    }

    fn handle_requires(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if self.lexicon.is_none() {
            return Err(self.fail(line, "Requires outside a Lexicon"));
        }
        self.require_parent("Requires", "Lexicon", Some("LexiconExtension"), line)?;
        let dependency = read_dependency(attrs, "Requires", self.resource_name, line)?;
        self.lexicon_mut().requires.push(dependency);
        Ok(())
    }

    fn handle_lexical_entry(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if let Some(ref outer) = self.entry {
            return Err(self.fail(line, format!("Nested LexicalEntry inside {}", outer.id)));
        }
        if self.lexicon.is_none() {
            return Err(self.fail(line, "LexicalEntry outside a Lexicon"));
        }
        self.require_parent("LexicalEntry", "Lexicon", Some("LexiconExtension"), line)?;
        let id = self.require_attr(attrs, "LexicalEntry", "id", line)?;
        self.claim_id(&id, "lexical entry", line)?;
        self.entry = Some(EntryBuilder { id, lemma: None, line });
        Ok(())
    }

    fn handle_external_lexical_entry(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if !self.in_extension() {
            return Err(self.fail(line, "ExternalLexicalEntry outside a LexiconExtension"));
        }
        self.require_parent("ExternalLexicalEntry", "LexiconExtension", None, line)?;
        let id = self.require_attr(attrs, "ExternalLexicalEntry", "id", line)?;
        self.claim_id(&id, "external lexical entry", line)?;
        self.external_entry = Some(ExternalEntryBuilder { id, line, external_senses: Vec::new() });
        Ok(())
    }

    fn handle_lemma(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        let Some(ref entry) = self.entry else {
            return Err(self.fail(line, "Lemma outside a LexicalEntry"));
        };
        if entry.lemma.is_some() {
            return Err(self.fail(line, format!("Duplicate Lemma in LexicalEntry {}", entry.id)));
        }
        self.require_parent("Lemma", "LexicalEntry", None, line)?;
        let written_form = self.require_attr(attrs, "Lemma", "writtenForm", line)?;
        let pos_code = self.require_attr(attrs, "Lemma", "partOfSpeech", line)?;
        let pos = parse_pos(&pos_code, self.resource_name, line)?;
        if let Some(ref mut entry) = self.entry {
            entry.lemma = Some((written_form, pos));
        }
        Ok(())
    }

    fn handle_sense(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if let Some(ref outer) = self.sense {
            return Err(self.fail(line, format!("Nested Sense inside {}", outer.id)));
        }
        let owner = if let Some(ref external_entry) = self.external_entry {
            external_entry.id.clone()
        } else if let Some(ref entry) = self.entry {
            if entry.lemma.is_none() {
                return Err(self.fail(line, format!("Sense before its entry's Lemma in LexicalEntry {}", entry.id)));
            }
            entry.id.clone()
        } else {
            return Err(self.fail(line, "Sense outside a LexicalEntry"));
        };
        self.require_parent("Sense", "LexicalEntry", Some("ExternalLexicalEntry"), line)?;
        let id = self.require_attr(attrs, "Sense", "id", line)?;
        let synset_id = self.require_attr(attrs, "Sense", "synset", line)?;
        self.claim_id(&id, "sense", line)?;
        self.sense = Some(SenseBuilder { id, entry_id: owner, synset_id, line, relations: Vec::new() });
        Ok(())
    }

    fn handle_external_sense(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if self.external_entry.is_none() {
            return Err(self.fail(line, "ExternalSense outside an ExternalLexicalEntry"));
        }
        self.require_parent("ExternalSense", "ExternalLexicalEntry", None, line)?;
        let id = self.require_attr(attrs, "ExternalSense", "id", line)?;
        self.claim_id(&id, "external sense", line)?;
        self.external_sense = Some(ExternalSenseBuilder { id, line, relations: Vec::new() });
        Ok(())
    }

    fn handle_sense_relation(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if self.external_sense.is_none() && self.sense.is_none() {
            return Err(self.fail(line, "SenseRelation outside a Sense"));
        }
        self.require_parent("SenseRelation", "Sense", Some("ExternalSense"), line)?;
        let relation = RawSenseRelation {
            rel_type: self.require_attr(attrs, "SenseRelation", "relType", line)?,
            target: self.require_attr(attrs, "SenseRelation", "target", line)?,
            line,
        };
        if let Some(ref mut external_sense) = self.external_sense {
            external_sense.relations.push(relation);
        } else if let Some(ref mut sense) = self.sense {
            sense.relations.push(relation);
        }
        Ok(())
    }

    fn handle_synset(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if let Some(ref outer) = self.synset {
            return Err(self.fail(line, format!("Nested Synset inside {}", outer.id)));
        }
        if self.lexicon.is_none() {
            return Err(self.fail(line, "Synset outside a Lexicon"));
        }
        self.require_parent("Synset", "Lexicon", Some("LexiconExtension"), line)?;
        let id = self.require_attr(attrs, "Synset", "id", line)?;
        let pos_code = self.require_attr(attrs, "Synset", "partOfSpeech", line)?;
        let pos = parse_pos(&pos_code, self.resource_name, line)?;
        self.claim_id(&id, "synset", line)?;
        let members = get_attr(attrs, "members").map(str::to_string);
        self.synset = Some(SynsetBuilder { id, pos, members, line, relations: Vec::new(), definitions: Vec::new() });
        Ok(())
    }

    fn handle_external_synset(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if !self.in_extension() {
            return Err(self.fail(line, "ExternalSynset outside a LexiconExtension"));
        }
        self.require_parent("ExternalSynset", "LexiconExtension", None, line)?;
        let id = self.require_attr(attrs, "ExternalSynset", "id", line)?;
        self.claim_id(&id, "external synset", line)?;
        self.external_synset = Some(ExternalSynsetBuilder { id, line, relations: Vec::new(), definition: None });
        Ok(())
    }

    fn handle_definition(&mut self, line: usize) -> Result<(), WordNetError> {
        if self.synset.is_none() && self.external_synset.is_none() {
            return Err(self.fail(line, "Definition outside a Synset"));
        }
        self.require_parent("Definition", "Synset", Some("ExternalSynset"), line)?;
        self.definition = Some(String::new());
        Ok(())
    }

    fn handle_synset_relation(&mut self, attrs: &[XmlAttr], line: usize) -> Result<(), WordNetError> {
        if self.synset.is_none() && self.external_synset.is_none() {
            return Err(self.fail(line, "SynsetRelation outside a Synset"));
        }
        self.require_parent("SynsetRelation", "Synset", Some("ExternalSynset"), line)?;
        let rel_type = self.require_attr(attrs, "SynsetRelation", "relType", line)?;
        let target = self.require_attr(attrs, "SynsetRelation", "target", line)?;
        if rel_type == OTHER_RELATION {
            return Ok(());
        }
        let relation = RawRelation { rel_type, target, line };
        if let Some(ref mut synset) = self.synset {
            synset.relations.push(relation);
        } else if let Some(ref mut external_synset) = self.external_synset {
            external_synset.relations.push(relation);
        }
        Ok(())
    }

    fn end_definition(&mut self) {
        let Some(definition) = self.definition.take() else {
            return;
        };
        if let Some(ref mut synset) = self.synset {
            synset.definitions.push(definition);
        } else if let Some(ref mut external_synset) = self.external_synset
            && external_synset.definition.is_none()
        {
            external_synset.definition = Some(definition);
        }
    }

    fn end_lexical_entry(&mut self) -> Result<(), WordNetError> {
        let entry = self.entry.take().expect("LexicalEntry end matches a handled start");
        let Some((lemma, pos)) = entry.lemma else {
            return Err(self.fail(entry.line, format!("LexicalEntry {} has no Lemma", entry.id)));
        };
        let lexicon = self.lexicon_mut();
        lexicon.entry_order.push(entry.id.clone());
        lexicon.entries.insert(entry.id.clone(), RawEntry { id: entry.id, lemma, pos, line: entry.line });
        Ok(())
    }

    fn end_sense(&mut self) {
        let Some(sense) = self.sense.take() else {
            return;
        };
        let lexicon = self.lexicon_mut();
        lexicon.sense_order.push(sense.id.clone());
        lexicon.senses.insert(
            sense.id.clone(),
            RawSense {
                id: sense.id,
                entry_id: sense.entry_id,
                synset_id: sense.synset_id,
                relations: sense.relations,
                line: sense.line,
            },
        );
    }

    fn end_external_sense(&mut self) {
        if let (Some(external_entry), Some(external_sense)) = (self.external_entry.as_mut(), self.external_sense.take())
        {
            external_entry.external_senses.push(RawExternalSense {
                id: external_sense.id,
                relations: external_sense.relations,
                line: external_sense.line,
            });
        }
    }

    fn end_external_lexical_entry(&mut self) {
        let Some(external_entry) = self.external_entry.take() else {
            return;
        };
        self.lexicon_mut().external_entries.push(RawExternalEntry {
            id: external_entry.id,
            external_senses: external_entry.external_senses,
            line: external_entry.line,
        });
    }

    fn end_synset(&mut self) {
        let Some(synset) = self.synset.take() else {
            return;
        };
        let lexicon = self.lexicon_mut();
        lexicon.synset_order.push(synset.id.clone());
        lexicon.synsets.insert(
            synset.id.clone(),
            RawSynset {
                id: synset.id,
                pos: synset.pos,
                members: synset.members,
                extra_members: Vec::new(),
                gloss: synset.definitions.join("; "),
                relations: synset.relations,
                line: synset.line,
            },
        );
    }

    fn end_external_synset(&mut self) {
        let Some(external_synset) = self.external_synset.take() else {
            return;
        };
        self.lexicon_mut().external_synsets.push(RawExternalSynset {
            id: external_synset.id,
            definition: external_synset.definition,
            relations: external_synset.relations,
            line: external_synset.line,
        });
    }

    fn end_lexicon(&mut self, line: usize) -> Result<(), WordNetError> {
        let Some(lexicon) = self.lexicon.take() else {
            return Ok(());
        };
        if lexicon.extension && lexicon.extends_ref.is_none() {
            return Err(self.fail(line, format!("LexiconExtension {} is missing its Extends element", lexicon.id)));
        }
        self.lexicons.push(lexicon);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::super::{cat_lexicon, wrap_lexicon};
    use super::*;

    fn raw(doc: &str) -> Result<Vec<RawLexicon>, WordNetError> {
        parse_raw(Cursor::new(doc.as_bytes()), "test.xml", true)
    }

    fn assert_err_contains(result: Result<impl std::fmt::Debug, WordNetError>, fragment: &str) {
        let err = result.expect_err("expected a parse error");
        assert!(err.to_string().contains(fragment), "expected '{fragment}' in '{err}'");
    }

    /// Hands the document to `parse_raw` one byte per `read` call, splitting every multibyte character.
    struct OneByteReader<'a>(&'a [u8]);

    impl Read for OneByteReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let Some((&first, rest)) = self.0.split_first() else { return Ok(0) };
            if buf.is_empty() {
                return Ok(0);
            }
            buf[0] = first;
            self.0 = rest;
            Ok(1)
        }
    }

    #[test]
    fn records_entries_senses_and_synsets_in_document_order() {
        let lexicons = raw(&cat_lexicon(
            " members=\"t-cat-n-1\"",
            "<Definition>a feline</Definition><Definition>a pet</Definition>\
             <SynsetRelation relType=\"other\" target=\"t-1\"/><SynsetRelation relType=\"hypernym\" target=\"t-2\"/>",
        ))
        .unwrap();
        assert_eq!(1, lexicons.len());
        let lexicon = &lexicons[0];
        assert_eq!(("t", "t", "en", "1"), (&*lexicon.id, &*lexicon.label, &*lexicon.language, &*lexicon.version));
        assert_eq!(vec!["t-cat-n"], lexicon.entry_order);
        assert_eq!(("cat", WordNetPos::Noun), (&*lexicon.entries["t-cat-n"].lemma, lexicon.entries["t-cat-n"].pos));
        assert_eq!("t-cat-n", lexicon.senses["t-cat-n-1"].entry_id);
        let synset = &lexicon.synsets["t-1"];
        assert_eq!("a feline; a pet", synset.gloss);
        assert_eq!(Some("t-cat-n-1"), synset.members.as_deref());
        assert_eq!(1, synset.relations.len(), "relType other is dropped while parsing");
        assert_eq!(("hypernym", "t-2"), (&*synset.relations[0].rel_type, &*synset.relations[0].target));
    }

    #[test]
    fn unmodelled_elements_are_skipped_with_their_whole_subtree() {
        let lexicons = raw(&wrap_lexicon(
            "<LexicalEntry id=\"t-run-v\"><Lemma writtenForm=\"run\" partOfSpeech=\"v\">\
             <Pronunciation>r\u{28c}n</Pronunciation><Tag category=\"x\">y</Tag></Lemma>\
             <Form writtenForm=\"ran\"/>\
             <Sense id=\"t-run-v-1\" synset=\"t-1\"><Example>they <Synset id=\"bogus\"/> run</Example>\
             <Count>3</Count></Sense>\
             <SyntacticBehaviour subcategorizationFrame=\"Somebody ----s\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"v\"><ILIDefinition>move fast</ILIDefinition>\
             <Definition>move fast</Definition><Example>run home</Example></Synset>\
             <SyntacticBehaviour id=\"t-frame-1\" subcategorizationFrame=\"Something ----s\"/>",
        ))
        .unwrap();
        let lexicon = &lexicons[0];
        assert_eq!(vec!["t-1"], lexicon.synset_order, "a Synset nested in an unmodelled Example is skipped");
        assert_eq!("move fast", lexicon.synsets["t-1"].gloss);
        assert_eq!("run", lexicon.entries["t-run-v"].lemma);
    }

    #[test]
    fn foreign_namespace_elements_are_not_mistaken_for_wn_lmf_elements() {
        let lexicons = raw(&cat_lexicon(
            "",
            "<o:Definition xmlns:o=\"urn:example:other\">not a gloss</o:Definition>\
             <Definition>a feline</Definition>\
             <o:SynsetRelation xmlns:o=\"urn:example:other\" relType=\"hypernym\" target=\"t-9\"/>",
        ))
        .unwrap();
        let synset = &lexicons[0].synsets["t-1"];
        assert_eq!("a feline", synset.gloss);
        assert!(synset.relations.is_empty(), "{:?}", synset.relations);

        let lexicons = raw(&wrap_lexicon(
            "<x:LexicalEntry xmlns:x=\"urn:example:other\" id=\"t-dog-n\">\
             <x:Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/></x:LexicalEntry>",
        ))
        .unwrap();
        assert!(lexicons[0].entries.is_empty(), "{:?}", lexicons[0].entry_order);
    }

    #[test]
    fn extension_namespace_qualified_elements_are_wn_lmf_elements() {
        let doc = format!(
            "<LexicalResource xmlns:ext=\"{EXT_NAMESPACE}\">\
             <ext:LexiconExtension id=\"x\" label=\"x\" language=\"en\" version=\"1\">\
             <ext:Extends ref=\"base\" version=\"2\"/>\
             <ext:ExternalSynset id=\"base-1\"><SynsetRelation relType=\"hypernym\" target=\"x-1\"/></ext:ExternalSynset>\
             </ext:LexiconExtension></LexicalResource>"
        );
        let lexicons = raw(&doc).unwrap();
        assert!(lexicons[0].extension);
        assert_eq!(Some("base"), lexicons[0].extends_ref.as_ref().map(WnLmfDependency::id));
        assert_eq!("base-1", lexicons[0].external_synsets[0].id);
    }

    #[test]
    fn ids_are_unique_across_every_kind_and_every_lexicon_in_a_document() {
        assert_err_contains(
            raw(&cat_lexicon("", "").replace("<Synset id=\"t-1\"", "<Synset id=\"t-cat-n-1\"")),
            "Duplicate synset id t-cat-n-1",
        );
        assert_err_contains(raw(&cat_lexicon("", "").replace("id=\"t-1\"", "id=\"t\"")), "Duplicate synset id t");
        let two = "<LexicalResource>\
             <Lexicon id=\"a\" label=\"a\" language=\"en\" version=\"1\"><Synset id=\"s\" partOfSpeech=\"n\"/></Lexicon>\
             <Lexicon id=\"b\" label=\"b\" language=\"en\" version=\"1\"><Synset id=\"s\" partOfSpeech=\"n\"/></Lexicon>\
             </LexicalResource>";
        assert_err_contains(raw(two), "Duplicate synset id s");
        assert_err_contains(raw(&two.replace("id=\"b\"", "id=\"a\"")), "Duplicate lexicon id a");
    }

    #[test]
    fn lexicons_in_one_resource_keep_separate_tables() {
        let doc = "<LexicalResource>\
             <Lexicon id=\"a\" label=\"a\" language=\"en\" version=\"1\"><Synset id=\"a-1\" partOfSpeech=\"n\"/></Lexicon>\
             <Lexicon id=\"b\" label=\"b\" language=\"de\" version=\"2\"><Synset id=\"b-1\" partOfSpeech=\"v\"/></Lexicon>\
             </LexicalResource>";
        let lexicons = raw(doc).unwrap();
        assert_eq!(vec![vec!["a-1"], vec!["b-1"]], lexicons.iter().map(|l| l.synset_order.clone()).collect::<Vec<_>>());
    }

    #[test]
    fn a_second_lexical_resource_after_the_first_is_rejected() {
        let doc = "<LexicalResource><Lexicon id=\"a\" label=\"a\" language=\"en\" version=\"1\"/></LexicalResource>\
                   <LexicalResource><Lexicon id=\"b\" label=\"b\" language=\"en\" version=\"1\"/></LexicalResource>";
        assert_err_contains(raw(doc), "second root element <LexicalResource>");
    }

    #[test]
    fn multibyte_characters_split_across_reads_are_reassembled() {
        let doc = cat_lexicon("", "<Definition>\u{e9}t\u{e9} \u{65e5}\u{672c} \u{1F431}</Definition>")
            .replace("writtenForm=\"cat\"", "writtenForm=\"ch\u{e2}t\"");
        let lexicons = parse_raw(OneByteReader(doc.as_bytes()), "test.xml", false).unwrap();
        assert_eq!("ch\u{e2}t", lexicons[0].entries["t-cat-n"].lemma);
        assert_eq!("\u{e9}t\u{e9} \u{65e5}\u{672c} \u{1F431}", lexicons[0].synsets["t-1"].gloss);
    }

    #[test]
    fn invalid_utf8_is_reported_as_a_malformed_document_naming_the_resource() {
        let mut bytes = cat_lexicon("", "").into_bytes();
        let at = bytes.iter().position(|&b| b == b'c').unwrap();
        bytes[at] = 0xC3;
        let err = parse_raw(Cursor::new(bytes), "bad-utf8.xml", false).unwrap_err();
        assert!(matches!(err, WordNetError::Parse(_)), "{err:?}");
        assert!(err.to_string().contains("bad-utf8.xml"), "{err}");
        assert!(err.to_string().contains("UTF-8"), "{err}");
    }

    #[test]
    fn extension_elements_need_a_resolver_and_an_extends() {
        let ext = "<LexicalResource><LexiconExtension id=\"x\" label=\"x\" language=\"en\" version=\"1\">\
                   </LexiconExtension></LexicalResource>";
        assert_err_contains(parse_raw(Cursor::new(ext.as_bytes()), "test.xml", false), "without a resolver");
        assert_err_contains(raw(ext), "LexiconExtension x is missing its Extends element");
        assert_err_contains(raw(&wrap_lexicon("<ExternalSynset id=\"s\"/>")), "outside a LexiconExtension");
        assert_err_contains(raw(&wrap_lexicon("<Extends ref=\"b\" version=\"1\"/>")), "outside a LexiconExtension");
    }
}
