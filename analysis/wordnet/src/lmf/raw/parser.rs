//! Event-driven state machine (`RawParser`) for WN-LMF `<Lexicon>` and `<LexiconExtension>` parsing.

use std::collections::{HashMap, HashSet};

use super::{
    EntryBuilder, ExternalEntryBuilder, ExternalSenseBuilder, ExternalSynsetBuilder, RawEntry, RawExternalEntry,
    RawExternalSense, RawExternalSynset, RawLexicon, RawRelation, RawSense, RawSenseRelation, RawSynset, SenseBuilder,
    SynsetBuilder, claim_document_id, get_attr, parse_pos, read_dependency, read_lexicon_metadata, require_attr,
    require_parent,
};
use crate::error::WordNetError;
use crate::lmf::xml::XmlAttr;
use crate::lmf::{EXT_NAMESPACE, OTHER_RELATION, malformed};

/// Event-driven state for [`super::parse_raw`]: one `handle_*` method per opening element and one
/// `end_*` method per closing element that carries state.
pub(super) struct RawParser<'a> {
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
    pub(super) fn new(resource_name: &'a str, allow_extensions: bool) -> Self {
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

    pub(super) fn start(
        &mut self,
        namespace_uri: &str,
        name: String,
        attrs: &[XmlAttr],
        line: usize,
    ) -> Result<(), WordNetError> {
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

    pub(super) fn text(&mut self, text: &str) {
        if self.skip_depth == 0
            && let Some(ref mut definition) = self.definition
        {
            definition.push_str(text);
        }
    }

    pub(super) fn end(&mut self, name: &str, line: usize) -> Result<(), WordNetError> {
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

    pub(super) fn finish(self, line: usize) -> Result<Vec<RawLexicon>, WordNetError> {
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
