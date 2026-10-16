//! WN-LMF XML reader (`wn-lmf-1.3.dtd` / `wn-lmf-1.4.dtd`) and the
//! OpenNLP WN-LMF composition extension (`http://opennlp.apache.org/wn-lmf-ext/1.0`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use crate::error::WordNetError;
use crate::folding::{fold, split_on_spaces};
use crate::lexicon::{InMemoryWordNetLexicon, LemmaKey, Synset};
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;

/// Namespace URI for the OpenNLP WN-LMF 1.4 composition extension.
pub const EXT_NAMESPACE: &str = "http://opennlp.apache.org/wn-lmf-ext/1.0";

const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
const MALFORMED_PREFIX: &str = "Malformed WN-LMF document ";
const OTHER_RELATION: &str = "other";

/// Maximum extension levels in a dependency chain, excluding the plain base lexicon.
pub const MAX_EXTENSION_DEPTH: usize = 16;

/// Qualified XML attribute or element name `(namespace_uri, local_name)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QName {
    namespace_uri: String,
    local_name: String,
}

impl QName {
    /// Creates a new qualified name.
    pub fn new(namespace_uri: impl Into<String>, local_name: impl Into<String>) -> Self {
        Self { namespace_uri: namespace_uri.into(), local_name: local_name.into() }
    }

    /// Creates a qualified name in the default (empty) namespace.
    pub fn local(local_name: impl Into<String>) -> Self {
        Self::new(String::new(), local_name)
    }

    /// Returns the namespace URI (empty if unprefixed).
    pub fn namespace_uri(&self) -> &str {
        &self.namespace_uri
    }

    /// Returns the local attribute/element name.
    pub fn local_name(&self) -> &str {
        &self.local_name
    }
}

/// Dependency reference declared by `<Requires>` or `<Extends>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WnLmfDependency {
    id: String,
    version: String,
    url: Option<String>,
}

impl WnLmfDependency {
    /// Creates a new dependency declaration.
    pub fn new(id: impl Into<String>, version: impl Into<String>, url: Option<String>) -> Result<Self, WordNetError> {
        let id = id.into();
        let version = version.into();
        if id.is_empty() {
            return Err(WordNetError::InvalidInput("WnLmfDependency ref/id must not be empty".to_string()));
        }
        if version.is_empty() {
            return Err(WordNetError::InvalidInput("WnLmfDependency version must not be empty".to_string()));
        }
        if let Some(ref u) = url
            && u.is_empty()
        {
            return Err(WordNetError::InvalidInput("WnLmfDependency url must not be empty when present".to_string()));
        }
        Ok(Self { id, version, url })
    }

    /// Returns the referenced lexicon identifier (`ref` attribute).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the referenced lexicon identifier (`ref` attribute).
    pub fn r#ref(&self) -> &str {
        &self.id
    }

    /// Returns the required lexicon version (`version` attribute).
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Returns the optional lexicon URL (`url` attribute).
    pub fn url(&self) -> Option<&str> {
        self.url.as_deref()
    }
}

/// Materialized `<Lexicon>` or composed `<LexiconExtension>` entry within a [`WnLmfResource`].
#[derive(Debug, Clone)]
pub struct WnLmfLexicon {
    id: String,
    label: String,
    language: String,
    version: String,
    metadata: BTreeMap<QName, String>,
    dependencies: Vec<WnLmfDependency>,
    extension_of: Option<WnLmfDependency>,
    knowledge_base: InMemoryWordNetLexicon,
}

impl WnLmfLexicon {
    /// Returns the lexicon identifier (`id`).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the human-readable lexicon label (`label`).
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the BCP-47 language tag (`language`).
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Returns the lexicon version string (`version`).
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Returns non-identity attributes declared on `<Lexicon>` or `<LexiconExtension>`
    /// (such as `email`, `license`, `url`, `citation`, `logo`, and `dc:*`), keyed by [`QName`].
    pub fn metadata(&self) -> &BTreeMap<QName, String> {
        &self.metadata
    }

    /// Returns the `<Requires>` dependencies declared by this lexicon.
    pub fn dependencies(&self) -> &[WnLmfDependency] {
        &self.dependencies
    }

    /// Returns the `<Extends>` base lexicon reference if this lexicon was produced from a `<LexiconExtension>`.
    pub fn extension_of(&self) -> Option<&WnLmfDependency> {
        self.extension_of.as_ref()
    }

    /// Returns the materialized [`InMemoryWordNetLexicon`].
    pub fn knowledge_base(&self) -> &InMemoryWordNetLexicon {
        &self.knowledge_base
    }

    /// Consumes this lexicon wrapper and returns the underlying [`InMemoryWordNetLexicon`].
    pub fn into_knowledge_base(self) -> InMemoryWordNetLexicon {
        self.knowledge_base
    }
}

/// Parsed `<LexicalResource>` containing one or more materialized [`WnLmfLexicon`] entries.
#[derive(Debug, Clone)]
pub struct WnLmfResource {
    lexicons: Vec<WnLmfLexicon>,
    by_id: HashMap<String, usize>,
}

impl WnLmfResource {
    fn new(lexicons: Vec<WnLmfLexicon>) -> Result<Self, WordNetError> {
        let mut by_id = HashMap::with_capacity(lexicons.len());
        for (idx, lex) in lexicons.iter().enumerate() {
            if by_id.insert(lex.id().to_string(), idx).is_some() {
                return Err(WordNetError::Parse(format!("Duplicate lexicon id: {}", lex.id())));
            }
        }
        Ok(Self { lexicons, by_id })
    }

    /// Returns all materialized lexicons in document order.
    pub fn lexicons(&self) -> &[WnLmfLexicon] {
        &self.lexicons
    }

    /// Consumes this resource and returns all materialized lexicons.
    pub fn into_lexicons(self) -> Vec<WnLmfLexicon> {
        self.lexicons
    }

    /// Looks up a lexicon by its `id` attribute.
    pub fn lexicon(&self, id: &str) -> Option<&WnLmfLexicon> {
        self.by_id.get(id).map(|&idx| &self.lexicons[idx])
    }
}

/// Source returned by a [`WnLmfResolver`] when resolving external lexicon dependencies.
pub enum WnLmfSource {
    /// Named raw XML bytes `(name, bytes)`.
    NamedBytes(String, Vec<u8>),
    /// Raw XML bytes (e.g. in-memory string or byte buffer).
    Bytes(Vec<u8>),
    /// Filesystem path to an XML file.
    File(PathBuf),
    /// Named boxed [`Read`] stream `(name, reader)`.
    NamedReader(String, Box<dyn Read>),
    /// Boxed [`Read`] stream.
    Reader(Box<dyn Read>),
}

impl WnLmfSource {
    /// Creates a named byte source.
    pub fn named_bytes(name: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Result<Self, WordNetError> {
        let name = name.into();
        if name.is_empty() {
            return Err(WordNetError::InvalidInput("WnLmfSource name must not be empty".to_string()));
        }
        Ok(Self::NamedBytes(name, bytes.into()))
    }

    /// Returns the source name used in diagnostic messages.
    pub fn name(&self) -> &str {
        match self {
            Self::NamedBytes(name, _) | Self::NamedReader(name, _) => name,
            Self::File(path) => path.to_str().unwrap_or("<file>"),
            Self::Bytes(_) | Self::Reader(_) => "<resolved>",
        }
    }
}

/// Callback resolver for `<Extends>` dependencies during multi-file composition.
pub trait WnLmfResolver {
    /// Resolves a dependency declaration to an XML stream, or returns `Ok(None)` if unknown.
    fn resolve(&self, dependency: &WnLmfDependency) -> Result<Option<WnLmfSource>, WordNetError>;
}

impl<F> WnLmfResolver for F
where
    F: Fn(&WnLmfDependency) -> Result<Option<WnLmfSource>, WordNetError>,
{
    fn resolve(&self, dependency: &WnLmfDependency) -> Result<Option<WnLmfSource>, WordNetError> {
        (self)(dependency)
    }
}

/// Exact 59 synset-only relation names in WN-LMF 1.4.
fn is_synset_only_rel_type(rel_type: &str) -> bool {
    matches!(
        rel_type,
        "attribute"
            | "be_in_state"
            | "causes"
            | "classified_by"
            | "classifies"
            | "co_agent_instrument"
            | "co_agent_patient"
            | "co_agent_result"
            | "co_instrument_agent"
            | "co_instrument_patient"
            | "co_instrument_result"
            | "co_patient_agent"
            | "co_patient_instrument"
            | "co_result_agent"
            | "co_result_instrument"
            | "co_role"
            | "direction"
            | "entails"
            | "eq_synonym"
            | "holo_location"
            | "holo_member"
            | "holo_part"
            | "holo_portion"
            | "holo_substance"
            | "holonym"
            | "hypernym"
            | "hyponym"
            | "in_manner"
            | "instance_hypernym"
            | "instance_hyponym"
            | "involved"
            | "involved_agent"
            | "involved_direction"
            | "involved_instrument"
            | "involved_location"
            | "involved_patient"
            | "involved_result"
            | "involved_source_direction"
            | "involved_target_direction"
            | "ir_synonym"
            | "is_caused_by"
            | "is_entailed_by"
            | "is_subevent_of"
            | "manner_of"
            | "mero_location"
            | "mero_member"
            | "mero_part"
            | "mero_portion"
            | "mero_substance"
            | "meronym"
            | "patient"
            | "restricted_by"
            | "restricts"
            | "role"
            | "source_direction"
            | "state_of"
            | "subevent"
            | "target_direction"
    )
}

/// Exact 21 sense-only relation names in WN-LMF 1.4.
fn is_sense_only_rel_type(rel_type: &str) -> bool {
    matches!(
        rel_type,
        "body_part"
            | "by_means_of"
            | "derivation"
            | "destination"
            | "event"
            | "has_metaphor"
            | "has_metonym"
            | "material"
            | "metaphor"
            | "metonym"
            | "participle"
            | "pertainym"
            | "property"
            | "secondary_aspect_ip"
            | "secondary_aspect_pi"
            | "simple_aspect_ip"
            | "simple_aspect_pi"
            | "state"
            | "undergoer"
            | "uses"
            | "vehicle"
    )
}

fn malformed(resource_name: &str, line: Option<usize>, message: impl AsRef<str>) -> WordNetError {
    match line {
        Some(l) if l > 0 => {
            WordNetError::Parse(format!("{MALFORMED_PREFIX}{resource_name} at line {l}: {}", message.as_ref()))
        }
        _ => WordNetError::Parse(format!("{MALFORMED_PREFIX}{resource_name}: {}", message.as_ref())),
    }
}

#[derive(Debug, Clone)]
struct RawRelation {
    rel_type: String,
    target: String,
    line: usize,
}

#[derive(Debug, Clone)]
struct RawSenseRelation {
    rel_type: String,
    target: String,
    line: usize,
}

#[derive(Debug, Clone)]
struct RawEntry {
    id: String,
    lemma: String,
    pos: WordNetPos,
    line: usize,
}

#[derive(Debug, Clone)]
struct RawSense {
    id: String,
    entry_id: String,
    synset_id: String,
    relations: Vec<RawSenseRelation>,
    line: usize,
}

#[derive(Debug, Clone)]
struct RawExternalSense {
    id: String,
    relations: Vec<RawSenseRelation>,
    line: usize,
}

#[derive(Debug, Clone)]
struct RawExternalEntry {
    id: String,
    external_senses: Vec<RawExternalSense>,
    line: usize,
}

#[derive(Debug, Clone)]
struct RawSynset {
    id: String,
    pos: WordNetPos,
    members: Option<String>,
    extra_members: Vec<String>,
    gloss: String,
    relations: Vec<RawRelation>,
    line: usize,
}

#[derive(Debug, Clone)]
struct RawExternalSynset {
    id: String,
    definition: Option<String>,
    relations: Vec<RawRelation>,
    line: usize,
}

#[derive(Debug, Clone)]
struct RawLexicon {
    extension: bool,
    id: String,
    label: String,
    language: String,
    version: String,
    metadata: BTreeMap<QName, String>,
    requires: Vec<WnLmfDependency>,
    extends_ref: Option<WnLmfDependency>,
    entries: HashMap<String, RawEntry>,
    entry_order: Vec<String>,
    senses: HashMap<String, RawSense>,
    sense_order: Vec<String>,
    synsets: HashMap<String, RawSynset>,
    synset_order: Vec<String>,
    external_entries: Vec<RawExternalEntry>,
    external_synsets: Vec<RawExternalSynset>,
    line: usize,
}

/// Reader for WordNet LMF (`WN-LMF`) XML files and the OpenNLP WN-LMF composition extension.
pub struct WnLmfReader;

impl WnLmfReader {
    /// Reads a single-lexicon WN-LMF XML file from `path` into an [`InMemoryWordNetLexicon`].
    pub fn read(path: impl AsRef<Path>) -> Result<InMemoryWordNetLexicon, WordNetError> {
        let p = path.as_ref();
        if !p.is_file() {
            return Err(WordNetError::InvalidInput(format!(
                "file does not exist or is not a regular file: {}",
                p.display()
            )));
        }
        let resource = Self::read_resource(p)?;
        only_lexicon(resource, &p.display().to_string())
    }

    /// Reads a single-lexicon WN-LMF XML file from `path`, composing a `<LexiconExtension>`
    /// against base lexicons supplied by `resolver`.
    pub fn read_with_resolver<Res: WnLmfResolver>(
        path: impl AsRef<Path>,
        resolver: &Res,
    ) -> Result<InMemoryWordNetLexicon, WordNetError> {
        let p = path.as_ref();
        if !p.is_file() {
            return Err(WordNetError::InvalidInput(format!(
                "file does not exist or is not a regular file: {}",
                p.display()
            )));
        }
        let resource = Self::read_resource_with_resolver(p, resolver)?;
        only_lexicon(resource, &p.display().to_string())
    }

    /// Reads a single-lexicon WN-LMF XML stream from `reader` into an [`InMemoryWordNetLexicon`].
    pub fn read_from_reader<R: Read>(reader: R) -> Result<InMemoryWordNetLexicon, WordNetError> {
        Self::read_named(reader, "stream.xml")
    }

    /// Reads a single-lexicon WN-LMF XML stream with an explicit `resource_name` for diagnostics.
    pub fn read_named<R: Read>(reader: R, resource_name: &str) -> Result<InMemoryWordNetLexicon, WordNetError> {
        let resource = Self::read_resource_named(reader, resource_name)?;
        only_lexicon(resource, resource_name)
    }

    /// Reads a single-lexicon WN-LMF XML stream with `resource_name` and `resolver`.
    pub fn read_named_with_resolver<R: Read, Res: WnLmfResolver>(
        reader: R,
        resource_name: &str,
        resolver: &Res,
    ) -> Result<InMemoryWordNetLexicon, WordNetError> {
        let resource = Self::read_resource_named_with_resolver(reader, resource_name, resolver)?;
        only_lexicon(resource, resource_name)
    }

    /// Reads a `<LexicalResource>` from `path` without an external dependency resolver.
    pub fn read_resource(path: impl AsRef<Path>) -> Result<WnLmfResource, WordNetError> {
        let p = path.as_ref();
        if !p.is_file() {
            return Err(WordNetError::InvalidInput(format!(
                "file does not exist or is not a regular file: {}",
                p.display()
            )));
        }
        let file = File::open(p)?;
        read_resource_internal::<_, fn(&WnLmfDependency) -> Result<Option<WnLmfSource>, WordNetError>>(
            BufReader::new(file),
            &p.display().to_string(),
            None,
        )
    }

    /// Reads a `<LexicalResource>` from `reader` without an external dependency resolver.
    pub fn read_resource_from_reader<R: Read>(reader: R) -> Result<WnLmfResource, WordNetError> {
        Self::read_resource_named(reader, "stream.xml")
    }

    /// Reads a `<LexicalResource>` from `reader` with `resource_name` and no resolver.
    pub fn read_resource_named<R: Read>(reader: R, resource_name: &str) -> Result<WnLmfResource, WordNetError> {
        read_resource_internal::<_, fn(&WnLmfDependency) -> Result<Option<WnLmfSource>, WordNetError>>(
            reader,
            resource_name,
            None,
        )
    }

    /// Reads a `<LexicalResource>` from `path`, resolving `<Extends>` dependencies via `resolver`.
    pub fn read_resource_with_resolver<Res: WnLmfResolver>(
        path: impl AsRef<Path>,
        resolver: &Res,
    ) -> Result<WnLmfResource, WordNetError> {
        let p = path.as_ref();
        if !p.is_file() {
            return Err(WordNetError::InvalidInput(format!(
                "file does not exist or is not a regular file: {}",
                p.display()
            )));
        }
        let file = File::open(p)?;
        read_resource_internal(BufReader::new(file), &p.display().to_string(), Some(resolver))
    }

    /// Reads a `<LexicalResource>` from `reader`, resolving `<Extends>` dependencies via `resolver`.
    pub fn read_resource_with_resolver_from_reader<R: Read, Res: WnLmfResolver>(
        reader: R,
        resolver: &Res,
    ) -> Result<WnLmfResource, WordNetError> {
        Self::read_resource_named_with_resolver(reader, "stream.xml", resolver)
    }

    /// Reads a `<LexicalResource>` from `reader` with `resource_name` and `resolver`.
    pub fn read_resource_named_with_resolver<R: Read, Res: WnLmfResolver>(
        reader: R,
        resource_name: &str,
        resolver: &Res,
    ) -> Result<WnLmfResource, WordNetError> {
        read_resource_internal(reader, resource_name, Some(resolver))
    }
}

fn only_lexicon(resource: WnLmfResource, resource_name: &str) -> Result<InMemoryWordNetLexicon, WordNetError> {
    let mut lexicons = resource.into_lexicons();
    if lexicons.len() != 1 {
        return Err(WordNetError::Parse(format!(
            "WN-LMF resource {resource_name} contains {} lexicons; use WnLmfReader::read_resource to preserve their boundaries",
            lexicons.len()
        )));
    }
    Ok(lexicons.remove(0).into_knowledge_base())
}

fn read_resource_internal<R: Read, Res: WnLmfResolver>(
    reader: R,
    resource_name: &str,
    resolver: Option<&Res>,
) -> Result<WnLmfResource, WordNetError> {
    let raw_lexicons = parse_raw(reader, resource_name, resolver.is_some())?;
    reject_same_document_base(&raw_lexicons, resource_name)?;

    let mut composition = resolver.map(Composition::new);
    let mut out = Vec::with_capacity(raw_lexicons.len());
    for raw in raw_lexicons {
        if !raw.extension {
            out.push(descriptor(raw, resource_name)?);
        } else {
            let comp = composition.as_mut().unwrap();
            let composed = comp.compose(&raw, resource_name)?;
            out.push(descriptor(composed.lexicon, resource_name)?);
        }
    }
    WnLmfResource::new(out)
}

fn descriptor(lexicon: RawLexicon, resource_name: &str) -> Result<WnLmfLexicon, WordNetError> {
    let kb = Materializer::new(&lexicon, resource_name).materialize()?;
    Ok(WnLmfLexicon {
        id: lexicon.id,
        label: lexicon.label,
        language: lexicon.language,
        version: lexicon.version,
        metadata: lexicon.metadata,
        dependencies: lexicon.requires,
        extension_of: lexicon.extends_ref,
        knowledge_base: kb,
    })
}

fn reject_same_document_base(lexicons: &[RawLexicon], resource_name: &str) -> Result<(), WordNetError> {
    for (i, ext) in lexicons.iter().enumerate() {
        if !ext.extension {
            continue;
        }
        let ext_ref = ext.extends_ref.as_ref().unwrap();
        for (j, other) in lexicons.iter().enumerate() {
            if i != j && other.id == ext_ref.id() && other.version == ext_ref.version() {
                return Err(malformed(
                    resource_name,
                    Some(ext.line),
                    format!(
                        "LexiconExtension {} extends {} version {}, which occurs in the same document; a base lexicon may not share a file with its extension",
                        ext.id, other.id, other.version
                    ),
                ));
            }
        }
    }
    Ok(())
}

// ============================================================================
// Self-contained Namespace-Aware XML Pull Parser with Line Numbers
// ============================================================================

#[derive(Debug, Clone)]
struct XmlAttr {
    namespace_uri: String,
    local_name: String,
    value: String,
}

#[derive(Debug, Clone)]
enum XmlEvent {
    Start { namespace_uri: String, local_name: String, attrs: Vec<XmlAttr>, line: usize },
    End { namespace_uri: String, local_name: String, line: usize },
    Text(String),
    Eof { line: usize },
}

type XmlElementFrame = (String, String, String, Vec<(String, String)>);

struct XmlPullParser<'a> {
    input: &'a str,
    resource_name: &'a str,
    pos: usize,
    line: usize,
    pending_end: Option<(String, String, usize)>,
    element_stack: Vec<XmlElementFrame>,
}

impl<'a> XmlPullParser<'a> {
    fn new(input: &'a str, resource_name: &'a str) -> Self {
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

    fn next_event(&mut self) -> Result<XmlEvent, WordNetError> {
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
fn unescape_xml(
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

// ============================================================================
// Raw Document Parser
// ============================================================================

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

fn parse_raw<R: Read>(
    mut reader: R,
    resource_name: &str,
    allow_extensions: bool,
) -> Result<Vec<RawLexicon>, WordNetError> {
    let mut xml_string = String::new();
    reader.read_to_string(&mut xml_string)?;
    // XML 1.0 section 2.11: a parser hands `\r\n` and a lone `\r` to the application as `\n`.
    if xml_string.contains('\r') {
        xml_string = xml_string.replace("\r\n", "\n").replace('\r', "\n");
    }
    let mut parser = XmlPullParser::new(&xml_string, resource_name);

    let mut lexicons: Vec<RawLexicon> = Vec::new();
    let mut lexicon_ids: HashSet<String> = HashSet::new();
    let mut document_ids: HashSet<String> = HashSet::new();
    let mut elements: Vec<String> = Vec::new();

    let mut current_extension = false;
    let mut current_lexicon_id: Option<String> = None;
    let mut current_lexicon_label = String::new();
    let mut current_lexicon_language = String::new();
    let mut current_lexicon_version = String::new();
    let mut current_lexicon_metadata: BTreeMap<QName, String> = BTreeMap::new();
    let mut current_lexicon_line = 0usize;
    let mut current_requires: Vec<WnLmfDependency> = Vec::new();
    let mut current_extends: Option<WnLmfDependency> = None;

    let mut entries: HashMap<String, RawEntry> = HashMap::new();
    let mut entry_order: Vec<String> = Vec::new();
    let mut senses: HashMap<String, RawSense> = HashMap::new();
    let mut sense_order: Vec<String> = Vec::new();
    let mut synsets: HashMap<String, RawSynset> = HashMap::new();
    let mut synset_order: Vec<String> = Vec::new();
    let mut external_entries: Vec<RawExternalEntry> = Vec::new();
    let mut external_synsets: Vec<RawExternalSynset> = Vec::new();

    let mut current_entry_id: Option<String> = None;
    let mut current_entry_lemma: Option<String> = None;
    let mut current_entry_pos: Option<WordNetPos> = None;
    let mut current_entry_line = 0usize;

    let mut current_sense_id: Option<String> = None;
    let mut current_sense_entry_id: Option<String> = None;
    let mut current_sense_synset: Option<String> = None;
    let mut current_sense_line = 0usize;
    let mut current_sense_relations: Vec<RawSenseRelation> = Vec::new();

    let mut current_synset: Option<SynsetBuilder> = None;
    let mut current_external_entry: Option<ExternalEntryBuilder> = None;
    let mut current_external_sense: Option<ExternalSenseBuilder> = None;
    let mut current_external_synset: Option<ExternalSynsetBuilder> = None;
    let mut definition_buffer: Option<String> = None;

    let mut skip_depth = 0usize;

    loop {
        match parser.next_event()? {
            XmlEvent::Start { namespace_uri: _ns_uri, local_name: name, attrs, line } => {
                if skip_depth > 0 {
                    skip_depth += 1;
                    continue;
                }
                if definition_buffer.is_some() {
                    return Err(malformed(
                        resource_name,
                        Some(line),
                        format!("Unexpected child element <{name}> inside Definition"),
                    ));
                }

                match name.as_str() {
                    "LexicalResource" => {
                        if !elements.is_empty() {
                            return Err(malformed(resource_name, Some(line), "Nested LexicalResource"));
                        }
                        elements.push(name);
                    }
                    "Lexicon" | "LexiconExtension" => {
                        let is_ext = name == "LexiconExtension";
                        if is_ext && !allow_extensions {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                "LexiconExtension is not supported without a resolver; use a read or read_resource overload that takes a WnLmfResolver",
                            ));
                        }
                        if let Some(ref outer_id) = current_lexicon_id {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                format!("Nested {name} inside {outer_id}"),
                            ));
                        }
                        require_parent(&elements, &name, "LexicalResource", None, resource_name, line)?;
                        let id = require_attr(&attrs, &name, "id", resource_name, line)?;
                        if !lexicon_ids.insert(id.clone()) {
                            return Err(malformed(resource_name, Some(line), format!("Duplicate lexicon id {id}")));
                        }
                        claim_document_id(&mut document_ids, &id, "lexicon", resource_name, line)?;
                        current_extension = is_ext;
                        current_lexicon_id = Some(id);
                        current_lexicon_label = require_attr(&attrs, &name, "label", resource_name, line)?;
                        current_lexicon_language = require_attr(&attrs, &name, "language", resource_name, line)?;
                        current_lexicon_version = require_attr(&attrs, &name, "version", resource_name, line)?;
                        current_lexicon_line = line;
                        current_lexicon_metadata = read_lexicon_metadata(&attrs);
                        elements.push(name);
                    }
                    "Extends" => {
                        if current_lexicon_id.is_none() || !current_extension {
                            return Err(malformed(resource_name, Some(line), "Extends outside a LexiconExtension"));
                        }
                        require_parent(&elements, "Extends", "LexiconExtension", None, resource_name, line)?;
                        if current_extends.is_some() {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                format!(
                                    "Repeated Extends in LexiconExtension {}",
                                    current_lexicon_id.as_deref().unwrap_or_default()
                                ),
                            ));
                        }
                        current_extends = Some(read_dependency(&attrs, "Extends", resource_name, line)?);
                        elements.push(name);
                    }
                    "Requires" => {
                        if current_lexicon_id.is_none() {
                            return Err(malformed(resource_name, Some(line), "Requires outside a Lexicon"));
                        }
                        require_parent(
                            &elements,
                            "Requires",
                            "Lexicon",
                            Some("LexiconExtension"),
                            resource_name,
                            line,
                        )?;
                        current_requires.push(read_dependency(&attrs, "Requires", resource_name, line)?);
                        elements.push(name);
                    }
                    "LexicalEntry" => {
                        if let Some(ref outer_id) = current_entry_id {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                format!("Nested LexicalEntry inside {outer_id}"),
                            ));
                        }
                        if current_lexicon_id.is_none() {
                            return Err(malformed(resource_name, Some(line), "LexicalEntry outside a Lexicon"));
                        }
                        require_parent(
                            &elements,
                            "LexicalEntry",
                            "Lexicon",
                            Some("LexiconExtension"),
                            resource_name,
                            line,
                        )?;
                        let id = require_attr(&attrs, "LexicalEntry", "id", resource_name, line)?;
                        claim_document_id(&mut document_ids, &id, "lexical entry", resource_name, line)?;
                        current_entry_id = Some(id);
                        current_entry_lemma = None;
                        current_entry_pos = None;
                        current_entry_line = line;
                        elements.push(name);
                    }
                    "ExternalLexicalEntry" => {
                        if current_lexicon_id.is_none() || !current_extension {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                "ExternalLexicalEntry outside a LexiconExtension",
                            ));
                        }
                        require_parent(
                            &elements,
                            "ExternalLexicalEntry",
                            "LexiconExtension",
                            None,
                            resource_name,
                            line,
                        )?;
                        let id = require_attr(&attrs, "ExternalLexicalEntry", "id", resource_name, line)?;
                        claim_document_id(&mut document_ids, &id, "external lexical entry", resource_name, line)?;
                        current_external_entry = Some(ExternalEntryBuilder { id, line, external_senses: Vec::new() });
                        elements.push(name);
                    }
                    "Lemma" => {
                        let Some(ref entry_id) = current_entry_id else {
                            return Err(malformed(resource_name, Some(line), "Lemma outside a LexicalEntry"));
                        };
                        if current_entry_lemma.is_some() {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                format!("Duplicate Lemma in LexicalEntry {entry_id}"),
                            ));
                        }
                        require_parent(&elements, "Lemma", "LexicalEntry", None, resource_name, line)?;
                        let wf = require_attr(&attrs, "Lemma", "writtenForm", resource_name, line)?;
                        let pos_code = require_attr(&attrs, "Lemma", "partOfSpeech", resource_name, line)?;
                        let pos = parse_pos(&pos_code, resource_name, line)?;
                        current_entry_lemma = Some(wf);
                        current_entry_pos = Some(pos);
                        elements.push(name);
                    }
                    "Sense" => {
                        if let Some(ref outer_sense) = current_sense_id {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                format!("Nested Sense inside {outer_sense}"),
                            ));
                        }
                        let owner = if let Some(ref ext_entry) = current_external_entry {
                            ext_entry.id.clone()
                        } else if let Some(ref entry_id) = current_entry_id {
                            if current_entry_lemma.is_none() {
                                return Err(malformed(
                                    resource_name,
                                    Some(line),
                                    format!("Sense before its entry's Lemma in LexicalEntry {entry_id}"),
                                ));
                            }
                            entry_id.clone()
                        } else {
                            return Err(malformed(resource_name, Some(line), "Sense outside a LexicalEntry"));
                        };
                        require_parent(
                            &elements,
                            "Sense",
                            "LexicalEntry",
                            Some("ExternalLexicalEntry"),
                            resource_name,
                            line,
                        )?;
                        let id = require_attr(&attrs, "Sense", "id", resource_name, line)?;
                        let synset_id = require_attr(&attrs, "Sense", "synset", resource_name, line)?;
                        claim_document_id(&mut document_ids, &id, "sense", resource_name, line)?;
                        current_sense_id = Some(id);
                        current_sense_entry_id = Some(owner);
                        current_sense_synset = Some(synset_id);
                        current_sense_line = line;
                        current_sense_relations.clear();
                        elements.push(name);
                    }
                    "ExternalSense" => {
                        if current_external_entry.is_none() {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                "ExternalSense outside an ExternalLexicalEntry",
                            ));
                        }
                        require_parent(&elements, "ExternalSense", "ExternalLexicalEntry", None, resource_name, line)?;
                        let id = require_attr(&attrs, "ExternalSense", "id", resource_name, line)?;
                        claim_document_id(&mut document_ids, &id, "external sense", resource_name, line)?;
                        current_external_sense = Some(ExternalSenseBuilder { id, line, relations: Vec::new() });
                        elements.push(name);
                    }
                    "SenseRelation" => {
                        if current_external_sense.is_none() && current_sense_id.is_none() {
                            return Err(malformed(resource_name, Some(line), "SenseRelation outside a Sense"));
                        }
                        require_parent(
                            &elements,
                            "SenseRelation",
                            "Sense",
                            Some("ExternalSense"),
                            resource_name,
                            line,
                        )?;
                        let rel = RawSenseRelation {
                            rel_type: require_attr(&attrs, "SenseRelation", "relType", resource_name, line)?,
                            target: require_attr(&attrs, "SenseRelation", "target", resource_name, line)?,
                            line,
                        };
                        if let Some(ref mut ext_sense) = current_external_sense {
                            ext_sense.relations.push(rel);
                        } else {
                            current_sense_relations.push(rel);
                        }
                        elements.push(name);
                    }
                    "Synset" => {
                        if let Some(ref outer_synset) = current_synset {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                format!("Nested Synset inside {}", outer_synset.id),
                            ));
                        }
                        if current_lexicon_id.is_none() {
                            return Err(malformed(resource_name, Some(line), "Synset outside a Lexicon"));
                        }
                        require_parent(&elements, "Synset", "Lexicon", Some("LexiconExtension"), resource_name, line)?;
                        let id = require_attr(&attrs, "Synset", "id", resource_name, line)?;
                        let pos_code = require_attr(&attrs, "Synset", "partOfSpeech", resource_name, line)?;
                        let pos = parse_pos(&pos_code, resource_name, line)?;
                        claim_document_id(&mut document_ids, &id, "synset", resource_name, line)?;
                        let members = get_attr(&attrs, "members").map(str::to_string);
                        current_synset = Some(SynsetBuilder {
                            id,
                            pos,
                            members,
                            line,
                            relations: Vec::new(),
                            definitions: Vec::new(),
                        });
                        elements.push(name);
                    }
                    "ExternalSynset" => {
                        if current_lexicon_id.is_none() || !current_extension {
                            return Err(malformed(
                                resource_name,
                                Some(line),
                                "ExternalSynset outside a LexiconExtension",
                            ));
                        }
                        require_parent(&elements, "ExternalSynset", "LexiconExtension", None, resource_name, line)?;
                        let id = require_attr(&attrs, "ExternalSynset", "id", resource_name, line)?;
                        claim_document_id(&mut document_ids, &id, "external synset", resource_name, line)?;
                        current_external_synset =
                            Some(ExternalSynsetBuilder { id, line, relations: Vec::new(), definition: None });
                        elements.push(name);
                    }
                    "Definition" => {
                        if current_synset.is_none() && current_external_synset.is_none() {
                            return Err(malformed(resource_name, Some(line), "Definition outside a Synset"));
                        }
                        require_parent(&elements, "Definition", "Synset", Some("ExternalSynset"), resource_name, line)?;
                        definition_buffer = Some(String::new());
                        elements.push(name);
                    }
                    "SynsetRelation" => {
                        if current_synset.is_none() && current_external_synset.is_none() {
                            return Err(malformed(resource_name, Some(line), "SynsetRelation outside a Synset"));
                        }
                        require_parent(
                            &elements,
                            "SynsetRelation",
                            "Synset",
                            Some("ExternalSynset"),
                            resource_name,
                            line,
                        )?;
                        let rel_type = require_attr(&attrs, "SynsetRelation", "relType", resource_name, line)?;
                        let target = require_attr(&attrs, "SynsetRelation", "target", resource_name, line)?;
                        if rel_type != OTHER_RELATION {
                            let rel = RawRelation { rel_type, target, line };
                            if let Some(ref mut syn) = current_synset {
                                syn.relations.push(rel);
                            } else if let Some(ref mut ext_syn) = current_external_synset {
                                ext_syn.relations.push(rel);
                            }
                        }
                        elements.push(name);
                    }
                    _ => {
                        skip_depth = 1;
                    }
                }
            }
            XmlEvent::Text(text) => {
                if skip_depth == 0
                    && let Some(ref mut def) = definition_buffer
                {
                    def.push_str(&text);
                }
            }
            XmlEvent::End { namespace_uri: _ns_uri, local_name: name, line } => {
                if skip_depth > 0 {
                    skip_depth -= 1;
                    continue;
                }

                elements.pop();
                match name.as_str() {
                    "Definition" => {
                        if let Some(def) = definition_buffer.take() {
                            if let Some(ref mut syn) = current_synset {
                                syn.definitions.push(def);
                            } else if let Some(ref mut ext_syn) = current_external_synset
                                && ext_syn.definition.is_none()
                            {
                                ext_syn.definition = Some(def);
                            }
                        }
                    }
                    "LexicalEntry" => {
                        let entry_id = current_entry_id.take().unwrap();
                        let Some(lemma) = current_entry_lemma.take() else {
                            return Err(malformed(
                                resource_name,
                                Some(current_entry_line),
                                format!("LexicalEntry {entry_id} has no Lemma"),
                            ));
                        };
                        let pos = current_entry_pos.take().unwrap();
                        entry_order.push(entry_id.clone());
                        entries
                            .insert(entry_id.clone(), RawEntry { id: entry_id, lemma, pos, line: current_entry_line });
                    }
                    "Sense" => {
                        if let Some(id) = current_sense_id.take() {
                            let entry_id = current_sense_entry_id.take().unwrap();
                            let synset_id = current_sense_synset.take().unwrap();
                            let relations = std::mem::take(&mut current_sense_relations);
                            sense_order.push(id.clone());
                            senses.insert(
                                id.clone(),
                                RawSense { id, entry_id, synset_id, relations, line: current_sense_line },
                            );
                        }
                    }
                    "ExternalSense" => {
                        if let (Some(ext_entry), Some(ext_sense)) =
                            (current_external_entry.as_mut(), current_external_sense.take())
                        {
                            ext_entry.external_senses.push(RawExternalSense {
                                id: ext_sense.id,
                                relations: ext_sense.relations,
                                line: ext_sense.line,
                            });
                        }
                    }
                    "ExternalLexicalEntry" => {
                        if let Some(ext_entry) = current_external_entry.take() {
                            external_entries.push(RawExternalEntry {
                                id: ext_entry.id,
                                external_senses: ext_entry.external_senses,
                                line: ext_entry.line,
                            });
                        }
                    }
                    "Synset" => {
                        if let Some(syn) = current_synset.take() {
                            synset_order.push(syn.id.clone());
                            synsets.insert(
                                syn.id.clone(),
                                RawSynset {
                                    id: syn.id,
                                    pos: syn.pos,
                                    members: syn.members,
                                    extra_members: Vec::new(),
                                    gloss: syn.definitions.join("; "),
                                    relations: syn.relations,
                                    line: syn.line,
                                },
                            );
                        }
                    }
                    "ExternalSynset" => {
                        if let Some(ext_syn) = current_external_synset.take() {
                            external_synsets.push(RawExternalSynset {
                                id: ext_syn.id,
                                definition: ext_syn.definition,
                                relations: ext_syn.relations,
                                line: ext_syn.line,
                            });
                        }
                    }
                    "Lexicon" | "LexiconExtension" => {
                        if let Some(id) = current_lexicon_id.take() {
                            if current_extension && current_extends.is_none() {
                                return Err(malformed(
                                    resource_name,
                                    Some(line),
                                    format!("LexiconExtension {id} is missing its Extends element"),
                                ));
                            }
                            lexicons.push(RawLexicon {
                                extension: current_extension,
                                id,
                                label: std::mem::take(&mut current_lexicon_label),
                                language: std::mem::take(&mut current_lexicon_language),
                                version: std::mem::take(&mut current_lexicon_version),
                                metadata: std::mem::take(&mut current_lexicon_metadata),
                                requires: std::mem::take(&mut current_requires),
                                extends_ref: current_extends.take(),
                                entries: std::mem::take(&mut entries),
                                entry_order: std::mem::take(&mut entry_order),
                                senses: std::mem::take(&mut senses),
                                sense_order: std::mem::take(&mut sense_order),
                                synsets: std::mem::take(&mut synsets),
                                synset_order: std::mem::take(&mut synset_order),
                                external_entries: std::mem::take(&mut external_entries),
                                external_synsets: std::mem::take(&mut external_synsets),
                                line: current_lexicon_line,
                            });
                            current_extension = false;
                        }
                    }
                    _ => {}
                }
            }
            XmlEvent::Eof { line } => {
                if let Some(id) = current_lexicon_id {
                    return Err(malformed(resource_name, Some(line), format!("Unclosed Lexicon {id}")));
                }
                if lexicons.is_empty() {
                    return Err(malformed(resource_name, Some(line), "Document contains no Lexicon"));
                }
                break;
            }
        }
    }

    Ok(lexicons)
}

// ============================================================================
// Materializer
// ============================================================================

struct Materializer<'a> {
    lexicon: &'a RawLexicon,
    resource_name: &'a str,
}

impl<'a> Materializer<'a> {
    fn new(lexicon: &'a RawLexicon, resource_name: &'a str) -> Self {
        Self { lexicon, resource_name }
    }

    fn fail(&self, line: Option<usize>, message: impl AsRef<str>) -> WordNetError {
        malformed(self.resource_name, line, message)
    }

    fn materialize(&self) -> Result<InMemoryWordNetLexicon, WordNetError> {
        let senses_by_synset = self.senses_by_synset()?;
        let mut sense_order: HashMap<LemmaKey, Vec<String>> = HashMap::new();
        for sense_id in &self.lexicon.sense_order {
            let sense = &self.lexicon.senses[sense_id];
            let entry = self.entry(sense)?;
            let key = LemmaKey::of(&entry.lemma, entry.pos);
            let order = sense_order.entry(key).or_default();
            if !order.contains(&sense.synset_id) {
                order.push(sense.synset_id.clone());
            }
        }

        let mut synsets_by_id: HashMap<String, Synset> = HashMap::with_capacity(self.lexicon.synsets.len());
        for synset_id in &self.lexicon.synset_order {
            let raw = &self.lexicon.synsets[synset_id];
            let senses = senses_by_synset.get(raw.id.as_str()).map(Vec::as_slice).unwrap_or(&[]);
            let relations = self.resolve_relations(raw, senses)?;
            let lemmas = self.member_lemmas(raw, senses)?;
            synsets_by_id
                .insert(raw.id.clone(), Synset::new(raw.id.clone(), raw.pos, lemmas, raw.gloss.clone(), relations)?);
        }
        InMemoryWordNetLexicon::new(synsets_by_id, sense_order)
    }

    fn validate(&self) -> Result<(), WordNetError> {
        let senses_by_synset = self.senses_by_synset()?;
        for synset_id in &self.lexicon.synset_order {
            let raw = &self.lexicon.synsets[synset_id];
            let senses = senses_by_synset.get(raw.id.as_str()).map(Vec::as_slice).unwrap_or(&[]);
            self.resolve_relations(raw, senses)?;
            self.member_lemmas(raw, senses)?;
        }
        Ok(())
    }

    fn senses_by_synset(&self) -> Result<HashMap<&str, Vec<&RawSense>>, WordNetError> {
        let mut out: HashMap<&str, Vec<&RawSense>> = HashMap::new();
        for sense_id in &self.lexicon.sense_order {
            let sense = &self.lexicon.senses[sense_id];
            let Some(synset) = self.lexicon.synsets.get(&sense.synset_id) else {
                return Err(self.fail(
                    Some(sense.line),
                    format!(
                        "Sense {} at line {} references undeclared synset {}",
                        sense.id, sense.line, sense.synset_id
                    ),
                ));
            };
            let entry = self.entry(sense)?;
            if entry.pos != synset.pos {
                return Err(self.fail(
                    Some(sense.line),
                    format!(
                        "Sense {} at line {} has part of speech {:?} ({}) but synset {} has {:?} ({})",
                        sense.id,
                        sense.line,
                        entry.pos,
                        pos_upper_name(entry.pos),
                        synset.id,
                        synset.pos,
                        pos_upper_name(synset.pos)
                    ),
                ));
            }
            out.entry(synset.id.as_str()).or_default().push(sense);
        }
        Ok(out)
    }

    fn entry(&self, sense: &RawSense) -> Result<&RawEntry, WordNetError> {
        self.lexicon.entries.get(&sense.entry_id).ok_or_else(|| {
            self.fail(
                Some(sense.line),
                format!("Sense {} belongs to unresolved lexical entry {}", sense.id, sense.entry_id),
            )
        })
    }

    fn resolve_relations(
        &self,
        raw: &RawSynset,
        senses: &[&RawSense],
    ) -> Result<BTreeMap<WordNetRelation, Vec<String>>, WordNetError> {
        let mut combined: Vec<(&str, &str, usize, bool)> = Vec::with_capacity(raw.relations.len());
        for rel in &raw.relations {
            combined.push((&rel.rel_type, &rel.target, rel.line, false));
        }
        for sense in senses {
            for rel in &sense.relations {
                if rel.rel_type == OTHER_RELATION {
                    continue;
                }
                let Some(target_sense) = self.lexicon.senses.get(&rel.target) else {
                    return Err(self.fail(
                        Some(rel.line),
                        format!(
                            "SenseRelation at line {} from sense {} references undeclared sense {}",
                            rel.line, sense.id, rel.target
                        ),
                    ));
                };
                combined.push((&rel.rel_type, &target_sense.synset_id, rel.line, true));
            }
        }

        let mut typed: BTreeMap<WordNetRelation, Vec<String>> = BTreeMap::new();
        for (rel_type, target_id, line, sense_relation) in combined {
            let rel_enum = self.parse_relation(rel_type, raw.pos, line, sense_relation)?;
            let Some(target) = self.lexicon.synsets.get(target_id) else {
                return Err(self.fail(
                    Some(line),
                    format!(
                        "Relation {rel_type} at line {line} on synset {} references undeclared synset {target_id}",
                        raw.id
                    ),
                ));
            };
            let list = typed.entry(rel_enum).or_default();
            if !list.contains(&target.id) {
                list.push(target.id.clone());
            }
        }
        Ok(typed)
    }

    fn member_lemmas(&self, raw: &RawSynset, senses: &[&RawSense]) -> Result<Vec<String>, WordNetError> {
        let mut member_ids: Vec<&str> = Vec::new();
        if let Some(ref members) = raw.members {
            if !members.is_empty() {
                member_ids.extend(split_on_spaces(members));
            } else {
                for sense in senses {
                    member_ids.push(&sense.entry_id);
                }
            }
        } else {
            for sense in senses {
                member_ids.push(&sense.entry_id);
            }
        }
        for extra in &raw.extra_members {
            member_ids.push(extra);
        }

        let mut lemmas: Vec<String> = Vec::with_capacity(member_ids.len());
        let mut folded_lemmas: HashSet<String> = HashSet::with_capacity(member_ids.len());
        for member_id in member_ids {
            let sense = self.lexicon.senses.get(member_id);
            if let Some(s) = sense
                && s.synset_id != raw.id
            {
                return Err(self.fail(
                    Some(raw.line),
                    format!(
                        "Synset {} at line {} lists member sense {} assigned to synset {}",
                        raw.id, raw.line, member_id, s.synset_id
                    ),
                ));
            }
            let entry_id = sense.map_or(member_id, |s| s.entry_id.as_str());
            let Some(entry) = self.lexicon.entries.get(entry_id) else {
                return Err(self.fail(
                    Some(raw.line),
                    format!(
                        "Synset {} at line {} lists undeclared member sense or entry {}",
                        raw.id, raw.line, member_id
                    ),
                ));
            };
            if raw.pos != entry.pos {
                return Err(self.fail(
                    Some(raw.line),
                    format!(
                        "Synset {} at line {} has part of speech {} but member entry {} has {}",
                        raw.id,
                        raw.line,
                        pos_upper_name(raw.pos),
                        entry.id,
                        pos_upper_name(entry.pos)
                    ),
                ));
            }
            if !lemmas.contains(&entry.lemma) {
                lemmas.push(entry.lemma.clone());
                folded_lemmas.insert(fold(&entry.lemma));
            }
        }

        for sense in senses {
            let entry = self.entry(sense)?;
            if !folded_lemmas.contains(&fold(&entry.lemma)) {
                return Err(self.fail(
                    Some(sense.line),
                    format!(
                        "Sense {} at line {} has lemma {} missing from members of synset {}",
                        sense.id, sense.line, entry.lemma, raw.id
                    ),
                ));
            }
        }

        Ok(lemmas)
    }

    fn parse_relation(
        &self,
        rel_type: &str,
        source_pos: WordNetPos,
        line: usize,
        sense_relation: bool,
    ) -> Result<WordNetRelation, WordNetError> {
        if sense_relation && is_synset_only_rel_type(rel_type) {
            return Err(
                self.fail(Some(line), format!("Relation type {rel_type} is not legal on SenseRelation at line {line}"))
            );
        }
        if !sense_relation && is_sense_only_rel_type(rel_type) {
            return Err(self
                .fail(Some(line), format!("Relation type {rel_type} is not legal on SynsetRelation at line {line}")));
        }
        if rel_type == "similar" {
            return Ok(if !sense_relation && source_pos == WordNetPos::Verb {
                WordNetRelation::VerbGroup
            } else {
                WordNetRelation::SimilarTo
            });
        }
        WordNetRelation::from_lmf_name(rel_type)
            .ok_or_else(|| self.fail(Some(line), format!("Unknown relation type {rel_type} at line {line}")))
    }
}

fn pos_upper_name(pos: WordNetPos) -> &'static str {
    match pos {
        WordNetPos::Noun => "NOUN",
        WordNetPos::Verb => "VERB",
        WordNetPos::Adjective => "ADJECTIVE",
        WordNetPos::Adverb => "ADVERB",
    }
}

// ============================================================================
// Extension Composition
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResolutionKey {
    id: String,
    version: String,
}

impl std::fmt::Display for ResolutionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id, self.version)
    }
}

#[derive(Debug, Clone)]
struct ResolvedLexicon {
    lexicon: RawLexicon,
    extension_path: Vec<ResolutionKey>,
}

struct Composition<'a, Res: WnLmfResolver> {
    resolver: &'a Res,
    cache: HashMap<ResolutionKey, ResolvedLexicon>,
    stack: Vec<ResolutionKey>,
}

impl<'a, Res: WnLmfResolver> Composition<'a, Res> {
    fn new(resolver: &'a Res) -> Self {
        Self { resolver, cache: HashMap::new(), stack: Vec::new() }
    }

    fn compose(&mut self, extension: &RawLexicon, resource_name: &str) -> Result<ResolvedLexicon, WordNetError> {
        let key = ResolutionKey { id: extension.id.clone(), version: extension.version.clone() };
        self.validate_path(std::slice::from_ref(&key), resource_name)?;
        self.stack.push(key.clone());
        let res = (|| {
            let ext_ref = extension.extends_ref.as_ref().unwrap();
            let base = self.base(ext_ref, resource_name)?;
            let mut extension_path = Vec::with_capacity(base.extension_path.len() + 1);
            extension_path.push(key.clone());
            extension_path.extend(base.extension_path);
            let merged = self.merge(&base.lexicon, extension, resource_name)?;
            Ok(ResolvedLexicon { lexicon: merged, extension_path })
        })();
        self.stack.pop();
        res
    }

    fn base(&mut self, reference: &WnLmfDependency, resource_name: &str) -> Result<ResolvedLexicon, WordNetError> {
        let key = ResolutionKey { id: reference.id().to_string(), version: reference.version().to_string() };
        if let Some(cached) = self.cache.get(&key).cloned() {
            self.validate_path(&cached.extension_path, resource_name)?;
            return Ok(cached);
        }

        let source = self.resolver.resolve(reference)?.ok_or_else(|| {
            WordNetError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Unresolved base lexicon '{}' version '{}'", reference.id(), reference.version()),
            ))
        })?;
        let source_name = source.name().to_string();
        let resolved_lexicons = match source {
            WnLmfSource::NamedBytes(_, bytes) | WnLmfSource::Bytes(bytes) => {
                parse_raw(std::io::Cursor::new(bytes), &source_name, true)?
            }
            WnLmfSource::File(path) => {
                let file = File::open(&path)?;
                parse_raw(BufReader::new(file), &source_name, true)?
            }
            WnLmfSource::NamedReader(_, r) | WnLmfSource::Reader(r) => parse_raw(r, &source_name, true)?,
        };

        reject_same_document_base(&resolved_lexicons, &source_name)?;
        let matched = resolved_lexicons
            .into_iter()
            .find(|lex| lex.id == reference.id() && lex.version == reference.version())
            .ok_or_else(|| {
                malformed(
                    resource_name,
                    None,
                    format!(
                        "Resolved resource {source_name} does not contain lexicon {} version {}",
                        reference.id(),
                        reference.version()
                    ),
                )
            })?;

        let result = if matched.extension {
            self.compose(&matched, &source_name)?
        } else {
            ResolvedLexicon { lexicon: matched, extension_path: Vec::new() }
        };
        Materializer::new(&result.lexicon, &source_name).validate()?;
        self.cache.insert(key, result.clone());
        Ok(result)
    }

    fn validate_path(&self, extension_path: &[ResolutionKey], resource_name: &str) -> Result<(), WordNetError> {
        for (i, key) in extension_path.iter().enumerate() {
            if self.stack.contains(key) {
                let mut chain: Vec<String> = self.stack.iter().map(ToString::to_string).collect();
                for k in &extension_path[..=i] {
                    chain.push(k.to_string());
                }
                return Err(malformed(
                    resource_name,
                    None,
                    format!("Extension dependency cycle: {}", chain.join(" -> ")),
                ));
            }
            if self.stack.len() + i >= MAX_EXTENSION_DEPTH {
                return Err(malformed(
                    resource_name,
                    None,
                    format!("Extension chain at {key} exceeds the maximum composition depth of {MAX_EXTENSION_DEPTH}"),
                ));
            }
        }
        Ok(())
    }

    fn merge(
        &self,
        base: &RawLexicon,
        extension: &RawLexicon,
        resource_name: &str,
    ) -> Result<RawLexicon, WordNetError> {
        for entry_id in &extension.entry_order {
            let entry = &extension.entries[entry_id];
            self.require_new_id(base, extension, &entry.id, "lexical entry", entry.line, resource_name)?;
        }
        for sense_id in &extension.sense_order {
            let sense = &extension.senses[sense_id];
            self.require_new_id(base, extension, &sense.id, "sense", sense.line, resource_name)?;
        }
        for synset_id in &extension.synset_order {
            let synset = &extension.synsets[synset_id];
            self.require_new_id(base, extension, &synset.id, "synset", synset.line, resource_name)?;
        }

        let mut external_synsets: HashMap<String, &RawExternalSynset> = HashMap::new();
        for ext_syn in &extension.external_synsets {
            if !base.synsets.contains_key(&ext_syn.id) {
                return Err(malformed(
                    resource_name,
                    Some(ext_syn.line),
                    format!(
                        "ExternalSynset {} at line {} does not resolve to a synset in base {} version {}{}",
                        ext_syn.id,
                        ext_syn.line,
                        base.id,
                        base.version,
                        kind_hint(base, &ext_syn.id)
                    ),
                ));
            }
            external_synsets.insert(ext_syn.id.clone(), ext_syn);
        }

        let mut declared_external_entries: HashSet<String> = HashSet::new();
        let mut declared_external_senses: HashSet<String> = HashSet::new();
        let mut added_sense_relations: HashMap<String, Vec<RawSenseRelation>> = HashMap::new();
        for ext_entry in &extension.external_entries {
            if !base.entries.contains_key(&ext_entry.id) {
                return Err(malformed(
                    resource_name,
                    Some(ext_entry.line),
                    format!(
                        "ExternalLexicalEntry {} at line {} does not resolve to a lexical entry in base {} version {}{}",
                        ext_entry.id,
                        ext_entry.line,
                        base.id,
                        base.version,
                        kind_hint(base, &ext_entry.id)
                    ),
                ));
            }
            declared_external_entries.insert(ext_entry.id.clone());
            for ext_sense in &ext_entry.external_senses {
                let Some(base_sense) = base.senses.get(&ext_sense.id) else {
                    return Err(malformed(
                        resource_name,
                        Some(ext_sense.line),
                        format!(
                            "ExternalSense {} at line {} does not resolve to a sense in base {} version {}{}",
                            ext_sense.id,
                            ext_sense.line,
                            base.id,
                            base.version,
                            kind_hint(base, &ext_sense.id)
                        ),
                    ));
                };
                if base_sense.entry_id != ext_entry.id {
                    return Err(malformed(
                        resource_name,
                        Some(ext_sense.line),
                        format!(
                            "ExternalSense {} at line {} is not a sense of lexical entry {}; it belongs to {}",
                            ext_sense.id, ext_sense.line, ext_entry.id, base_sense.entry_id
                        ),
                    ));
                }
                declared_external_senses.insert(ext_sense.id.clone());
                added_sense_relations
                    .entry(ext_sense.id.clone())
                    .or_default()
                    .extend(ext_sense.relations.iter().cloned());
            }
        }

        for sense_id in &extension.sense_order {
            let sense = &extension.senses[sense_id];
            self.require_synset_target(
                extension,
                &external_synsets,
                &sense.synset_id,
                &format!("Sense {}", sense.id),
                sense.line,
                resource_name,
            )?;
            self.require_sense_targets(
                extension,
                &declared_external_senses,
                &sense.id,
                &sense.relations,
                resource_name,
            )?;
        }
        for synset_id in &extension.synset_order {
            let synset = &extension.synsets[synset_id];
            self.require_member_targets(
                extension,
                synset,
                &declared_external_entries,
                &declared_external_senses,
                resource_name,
            )?;
            for rel in &synset.relations {
                self.require_synset_target(
                    extension,
                    &external_synsets,
                    &rel.target,
                    &format!("Relation {} on synset {}", rel.rel_type, synset.id),
                    rel.line,
                    resource_name,
                )?;
            }
        }
        for ext_syn in &extension.external_synsets {
            for rel in &ext_syn.relations {
                self.require_synset_target(
                    extension,
                    &external_synsets,
                    &rel.target,
                    &format!("Relation {} on ExternalSynset {}", rel.rel_type, ext_syn.id),
                    rel.line,
                    resource_name,
                )?;
            }
        }
        for (source_id, rels) in &added_sense_relations {
            self.require_sense_targets(extension, &declared_external_senses, source_id, rels, resource_name)?;
        }

        let mut entries = base.entries.clone();
        let mut entry_order = base.entry_order.clone();
        for id in &extension.entry_order {
            entry_order.push(id.clone());
            entries.insert(id.clone(), extension.entries[id].clone());
        }

        let mut senses = base.senses.clone();
        let mut sense_order = base.sense_order.clone();
        for (sense_id, added) in added_sense_relations {
            let base_sense = senses.get_mut(&sense_id).unwrap();
            base_sense.relations.extend(added);
        }
        for id in &extension.sense_order {
            sense_order.push(id.clone());
            senses.insert(id.clone(), extension.senses[id].clone());
        }

        let mut extra_members: HashMap<String, Vec<String>> = HashMap::new();
        for id in &extension.sense_order {
            let sense = &extension.senses[id];
            if external_synsets.contains_key(&sense.synset_id) {
                extra_members.entry(sense.synset_id.clone()).or_default().push(sense.entry_id.clone());
            }
        }

        let mut synsets = base.synsets.clone();
        let mut synset_order = base.synset_order.clone();
        for ext_syn in &extension.external_synsets {
            let base_synset = synsets.get_mut(&ext_syn.id).unwrap();
            base_synset.relations.extend(ext_syn.relations.iter().cloned());
            if base_synset.gloss.is_empty()
                && let Some(ref def) = ext_syn.definition
            {
                base_synset.gloss = def.clone();
            }
            if let Some(extras) = extra_members.get(&ext_syn.id) {
                base_synset.extra_members.extend(extras.iter().cloned());
            }
        }
        for id in &extension.synset_order {
            synset_order.push(id.clone());
            synsets.insert(id.clone(), extension.synsets[id].clone());
        }

        Ok(RawLexicon {
            extension: true,
            id: extension.id.clone(),
            label: extension.label.clone(),
            language: extension.language.clone(),
            version: extension.version.clone(),
            metadata: extension.metadata.clone(),
            requires: extension.requires.clone(),
            extends_ref: extension.extends_ref.clone(),
            entries,
            entry_order,
            senses,
            sense_order,
            synsets,
            synset_order,
            external_entries: Vec::new(),
            external_synsets: Vec::new(),
            line: extension.line,
        })
    }

    fn require_new_id(
        &self,
        base: &RawLexicon,
        extension: &RawLexicon,
        id: &str,
        kind: &str,
        line: usize,
        resource_name: &str,
    ) -> Result<(), WordNetError> {
        if base.entries.contains_key(id) || base.senses.contains_key(id) || base.synsets.contains_key(id) {
            return Err(malformed(
                resource_name,
                Some(line),
                format!(
                    "New {kind} id {id} at line {line} in extension {} collides with content of base {} version {}",
                    extension.id, base.id, base.version
                ),
            ));
        }
        Ok(())
    }

    fn require_synset_target(
        &self,
        extension: &RawLexicon,
        external_synsets: &HashMap<String, &RawExternalSynset>,
        target: &str,
        what: &str,
        line: usize,
        resource_name: &str,
    ) -> Result<(), WordNetError> {
        if !extension.synsets.contains_key(target) && !external_synsets.contains_key(target) {
            return Err(malformed(
                resource_name,
                Some(line),
                format!(
                    "{what} at line {line} references synset {target}, which is neither declared in the extension nor as an ExternalSynset"
                ),
            ));
        }
        Ok(())
    }

    fn require_sense_targets(
        &self,
        extension: &RawLexicon,
        declared_external_senses: &HashSet<String>,
        source_id: &str,
        relations: &[RawSenseRelation],
        resource_name: &str,
    ) -> Result<(), WordNetError> {
        for rel in relations {
            if rel.rel_type == OTHER_RELATION {
                continue;
            }
            if !extension.senses.contains_key(&rel.target) && !declared_external_senses.contains(&rel.target) {
                return Err(malformed(
                    resource_name,
                    Some(rel.line),
                    format!(
                        "SenseRelation {} at line {} from sense {source_id} references sense {}, which is neither declared in the extension nor as an ExternalSense",
                        rel.rel_type, rel.line, rel.target
                    ),
                ));
            }
        }
        Ok(())
    }

    fn require_member_targets(
        &self,
        extension: &RawLexicon,
        synset: &RawSynset,
        external_entries: &HashSet<String>,
        external_senses: &HashSet<String>,
        resource_name: &str,
    ) -> Result<(), WordNetError> {
        let Some(ref members) = synset.members else {
            return Ok(());
        };
        for member in split_on_spaces(members) {
            if !extension.entries.contains_key(member)
                && !extension.senses.contains_key(member)
                && !external_entries.contains(member)
                && !external_senses.contains(member)
            {
                return Err(malformed(
                    resource_name,
                    Some(synset.line),
                    format!(
                        "Synset {} at line {} lists member {member}, which needs a sense or entry declaration in the extension, or an ExternalSense or ExternalLexicalEntry",
                        synset.id, synset.line
                    ),
                ));
            }
        }
        Ok(())
    }
}

fn kind_hint(base: &RawLexicon, id: &str) -> &'static str {
    if base.entries.contains_key(id) {
        "; it names a lexical entry"
    } else if base.senses.contains_key(id) {
        "; it names a sense"
    } else if base.synsets.contains_key(id) {
        "; it names a synset"
    } else {
        ""
    }
}
