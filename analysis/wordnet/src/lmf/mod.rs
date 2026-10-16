//! WN-LMF XML reader (`wn-lmf-1.3.dtd` / `wn-lmf-1.4.dtd`) and the
//! OpenNLP WN-LMF composition extension (`http://opennlp.apache.org/wn-lmf-ext/1.0`).

//!
//! The reader is split by stage: [`xml`] tokenizes the document, [`raw`] turns XML events into
//! unresolved per-lexicon tables, [`merge`] composes a `<LexiconExtension>` onto its resolved
//! base, and [`materialize`] validates references and builds the [`InMemoryWordNetLexicon`].

mod materialize;
mod merge;
mod raw;
mod xml;

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use crate::error::WordNetError;
use crate::lexicon::InMemoryWordNetLexicon;

use materialize::Materializer;
use merge::Composition;
use raw::{RawLexicon, parse_raw};

/// Namespace URI for the OpenNLP WN-LMF 1.4 composition extension.
pub const EXT_NAMESPACE: &str = "http://opennlp.apache.org/wn-lmf-ext/1.0";

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

fn malformed(resource_name: &str, line: Option<usize>, message: impl AsRef<str>) -> WordNetError {
    match line {
        Some(l) if l > 0 => {
            WordNetError::Parse(format!("{MALFORMED_PREFIX}{resource_name} at line {l}: {}", message.as_ref()))
        }
        _ => WordNetError::Parse(format!("{MALFORMED_PREFIX}{resource_name}: {}", message.as_ref())),
    }
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

/// Wraps `body` in a `<LexicalResource>` with one `<Lexicon id="t">`.
#[cfg(test)]
fn wrap_lexicon(body: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<LexicalResource>\n\
         <Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">\n{body}\n</Lexicon>\n</LexicalResource>\n"
    )
}

/// One noun entry `t-cat-n` with sense `t-cat-n-1` in synset `t-1`, plus `synset_body` inside `t-1`.
#[cfg(test)]
fn cat_lexicon(synset_attrs: &str, synset_body: &str) -> String {
    wrap_lexicon(&format!(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"n\"{synset_attrs}>{synset_body}</Synset>"
    ))
}
