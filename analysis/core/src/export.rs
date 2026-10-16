//! The document graph export boundary: a [`Document`] as one typed,
//! versioned record a search index, a coordinator or another process can
//! hold and hand back.
//!
//! A graph carries the document's source bytes and the alignment that
//! decoded them, every text view with its parent and alignment edits,
//! and one record per layer instance in the order the layers were
//! added, each with its full layer name, instance name, [`Layer::SCHEMA`],
//! status and provenance. Items are named by [`NodeRef`] (layer,
//! instance and index), never by pointer or position in this record, so
//! a consumer can resolve an edge endpoint or a group member against
//! exactly the instance it was made on. Internal typed layers stay
//! what they are: a layer crosses the boundary only through a
//! [`LayerCodec`] registered for its type in [`Codecs`], and its body
//! is that codec's bytes. A layer with no codec refuses the whole
//! export or import rather than being dropped or flattened.
//!
//! # Versioning and recovery
//!
//! Two numbers version a graph: [`FORMAT_VERSION`] for this boundary
//! (the record shapes and the byte encoding together) and each layer
//! record's `schema`, copied from [`Layer::SCHEMA`]. Import is atomic
//! and strict: the format version, every layer schema, codec coverage,
//! duplicate instances, view parents and, through the codecs, every
//! edge and group endpoint are checked, and any failure returns a typed
//! [`ExportError`] with no document. A consumer that meets a newer
//! format or schema upgrades; there is no partial recovery mode and no
//! silent loss. Skipped and failed layers are records too and import
//! back as skipped and failed entries with the same reasons and the
//! same errors, variant for variant. Importing rebuilds provenance names as
//! `&'static str` through a process-wide interner that leaks one copy
//! of each distinct name ever seen, so importing without end leaks the
//! vocabulary of annotator, version and instance names, not a copy per
//! document.
//!
//! [`DocumentGraph::to_bytes`] is the canonical encoding of the record:
//! deterministic for a given graph, so two exports of one document
//! through one codec set are byte equal. It is a transport encoding of
//! this boundary, not a persistence promise beyond [`FORMAT_VERSION`].

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use crate::doc::{Document, Status, View};
use crate::error::Error;
use crate::graph::{Edge, Edges, Groups};
use crate::layer::{Execution, Key, KeyId, Layer, ModelRef, Provenance, Tag, TagSet};
use crate::span::{Ix, Span, Spans};
use crate::text::{Alignment, Edit, ViewId};

/// The version of this boundary: the record shapes and byte encoding.
/// Bumped when either changes meaning; older bytes are refused, not
/// migrated.
pub const FORMAT_VERSION: u16 = 1;

const MAGIC: &[u8] = b"TGRF";

/// One layer instance on a document, named as a [`Key`] names it: the
/// layer's full name and the instance name (`""` for the default one).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerRef {
    pub layer: String,
    pub instance: String,
}

impl fmt::Display for LayerRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.layer)?;
        if !self.instance.is_empty() {
            write!(f, "@{}", self.instance)?;
        }
        Ok(())
    }
}

/// One item of one layer instance: the typed node identity a consumer
/// resolves edge endpoints and group members by.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeRef {
    pub layer: LayerRef,
    pub index: u32,
}

impl fmt::Display for NodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", self.layer, self.index)
    }
}

/// The [`LayerRef`] of a typed key.
pub fn layer_ref<L: Layer>(key: &Key<L>) -> LayerRef {
    LayerRef { layer: L::NAME.to_string(), instance: key.name.to_owned() }
}

/// The [`NodeRef`] of item `index` of the instance `key` names.
pub fn node_ref<L: Layer>(key: &Key<L>, index: Ix<L>) -> NodeRef {
    NodeRef { layer: layer_ref(key), index: index.get() as u32 }
}

/// Where a layer came from, as owned strings: [`Provenance`] with its
/// key references as [`LayerRef`]s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub annotator: String,
    pub version: String,
    pub config: u64,
    pub model: Option<ModelRef>,
    pub execution: Option<Execution>,
    pub reads: Vec<LayerRef>,
    pub on: Vec<LayerRef>,
}

/// What became of a layer, as a record: [`Status`] with a failed
/// layer's error kept whole, so it restores as the same [`Error`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StatusRecord {
    Complete,
    Skipped(String),
    Failed(ErrorRecord),
}

/// An [`Error`] as a record: the same variants, with layer keys as
/// [`LayerRef`]s, a view as its instance name, and names as owned
/// strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorRecord {
    Missing(LayerRef),
    Duplicate(LayerRef),
    Clash(LayerRef),
    Bounds { layer: LayerRef, view: String, detail: String },
    Misaligned { layer: LayerRef, anchor: LayerRef, len: u32, anchor_len: u32 },
    Unaligned { layer: LayerRef, on: LayerRef, read: String },
    Invalid { layer: LayerRef, detail: String },
    Pipeline(String),
    Annotator { annotator: String, detail: String },
    Skipped(String),
    Offset { unit: crate::offsets::OffsetUnit, at: u32, reason: crate::offsets::OffsetError },
    Source(crate::bytes::SourceError),
}

/// The bytes the document was read from and the text they decoded to.
/// `alignment` is the decoder's alignment from the bytes to the text,
/// present exactly when the document was made by [`Document::decoded`];
/// without it the bytes are the text's own UTF-8.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRecord {
    pub bytes: Vec<u8>,
    pub text: String,
    pub alignment: Option<Vec<Edit>>,
}

/// One text view: its instance name, status and provenance, and for a
/// view that is here, its parent view's instance name, its text and the
/// edits that align it to its parent. The source view is the one with
/// no parent; its text is the graph's [`SourceRecord::text`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewRecord {
    pub instance: String,
    pub status: StatusRecord,
    pub provenance: ProvenanceRecord,
    pub parent: Option<String>,
    pub text: Option<String>,
    pub edits: Vec<Edit>,
}

/// One layer instance: its name, schema, item count, status and
/// provenance, and for a complete layer the bytes its codec wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerRecord {
    pub layer: String,
    pub instance: String,
    pub schema: u16,
    pub len: u32,
    pub status: StatusRecord,
    pub provenance: ProvenanceRecord,
    pub body: Option<Vec<u8>>,
}

/// A whole document at the export boundary, views and layers each in
/// the order they were added to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentGraph {
    pub version: u16,
    pub source: SourceRecord,
    pub views: Vec<ViewRecord>,
    pub layers: Vec<LayerRecord>,
}

/// Why an export or an import was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportError {
    /// The graph's format version is not this build's.
    UnsupportedVersion { found: u16, supported: u16 },
    /// A layer record's schema is not its type's current `SCHEMA`.
    IncompatibleSchema { layer: String, exported: u16, supported: u16 },
    /// No codec is registered for the layer type.
    NoCodec { layer: String },
    /// Two records name one layer instance (or two views).
    DuplicateLayer { layer: String, instance: String },
    /// A view instance the graph does not hold, named as a parent or
    /// by a span codec.
    UnknownView { instance: String },
    /// A reference to a layer instance the graph does not restore:
    /// an endpoint's layer, or one provenance names.
    DanglingReference { layer: String, reference: LayerRef },
    /// An edge or group endpoint past the end of its layer.
    DanglingEndpoint { layer: String, endpoint: NodeRef },
    /// The source record does not hold together: bytes that are not
    /// the text's, or an alignment that does not decode them to it.
    InvalidSource(String),
    /// A codec's layer was refused by the document, or its payload
    /// disagrees with its record.
    InvalidLayer { layer: String, detail: String },
    /// Bytes that are not a graph: truncated, trailing, or a field out
    /// of range.
    Malformed(String),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ExportError::*;
        match self {
            UnsupportedVersion { found, supported } => {
                write!(f, "document graph format {found}, this build reads {supported}")
            }
            IncompatibleSchema { layer, exported, supported } => {
                write!(f, "layer {layer} was exported at schema {exported}, this build reads {supported}")
            }
            NoCodec { layer } => write!(f, "no codec registered for layer {layer}"),
            DuplicateLayer { layer, instance } => write!(f, "two records for layer instance {layer}@{instance}"),
            UnknownView { instance } => write!(f, "no view instance {instance:?}"),
            DanglingReference { layer, reference } => {
                write!(f, "layer {layer} refers to {reference}, which the graph does not restore")
            }
            DanglingEndpoint { layer, endpoint } => {
                write!(f, "layer {layer} has an endpoint {endpoint} past the end of its layer")
            }
            InvalidSource(detail) => f.write_str(detail),
            InvalidLayer { layer, detail } => write!(f, "layer {layer}: {detail}"),
            Malformed(detail) => write!(f, "malformed document graph: {detail}"),
        }
    }
}

impl std::error::Error for ExportError {}

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

/// Writes the fields of the boundary encoding: little-endian counts and
/// length-prefixed bytes, with no padding. Codecs write their payloads
/// with the same writer, so a payload and its record share one set of
/// field rules.
#[derive(Default)]
pub struct Writer {
    out: Vec<u8>,
}

impl Writer {
    pub fn new() -> Writer {
        Writer::default()
    }

    pub fn u8(&mut self, v: u8) {
        self.out.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }

    pub fn usize(&mut self, v: usize) -> Result<(), ExportError> {
        self.u32(u32::try_from(v).map_err(|_| malformed("a count longer than a u32 can hold"))?);
        Ok(())
    }

    pub fn bytes(&mut self, b: &[u8]) -> Result<(), ExportError> {
        self.usize(b.len())?;
        self.out.extend_from_slice(b);
        Ok(())
    }

    pub fn str(&mut self, s: &str) -> Result<(), ExportError> {
        self.bytes(s.as_bytes())
    }

    pub fn span(&mut self, s: Span) {
        self.u32(s.start);
        self.u32(s.end);
    }

    pub fn finish(self) -> Vec<u8> {
        self.out
    }
}

/// Reads what [`Writer`] wrote, refusing anything truncated or out of
/// range as [`ExportError::Malformed`] rather than panicking.
pub struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ExportError> {
        let end = self.at.checked_add(n).ok_or_else(|| malformed("truncated"))?;
        let some = self.buf.get(self.at..end).ok_or_else(|| malformed("truncated"))?;
        self.at = end;
        Ok(some)
    }

    pub fn u8(&mut self) -> Result<u8, ExportError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, ExportError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, ExportError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, ExportError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn f64(&mut self) -> Result<f64, ExportError> {
        Ok(f64::from_bits(self.u64()?))
    }

    pub fn usize(&mut self) -> Result<usize, ExportError> {
        usize::try_from(self.u32()?).map_err(|_| malformed("a count longer than a usize can hold"))
    }

    pub fn bytes(&mut self) -> Result<&'a [u8], ExportError> {
        let n = self.usize()?;
        self.take(n)
    }

    pub fn string(&mut self) -> Result<String, ExportError> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| malformed("a string that is not UTF-8"))
    }

    pub fn span(&mut self) -> Result<Span, ExportError> {
        let (start, end) = (self.u32()?, self.u32()?);
        if start > end {
            return Err(malformed("a span that ends before it starts"));
        }
        Ok(Span { start, end })
    }

    /// Whether the whole input has been read.
    pub fn is_done(&self) -> bool {
        self.at == self.buf.len()
    }
}

/// How one layer type crosses the boundary. The payload is the codec's
/// own, written with [`Writer`] and read with [`Reader`]; span layers
/// and tag columns use [`put_spans`], [`get_spans`], [`put_tagset`] and
/// [`get_tagset`] so those fields agree across codecs. An import must
/// refuse a payload that does not hold together (a view that is not
/// here, an endpoint past its layer) with the typed errors on
/// [`ImportCx`], never by clamping or defaulting.
pub trait LayerCodec: Layer {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError>;
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError>;
}

/// What an exporting codec needs beyond its own layer: the instance
/// name of each view, so span columns name their view across the
/// boundary instead of carrying a process-local [`ViewId`], and the
/// instance a layer was actually made on (see [`ExportCx::resolve`]).
pub struct ExportCx {
    views: HashMap<ViewId, String>,
    on: Vec<KeyId>,
    reads: Vec<KeyId>,
}

impl ExportCx {
    pub fn view_instance(&self, view: ViewId) -> Result<&str, ExportError> {
        self.views
            .get(&view)
            .map(String::as_str)
            .ok_or_else(|| ExportError::UnknownView { instance: format!("view {}", view.0) })
    }

    /// The instance a reference through `key` actually names on the
    /// document being exported. Relation layers built with
    /// `Edges::new` or `Groups::new` carry the default key even when
    /// the step that wrote them was bound to a named instance; the
    /// document resolved the empty name through the layer's provenance
    /// when it took the layer, and this applies the same resolution,
    /// so an exported endpoint names the instance it indexes. Codecs
    /// that write layer references (relation endpoints above all) must
    /// resolve them through here.
    pub fn resolve<L: Layer>(&self, key: &Key<L>) -> LayerRef {
        if key.name.is_empty() {
            let ty = TypeId::of::<L>();
            if let Some(found) = self.on.iter().chain(&self.reads).find(|r| r.ty == ty) {
                return ref_of(*found);
            }
        }
        layer_ref(key)
    }
}

/// What an importing codec needs: view instances resolved to the new
/// document's [`ViewId`]s, the item counts of the layers restored so
/// far, and `&'static str` names for keys from a process-wide interner.
pub struct ImportCx {
    views: HashMap<String, ViewId>,
    lens: HashMap<(String, String), usize>,
    known: HashSet<(String, String)>,
    layer: String,
}

impl ImportCx {
    /// The view a record names, in the document being built.
    pub fn view(&self, instance: &str) -> Result<ViewId, ExportError> {
        self.views.get(instance).copied().ok_or_else(|| ExportError::UnknownView { instance: instance.to_owned() })
    }

    /// How many items a restored layer instance holds.
    pub fn layer_len(&self, layer: &LayerRef) -> Result<usize, ExportError> {
        self.lens
            .get(&(layer.layer.clone(), layer.instance.clone()))
            .copied()
            .ok_or_else(|| ExportError::DanglingReference { layer: self.layer.clone(), reference: layer.clone() })
    }

    /// Refuses unless the node is an item of a restored layer instance.
    pub fn require_node(&self, node: &NodeRef) -> Result<(), ExportError> {
        if node.index as usize >= self.layer_len(&node.layer)? {
            return Err(ExportError::DanglingEndpoint { layer: self.layer.clone(), endpoint: node.clone() });
        }
        Ok(())
    }

    /// An instance name as the `&'static str` a [`Key`] needs. Names
    /// are interned process-wide: one leaked copy per distinct name
    /// ever imported, however many documents cross.
    pub fn intern(&mut self, name: &str) -> &'static str {
        intern_name(name)
    }

    /// The key of an instance of `L`, by the name a record carries.
    pub fn key<L: Layer>(&mut self, instance: &str) -> Key<L> {
        Key::named(self.intern(instance))
    }
}

/// The process-wide name interner behind [`ImportCx::intern`].
fn intern_name(name: &str) -> &'static str {
    if name.is_empty() {
        return "";
    }
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES.get_or_init(|| Mutex::new(HashSet::new())).lock().expect("the name interner is not poisoned");
    match names.get(name) {
        Some(&found) => found,
        None => {
            let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
            names.insert(leaked);
            leaked
        }
    }
}

type Exporter = fn(&dyn Any, &ExportCx) -> Result<Vec<u8>, ExportError>;
type Importer = fn(&mut Document, &LayerRecord, Vec<u8>, &mut ImportCx, Provenance) -> Result<usize, ExportError>;

struct Codec {
    schema: u16,
    export: Exporter,
    import: Importer,
    key: fn(&'static str) -> KeyId,
}

/// The codecs a caller works with, one per layer type. Export and
/// import refuse a layer with none, so what crosses the boundary is
/// exactly what the caller registered.
#[derive(Default)]
pub struct Codecs {
    by_type: HashMap<TypeId, Codec>,
    by_name: HashMap<String, TypeId>,
}

impl Codecs {
    pub fn new() -> Codecs {
        Codecs::default()
    }

    /// Registers `L`'s codec under its full layer name.
    pub fn register<L: LayerCodec>(&mut self) -> &mut Codecs {
        let codec = Codec {
            schema: L::SCHEMA,
            export: |value, cx| {
                let layer =
                    value.downcast_ref::<L>().ok_or_else(|| malformed("a layer value that is not its key's type"))?;
                L::export(layer, cx)
            },
            import: |doc, record, body, cx, provenance| {
                let layer = L::import(&body, cx)?;
                let len = layer.len();
                let key = Key::<L>::named(cx.intern(&record.instance));
                doc.add(&key, layer, provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: record.layer.clone(), detail: e.to_string() })?;
                Ok(len)
            },
            key: |instance| Key::<L>::named(instance).id(),
        };
        self.by_name.insert(L::NAME.to_string(), TypeId::of::<L>());
        self.by_type.insert(TypeId::of::<L>(), codec);
        self
    }

    fn by_type_id(&self, ty: TypeId) -> Option<&Codec> {
        self.by_type.get(&ty)
    }

    fn by_layer_name(&self, name: &str) -> Option<&Codec> {
        self.by_name.get(name).and_then(|ty| self.by_type.get(ty))
    }

    fn key_id(&self, layer: &LayerRef, cx: &mut ImportCx) -> Result<KeyId, ExportError> {
        if layer.layer == View::ID {
            return Ok(Key::<View>::named(cx.intern(&layer.instance)).id());
        }
        let codec =
            self.by_layer_name(&layer.layer).ok_or_else(|| ExportError::NoCodec { layer: layer.layer.clone() })?;
        Ok((codec.key)(cx.intern(&layer.instance)))
    }
}

/// Writes a span layer's columns: its view's instance name, then per
/// item its fragments (one for a continuous span), so discontinuous
/// items round-trip exactly.
pub fn put_spans<L>(w: &mut Writer, spans: &Spans<L>, cx: &ExportCx) -> Result<(), ExportError> {
    w.str(cx.view_instance(spans.view())?)?;
    w.usize(spans.len())?;
    for (i, _) in spans.iter() {
        let fragments: Vec<Span> = spans.fragments(i).collect();
        w.usize(fragments.len())?;
        for fragment in fragments {
            w.span(fragment);
        }
    }
    Ok(())
}

/// Reads what [`put_spans`] wrote.
pub fn get_spans<L>(r: &mut Reader<'_>, cx: &mut ImportCx) -> Result<Spans<L>, ExportError> {
    let instance = r.string()?;
    let mut spans = Spans::new(cx.view(&instance)?);
    for _ in 0..r.usize()? {
        let mut fragments = Vec::new();
        for _ in 0..r.usize()? {
            fragments.push(r.span()?);
        }
        match &fragments[..] {
            [] => return Err(malformed("a span item with no fragments")),
            [one] => {
                spans.push(*one);
            }
            many => {
                spans
                    .try_push_fragments(many)
                    .map_err(|e| ExportError::InvalidLayer { layer: cx.layer.clone(), detail: e.to_string() })?;
            }
        }
    }
    Ok(spans)
}

/// Writes a tag set: its name and its labels in tag order.
pub fn put_tagset(w: &mut Writer, kinds: &TagSet) -> Result<(), ExportError> {
    w.str(&kinds.name)?;
    w.usize(kinds.len())?;
    for i in 0..kinds.len() {
        w.str(kinds.label(Tag(u16::try_from(i).map_err(|_| malformed("a tag set longer than a u16 can index"))?)))?;
    }
    Ok(())
}

/// Reads what [`put_tagset`] wrote. A tag is a `u16`, so a set of more
/// than 65,536 labels cannot be indexed and is refused instead of
/// wrapping an index onto another label.
pub fn get_tagset(r: &mut Reader<'_>) -> Result<Arc<TagSet>, ExportError> {
    let name = r.string()?;
    let count = r.usize()?;
    if count > usize::from(u16::MAX) + 1 {
        return Err(malformed("a tag set longer than a u16 can index"));
    }
    let mut labels = Vec::new();
    for _ in 0..count {
        labels.push(r.string()?);
    }
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    Ok(TagSet::new(&name, &labels))
}

fn put_layer_ref(w: &mut Writer, layer: &LayerRef) -> Result<(), ExportError> {
    w.str(&layer.layer)?;
    w.str(&layer.instance)
}

fn get_layer_ref(r: &mut Reader<'_>) -> Result<LayerRef, ExportError> {
    Ok(LayerRef { layer: r.string()?, instance: r.string()? })
}

impl<From: Layer, To: Layer> LayerCodec for Edges<From, To> {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_layer_ref(&mut w, &cx.resolve(&self.from_key))?;
        put_layer_ref(&mut w, &cx.resolve(&self.to_key))?;
        put_tagset(&mut w, &self.kinds)?;
        w.usize(self.len())?;
        for (_, edge) in self.iter() {
            w.u32(edge.from.get() as u32);
            w.u32(edge.to.get() as u32);
            w.u32(u32::from(edge.kind.0));
            w.f64(edge.score);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let from = get_layer_ref(&mut r)?;
        let to = get_layer_ref(&mut r)?;
        for (named, typed) in [(&from, From::NAME.to_string()), (&to, To::NAME.to_string())] {
            if named.layer != typed {
                return Err(ExportError::InvalidLayer {
                    layer: cx.layer.clone(),
                    detail: format!("an endpoint on {}, not {typed}", named.layer),
                });
            }
        }
        cx.layer_len(&from)?;
        cx.layer_len(&to)?;
        let kinds = get_tagset(&mut r)?;
        let mut rows = Vec::new();
        for _ in 0..r.usize()? {
            let (a, b, kind, score) = (r.u32()?, r.u32()?, r.u32()?, r.f64()?);
            cx.require_node(&NodeRef { layer: from.clone(), index: a })?;
            cx.require_node(&NodeRef { layer: to.clone(), index: b })?;
            let kind = Tag(u16::try_from(kind).map_err(|_| malformed("an edge kind outside a u16"))?);
            rows.push(Edge::new(Ix::new(a), Ix::new(b), kind, score));
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in an edges payload"));
        }
        let (from_key, to_key) = (cx.key::<From>(&from.instance.clone()), cx.key::<To>(&to.instance.clone()));
        Ok(Edges::between(from_key, to_key, kinds, rows))
    }
}

impl<L: Layer> LayerCodec for Groups<L> {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_layer_ref(&mut w, &cx.resolve(&self.of_key))?;
        w.usize(self.len())?;
        for (_, members) in self.iter() {
            w.usize(members.len())?;
            for member in members {
                w.u32(member.get() as u32);
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let of = get_layer_ref(&mut r)?;
        if of.layer != L::NAME.to_string() {
            return Err(ExportError::InvalidLayer {
                layer: cx.layer.clone(),
                detail: format!("groups of {}, not {}", of.layer, L::NAME),
            });
        }
        cx.layer_len(&of)?;
        let mut groups: Vec<Vec<Ix<L>>> = Vec::new();
        for _ in 0..r.usize()? {
            let mut members = Vec::new();
            for _ in 0..r.usize()? {
                let index = r.u32()?;
                cx.require_node(&NodeRef { layer: of.clone(), index })?;
                members.push(Ix::new(index));
            }
            groups.push(members);
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a groups payload"));
        }
        let of_key = cx.key::<L>(&of.instance.clone());
        Ok(Groups::of(of_key, groups))
    }
}

fn ref_of(key: KeyId) -> LayerRef {
    LayerRef { layer: key.layer.to_string(), instance: key.name.to_owned() }
}

fn provenance_record(p: &Provenance) -> ProvenanceRecord {
    ProvenanceRecord {
        annotator: p.annotator.to_owned(),
        version: p.version.to_owned(),
        config: p.config,
        model: p.model.clone(),
        execution: p.execution.clone(),
        reads: p.reads.iter().copied().map(ref_of).collect(),
        on: p.on.iter().copied().map(ref_of).collect(),
    }
}

fn error_record(error: &Error, cx: &ExportCx) -> Result<ErrorRecord, ExportError> {
    Ok(match error {
        Error::Missing(k) => ErrorRecord::Missing(ref_of(*k)),
        Error::Duplicate(k) => ErrorRecord::Duplicate(ref_of(*k)),
        Error::Clash(k) => ErrorRecord::Clash(ref_of(*k)),
        Error::Bounds { layer, view, detail } => ErrorRecord::Bounds {
            layer: ref_of(*layer),
            view: cx.view_instance(*view)?.to_owned(),
            detail: detail.clone(),
        },
        Error::Misaligned { layer, anchor, len, anchor_len } => ErrorRecord::Misaligned {
            layer: ref_of(*layer),
            anchor: ref_of(*anchor),
            len: u32::try_from(*len).map_err(|_| malformed("a length longer than a u32 can count"))?,
            anchor_len: u32::try_from(*anchor_len).map_err(|_| malformed("a length longer than a u32 can count"))?,
        },
        Error::Unaligned { layer, on, read } => {
            ErrorRecord::Unaligned { layer: ref_of(*layer), on: ref_of(*on), read: read.to_string() }
        }
        Error::Invalid { layer, detail } => ErrorRecord::Invalid { layer: ref_of(*layer), detail: detail.clone() },
        Error::Pipeline(detail) => ErrorRecord::Pipeline(detail.clone()),
        Error::Annotator { annotator, detail } => {
            ErrorRecord::Annotator { annotator: annotator.to_string(), detail: detail.clone() }
        }
        Error::Skipped(reason) => ErrorRecord::Skipped(reason.clone()),
        Error::Offset { unit, at, reason } => ErrorRecord::Offset { unit: *unit, at: *at, reason: *reason },
        Error::Source(e) => ErrorRecord::Source(e.clone()),
    })
}

fn status_record(status: &Status, cx: &ExportCx) -> Result<StatusRecord, ExportError> {
    Ok(match status {
        Status::Complete => StatusRecord::Complete,
        Status::Skipped(reason) => StatusRecord::Skipped(reason.clone()),
        Status::Failed(error) => StatusRecord::Failed(error_record(error, cx)?),
    })
}

fn rebuild_error(record: &ErrorRecord, codecs: &Codecs, cx: &mut ImportCx, layer: &str) -> Result<Error, ExportError> {
    let key = |r: &LayerRef, cx: &mut ImportCx| -> Result<KeyId, ExportError> {
        codecs.key_id(r, cx).map_err(|e| match e {
            ExportError::NoCodec { .. } => {
                ExportError::DanglingReference { layer: layer.to_owned(), reference: r.clone() }
            }
            other => other,
        })
    };
    Ok(match record {
        ErrorRecord::Missing(r) => Error::Missing(key(r, cx)?),
        ErrorRecord::Duplicate(r) => Error::Duplicate(key(r, cx)?),
        ErrorRecord::Clash(r) => Error::Clash(key(r, cx)?),
        ErrorRecord::Bounds { layer: l, view, detail } => {
            Error::Bounds { layer: key(l, cx)?, view: cx.view(view)?, detail: detail.clone() }
        }
        ErrorRecord::Misaligned { layer: l, anchor, len, anchor_len } => Error::Misaligned {
            layer: key(l, cx)?,
            anchor: key(anchor, cx)?,
            len: *len as usize,
            anchor_len: *anchor_len as usize,
        },
        ErrorRecord::Unaligned { layer: l, on, read } => {
            Error::Unaligned { layer: key(l, cx)?, on: key(on, cx)?, read: cx.intern(&read.clone()) }
        }
        ErrorRecord::Invalid { layer: l, detail } => Error::Invalid { layer: key(l, cx)?, detail: detail.clone() },
        ErrorRecord::Pipeline(detail) => Error::Pipeline(detail.clone()),
        ErrorRecord::Annotator { annotator, detail } => {
            Error::Annotator { annotator: cx.intern(&annotator.clone()), detail: detail.clone() }
        }
        ErrorRecord::Skipped(reason) => Error::Skipped(reason.clone()),
        ErrorRecord::Offset { unit, at, reason } => Error::Offset { unit: *unit, at: *at, reason: *reason },
        ErrorRecord::Source(e) => Error::Source(e.clone()),
    })
}

/// Exports a document as a graph through the registered codecs. Every
/// layer instance on the document needs a codec, complete, skipped and
/// failed alike; the first one without refuses the export.
pub fn export(doc: &Document, codecs: &Codecs) -> Result<DocumentGraph, ExportError> {
    let mut view_instances: HashMap<ViewId, String> = HashMap::new();
    for entry in doc.entries() {
        if entry.key.ty == TypeId::of::<View>()
            && let Some(view) = entry.value.as_ref().and_then(|v| v.downcast_ref::<View>())
        {
            view_instances.insert(view.id(), entry.key.name.to_owned());
        }
    }
    let mut views = Vec::new();
    let mut layers = Vec::new();
    for entry in doc.entries() {
        let cx = ExportCx {
            views: view_instances.clone(),
            on: entry.provenance.on.clone(),
            reads: entry.provenance.reads.clone(),
        };
        let provenance = provenance_record(&entry.provenance);
        let status = status_record(&entry.status, &cx)?;
        if entry.key.ty == TypeId::of::<View>() {
            let (parent, text, edits) = match entry.value.as_ref().and_then(|v| v.downcast_ref::<View>()) {
                Some(view) => {
                    let parent = match view.parent() {
                        None => None,
                        Some(p) => Some(
                            view_instances
                                .get(&p)
                                .ok_or_else(|| ExportError::UnknownView { instance: format!("view {}", p.0) })?
                                .clone(),
                        ),
                    };
                    (parent, Some(view.text().to_owned()), view.alignment().edits().to_vec())
                }
                None => (None, None, Vec::new()),
            };
            views.push(ViewRecord { instance: entry.key.name.to_owned(), status, provenance, parent, text, edits });
            continue;
        }
        let name = entry.key.layer.to_string();
        let codec = codecs.by_type_id(entry.key.ty).ok_or_else(|| ExportError::NoCodec { layer: name.clone() })?;
        let body = match (&entry.status, &entry.value) {
            (Status::Complete, Some(value)) => Some((codec.export)(value.as_ref(), &cx)?),
            _ => None,
        };
        layers.push(LayerRecord {
            layer: name,
            instance: entry.key.name.to_owned(),
            schema: codec.schema,
            len: u32::try_from(entry.len).map_err(|_| malformed("a layer longer than a u32 can count"))?,
            status,
            provenance,
            body,
        });
    }
    let source = match doc.source_bytes() {
        Some(bytes) => SourceRecord {
            bytes: bytes.bytes().to_vec(),
            text: doc.source().to_owned(),
            alignment: Some(bytes.alignment().edits().to_vec()),
        },
        None => SourceRecord { bytes: doc.bytes().to_vec(), text: doc.source().to_owned(), alignment: None },
    };
    Ok(DocumentGraph { version: FORMAT_VERSION, source, views, layers })
}

/// Checks an alignment against the two texts it joins, with the same
/// rules the source record's decoder alignment is held to: every edit
/// inside both texts and on character boundaries on both sides, and
/// the unchanged stretches (before, between and after the edits) byte
/// equal. Without this, an imported view could claim to map to its
/// parent while mapping spans to the wrong bytes.
fn check_view_alignment(parent: &str, text: &str, edits: &[Edit]) -> Result<(), String> {
    let (mut p, mut c) = (0usize, 0usize);
    for (i, edit) in edits.iter().enumerate() {
        let unchanged_parent = parent
            .get(p..edit.parent.start as usize)
            .ok_or_else(|| format!("edit {i} starts past the parent text or off a character boundary"))?;
        let unchanged_child = text
            .get(c..edit.child.start as usize)
            .ok_or_else(|| format!("edit {i} starts past the view text or off a character boundary"))?;
        if unchanged_parent != unchanged_child {
            return Err(format!("the unchanged text before edit {i} differs between the view and its parent"));
        }
        if parent.get(edit.parent.start as usize..edit.parent.end as usize).is_none()
            || text.get(edit.child.start as usize..edit.child.end as usize).is_none()
        {
            return Err(format!("edit {i} ends past a text or off a character boundary"));
        }
        p = edit.parent.end as usize;
        c = edit.child.end as usize;
    }
    match (parent.get(p..), text.get(c..)) {
        (Some(a), Some(b)) if a == b => Ok(()),
        _ => Err("the unchanged text after the last edit differs between the view and its parent".into()),
    }
}

fn rebuild_provenance(
    record: &ProvenanceRecord,
    codecs: &Codecs,
    cx: &mut ImportCx,
    layer: &str,
    strict: bool,
) -> Result<Provenance, ExportError> {
    let resolve = |refs: &[LayerRef], cx: &mut ImportCx| -> Result<Vec<KeyId>, ExportError> {
        refs.iter()
            .map(|r| {
                // A complete layer read what its provenance names, so
                // every reference must be an instance the graph holds.
                // A failed layer may name the required input whose
                // absence failed it, which no record holds by
                // construction; its references rebuild by type alone.
                if strict && !cx.known.contains(&(r.layer.clone(), r.instance.clone())) {
                    return Err(ExportError::DanglingReference { layer: layer.to_owned(), reference: r.clone() });
                }
                codecs.key_id(r, cx).map_err(|e| match e {
                    ExportError::NoCodec { .. } => {
                        ExportError::DanglingReference { layer: layer.to_owned(), reference: r.clone() }
                    }
                    other => other,
                })
            })
            .collect()
    };
    let reads = resolve(&record.reads, cx)?;
    let on = resolve(&record.on, cx)?;
    Ok(Provenance {
        annotator: cx.intern(&record.annotator.clone()),
        version: cx.intern(&record.version.clone()),
        config: record.config,
        model: record.model.clone(),
        execution: record.execution.clone(),
        reads,
        on,
    })
}

/// Imports a graph back into a document, atomically: any refusal (a
/// version or schema this build does not read, a layer with no codec, a
/// duplicate instance, a view parent that is not here, a provenance
/// reference or an endpoint that does not resolve, a layer the
/// document itself refuses) returns the typed error and no document.
pub fn import(graph: &DocumentGraph, codecs: &Codecs) -> Result<Document, ExportError> {
    if graph.version != FORMAT_VERSION {
        return Err(ExportError::UnsupportedVersion { found: graph.version, supported: FORMAT_VERSION });
    }
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for view in &graph.views {
        let id = (View::ID.to_owned(), view.instance.clone());
        if !seen.insert(id.clone()) {
            return Err(ExportError::DuplicateLayer { layer: id.0, instance: id.1 });
        }
    }
    for record in &graph.layers {
        if record.layer == View::ID {
            return Err(ExportError::InvalidLayer {
                layer: record.layer.clone(),
                detail: "a view among the layer records; views cross as view records".into(),
            });
        }
        let id = (record.layer.clone(), record.instance.clone());
        if !seen.insert(id.clone()) {
            return Err(ExportError::DuplicateLayer { layer: id.0, instance: id.1 });
        }
        let codec =
            codecs.by_layer_name(&record.layer).ok_or_else(|| ExportError::NoCodec { layer: record.layer.clone() })?;
        if codec.schema != record.schema {
            return Err(ExportError::IncompatibleSchema {
                layer: record.layer.clone(),
                exported: record.schema,
                supported: codec.schema,
            });
        }
        if matches!(record.status, StatusRecord::Complete) != record.body.is_some() {
            return Err(malformed(format!("layer {} is complete exactly when it has a body", record.layer)));
        }
    }

    let mut doc = match &graph.source.alignment {
        None => {
            if graph.source.bytes != graph.source.text.as_bytes() {
                return Err(ExportError::InvalidSource(
                    "the source bytes are not the source text's UTF-8 and no alignment is given".into(),
                ));
            }
            Document::new(graph.source.text.clone())
        }
        Some(edits) => {
            let alignment = Alignment::from_edits(edits.clone()).map_err(ExportError::InvalidSource)?;
            Document::decoded(graph.source.bytes.clone(), graph.source.text.clone(), alignment)
                .map_err(|e| ExportError::InvalidSource(e.to_string()))?
        }
    };

    // The first view record is the source view, whatever follows it:
    // complete, the default instance, no parent, no edits, and the
    // source text itself. Its provenance is fixed by the format (the
    // document gives the source view `turbo:source` and nothing else),
    // so a record claiming another is refused rather than silently
    // replaced on import.
    match graph.views.first() {
        None => return Err(ExportError::InvalidSource("the graph holds no views".into())),
        Some(first_view) => {
            let p = &first_view.provenance;
            let fixed = Provenance::by("turbo:source");
            if !matches!(first_view.status, StatusRecord::Complete)
                || !first_view.instance.is_empty()
                || first_view.parent.is_some()
                || !first_view.edits.is_empty()
                || first_view.text.as_deref() != Some(graph.source.text.as_str())
                || p.annotator != fixed.annotator
                || p.version != fixed.version
                || p.config != fixed.config
                || p.model.is_some()
                || p.execution.is_some()
                || !p.reads.is_empty()
                || !p.on.is_empty()
            {
                return Err(ExportError::InvalidSource(
                    "the first view is not the source view of the source text".into(),
                ));
            }
        }
    }

    let mut cx = ImportCx {
        views: HashMap::from([(String::new(), ViewId::SOURCE)]),
        lens: HashMap::new(),
        known: seen,
        layer: View::ID.to_owned(),
    };
    cx.lens.insert((View::ID.to_owned(), String::new()), 1);

    for (index, record) in graph.views.iter().enumerate() {
        cx.layer = View::ID.to_owned();
        let provenance = rebuild_provenance(
            &record.provenance,
            codecs,
            &mut cx,
            View::ID,
            !matches!(record.status, StatusRecord::Failed(_)),
        )?;
        let key = Key::<View>::named(cx.intern(&record.instance.clone()));
        match (&record.status, &record.text) {
            (StatusRecord::Complete, Some(text)) => {
                if index == 0 {
                    // The source view, already on the document.
                } else {
                    let parent = record.parent.as_deref().ok_or_else(|| malformed("a derived view with no parent"))?;
                    let parent_key = Key::<View>::named(cx.intern(parent));
                    let parent_view = doc
                        .get_key(&parent_key)
                        .ok_or_else(|| ExportError::UnknownView { instance: parent.to_owned() })?;
                    let alignment = Alignment::from_edits(record.edits.clone())
                        .map_err(|detail| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail })?;
                    check_view_alignment(parent_view.text(), text, alignment.edits())
                        .map_err(|detail| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail })?;
                    let view = View::derived(parent_view, text.clone(), alignment);
                    doc.add(&key, view, provenance)
                        .map_err(|e| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail: e.to_string() })?;
                    let added = doc.get_key(&key).expect("the view just added");
                    cx.views.insert(record.instance.clone(), added.id());
                    cx.lens.insert((View::ID.to_owned(), record.instance.clone()), 1);
                }
            }
            (StatusRecord::Complete, None) => {
                return Err(malformed("a complete view with no text"));
            }
            (StatusRecord::Skipped(reason), _) => {
                doc.skip(key.id(), reason.clone(), provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail: e.to_string() })?;
            }
            (StatusRecord::Failed(error), _) => {
                let error = rebuild_error(error, codecs, &mut cx, View::ID)?;
                doc.fail(key.id(), error, provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail: e.to_string() })?;
            }
        }
    }

    for record in &graph.layers {
        cx.layer = record.layer.clone();
        let provenance = rebuild_provenance(
            &record.provenance,
            codecs,
            &mut cx,
            &record.layer,
            !matches!(record.status, StatusRecord::Failed(_)),
        )?;
        let codec = codecs.by_layer_name(&record.layer).expect("checked above");
        match &record.status {
            StatusRecord::Complete => {
                let body = record.body.clone().expect("checked above");
                let len = (codec.import)(&mut doc, record, body, &mut cx, provenance)?;
                if len != record.len as usize {
                    return Err(ExportError::InvalidLayer {
                        layer: record.layer.clone(),
                        detail: format!("the record counts {} items, the codec restored {len}", record.len),
                    });
                }
                cx.lens.insert((record.layer.clone(), record.instance.clone()), len);
            }
            StatusRecord::Skipped(reason) => {
                let key = (codec.key)(cx.intern(&record.instance.clone()));
                doc.skip(key, reason.clone(), provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: record.layer.clone(), detail: e.to_string() })?;
            }
            StatusRecord::Failed(error) => {
                let key = (codec.key)(cx.intern(&record.instance.clone()));
                let error = rebuild_error(error, codecs, &mut cx, &record.layer)?;
                doc.fail(key, error, provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: record.layer.clone(), detail: e.to_string() })?;
            }
        }
    }
    Ok(doc)
}

fn put_error(w: &mut Writer, error: &ErrorRecord) -> Result<(), ExportError> {
    use crate::bytes::SourceError;
    use crate::offsets::{OffsetError, OffsetUnit};
    match error {
        ErrorRecord::Missing(r) => {
            w.u8(0);
            put_layer_ref(w, r)?;
        }
        ErrorRecord::Duplicate(r) => {
            w.u8(1);
            put_layer_ref(w, r)?;
        }
        ErrorRecord::Clash(r) => {
            w.u8(2);
            put_layer_ref(w, r)?;
        }
        ErrorRecord::Bounds { layer, view, detail } => {
            w.u8(3);
            put_layer_ref(w, layer)?;
            w.str(view)?;
            w.str(detail)?;
        }
        ErrorRecord::Misaligned { layer, anchor, len, anchor_len } => {
            w.u8(4);
            put_layer_ref(w, layer)?;
            put_layer_ref(w, anchor)?;
            w.u32(*len);
            w.u32(*anchor_len);
        }
        ErrorRecord::Unaligned { layer, on, read } => {
            w.u8(5);
            put_layer_ref(w, layer)?;
            put_layer_ref(w, on)?;
            w.str(read)?;
        }
        ErrorRecord::Invalid { layer, detail } => {
            w.u8(6);
            put_layer_ref(w, layer)?;
            w.str(detail)?;
        }
        ErrorRecord::Pipeline(detail) => {
            w.u8(7);
            w.str(detail)?;
        }
        ErrorRecord::Annotator { annotator, detail } => {
            w.u8(8);
            w.str(annotator)?;
            w.str(detail)?;
        }
        ErrorRecord::Skipped(reason) => {
            w.u8(9);
            w.str(reason)?;
        }
        ErrorRecord::Offset { unit, at, reason } => {
            w.u8(10);
            w.u8(match unit {
                OffsetUnit::Utf16 => 0,
                OffsetUnit::CodePoint => 1,
            });
            w.u32(*at);
            match reason {
                OffsetError::PastEnd { len } => {
                    w.u8(0);
                    w.u32(*len);
                }
                OffsetError::InsideCharacter => w.u8(1),
            }
        }
        ErrorRecord::Source(e) => {
            w.u8(11);
            match e {
                SourceError::TooLong { len } => {
                    w.u8(0);
                    w.u64(*len);
                }
                SourceError::Misaligned(detail) => {
                    w.u8(1);
                    w.str(detail)?;
                }
            }
        }
    }
    Ok(())
}

fn get_error(r: &mut Reader<'_>) -> Result<ErrorRecord, ExportError> {
    use crate::bytes::SourceError;
    use crate::offsets::{OffsetError, OffsetUnit};
    Ok(match r.u8()? {
        0 => ErrorRecord::Missing(get_layer_ref(r)?),
        1 => ErrorRecord::Duplicate(get_layer_ref(r)?),
        2 => ErrorRecord::Clash(get_layer_ref(r)?),
        3 => ErrorRecord::Bounds { layer: get_layer_ref(r)?, view: r.string()?, detail: r.string()? },
        4 => {
            let layer = get_layer_ref(r)?;
            let anchor = get_layer_ref(r)?;
            ErrorRecord::Misaligned { layer, anchor, len: r.u32()?, anchor_len: r.u32()? }
        }
        5 => {
            let layer = get_layer_ref(r)?;
            let on = get_layer_ref(r)?;
            ErrorRecord::Unaligned { layer, on, read: r.string()? }
        }
        6 => ErrorRecord::Invalid { layer: get_layer_ref(r)?, detail: r.string()? },
        7 => ErrorRecord::Pipeline(r.string()?),
        8 => ErrorRecord::Annotator { annotator: r.string()?, detail: r.string()? },
        9 => ErrorRecord::Skipped(r.string()?),
        10 => {
            let unit = match r.u8()? {
                0 => OffsetUnit::Utf16,
                1 => OffsetUnit::CodePoint,
                tag => return Err(malformed(format!("an offset unit tag of {tag}"))),
            };
            let at = r.u32()?;
            let reason = match r.u8()? {
                0 => OffsetError::PastEnd { len: r.u32()? },
                1 => OffsetError::InsideCharacter,
                tag => return Err(malformed(format!("an offset reason tag of {tag}"))),
            };
            ErrorRecord::Offset { unit, at, reason }
        }
        11 => ErrorRecord::Source(match r.u8()? {
            0 => SourceError::TooLong { len: r.u64()? },
            1 => SourceError::Misaligned(r.string()?),
            tag => return Err(malformed(format!("a source error tag of {tag}"))),
        }),
        tag => return Err(malformed(format!("an error tag of {tag}"))),
    })
}

fn put_status(w: &mut Writer, status: &StatusRecord) -> Result<(), ExportError> {
    match status {
        StatusRecord::Complete => w.u8(0),
        StatusRecord::Skipped(reason) => {
            w.u8(1);
            w.str(reason)?;
        }
        StatusRecord::Failed(error) => {
            w.u8(2);
            put_error(w, error)?;
        }
    }
    Ok(())
}

fn get_status(r: &mut Reader<'_>) -> Result<StatusRecord, ExportError> {
    Ok(match r.u8()? {
        0 => StatusRecord::Complete,
        1 => StatusRecord::Skipped(r.string()?),
        2 => StatusRecord::Failed(get_error(r)?),
        tag => return Err(malformed(format!("a status tag of {tag}"))),
    })
}

fn put_provenance(w: &mut Writer, p: &ProvenanceRecord) -> Result<(), ExportError> {
    w.str(&p.annotator)?;
    w.str(&p.version)?;
    w.u64(p.config);
    match &p.model {
        None => w.u8(0),
        Some(model) => {
            w.u8(1);
            w.str(&model.id)?;
            w.str(&model.revision)?;
            w.str(&model.sha256)?;
        }
    }
    match &p.execution {
        None => w.u8(0),
        Some(execution) => {
            w.u8(1);
            w.str(&execution.backend)?;
            w.str(&execution.device)?;
            w.str(&execution.precision)?;
        }
    }
    for refs in [&p.reads, &p.on] {
        w.usize(refs.len())?;
        for r in refs {
            put_layer_ref(w, r)?;
        }
    }
    Ok(())
}

fn get_provenance(r: &mut Reader<'_>) -> Result<ProvenanceRecord, ExportError> {
    let annotator = r.string()?;
    let version = r.string()?;
    let config = r.u64()?;
    let model = match r.u8()? {
        0 => None,
        1 => Some(ModelRef { id: r.string()?, revision: r.string()?, sha256: r.string()? }),
        tag => return Err(malformed(format!("a model tag of {tag}"))),
    };
    let execution = match r.u8()? {
        0 => None,
        1 => Some(Execution { backend: r.string()?, device: r.string()?, precision: r.string()? }),
        tag => return Err(malformed(format!("an execution tag of {tag}"))),
    };
    let mut read_refs = || -> Result<Vec<LayerRef>, ExportError> {
        let mut out = Vec::new();
        for _ in 0..r.usize()? {
            out.push(get_layer_ref(r)?);
        }
        Ok(out)
    };
    let reads = read_refs()?;
    let on = read_refs()?;
    Ok(ProvenanceRecord { annotator, version, config, model, execution, reads, on })
}

fn put_edits(w: &mut Writer, edits: &[Edit]) -> Result<(), ExportError> {
    w.usize(edits.len())?;
    for edit in edits {
        w.span(edit.parent);
        w.span(edit.child);
    }
    Ok(())
}

fn get_edits(r: &mut Reader<'_>) -> Result<Vec<Edit>, ExportError> {
    let mut out = Vec::new();
    for _ in 0..r.usize()? {
        out.push(Edit { parent: r.span()?, child: r.span()? });
    }
    Ok(out)
}

impl DocumentGraph {
    /// The canonical bytes of this graph: deterministic for a given
    /// graph, versioned by [`FORMAT_VERSION`]. A hand-built graph whose
    /// counts overflow a `u32` is refused rather than truncated.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ExportError> {
        fn put(graph: &DocumentGraph) -> Result<Vec<u8>, ExportError> {
            let mut w = Writer::new();
            w.out.extend_from_slice(MAGIC);
            w.u16(graph.version);
            w.bytes(&graph.source.bytes)?;
            w.str(&graph.source.text)?;
            match &graph.source.alignment {
                None => w.u8(0),
                Some(edits) => {
                    w.u8(1);
                    put_edits(&mut w, edits)?;
                }
            }
            w.usize(graph.views.len())?;
            for view in &graph.views {
                w.str(&view.instance)?;
                put_status(&mut w, &view.status)?;
                put_provenance(&mut w, &view.provenance)?;
                match &view.parent {
                    None => w.u8(0),
                    Some(parent) => {
                        w.u8(1);
                        w.str(parent)?;
                    }
                }
                match &view.text {
                    None => w.u8(0),
                    Some(text) => {
                        w.u8(1);
                        w.str(text)?;
                    }
                }
                put_edits(&mut w, &view.edits)?;
            }
            w.usize(graph.layers.len())?;
            for layer in &graph.layers {
                w.str(&layer.layer)?;
                w.str(&layer.instance)?;
                w.u16(layer.schema);
                w.u32(layer.len);
                put_status(&mut w, &layer.status)?;
                put_provenance(&mut w, &layer.provenance)?;
                match &layer.body {
                    None => w.u8(0),
                    Some(body) => {
                        w.u8(1);
                        w.bytes(body)?;
                    }
                }
            }
            Ok(w.finish())
        }
        put(self)
    }

    /// Reads what [`DocumentGraph::to_bytes`] wrote.
    pub fn from_bytes(bytes: &[u8]) -> Result<DocumentGraph, ExportError> {
        let mut r = Reader::new(bytes);
        if r.take(MAGIC.len())? != MAGIC {
            return Err(malformed("not a document graph (bad magic)"));
        }
        let version = r.u16()?;
        if version != FORMAT_VERSION {
            return Err(ExportError::UnsupportedVersion { found: version, supported: FORMAT_VERSION });
        }
        let source_bytes = r.bytes()?.to_vec();
        let source_text = r.string()?;
        let alignment = match r.u8()? {
            0 => None,
            1 => Some(get_edits(&mut r)?),
            tag => return Err(malformed(format!("an alignment tag of {tag}"))),
        };
        let mut views = Vec::new();
        for _ in 0..r.usize()? {
            let instance = r.string()?;
            let status = get_status(&mut r)?;
            let provenance = get_provenance(&mut r)?;
            let parent = match r.u8()? {
                0 => None,
                1 => Some(r.string()?),
                tag => return Err(malformed(format!("a parent tag of {tag}"))),
            };
            let text = match r.u8()? {
                0 => None,
                1 => Some(r.string()?),
                tag => return Err(malformed(format!("a text tag of {tag}"))),
            };
            let edits = get_edits(&mut r)?;
            views.push(ViewRecord { instance, status, provenance, parent, text, edits });
        }
        let mut layers = Vec::new();
        for _ in 0..r.usize()? {
            let layer = r.string()?;
            let instance = r.string()?;
            let schema = r.u16()?;
            let len = r.u32()?;
            let status = get_status(&mut r)?;
            let provenance = get_provenance(&mut r)?;
            let body = match r.u8()? {
                0 => None,
                1 => Some(r.bytes()?.to_vec()),
                tag => return Err(malformed(format!("a body tag of {tag}"))),
            };
            layers.push(LayerRecord { layer, instance, schema, len, status, provenance, body });
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes"));
        }
        Ok(DocumentGraph {
            version,
            source: SourceRecord { bytes: source_bytes, text: source_text, alignment },
            views,
            layers,
        })
    }
}
