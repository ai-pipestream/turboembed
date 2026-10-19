use super::*;

/// The version of this boundary: the record shapes and byte encoding.
/// Bumped when either changes meaning; older bytes are refused, not
/// migrated.
pub const FORMAT_VERSION: u16 = 1;

pub(super) const MAGIC: &[u8] = b"TGRF";

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
