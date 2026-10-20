//! Typed, layered document analysis: a document's text, the views made
//! from it, and the layers annotators add to it.
//!
//! Every layer is its own Rust type, stored by column, so reading one
//! gives that type and an index into one layer cannot index another.
//! Every offset is a UTF-8 byte offset into one text view, and every view
//! maps back to exact source bytes. The layers add-ons share are in
//! turbo-analysis-layers; each add-on, the chunker first, is a crate of
//! its own. docs/analysis.md describes the model.

pub mod alternatives;
pub mod analyzer;
pub mod annotate;
pub mod bytes;
pub mod doc;
pub mod error;
pub mod evidence;
pub mod export;
pub mod graph;
pub mod layer;
pub mod markup;
pub mod offsets;
pub mod predicate;
pub mod registry;
pub mod span;
pub mod text;

pub use alternatives::{Alternative, Alternatives, ModeSet, NBest, SplitMode};
pub use analyzer::Analyzer;
pub use annotate::{Annotator, Bound, Refs};
pub use bytes::{DecodeBuilder, MAX_SOURCE_LEN, SourceBytes, SourceError};
pub use doc::{Document, LayerStatus, Status, View};
pub use error::Error;
pub use evidence::{Evidence, EvidenceError, EvidenceStep, Evidenced, Producer, Support};
pub use export::{
    Codecs, DocumentGraph, ErrorRecord, ExportCx, ExportError, FORMAT_VERSION, ImportCx, LayerCodec, LayerRecord,
    LayerRef, NodeRef, ProvenanceRecord, Reader, SourceRecord, StatusRecord, ViewRecord, Writer, export, get_spans,
    get_tagset, import, layer_ref, node_ref, put_spans, put_tagset,
};
pub use graph::{Edge, Edges, Groups};
pub use layer::{Anchor, Execution, Key, KeyId, Layer, LayerName, ModelRef, Provenance, Tag, TagSet};
pub use markup::{MarkupError, MarkupText, TagKind, TagSpan, TextRun, TextRunKind};
pub use offsets::{
    CodePointOffset, CodePointSpan, OffsetCursor, OffsetError, OffsetTable, OffsetUnit, Utf16Offset, Utf16Span,
};
pub use predicate::{
    ConditionalAnnotator, ConditionalStep, FilterAnnotator, FilterableLayer, SpanFilter, co_written_aligned_anchors,
    retain_aligned_anchor,
};
pub use span::{FilterRule, FragmentError, Fragments, Ix, IxRange, Overlap, Span, Spans, TextSpans, filter_spans};
pub use text::{Alignment, Edit, Side, ViewBuilder, ViewId};
