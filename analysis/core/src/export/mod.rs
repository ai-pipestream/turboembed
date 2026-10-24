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

mod alternatives;
mod columns;
mod context;
mod decode;
mod diff;
mod encode;
mod error;
mod error_wire;
mod graph;
mod provenance;
mod records;
mod registry;
mod relations;
mod slice;
mod view_ref;
mod wire;

pub use columns::{get_spans, get_spans_raw, get_tagset, put_spans, put_spans_raw, put_tagset, slice_spans_payload};
pub use context::{ExportCx, ImportCx, LayerCodec, SliceCx, SlicedBody};
pub use decode::import;
pub use diff::{GraphDiff, LayerDiffSummary};
pub use encode::export;
pub use error::ExportError;
pub use records::{
    DocumentGraph, ErrorRecord, FORMAT_VERSION, LayerRecord, LayerRef, NodeRef, ProvenanceRecord, SourceRecord,
    StatusRecord, ViewRecord, layer_ref, node_ref,
};
pub use registry::Codecs;
pub use view_ref::{DocumentGraphRef, LayerWireEntry, ViewWireEntry};
pub use wire::{Reader, Writer};

use columns::{get_layer_ref, put_layer_ref};
use error::malformed;
use error_wire::{get_error, put_error};
use provenance::{provenance_record, rebuild_error, rebuild_provenance, ref_of, status_record};
use records::MAGIC;
