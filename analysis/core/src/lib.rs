//! Typed, layered document analysis: a document's text, the views made
//! from it, and the layers annotators add to it.
//!
//! Every layer is its own Rust type, stored by column, so reading one
//! gives that type and an index into one layer cannot index another.
//! Every offset is a UTF-8 byte offset into one text view, and every view
//! maps back to exact source bytes. The layers add-ons share are in
//! turbo-analysis-layers; each add-on, the chunker first, is a crate of
//! its own. docs/analysis.md describes the model.

pub mod analyzer;
pub mod annotate;
pub mod doc;
pub mod error;
pub mod layer;
pub mod registry;
pub mod span;
pub mod text;

pub use analyzer::Analyzer;
pub use annotate::{Annotator, Bound, Refs};
pub use doc::{Document, Status, View};
pub use error::Error;
pub use layer::{Anchor, Execution, Key, KeyId, Layer, LayerName, ModelRef, Provenance};
pub use span::{Ix, IxRange, Span, Spans};
pub use text::{Alignment, ViewBuilder, ViewId};
