//! Schema-driven extraction with one GLiNER2 encoder pass: entity types,
//! document classification tasks and structured record types together
//! (docs/schema-extract.md).

pub mod decode;
pub mod model;
pub mod prompt;
pub mod schema;

pub use decode::{Corroborate, Extraction, decode};
pub use model::{ExtractError, Gliner2Extractor};
pub use schema::{
    Cardinality, Check, ClassificationTask, EntityType, ExtractionSchema, FieldSpec, Label, Limits, SchemaError,
    Source, StructureSpec,
};
