//! Classical model loading is disabled under #1511.
//!
//! ModelFile and MaxentModel readers return Error::Disabled before I/O or decoding.
//! Dictionary/archive utilities and in-memory classifiers remain for dependent
//! crates until the training-stack deletion in #1532. Replacement heads are
//! tracked in #1509.

pub mod container;
mod crc32;
pub mod dictionary;
pub mod error;
mod hash;
pub mod inflate;
pub mod java_data;
pub mod java_string;
pub mod model;
mod model_reader;
pub mod properties;
pub mod sha256 {
    pub use turbo_sha256::{Digest, sha256};
}
pub mod xml;
pub mod zip;

pub use container::ModelFile;
pub use crc32::crc32;
pub use dictionary::{Dictionary, Entries, Entry};
pub use error::{Error, Result};
pub use java_string::JavaString;
pub use model::{MaxentModel, ModelKind, best_outcome};
pub use properties::Properties;
pub use turbo_sha256::{Digest, sha256};
pub use zip::{Method, Zip};
