//! Reading OpenNLP 3.x model files and evaluating their classifiers.
//!
//! An OpenNLP `.bin` model is a zip archive. [`ModelFile`] opens one: it
//! reads the archive with this crate's own [`zip`] and [`inflate`]
//! modules, and the manifest as `java.util.Properties` reads it
//! ([`Properties`]). From the archive it reads the classifier
//! ([`MaxentModel`], for models of type `GIS`, `QN` and `Perceptron`) and
//! the dictionary XML ([`Dictionary`] as a set, [`Entries`] with each
//! entry's attributes).
//!
//! [`MaxentModel::eval`] takes a context's features as Java modified
//! UTF-8 bytes and writes one probability per outcome into the caller's
//! buffer. It computes them as the Java model of the same type does, in
//! `f64` throughout, and allocates nothing. docs/segmentation.md, section
//! "Model files", describes where this fits.
//!
//! Nothing here calls into a JVM; the tests compare against values the
//! JVM produced, kept as fixtures.

pub mod container;
mod crc32;
pub mod dictionary;
pub mod error;
mod hash;
pub mod inflate;
pub mod java_data;
pub mod model;
mod model_reader;
pub mod properties;
mod xml;
pub mod zip;

pub use container::ModelFile;
pub use crc32::crc32;
pub use dictionary::{Dictionary, Entries, Entry};
pub use error::{Error, Result};
pub use model::{MaxentModel, ModelKind};
pub use properties::Properties;
pub use zip::{Method, Zip};
