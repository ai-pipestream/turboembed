//! Word tokenizers ported from OpenNLP 3.x onto the analysis API.
//!
//! [`me`] is the maxent tokenizer, `TokenizerME`: it reads an OpenNLP
//! tokenizer model and writes the `Tokens` layer with a confidence per
//! token. [`sha256`] identifies a model file in the provenance of every
//! layer the tokenizer writes. Every offset is a UTF-8 byte offset into
//! the view, on a character boundary. docs/segmentation.md describes the
//! design and every place the port differs from OpenNLP on purpose.

pub mod me;
pub mod sha256;
