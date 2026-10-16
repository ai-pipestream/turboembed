//! Decoding bytes to text with an alignment back to the bytes, detecting
//! the encoding of unlabelled bytes, and the single-byte code pages the
//! mojibake repair re-encodes text with.
//!
//! The decoders implement the WHATWG Encoding Standard; the index tables
//! under `src/tables` are generated from its index files by
//! `turbo-encoding-gen` (see the README for how to regenerate them).

pub mod charmap;
pub mod decode;
pub mod detect;
mod open;
pub mod repair;
mod tables;

pub use decode::{Decoded, bom_sniff, decode, decode_without_bom_handling};
pub use detect::{Detection, EncodingDetector, detect};
pub use open::{OPEN_ID, open, settle};
pub use turbo_encoding_layers::{Encoding, EncodingEvidence, SourceEncoding};
