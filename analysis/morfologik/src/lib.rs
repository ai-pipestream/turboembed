//! Morfologik finite-state automata (`FSA5Reader`, `CFSA2Reader`), morphological
//! dictionary readers (`MorfologikDictionaryReader`, `PoliMorfDictionaryReader`),
//! lemmatizer (`MorfologikLemmatizer`), POS tag dictionary (`MorfologikTagDictionary`),
//! and dictionary builder (`MorfologikDictionaryBuilder`), ported from OpenNLP 3.x's
//! `opennlp-formats` and `opennlp-morfologik`.
//!
//! No third-party crate is used: FSA5 (`0x05`) and CFSA2 (`0xc6`) automata, `.info`
//! metadata properties, `fsa.dict.input-conversion` rules, single-byte character tables
//! (`ISO-8859-1`, `ISO-8859-2`, `windows-1250`, `windows-1251`, `windows-1252`), and the
//! four Morfologik base-form encoders (`NONE`, `SUFFIX`, `PREFIX`, `INFIX`) are decoded
//! and validated directly in this crate.

pub mod charset;
pub mod dictionary;
pub mod encoding;
pub mod error;
pub mod fsa;
mod identity;
mod lemmatizer;
pub mod metadata;
pub mod reader;

pub use charset::DictionaryCharset;
pub use dictionary::{
    MorfologikDictionary, MorfologikDictionaryBuilder, MorfologikLemmatizer, MorfologikTagDictionary, WordData,
    serialize_fsa5,
};
pub use encoding::{BaseFormEncoding, CONTROL_OFFSET, REPLACE_FORM, decode_base_form, encode_base_form};
pub use error::MorfologikError;
pub use fsa::{Cfsa2Reader, Fsa5Reader, FsaReader, MAGIC, VERSION_CFSA2, VERSION_FSA5};
pub use metadata::{
    KEY_ENCODER, KEY_ENCODING, KEY_INPUT_CONVERSION, KEY_SEPARATOR, MorfologikInputConversion, MorfologikMetadata,
};
pub use reader::{ExpandedDictionaryLemmatizer, MorfologikDictionaryReader, NOT_FOUND, PoliMorfDictionaryReader};
