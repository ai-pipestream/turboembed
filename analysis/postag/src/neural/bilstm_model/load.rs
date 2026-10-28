//! Retired ONLP BiLSTM import and character vocabulary helpers.
use super::BilstmPosModel;
use crate::neural::{NeuralError, encode_str_modified_utf8, encode_unit_modified_utf8};
use std::io::Read;
use std::path::Path;
/// Helper to build a character vocabulary entry list from `&str` keys.
pub fn char_vocabulary_from_strs<I, S>(entries: I) -> Vec<(Vec<u8>, usize)>
where
    I: IntoIterator<Item = (S, usize)>,
    S: AsRef<str>,
{
    entries.into_iter().map(|(k, v)| (encode_str_modified_utf8(k.as_ref()), v)).collect()
}

/// Helper to build a character vocabulary entry from a single UTF-16 code unit.
pub fn char_key_from_unit(unit: u16) -> Vec<u8> {
    encode_unit_modified_utf8(unit)
}
impl BilstmPosModel {
    /// Returns the retirement error without reading the supplied stream.
    pub fn load<R: Read>(_reader: R) -> Result<Self, NeuralError> {
        Err(NeuralError::Disabled)
    }
    /// Returns the retirement error without accessing the filesystem.
    pub fn load_from_path(_path: impl AsRef<Path>) -> Result<Self, NeuralError> {
        Err(NeuralError::Disabled)
    }
}
