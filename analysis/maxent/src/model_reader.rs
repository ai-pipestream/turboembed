//! Classical OpenNLP model readers were retired under #1511.
use crate::{Error, MaxentModel, Result};

impl MaxentModel {
    /// Returns the retirement error without decoding raw model parameters.
    /// Replacement model heads are tracked in #1509.
    pub fn read(_bytes: &[u8]) -> Result<Self> {
        Err(Error::Disabled)
    }
}
