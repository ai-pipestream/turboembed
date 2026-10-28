//! Retired Apache dependency model import.
use super::FeedforwardDependencyModel;
use crate::{Result, error};
impl FeedforwardDependencyModel {
    /// Returns the retirement error without decoding an Apache model.
    pub fn from_apache_bytes(_bytes: &[u8]) -> Result<Self> {
        Err(error(turbo_maxent::Error::Disabled))
    }
}
