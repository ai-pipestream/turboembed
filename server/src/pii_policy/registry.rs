//! The server's [`PolicyResolver`]: keyed-token strategies resolve their
//! key reference through #1909's token key registry (`--pii-token-key`),
//! every registered key loaded at start. Dictionaries have no registry
//! yet, so a dictionary reference is refused.

use std::sync::Arc;

use super::{PolicyResolver, ResolveError};
use crate::release::keys::{TokenKey, TokenKeyFile};

/// The registered token keys, by id and version. `Debug` names the
/// references only.
#[derive(Default)]
pub struct ServerResolver {
    keys: Vec<TokenKey>,
}

impl std::fmt::Debug for ServerResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.keys.iter().map(|k| k.key.to_string())).finish()
    }
}

impl ServerResolver {
    /// Reads every registered key file.
    pub fn load(files: &[TokenKeyFile]) -> Result<ServerResolver, String> {
        Ok(ServerResolver { keys: files.iter().map(TokenKey::load).collect::<Result<_, _>>()? })
    }

    pub fn shared(self) -> Arc<ServerResolver> {
        Arc::new(self)
    }
}

impl PolicyResolver for ServerResolver {
    fn hmac_key(&self, id: &str, version: u32) -> Result<&[u8], ResolveError> {
        self.keys
            .iter()
            .find(|k| k.key.id == id && k.key.version == version)
            .map(TokenKey::bytes)
            .ok_or(ResolveError::NotRegistered)
    }

    fn terms(&self, _: &str, _: &str) -> Result<Arc<[String]>, ResolveError> {
        Err(ResolveError::NoRegistry)
    }

    fn map_entries(&self, _: &str, _: &str) -> Result<Arc<[(String, String)]>, ResolveError> {
        Err(ResolveError::NoRegistry)
    }
}
