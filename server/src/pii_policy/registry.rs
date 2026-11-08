//! The server's [`PolicyResolver`]: keyed-token strategies resolve their
//! key reference through #1909's token key registry (`--pii-token-key`),
//! every registered key loaded at start, and dictionary references
//! resolve through the vocabulary store (`--vocabulary-dir`, #1882 part
//! C), pinned by content.
//!
//! A [`PiiDictionaryRef`](crate::proto::result::PiiDictionaryRef) names a
//! stored dictionary by artifact id and the SHA-256 of its bytes. It
//! resolves only when the store holds that artifact with exactly that
//! digest and in the format its use takes: `plain-word-list` for term
//! sources, `headword-definition-tsv` for map tables. Because the key is
//! the content, a resolved dictionary is cached by `(artifact_id,
//! sha256)` for the server's life. No refusal carries a term, a
//! replacement, a key or a path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{PolicyResolver, ResolveError};
use crate::release::keys::{TokenKey, TokenKeyFile};
use crate::vocabulary::{Store, Stored, formats};

/// The format a term source takes.
pub const TERMS_FORMAT: &str = "plain-word-list";
/// The format a map table takes.
pub const MAP_FORMAT: &str = "headword-definition-tsv";

#[derive(Clone)]
enum Resolved {
    Terms(Arc<[String]>),
    Map(Arc<[(String, String)]>),
}

/// The registered token keys, by id and version, and the vocabulary store
/// dictionaries resolve through. `Debug` names the key references and the
/// cached dictionary references only.
#[derive(Default)]
pub struct ServerResolver {
    keys: Vec<TokenKey>,
    store: Option<Arc<Store>>,
    cache: Mutex<HashMap<(String, String), Resolved>>,
}

impl std::fmt::Debug for ServerResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cached: Vec<String> = self.cache().keys().map(|(id, sha)| format!("{id}@{sha}")).collect();
        f.debug_struct("ServerResolver")
            .field("keys", &self.keys.iter().map(|k| k.key.to_string()).collect::<Vec<_>>())
            .field("vocabulary_store", &self.store.is_some())
            .field("dictionaries", &cached)
            .finish()
    }
}

impl ServerResolver {
    /// Reads every registered key file; dictionaries resolve through
    /// `store`, and are refused without one.
    pub(crate) fn load(files: &[TokenKeyFile], store: Option<Arc<Store>>) -> Result<ServerResolver, String> {
        let keys = files.iter().map(TokenKey::load).collect::<Result<_, _>>()?;
        Ok(ServerResolver { keys, store, cache: Mutex::default() })
    }

    pub fn shared(self) -> Arc<ServerResolver> {
        Arc::new(self)
    }

    fn cache(&self) -> std::sync::MutexGuard<'_, HashMap<(String, String), Resolved>> {
        self.cache.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The dictionary `id` pinned to `sha256` in `format`, read with
    /// `parse` the first time and from the cache after.
    fn dictionary(
        &self,
        id: &str,
        sha256: &str,
        format: &str,
        parse: impl FnOnce(&[u8]) -> Result<Resolved, String>,
    ) -> Result<Resolved, ResolveError> {
        let store = self.store.as_ref().ok_or(ResolveError::NoRegistry)?;
        let d = store.dictionary(id).ok_or(ResolveError::NotRegistered)?;
        if d.descriptor.sha256 != sha256 {
            return Err(ResolveError::DigestMismatch);
        }
        if d.descriptor.format_id != format {
            return Err(ResolveError::WrongFormat { format: d.descriptor.format_id });
        }
        let key = (id.to_owned(), sha256.to_owned());
        if let Some(hit) = self.cache().get(&key) {
            return Ok(hit.clone());
        }
        // The store checks the bytes against the descriptor's digest; its
        // message names a path, so it is dropped.
        let bytes = match store.read(id).map_err(|_| ResolveError::Unreadable)? {
            Some(Stored::Dictionary(desc, bytes)) if desc.sha256 == sha256 => bytes,
            Some(Stored::Dictionary(..)) => return Err(ResolveError::DigestMismatch),
            Some(Stored::Vocabulary(..)) | None => return Err(ResolveError::NotRegistered),
        };
        let resolved = parse(&bytes).map_err(|_| ResolveError::Unreadable)?;
        self.cache().insert(key, resolved.clone());
        Ok(resolved)
    }

    /// How many dictionaries are cached.
    pub fn cached_dictionaries(&self) -> usize {
        self.cache().len()
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

    fn terms(&self, id: &str, sha256: &str) -> Result<Arc<[String]>, ResolveError> {
        let parse = |b: &[u8]| formats::word_list(b).map(|t| Resolved::Terms(t.into()));
        match self.dictionary(id, sha256, TERMS_FORMAT, parse)? {
            Resolved::Terms(t) => Ok(t),
            Resolved::Map(_) => Err(ResolveError::WrongFormat { format: MAP_FORMAT.to_owned() }),
        }
    }

    fn map_entries(&self, id: &str, sha256: &str) -> Result<Arc<[(String, String)]>, ResolveError> {
        let parse = |b: &[u8]| {
            let mut rows = formats::map_table(b)?;
            // Sorted so the cached rows do not depend on file order; keys equal
            // once normalized are refused when the table is built.
            rows.sort_unstable();
            Ok(Resolved::Map(rows.into()))
        };
        match self.dictionary(id, sha256, MAP_FORMAT, parse)? {
            Resolved::Map(m) => Ok(m),
            Resolved::Terms(_) => Err(ResolveError::WrongFormat { format: TERMS_FORMAT.to_owned() }),
        }
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
