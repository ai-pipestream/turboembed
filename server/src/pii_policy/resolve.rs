//! Registered, versioned references: keys and dictionaries a policy names
//! but never carries, and the checks that keep a path or key material out
//! of a reference field.

use std::sync::Arc;

use super::error::{At, PolicyError};

/// Why a reference was not resolved. Never carries key bytes or terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveError {
    /// No key or dictionary is registered under the reference.
    NotRegistered,
    /// The server has no registry of this kind.
    NoRegistry,
}

impl ResolveError {
    pub(crate) fn reason(&self) -> String {
        match self {
            ResolveError::NotRegistered => "not registered".to_owned(),
            ResolveError::NoRegistry => "the server has no registry for it".to_owned(),
        }
    }
}

/// Resolves the keys and dictionaries a policy references. The server's
/// key and dictionary registries implement it (#1882 part B).
pub trait PolicyResolver {
    /// The bytes of the HMAC key registered as `id` at `version`.
    fn hmac_key(&self, id: &str, version: u32) -> Result<&[u8], ResolveError>;
    /// The terms of the dictionary `artifact_id` whose stored bytes hash to
    /// `sha256`.
    fn terms(&self, artifact_id: &str, sha256: &str) -> Result<Arc<[String]>, ResolveError>;
    /// The `(value, replacement)` entries of the map dictionary
    /// `artifact_id` whose stored bytes hash to `sha256`.
    fn map_entries(&self, artifact_id: &str, sha256: &str) -> Result<Arc<[(String, String)]>, ResolveError>;
}

/// A resolver with no registry: every reference is refused. A policy of
/// inline terms and masks still converts.
#[derive(Copy, Clone, Debug, Default)]
pub struct NoRegistry;

impl PolicyResolver for NoRegistry {
    fn hmac_key(&self, _: &str, _: u32) -> Result<&[u8], ResolveError> {
        Err(ResolveError::NoRegistry)
    }

    fn terms(&self, _: &str, _: &str) -> Result<Arc<[String]>, ResolveError> {
        Err(ResolveError::NoRegistry)
    }

    fn map_entries(&self, _: &str, _: &str) -> Result<Arc<[(String, String)]>, ResolveError> {
        Err(ResolveError::NoRegistry)
    }
}

/// The longest reference id accepted.
pub const MAX_REF_ID_BYTES: usize = 128;

/// A key id is refused as key material when it is this long or longer and
/// all hexadecimal digits: a registered id is a name, not a secret.
pub const KEY_MATERIAL_HEX_LEN: usize = 32;

/// Checks a reference id: 1 to [`MAX_REF_ID_BYTES`] ASCII letters, digits,
/// `.`, `_`, `-` or `:`, not starting with `.`. A path separator or `..`
/// is refused, so no field names a server file.
pub(crate) fn check_ref_id(at: At, field: &'static str, id: &str) -> Result<(), PolicyError> {
    let bad = |reason| Err(PolicyError::BadRef { at, field, reason });
    if id.is_empty() {
        return Err(PolicyError::Missing { at, field });
    }
    if id.len() > MAX_REF_ID_BYTES {
        return bad("is longer than 128 bytes");
    }
    if id.contains('/') || id.contains('\\') || id.contains("..") || id.starts_with('.') {
        return bad("looks like a file path; name a registered reference");
    }
    if !id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b':')) {
        return bad("has a character outside letters, digits, '.', '_', '-' and ':'");
    }
    Ok(())
}

/// Checks a key reference id: a reference id that does not look like key
/// material.
pub(crate) fn check_key_id(at: At, id: &str) -> Result<(), PolicyError> {
    check_ref_id(at, "key.id", id)?;
    if id.len() >= KEY_MATERIAL_HEX_LEN && id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(PolicyError::BadRef {
            at,
            field: "key.id",
            reason: "looks like key material; name a registered key",
        });
    }
    Ok(())
}

/// Checks a dictionary reference: a reference id and a 64-digit lowercase
/// hexadecimal SHA-256.
pub(crate) fn check_dictionary(at: At, artifact_id: &str, sha256: &str) -> Result<(), PolicyError> {
    check_ref_id(at, "dictionary.artifact_id", artifact_id)?;
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(PolicyError::BadRef {
            at,
            field: "dictionary.sha256",
            reason: "is not 64 lowercase hexadecimal digits",
        });
    }
    Ok(())
}
