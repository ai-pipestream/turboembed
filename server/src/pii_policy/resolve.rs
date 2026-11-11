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
    /// The server has no registry of this kind (a dictionary reference on
    /// a server without `--vocabulary-dir`).
    NoRegistry,
    /// The stored dictionary's bytes do not hash to the reference's
    /// SHA-256: the reference pins other content.
    DigestMismatch,
    /// The stored dictionary is in a format this use does not take; the
    /// format id is the store's, never the file's content.
    WrongFormat { format: String },
    /// The stored dictionary could not be read. The store's own message
    /// is dropped: it may name a path.
    Unreadable,
    /// The stored dictionary was read but does not parse as `format` for
    /// this use. `reason` is the parser's, which names a line or a byte
    /// offset and never a term.
    Unparsable { format: String, reason: String },
    /// An inline policy named an encryption key the server loaded without
    /// `--pii-grant` (#2111): such a key serves named `--pii-policy`
    /// policies only, since a caller choosing its own FF1 tweak would
    /// otherwise get a deterministic encryption oracle under it.
    NamedPoliciesOnly,
}

impl ResolveError {
    pub(crate) fn reason(&self) -> String {
        match self {
            ResolveError::NotRegistered => "not registered".to_owned(),
            ResolveError::NoRegistry => "the server has no registry for it".to_owned(),
            ResolveError::DigestMismatch => "the stored dictionary does not have this sha256".to_owned(),
            ResolveError::WrongFormat { format } => {
                format!("the stored dictionary's format {format} is not taken here")
            }
            ResolveError::Unreadable => "the stored dictionary cannot be read".to_owned(),
            ResolveError::NamedPoliciesOnly => {
                "a key loaded without --pii-grant serves named --pii-policy policies only".to_owned()
            }
            ResolveError::Unparsable { format, reason } => {
                format!("the stored dictionary does not parse as {format}: {reason}")
            }
        }
    }
}

/// Resolves the keys and dictionaries a policy references. The server's
/// key registry and its vocabulary store implement it (#1882 parts B
/// and C).
pub trait PolicyResolver {
    /// The bytes of the HMAC key registered as `id` at `version`.
    fn hmac_key(&self, id: &str, version: u32) -> Result<&[u8], ResolveError>;
    /// The bytes of the encryption key registered as `id` at `version`
    /// (`--pii-encryption-key`), for the AES-256-GCM and FF1 strategies.
    /// A token key of the same reference is another key and never
    /// answers.
    fn encryption_key(&self, id: &str, version: u32) -> Result<&[u8], ResolveError>;
    /// The terms of the `plain-word-list` dictionary `artifact_id` whose
    /// stored bytes hash to `sha256`.
    fn terms(&self, artifact_id: &str, sha256: &str) -> Result<Arc<[String]>, ResolveError>;
    /// The `(value, replacement)` entries of the `headword-definition-tsv`
    /// dictionary `artifact_id` whose stored bytes hash to `sha256`.
    fn map_entries(&self, artifact_id: &str, sha256: &str) -> Result<Arc<[(String, String)]>, ResolveError>;
    /// Whether a request's inline policy may use the encryption keys.
    /// False on a server that loaded them without `--pii-grant` (#2111):
    /// there callers need not be authenticated, so only the operator's
    /// named policies may encrypt under them.
    fn inline_encryption(&self) -> bool {
        true
    }
}

/// `inner` as an inline policy sees it on a server whose encryption keys
/// serve named policies only: every encryption reference is refused.
pub(crate) struct NamedOnlyEncryption<'a>(pub &'a dyn PolicyResolver);

impl PolicyResolver for NamedOnlyEncryption<'_> {
    fn hmac_key(&self, id: &str, version: u32) -> Result<&[u8], ResolveError> {
        self.0.hmac_key(id, version)
    }

    fn encryption_key(&self, _: &str, _: u32) -> Result<&[u8], ResolveError> {
        Err(ResolveError::NamedPoliciesOnly)
    }

    fn terms(&self, artifact_id: &str, sha256: &str) -> Result<Arc<[String]>, ResolveError> {
        self.0.terms(artifact_id, sha256)
    }

    fn map_entries(&self, artifact_id: &str, sha256: &str) -> Result<Arc<[(String, String)]>, ResolveError> {
        self.0.map_entries(artifact_id, sha256)
    }

    fn inline_encryption(&self) -> bool {
        false
    }
}

/// A resolver with no registry: every reference is refused. A policy of
/// inline terms and masks still converts.
#[derive(Copy, Clone, Debug, Default)]
pub struct NoRegistry;

impl PolicyResolver for NoRegistry {
    fn hmac_key(&self, _: &str, _: u32) -> Result<&[u8], ResolveError> {
        Err(ResolveError::NoRegistry)
    }

    fn encryption_key(&self, _: &str, _: u32) -> Result<&[u8], ResolveError> {
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
