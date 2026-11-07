//! Token keys: registered, versioned key references in server config whose
//! bytes are read from a file at start (#1909). The bytes stay in
//! [`KeyBytes`], whose `Debug` prints nothing of them; no message, error
//! or event carries them, only the reference.

use std::fmt;
use std::path::PathBuf;

use turbo_pii::checksum::sha256;

use crate::proto::result::{PiiPolicyKeyRef, PiiValueToken};

/// The fewest and most bytes a token key file may hold.
pub const MIN_KEY_BYTES: usize = 32;
pub const MAX_KEY_BYTES: usize = 1024;

/// What a token is a MAC over, before the value: tokens of this scheme
/// never equal a MAC of the bare value under the same key.
const TOKEN_DOMAIN: &[u8] = b"turbo-pii-token-v1\0";

/// A registered key: its id and version.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KeyRef {
    pub id: String,
    pub version: u32,
}

impl KeyRef {
    /// `ID:VERSION`; the id is ASCII letters, digits, `-`, `_` and `.`,
    /// and the version a number from 1.
    pub fn parse(s: &str) -> Result<KeyRef, String> {
        let (id, version) = s.split_once(':').ok_or_else(|| format!("`{s}` is not ID:VERSION"))?;
        KeyRef::new(id, version)
    }

    fn new(id: &str, version: &str) -> Result<KeyRef, String> {
        if id.is_empty() || id.len() > 64 || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b)) {
            return Err(format!("key id `{id}` is not 1 to 64 of A-Z, a-z, 0-9, -, _ and ."));
        }
        match version.parse::<u32>() {
            Ok(v) if v > 0 => Ok(KeyRef { id: id.to_owned(), version: v }),
            _ => Err(format!("key version `{version}` is not a number from 1")),
        }
    }

    /// The one key reference type of the API (turbo_pii_policy.proto).
    pub fn policy(&self) -> PiiPolicyKeyRef {
        PiiPolicyKeyRef { id: self.id.clone(), version: self.version }
    }

    /// A policy's key reference, refused without an id or a version.
    pub fn of_policy(r: &PiiPolicyKeyRef) -> Result<KeyRef, String> {
        if r.id.is_empty() || r.version == 0 {
            return Err("a key reference needs an id and a version from 1".into());
        }
        Ok(KeyRef { id: r.id.clone(), version: r.version })
    }

    /// The key a released token names: its `key_id` and `key_version`,
    /// the PiiPolicyKeyRef identity carried as plain fields.
    pub fn of_token(t: &PiiValueToken) -> Result<KeyRef, String> {
        KeyRef::of_policy(&PiiPolicyKeyRef { id: t.key_id.clone(), version: t.key_version })
    }
}

impl fmt::Display for KeyRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.id, self.version)
    }
}

/// One `--pii-token-key`: a key reference and the file holding its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenKeyFile {
    pub key: KeyRef,
    pub path: PathBuf,
}

impl TokenKeyFile {
    /// `ID:VERSION:PATH`; the path may hold `:`.
    pub fn parse(s: &str) -> Result<TokenKeyFile, String> {
        let mut parts = s.splitn(3, ':');
        let (Some(id), Some(version), Some(path)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(format!("`{s}` is not ID:VERSION:PATH"));
        };
        if path.is_empty() {
            return Err(format!("`{s}`: the path is empty"));
        }
        Ok(TokenKeyFile { key: KeyRef::new(id, version)?, path: PathBuf::from(path) })
    }
}

/// Key material. Its `Debug` names its length only.
pub struct KeyBytes(Vec<u8>);

impl fmt::Debug for KeyBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KeyBytes(<{} bytes withheld>)", self.0.len())
    }
}

/// The key tokens are issued under.
#[derive(Debug)]
pub struct TokenKey {
    pub key: KeyRef,
    bytes: KeyBytes,
}

impl TokenKey {
    /// Reads `file`. An error names the reference and the path, never the
    /// bytes.
    pub fn load(file: &TokenKeyFile) -> Result<TokenKey, String> {
        let bytes = std::fs::read(&file.path)
            .map_err(|e| format!("--pii-token-key {}: {}: {e}", file.key, file.path.display()))?;
        if !(MIN_KEY_BYTES..=MAX_KEY_BYTES).contains(&bytes.len()) {
            return Err(format!(
                "--pii-token-key {}: {} holds {} bytes, not {MIN_KEY_BYTES} to {MAX_KEY_BYTES}",
                file.key,
                file.path.display(),
                bytes.len()
            ));
        }
        Ok(TokenKey { key: file.key.clone(), bytes: KeyBytes(bytes) })
    }

    /// The token of `value`: HMAC-SHA-256 under this key, equal for equal
    /// values, with the key's reference beside it.
    pub fn token(&self, value: &str) -> PiiValueToken {
        let mut msg = Vec::with_capacity(TOKEN_DOMAIN.len() + value.len());
        msg.extend_from_slice(TOKEN_DOMAIN);
        msg.extend_from_slice(value.as_bytes());
        PiiValueToken {
            token: sha256::hmac(&self.bytes.0, &msg).to_vec(),
            key_id: self.key.id.clone(),
            key_version: self.key.version,
        }
    }
}

/// The token key the configured keys name: the one key, or the one
/// `--pii-token-key-active` names among several.
pub fn active<'a>(keys: &'a [TokenKeyFile], active: Option<&KeyRef>) -> Result<Option<&'a TokenKeyFile>, String> {
    for (i, k) in keys.iter().enumerate() {
        if keys[..i].iter().any(|o| o.key == k.key) {
            return Err(format!("--pii-token-key {} given twice", k.key));
        }
    }
    match (keys, active) {
        ([], None) => Ok(None),
        ([], Some(a)) => Err(format!("--pii-token-key-active {a} needs --pii-token-key")),
        ([one], None) => Ok(Some(one)),
        (_, None) => Err("several --pii-token-key values need --pii-token-key-active".into()),
        (keys, Some(a)) => keys
            .iter()
            .find(|k| &k.key == a)
            .map(Some)
            .ok_or_else(|| format!("--pii-token-key-active {a} is not a --pii-token-key")),
    }
}
