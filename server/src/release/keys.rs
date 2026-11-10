//! Token keys (#1909) and encryption keys (#1885): registered, versioned
//! key references in server config whose bytes are read from a file at
//! start. The bytes stay in
//! [`KeyBytes`], whose `Debug` prints nothing of them; no message, error
//! or event carries them, only the reference.

use std::fmt;
use std::path::PathBuf;

use crate::proto::result::{PiiPolicyKeyRef, PiiValueToken};

/// The fewest and most bytes a token key file may hold.
pub const MIN_KEY_BYTES: usize = 32;
pub const MAX_KEY_BYTES: usize = 1024;

/// The bytes an encryption key file holds: an AES-256 key, for the
/// AES-256-GCM and FF1 redaction strategies.
pub const ENCRYPTION_KEY_BYTES: usize = 32;

/// What a token is a MAC over, before the kind and the value: tokens of
/// this scheme never equal a MAC of the bare value under the same key.
const TOKEN_DOMAIN: &[u8] = b"turbo-pii-token-v1\0";

/// A registered key: its id and version.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KeyRef {
    pub id: String,
    pub version: u32,
}

impl KeyRef {
    fn valid_id(id: &str) -> bool {
        !id.is_empty() && id.len() <= 64 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    }

    /// `ID:VERSION`; the id is ASCII letters, digits, `-`, `_` and `.`,
    /// and the version a number from 1. An error never repeats the input,
    /// which may be key text pasted in the wrong place: it names the part
    /// that is wrong and its length.
    pub fn parse(s: &str) -> Result<KeyRef, String> {
        let (id, version) = s.split_once(':').ok_or_else(|| shape("ID:VERSION", s))?;
        KeyRef::new(id, version)
    }

    fn new(id: &str, version: &str) -> Result<KeyRef, String> {
        if !Self::valid_id(id) {
            let n = id.len();
            return Err(format!("the key id ({n} bytes, not shown) is not 1 to 64 of A-Z, a-z, 0-9, -, _ and ."));
        }
        match version.parse::<u32>() {
            Ok(v) if v > 0 => Ok(KeyRef { id: id.to_owned(), version: v }),
            _ => Err(format!("the key version ({} bytes, not shown) is not a number from 1", version.len())),
        }
    }

    /// The one key reference type of the API (turbo_pii_policy.proto).
    pub fn policy(&self) -> PiiPolicyKeyRef {
        PiiPolicyKeyRef { id: self.id.clone(), version: self.version }
    }

    /// A policy's key reference, with the same ID contract as the CLI.
    pub fn of_policy(r: &PiiPolicyKeyRef) -> Result<KeyRef, String> {
        if !Self::valid_id(&r.id) || r.version == 0 {
            return Err("a key reference needs a valid ID and a version from 1".into());
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
    /// `ID:VERSION:PATH`; the path may hold `:`. As with
    /// [`KeyRef::parse`], an error never repeats the input, not even an id
    /// that has the right shape: pasted key text can. Errors after a
    /// complete parse, such as a file that does not read, name the
    /// reference and the path.
    pub fn parse(s: &str) -> Result<TokenKeyFile, String> {
        let mut parts = s.splitn(3, ':');
        let (Some(id), Some(version), Some(path)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(shape("ID:VERSION:PATH", s));
        };
        let key = KeyRef::new(id, version)?;
        if path.is_empty() {
            return Err("the path is empty (nothing after ID:VERSION:)".into());
        }
        Ok(TokenKeyFile { key, path: PathBuf::from(path) })
    }
}

/// A value that does not split into `form`, described without its text.
fn shape(form: &str, s: &str) -> String {
    let parts = s.split(':').count();
    let n = s.len();
    format!("the value ({n} bytes in {parts} `:`-separated parts, not shown) is not {form}")
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

/// The bytes of a key file. A read error names `flag`, the reference and
/// the path's length: the path may be key text pasted in the wrong place.
/// `std::fs::read`'s error does not repeat the path.
fn read_key(flag: &str, file: &TokenKeyFile) -> Result<Vec<u8>, String> {
    std::fs::read(&file.path).map_err(|e| {
        let n = file.path.as_os_str().len();
        format!("{flag} {}: the key file (a path of {n} bytes, not shown) cannot be read: {e}", file.key)
    })
}

impl TokenKey {
    /// Reads `file`. An error names the reference and the path's length,
    /// never the path or the bytes: key text pasted where the path belongs
    /// would otherwise be printed.
    pub fn load(file: &TokenKeyFile) -> Result<TokenKey, String> {
        let bytes = read_key("--pii-token-key", file)?;
        if !(MIN_KEY_BYTES..=MAX_KEY_BYTES).contains(&bytes.len()) {
            return Err(format!(
                "--pii-token-key {}: the key file holds {} bytes, not {MIN_KEY_BYTES} to {MAX_KEY_BYTES}",
                file.key,
                bytes.len()
            ));
        }
        Ok(TokenKey { key: file.key.clone(), bytes: KeyBytes(bytes) })
    }

    /// The key material, for a redaction policy's keyed-token strategy
    /// (#1882); never logged or printed.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes.0
    }

    /// The token of `value` found as `kind`: HMAC-SHA-256 under this key
    /// over the domain, then the kind code and the value, each prefixed
    /// with its length as 4 big-endian bytes (#1930). Equal values of one
    /// kind give equal tokens; one value under two kinds gives two. The
    /// key's reference is beside it.
    pub fn token(&self, kind: &str, value: &str) -> PiiValueToken {
        let mut msg = Vec::with_capacity(TOKEN_DOMAIN.len() + 8 + kind.len() + value.len());
        msg.extend_from_slice(TOKEN_DOMAIN);
        for field in [kind.as_bytes(), value.as_bytes()] {
            msg.extend_from_slice(&(field.len() as u32).to_be_bytes());
            msg.extend_from_slice(field);
        }
        PiiValueToken {
            token: turbo_sha256::hmac_sha256(&self.bytes.0, &msg).0.to_vec(),
            key_id: self.key.id.clone(),
            key_version: self.key.version,
        }
    }
}

/// A key of the AES-256-GCM and FF1 redaction strategies
/// (`--pii-encryption-key`, #1885). `Debug` names its reference only.
#[derive(Debug)]
pub struct EncryptionKey {
    pub key: KeyRef,
    bytes: KeyBytes,
}

impl EncryptionKey {
    /// Reads `file`, which must hold exactly [`ENCRYPTION_KEY_BYTES`]. An
    /// error names the reference and the path's length, never the path or
    /// the bytes, as [`TokenKey::load`].
    pub fn load(file: &TokenKeyFile) -> Result<EncryptionKey, String> {
        let bytes = read_key("--pii-encryption-key", file)?;
        if bytes.len() != ENCRYPTION_KEY_BYTES {
            return Err(format!(
                "--pii-encryption-key {}: the key file holds {} bytes, not {ENCRYPTION_KEY_BYTES}",
                file.key,
                bytes.len()
            ));
        }
        Ok(EncryptionKey { key: file.key.clone(), bytes: KeyBytes(bytes) })
    }

    /// The key material, for a redaction policy's encryption strategy;
    /// never logged or printed.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes.0
    }
}

/// Refuses a reference `flag` lists twice.
pub fn distinct(flag: &str, keys: &[TokenKeyFile]) -> Result<(), String> {
    for (i, k) in keys.iter().enumerate() {
        if keys[..i].iter().any(|o| o.key == k.key) {
            return Err(format!("{flag} {} given twice", k.key));
        }
    }
    Ok(())
}

/// The token key the configured keys name: the one key, or the one
/// `--pii-token-key-active` names among several.
pub fn active<'a>(keys: &'a [TokenKeyFile], active: Option<&KeyRef>) -> Result<Option<&'a TokenKeyFile>, String> {
    distinct("--pii-token-key", keys)?;
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
