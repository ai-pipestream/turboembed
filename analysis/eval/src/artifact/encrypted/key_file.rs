//! File-backed caller key provider for CLI and local encrypted artifact stores.
//!
//! Each non-empty, non-comment (`#`) line in a key file is either:
//! - `<32-hex-key-id>:<64-hex-key-material>` (a 16-byte [`KeyId`] and 32-byte
//!   [`KeyMaterial`]); the first such line is the active key for new writes,
//!   and every listed key remains resolvable by its [`KeyId`] across rotation.
//! - `revoked:<32-hex-key-id>` (a 16-byte [`KeyId`] whose resolution fails
//!   closed with [`Error::RevokedKey`]).
//!
//! Diagnostic messages name 1-based line numbers only and never include key
//! material.

use std::fmt;
use std::fs;
use std::path::Path;

use super::{Error, KeyId, KeyMaterial, KeyProvider, PrincipalId};

enum Entry {
    Active([u8; 32]),
    Revoked,
}

/// Caller-supplied keys loaded from a key file or parsed from text.
pub struct KeyFile {
    active: Option<KeyId>,
    entries: Vec<(KeyId, Entry)>,
}

impl fmt::Debug for KeyFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyFile")
            .field("entries", &self.entries.len())
            .field("has_active", &self.active.is_some())
            .finish()
    }
}

impl KeyFile {
    /// Reads and parses a key file from `path`. Errors name the path and
    /// line number, never any key material.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Parses key definitions from `text`. Errors name the 1-based line
    /// number and never echo the line contents.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut active = None;
        let mut entries: Vec<(KeyId, Entry)> = Vec::new();
        for (idx, raw) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(id_hex) = line.strip_prefix("revoked:") {
                let id = KeyId(
                    parse_lower_hex::<16>(id_hex)
                        .ok_or_else(|| format!("line {line_no}: revoked entry needs 32 lower-case hex digits"))?,
                );
                if entries.iter().any(|(k, _)| *k == id) {
                    return Err(format!("line {line_no}: duplicate key id"));
                }
                entries.push((id, Entry::Revoked));
                continue;
            }
            let Some((id_hex, secret_hex)) = line.split_once(':') else {
                return Err(format!(
                    "line {line_no}: expected <32-hex-key-id>:<64-hex-key> or revoked:<32-hex-key-id>"
                ));
            };
            let (Some(id_bytes), Some(secret)) = (parse_lower_hex::<16>(id_hex), parse_lower_hex::<32>(secret_hex))
            else {
                return Err(format!("line {line_no}: expected <32-hex-key-id>:<64-hex-key> in lower-case hex"));
            };
            let id = KeyId(id_bytes);
            if entries.iter().any(|(k, _)| *k == id) {
                return Err(format!("line {line_no}: duplicate key id"));
            }
            if active.is_none() {
                active = Some(id);
            }
            entries.push((id, Entry::Active(secret)));
        }
        if entries.is_empty() {
            return Err("key file contains no keys".into());
        }
        Ok(Self { active, entries })
    }
}

impl KeyProvider for KeyFile {
    fn active(&self) -> Result<(KeyId, KeyMaterial), Error> {
        let Some(id) = self.active else {
            return Err(Error::RevokedKey);
        };
        Ok((id, self.resolve(id)?))
    }

    fn resolve(&self, id: KeyId) -> Result<KeyMaterial, Error> {
        match self.entries.iter().find(|(k, _)| *k == id) {
            Some((_, Entry::Active(bytes))) => Ok(KeyMaterial::new(*bytes)),
            Some((_, Entry::Revoked)) => Err(Error::RevokedKey),
            None => Err(Error::UnavailableKey),
        }
    }
}

impl PrincipalId {
    /// Parses a 32-character lower-case hexadecimal principal identifier.
    pub fn parse_hex(s: &str) -> Result<Self, String> {
        parse_lower_hex::<16>(s).map(Self).ok_or_else(|| "principal id needs 32 lower-case hex digits".into())
    }
}

pub(crate) fn parse_lower_hex<const N: usize>(s: &str) -> Option<[u8; N]> {
    let bytes = s.as_bytes();
    if bytes.len() != N * 2 || !bytes.iter().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b)) {
        return None;
    }
    let mut out = [0u8; N];
    for (i, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let digit = |b: u8| if b.is_ascii_digit() { b - b'0' } else { b - b'a' + 10 };
        out[i] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID_1: &str = "01010101010101010101010101010101";
    const ID_2: &str = "02020202020202020202020202020202";
    const ID_3: &str = "03030303030303030303030303030303";
    const KEY_1: &str = "4747474747474747474747474747474747474747474747474747474747474747";
    const KEY_2: &str = "4848484848484848484848484848484848484848484848484848484848484848";

    #[test]
    fn key_file_resolves_active_rotated_revoked_and_missing_keys() {
        let text = format!("# comment\n\n{ID_2}:{KEY_2}\n{ID_1}:{KEY_1}\nrevoked:{ID_3}\n");
        let kf = KeyFile::parse(&text).unwrap();
        let (active_id, active_material) = kf.active().unwrap();
        assert_eq!(active_id, KeyId([2; 16]));
        assert_eq!(active_material.0, [0x48; 32]);
        assert_eq!(kf.resolve(KeyId([1; 16])).unwrap().0, [0x47; 32]);
        assert!(matches!(kf.resolve(KeyId([3; 16])), Err(Error::RevokedKey)));
        assert!(matches!(kf.resolve(KeyId([9; 16])), Err(Error::UnavailableKey)));
        let debug = format!("{kf:?}");
        assert!(!debug.contains("4747") && !debug.contains("4848"));
    }

    #[test]
    fn key_file_rejects_malformed_or_duplicate_entries_without_echoing_secrets() {
        assert!(KeyFile::parse("   \n# only comment\n").is_err());
        let bad_secret = "deadbeefcafebabe";
        let err = KeyFile::parse(&format!("{ID_1}:{bad_secret}")).unwrap_err();
        assert!(err.contains("line 1") && !err.contains(bad_secret), "{err}");
        let dup = KeyFile::parse(&format!("{ID_1}:{KEY_1}\n{ID_1}:{KEY_2}\n")).unwrap_err();
        assert!(dup.contains("duplicate key id") && !dup.contains(KEY_1), "{dup}");
        let revoked_only = KeyFile::parse(&format!("revoked:{ID_1}\n")).unwrap();
        assert!(matches!(revoked_only.active(), Err(Error::RevokedKey)));
    }
}
