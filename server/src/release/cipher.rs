//! Ciphertext release (#1885, #1909): a value a service is granted
//! `ciphertext` for leaves encrypted under the active encryption key
//! (`--pii-encryption-key`, `--pii-encryption-key-active`), never in clear.
//!
//! The ciphertext is the UTF-8 text of a `turbo-pii` AES-256-GCM token
//! ([`SCHEME`]): the kind code as it is, its case kept (`builtin:3`,
//! `custom:Badge`), `:`, and the unpadded base64url of a 12-byte nonce,
//! the encrypted normalized value and the 16-byte tag. The associated
//! data binds the exact kind code and [`CONTEXT`], so a holder of the key
//! decrypts it with
//! `AesGcmEncryptor::new(key)?.with_context(CONTEXT).decrypt(token)`, and
//! a ciphertext moved to another kind fails authentication, also to a
//! custom kind whose name differs only by case (#2087). Each release
//! draws a fresh nonce: equal values give different ciphertexts, unlike
//! tokens. A server built without the `pii-reversible` feature loads no
//! cipher, and so releases no ciphertext.

use std::fmt;

use super::keys::{KeyRef, TokenKeyFile};
use crate::proto::result::PiiValueCiphertext;

/// `PiiValueCiphertext.scheme` of every ciphertext this server releases.
pub const SCHEME: &str = "turbo-pii-aes-256-gcm-v1";

/// The context every released ciphertext is bound to.
pub const CONTEXT: &[u8] = b"turbo-kserve.pii-release.v1";

/// The encryptor of ciphertext releases, under one registered key.
pub struct ReleaseCipher {
    key: KeyRef,
    #[cfg(feature = "pii-reversible")]
    enc: turbo_pii::rewrite::AesGcmEncryptor,
}

impl fmt::Debug for ReleaseCipher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReleaseCipher").field("key", &self.key).finish_non_exhaustive()
    }
}

impl ReleaseCipher {
    /// Reads `file`, an encryption key of exactly 32 bytes. An error names
    /// the reference and the path's length, never the path or the bytes.
    pub fn load(file: &TokenKeyFile) -> Result<ReleaseCipher, String> {
        #[cfg(feature = "pii-reversible")]
        {
            let key = super::keys::EncryptionKey::load(file)?;
            let enc = turbo_pii::rewrite::AesGcmEncryptor::new(key.bytes())
                .map_err(|e| format!("--pii-encryption-key {}: {e}", file.key))?
                .with_context(CONTEXT);
            Ok(ReleaseCipher { key: file.key.clone(), enc })
        }
        #[cfg(not(feature = "pii-reversible"))]
        {
            Err(format!("--pii-encryption-key {}: ciphertext release needs the pii-reversible feature", file.key))
        }
    }

    /// The key ciphertexts are made under.
    pub fn key(&self) -> &KeyRef {
        &self.key
    }

    /// The ciphertext of `value` found as the kind coded `kind`. An error
    /// names neither.
    pub fn seal(&self, kind: &str, value: &str) -> Result<PiiValueCiphertext, String> {
        #[cfg(feature = "pii-reversible")]
        {
            let token = self.enc.exact_token(kind, value).map_err(|e| format!("a ciphertext release failed: {e}"))?;
            Ok(PiiValueCiphertext {
                ciphertext: token.into_bytes(),
                key_id: self.key.id.clone(),
                scheme: SCHEME.to_owned(),
                key_version: self.key.version,
            })
        }
        #[cfg(not(feature = "pii-reversible"))]
        {
            let _ = (kind, value);
            Err("ciphertext release needs the pii-reversible feature".to_owned())
        }
    }
}
