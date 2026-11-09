//! AES-256-GCM tokens ([`AesGcmEncryptor`]): the kind in capitals, `:`,
//! and the unpadded base64url of a 12-byte nonce, the ciphertext of the
//! mention's surface and the 16-byte tag, such as `EMAIL:3q2-7w...`.
//!
//! The associated data binds the kind as the token writes it and a context
//! the caller chooses (a document id or a field name, say), so a token
//! moved to another kind or context fails authentication. Each token draws
//! a fresh nonce, so one value gives a different token every time: tokens
//! are not linkable, unlike [`crate::rewrite::HmacTokenizer`]'s labels.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use aes_gcm::aead::{AeadInOut, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiStrategyKind};

use super::{CryptError, base64url, check_key, from_base64url, random_seed};
use crate::checksum::sha256;
use crate::rewrite::replace::EditOrigin;
use crate::rewrite::{PiiRewrite, RewriteError, kind_name};

const NONCE_BYTES: usize = 12;
const TAG_BYTES: usize = 16;
const AAD_LABEL: &[u8] = b"turbo-pii:aes-256-gcm";

/// Replaces each mention with an AES-256-GCM token that a holder of the
/// key and the context decrypts back to the mention's surface.
pub struct AesGcmEncryptor {
    cipher: Aes256Gcm,
    context: Vec<u8>,
    refuse_overlaps: bool,
    // Nonces are HMAC-SHA-256 of a counter under a random seed of this
    // encryptor, cut to 96 bits: distinct within it and across encryptors
    // under one key with the probability of random 96-bit nonces.
    seed: [u8; 32],
    counter: AtomicU64,
}

impl fmt::Debug for AesGcmEncryptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The key and the nonce seed stay out of logs.
        f.debug_struct("AesGcmEncryptor")
            .field("context_bytes", &self.context.len())
            .field("refuse_overlaps", &self.refuse_overlaps)
            .finish_non_exhaustive()
    }
}

impl Clone for AesGcmEncryptor {
    /// The same key and context with a nonce seed of its own.
    fn clone(&self) -> AesGcmEncryptor {
        AesGcmEncryptor {
            cipher: self.cipher.clone(),
            context: self.context.clone(),
            refuse_overlaps: self.refuse_overlaps,
            seed: random_seed(),
            counter: AtomicU64::new(0),
        }
    }
}

impl AesGcmEncryptor {
    /// An encryptor under the 32-byte AES-256 `key`, with an empty context.
    pub fn new(key: &[u8]) -> Result<AesGcmEncryptor, CryptError> {
        check_key(key)?;
        let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| CryptError::KeyLength(key.len()))?;
        Ok(AesGcmEncryptor {
            cipher,
            context: Vec::new(),
            refuse_overlaps: false,
            seed: random_seed(),
            counter: AtomicU64::new(0),
        })
    }

    /// This encryptor binding `context` into every token, which decryption
    /// must then be given.
    pub fn with_context(mut self, context: &[u8]) -> AesGcmEncryptor {
        self.context = context.to_vec();
        self
    }

    /// This encryptor, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> AesGcmEncryptor {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    fn aad(&self, tag: &str) -> Vec<u8> {
        let mut aad = Vec::with_capacity(AAD_LABEL.len() + tag.len() + self.context.len() + 2);
        aad.extend_from_slice(AAD_LABEL);
        aad.push(0);
        aad.extend_from_slice(tag.as_bytes());
        aad.push(0);
        aad.extend_from_slice(&self.context);
        aad
    }

    fn next_nonce(&self) -> [u8; NONCE_BYTES] {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mac = sha256::hmac(&self.seed, &n.to_le_bytes());
        let mut nonce = [0u8; NONCE_BYTES];
        nonce.copy_from_slice(&mac[..NONCE_BYTES]);
        nonce
    }

    /// Appends the token of `value`, of the kind named `kind`, to `out`.
    pub fn write_token(&self, kind: &str, value: &str, out: &mut String) -> Result<(), CryptError> {
        if value.is_empty() {
            return Err(CryptError::EmptyValue);
        }
        let start = out.len();
        out.extend(kind.chars().map(|c| c.to_ascii_uppercase()));
        let aad = self.aad(&out[start..]);
        out.push(':');
        let nonce = self.next_nonce();
        let mut sealed = Vec::with_capacity(NONCE_BYTES + value.len() + TAG_BYTES);
        sealed.extend_from_slice(&nonce);
        sealed.extend_from_slice(value.as_bytes());
        let tag = self
            .cipher
            .encrypt_inout_detached(&Nonce::from(nonce), &aad, (&mut sealed[NONCE_BYTES..]).into())
            .map_err(|_| CryptError::TooLong)?;
        sealed.extend_from_slice(&tag);
        base64url(&sealed, out);
        Ok(())
    }

    /// The token of `value` of the kind named `kind`.
    pub fn token(&self, kind: &str, value: &str) -> Result<String, CryptError> {
        let mut out = String::new();
        self.write_token(kind, value, &mut out)?;
        Ok(out)
    }

    /// The kind as the token writes it (`EMAIL`) and the value of `token`,
    /// made by an encryptor with this key and context.
    pub fn decrypt(&self, token: &str) -> Result<(String, String), CryptError> {
        let (tag, body) = token.rsplit_once(':').ok_or(CryptError::Malformed)?;
        let mut sealed = from_base64url(body).ok_or(CryptError::Malformed)?;
        if sealed.len() <= NONCE_BYTES + TAG_BYTES {
            return Err(CryptError::Malformed);
        }
        let mac_at = sealed.len() - TAG_BYTES;
        let (head, mac) = sealed.split_at_mut(mac_at);
        let (nonce, ciphertext) = head.split_at_mut(NONCE_BYTES);
        let nonce = Nonce::try_from(&*nonce).map_err(|_| CryptError::Malformed)?;
        let mac = aes_gcm::Tag::try_from(&*mac).map_err(|_| CryptError::Malformed)?;
        self.cipher
            .decrypt_inout_detached(&nonce, &self.aad(tag), ciphertext.into(), &mac)
            .map_err(|_| CryptError::Authentication)?;
        let value = sealed[NONCE_BYTES..mac_at].to_vec();
        Ok((tag.to_string(), String::from_utf8(value).map_err(|_| CryptError::NotUtf8)?))
    }

    /// Replaces every mention of `pii`, a layer over `src`, with the token
    /// of its surface, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let text = src.text();
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |c, out| {
                let m = c.merged;
                let surface = &text[m.span.start as usize..m.span.end as usize];
                self.write_token(kind_name(&pii.custom_kinds, m.kind), surface, out)
                    .map_err(|e| RewriteError::Encrypt(m.span, e.to_string()))?;
                Ok(EditOrigin::only(PiiStrategyKind::Aes256Gcm))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use aes_gcm::aead::{AeadInOut, KeyInit};
    use aes_gcm::{Aes256Gcm, Nonce};

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// NIST CAVP `gcmEncryptExtIV256.rsp` vectors with a 96-bit IV, a
    /// 128-bit tag and both plaintext and associated data, through the
    /// same calls the encryptor makes.
    #[test]
    fn the_cipher_meets_nist_cavp_vectors() {
        for (key, iv, pt, aad, ct, tag) in [
            (
                "92e11dcdaa866f5ce790fd24501f92509aacf4cb8b1339d50c9c1240935dd08b",
                "ac93a1a6145299bde902f21a",
                "2d71bcfa914e4ac045b2aa60955fad24",
                "1e0889016f67601c8ebea4943bc23ad6",
                "8995ae2e6df3dbf96fac7b7137bae67f",
                "eca5aa77d51d4a0a14d9c51e1da474ab",
            ),
            (
                "7da3bccaffb3464178ca7c722379836db50ce0bfb47640b9572163865332e486",
                "c04fd2e701c3dc62b68738b3",
                "fd671cab1ee21f0df6bb610bf94f0e69",
                "fec0311013202e4ffdc4204926ae0ddf",
                "6be61b17b7f7d494a7cdf270562f37ba",
                "5e702a38323fe1160b780d17adad3e96",
            ),
        ] {
            let cipher = Aes256Gcm::new_from_slice(&hex(key)).unwrap();
            let nonce = Nonce::try_from(&hex(iv)[..]).unwrap();
            let mut buf = hex(pt);
            let t = cipher.encrypt_inout_detached(&nonce, &hex(aad), (&mut buf[..]).into()).unwrap();
            assert_eq!((buf.clone(), t.to_vec()), (hex(ct), hex(tag)));
            let t = aes_gcm::Tag::try_from(&hex(tag)[..]).unwrap();
            cipher.decrypt_inout_detached(&nonce, &hex(aad), (&mut buf[..]).into(), &t).unwrap();
            assert_eq!(buf, hex(pt));
        }
    }
}
