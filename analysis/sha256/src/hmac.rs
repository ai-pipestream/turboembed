//! HMAC-SHA-256 (RFC 2104).

use crate::{Digest, Sha256, sha256};

/// HMAC-SHA-256 of `message` under `key`. A key longer than the 64-byte
/// block is hashed first, as RFC 2104 says.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> Digest {
    let mut block = [0u8; 64];
    if key.len() > 64 {
        block[..32].copy_from_slice(&sha256(key).0);
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(&block.map(|b| b ^ 0x36)).update(message);
    let inner = inner.finish();
    let mut outer = Sha256::new();
    outer.update(&block.map(|b| b ^ 0x5c)).update(&inner.0);
    outer.finish()
}
