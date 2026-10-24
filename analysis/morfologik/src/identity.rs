use turbo_maxent::sha256;

/// Domain-separated digest of ordered inputs. Fixed-width component digests
/// prevent ambiguity between boundaries without copying a whole dictionary.
pub(crate) fn digest(parts: &[&[u8]]) -> String {
    let mut bytes = b"turbo-morfologik-identity-v1".to_vec();
    for part in parts {
        bytes.extend_from_slice(&sha256(part).0);
    }
    sha256(&bytes).to_hex()
}
