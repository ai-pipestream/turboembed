//! Output receipt framing: native-width token length, UTF-8 token bytes,
//! then one 0xff byte per input line.
pub(crate) fn feed_line(hash: u64, tokens: &[&str]) -> u64 {
    let mut hash = turbo_analysis::Fnv1a::with_seed(hash);
    for token in tokens {
        hash.bytes(&token.len().to_le_bytes()).bytes(token.as_bytes());
    }
    hash.u8(0xff).finish()
}
