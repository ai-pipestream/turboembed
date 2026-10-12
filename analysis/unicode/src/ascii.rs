//! The ASCII fast paths shared by the string functions.

/// The number of bytes at the start of `bytes` that are ASCII. Reads eight
/// bytes at a time, since most text has long ASCII stretches.
#[inline]
pub(crate) fn ascii_prefix_len(bytes: &[u8]) -> usize {
    const HIGH_BITS: u64 = 0x8080_8080_8080_8080;
    let mut chunks = bytes.chunks_exact(8);
    let mut len = 0;
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("chunks of eight bytes"));
        if word & HIGH_BITS != 0 {
            break;
        }
        len += 8;
    }
    len + bytes[len..].iter().position(|b| !b.is_ascii()).unwrap_or(bytes.len() - len)
}

/// Appends the run of ASCII bytes of `text` that starts at byte `start`,
/// lowercased, to `out`, and returns the byte offset where the run ends
/// (`start` itself when `text[start..]` begins with a non-ASCII character).
#[inline]
pub(crate) fn push_lowercase_run(text: &str, start: usize, out: &mut String) -> usize {
    let end = start + ascii_prefix_len(&text.as_bytes()[start..]);
    if end > start {
        let at = out.len();
        out.push_str(&text[start..end]);
        out[at..].make_ascii_lowercase();
    }
    end
}

#[cfg(test)]
mod tests {
    use super::ascii_prefix_len;

    #[test]
    fn prefix_length_at_every_position_of_a_word() {
        for len in 0..20 {
            for at in 0..=len {
                let mut bytes = vec![b'a'; len];
                if at < len {
                    bytes[at] = 0xC3;
                }
                assert_eq!(ascii_prefix_len(&bytes), at, "length {len}, non-ASCII at {at}");
            }
        }
    }
}
