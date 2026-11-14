//! SHA-1 (FIPS 180-4), for the one place a format outside ours fixes it:
//! git names a file by the SHA-1 of its blob, and a repository listing
//! gives that name. It is not collision resistant, so nothing of ours is
//! pinned by it; every file is still pinned by SHA-256 and SHA-512.

/// The SHA-1 of `parts` read one after another as a single message.
pub fn sha1(parts: &[&[u8]]) -> [u8; 20] {
    let mut state: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let mut block = [0u8; 64];
    let mut filled = 0;
    let mut length: u64 = 0;
    for part in parts {
        length = length.wrapping_add(part.len() as u64);
        let mut rest = *part;
        while !rest.is_empty() {
            let take = rest.len().min(64 - filled);
            block[filled..filled + take].copy_from_slice(&rest[..take]);
            filled += take;
            rest = &rest[take..];
            if filled == 64 {
                compress(&mut state, &block);
                filled = 0;
            }
        }
    }
    // The padding: one 1 bit, zeros, and the length in bits in the last
    // eight bytes of a block.
    block[filled] = 0x80;
    block[filled + 1..].fill(0);
    if filled + 1 > 56 {
        compress(&mut state, &block);
        block.fill(0);
    }
    block[56..].copy_from_slice(&length.wrapping_mul(8).to_be_bytes());
    compress(&mut state, &block);
    let mut out = [0u8; 20];
    for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *bytes = word.to_be_bytes();
    }
    out
}

/// The id git gives a file of these bytes: the SHA-1 of `blob <size>`, a
/// zero byte, and the bytes.
pub fn git_blob_id(bytes: &[u8]) -> [u8; 20] {
    let header = format!("blob {}\0", bytes.len());
    sha1(&[header.as_bytes(), bytes])
}

fn compress(state: &mut [u32; 5], block: &[u8; 64]) {
    let mut w = [0u32; 80];
    for (word, bytes) in w.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_be_bytes(*bytes);
    }
    for t in 16..80 {
        w[t] = (w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16]).rotate_left(1);
    }
    let [mut a, mut b, mut c, mut d, mut e] = *state;
    for (t, word) in w.iter().enumerate() {
        let (f, k) = match t / 20 {
            0 => ((b & c) | (!b & d), 0x5A82_7999),
            1 => (b ^ c ^ d, 0x6ED9_EBA1),
            2 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
            _ => (b ^ c ^ d, 0xCA62_C1D6),
        };
        let next = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*word);
        (e, d, c, b, a) = (d, c, b.rotate_left(30), a, next);
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d, e]) {
        *s = s.wrapping_add(v);
    }
}
