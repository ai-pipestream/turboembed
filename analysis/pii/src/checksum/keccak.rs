//! Keccak-256, the hash Ethereum uses (the original Keccak padding, not
//! the SHA-3 one), for the EIP-55 checksum.

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// The rotation of each step of the rho and pi walk.
const ROTATIONS: [u32; 24] =
    [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];

/// The state word each step of the walk moves to.
const WALK: [usize; 24] = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];

/// The bytes absorbed per permutation: 1600 bits less twice the output.
const RATE: usize = 136;

/// The Keccak-256 digest of `message`.
pub fn digest(message: &[u8]) -> [u8; 32] {
    let mut state = [0u64; 25];
    let mut blocks = message.chunks_exact(RATE);
    for block in &mut blocks {
        absorb(&mut state, block);
        permute(&mut state);
    }
    let rest = blocks.remainder();
    let mut last = [0u8; RATE];
    last[..rest.len()].copy_from_slice(rest);
    last[rest.len()] = 0x01;
    last[RATE - 1] |= 0x80;
    absorb(&mut state, &last);
    permute(&mut state);
    let mut out = [0u8; 32];
    for (o, w) in out.chunks_exact_mut(8).zip(state) {
        o.copy_from_slice(&w.to_le_bytes());
    }
    out
}

fn absorb(state: &mut [u64; 25], block: &[u8]) {
    for (s, c) in state.iter_mut().zip(block.chunks_exact(8)) {
        *s ^= u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
    }
}

fn permute(a: &mut [u64; 25]) {
    for rc in ROUND_CONSTANTS {
        // Theta.
        let mut parity = [0u64; 5];
        for (x, p) in parity.iter_mut().enumerate() {
            *p = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = parity[(x + 4) % 5] ^ parity[(x + 1) % 5].rotate_left(1);
            for y in (0..25).step_by(5) {
                a[x + y] ^= d;
            }
        }
        // Rho and pi.
        let mut current = a[1];
        for (&to, &r) in WALK.iter().zip(&ROTATIONS) {
            let moved = a[to];
            a[to] = current.rotate_left(r);
            current = moved;
        }
        // Chi.
        for y in (0..25).step_by(5) {
            let row = [a[y], a[y + 1], a[y + 2], a[y + 3], a[y + 4]];
            for x in 0..5 {
                a[y + x] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
            }
        }
        // Iota.
        a[0] ^= rc;
    }
}

#[cfg(test)]
mod tests {
    use super::digest;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn known_digests() {
        assert_eq!(hex(&digest(b"")), "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
        assert_eq!(hex(&digest(b"abc")), "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
        // 136 and 137 bytes: the padding in a block of its own, and split.
        assert_eq!(hex(&digest(&[b'a'; 136])), "a6c4d403279fe3e0af03729caada8374b5ca54d8065329a3ebcaeb4b60aa386e");
        assert_eq!(hex(&digest(&[b'a'; 135])), "34367dc248bbd832f4e3e69dfaac2f92638bd0bbd18f2912ba4ef454919cf446");
    }
}
