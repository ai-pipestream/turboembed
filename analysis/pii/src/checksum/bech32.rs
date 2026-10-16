//! Bech32 and bech32m (BIP 173, BIP 350) for segwit addresses, as
//! OpenNLP's `Bech32` checks them.

const CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

const VALUES: [i8; 128] = {
    let mut v = [-1i8; 128];
    let mut i = 0;
    while i < CHARSET.len() {
        v[CHARSET[i] as usize] = i as i8;
        i += 1;
    }
    v
};

const GENERATOR: [u32; 5] = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
const BECH32: u32 = 1;
const BECH32M: u32 = 0x2bc830a3;

/// The value of a data character in either case, or -1.
#[inline]
fn value(c: u8) -> i32 {
    if c < 128 { i32::from(VALUES[c.to_ascii_lowercase() as usize]) } else { -1 }
}

/// Whether `c` is a data character in either case.
#[inline]
pub fn is_data_char(c: u8) -> bool {
    value(c) >= 0
}

/// The witness version of a segwit address with human-readable part
/// `hrp` whose data part (after the `1`) is `data`: version 0 with a
/// 20 or 32 byte program and the bech32 checksum, or versions 1 to 16
/// with a 2 to 40 byte program and the bech32m checksum. `None` when any
/// check fails.
pub fn checked_witness_version(data: &[u8], hrp: &[u8]) -> Option<u8> {
    if data.len() <= 6 || data.iter().any(|&c| !is_data_char(c)) {
        return None;
    }
    let version = value(data[0]) as u32;
    if version > 16 {
        return None;
    }
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    let mut program = 0usize;
    for &c in &data[1..data.len() - 6] {
        accumulator = accumulator << 5 | value(c) as u32;
        bits += 5;
        while bits >= 8 {
            bits -= 8;
            program += 1;
            accumulator &= (1 << bits) - 1;
        }
    }
    if !(2..=40).contains(&program) || bits > 4 || accumulator != 0 {
        return None;
    }
    if version == 0 && program != 20 && program != 32 {
        return None;
    }
    let mut checksum = 1u32;
    let hrp_values = hrp.iter().map(|&c| u32::from(c) >> 5).chain([0]).chain(hrp.iter().map(|&c| u32::from(c) & 31));
    for v in hrp_values.chain(data.iter().map(|&c| value(c) as u32)) {
        checksum = polymod_step(checksum, v);
    }
    let residue = if version == 0 { BECH32 } else { BECH32M };
    (checksum == residue).then_some(version as u8)
}

fn polymod_step(checksum: u32, value: u32) -> u32 {
    let top = checksum >> 25;
    let mut c = (checksum & 0x1ffffff) << 5 ^ value;
    for (i, g) in GENERATOR.iter().enumerate() {
        if top >> i & 1 != 0 {
            c ^= g;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(address: &str) -> Option<u8> {
        let (hrp, data) = address.rsplit_once('1').unwrap();
        checked_witness_version(data.as_bytes(), hrp.to_ascii_lowercase().as_bytes())
    }

    #[test]
    fn bip_173_and_350_vectors() {
        assert_eq!(version("BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4"), Some(0));
        assert_eq!(version("bc1qrp33g0q5c5txsp9arysrx4k6zdkfs4nce4xj0gdcccefvpysxf3qccfmv3"), Some(0));
        assert_eq!(version("bc1pw508d6qejxtdg4y5r3zarvary0c5xw7kw508d6qejxtdg4y5r3zarvary0c5xw7kt5nd6y"), Some(1));
        assert_eq!(version("bc1sw50qgdz25j"), Some(16));
        assert_eq!(version("bc1zw508d6qejxtdg4y5r3zarvaryvaxxpcs"), Some(2));
        // A version 0 program with a bech32m checksum, and the reverse.
        assert_eq!(version("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kemeawh"), None);
        assert_eq!(version("bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqh2y7hd"), None);
        // A wrong checksum character.
        assert_eq!(version("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5"), None);
        // Program lengths outside 2 to 40, and non-zero padding.
        assert_eq!(version("bc1gmk9yu"), None);
        assert_eq!(version("bc1zw508d6qejxtdg4y5r3zarvaryvqyzf3du"), None);
        assert!(is_data_char(b'Q') && !is_data_char(b'b') && !is_data_char(b'1') && !is_data_char(0xC3));
    }
}
