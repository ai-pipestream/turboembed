//! Base58check, the encoding of legacy Bitcoin addresses, as OpenNLP's
//! `Base58Check` decodes it.

use super::sha256;

const ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// The value of each ASCII byte in the alphabet, or -1.
const VALUES: [i8; 128] = {
    let mut v = [-1i8; 128];
    let mut i = 0;
    while i < ALPHABET.len() {
        v[ALPHABET[i] as usize] = i as i8;
        i += 1;
    }
    v
};

/// The decoded length of an address: a version byte, a 20-byte hash and
/// a 4-byte checksum.
const PAYLOAD: usize = 25;

/// Whether `c` is a base58 character.
#[inline]
pub fn is_char(c: u8) -> bool {
    c < 128 && VALUES[c as usize] >= 0
}

/// The version byte of a base58check value that decodes to 25 bytes whose
/// last four are the start of the double SHA-256 of the first 21, or
/// `None`. `encoded` holds only base58 characters.
pub fn checked_version(encoded: &[u8]) -> Option<u8> {
    // One output byte per character is always enough.
    let mut reversed = [0u8; 64];
    if encoded.len() >= reversed.len() {
        return None;
    }
    let mut length = 0;
    for &c in encoded {
        let mut carry = VALUES[c as usize] as u32;
        for b in &mut reversed[..length] {
            carry += u32::from(*b) * 58;
            *b = carry as u8;
            carry >>= 8;
        }
        while carry > 0 {
            reversed[length] = carry as u8;
            length += 1;
            carry >>= 8;
        }
    }
    // Each leading '1' stands for a zero byte.
    for _ in encoded.iter().take_while(|&&c| c == ALPHABET[0]) {
        reversed[length] = 0;
        length += 1;
    }
    if length != PAYLOAD {
        return None;
    }
    let mut payload = [0u8; PAYLOAD];
    for (p, r) in payload.iter_mut().zip(reversed[..PAYLOAD].iter().rev()) {
        *p = *r;
    }
    let check = sha256::digest(&sha256::digest(&payload[..PAYLOAD - 4]));
    (check[..4] == payload[PAYLOAD - 4..]).then_some(payload[0])
}

/// Encodes `payload` (a version byte and a hash) with its checksum, the
/// inverse of [`checked_version`]; for making test values.
pub fn encode_checked(payload: &[u8]) -> String {
    let check = sha256::digest(&sha256::digest(payload));
    let mut bytes = payload.to_vec();
    bytes.extend_from_slice(&check[..4]);
    let zeros = bytes.iter().take_while(|&&b| b == 0).count();
    let mut digits: Vec<u8> = Vec::new();
    for &b in &bytes {
        let mut carry = u32::from(b);
        for d in &mut digits {
            carry += u32::from(*d) << 8;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }
    let mut out = String::with_capacity(zeros + digits.len());
    out.extend(std::iter::repeat_n('1', zeros));
    out.extend(digits.iter().rev().map(|&d| ALPHABET[d as usize] as char));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_versions() {
        let mut payload = [0u8; 21];
        for (i, b) in payload.iter_mut().enumerate().skip(1) {
            *b = (i * 37 % 251) as u8;
        }
        let p2pkh = encode_checked(&payload);
        assert!(p2pkh.starts_with('1'));
        assert_eq!(checked_version(p2pkh.as_bytes()), Some(0));
        payload[0] = 5;
        let p2sh = encode_checked(&payload);
        assert!(p2sh.starts_with('3'));
        assert_eq!(checked_version(p2sh.as_bytes()), Some(5));
        // One changed character breaks the checksum.
        let mut broken = p2sh.into_bytes();
        let last = broken.len() - 1;
        broken[last] = if broken[last] == b'z' { b'y' } else { b'z' };
        assert_eq!(checked_version(&broken), None);
    }

    #[test]
    fn lengths_other_than_25_bytes_fail() {
        assert_eq!(checked_version(b"1111"), None);
        assert_eq!(checked_version(b""), None);
        assert_eq!(checked_version(&[b'z'; 70]), None);
        assert!(is_char(b'z') && !is_char(b'0') && !is_char(b'O') && !is_char(b'I') && !is_char(b'l'));
        assert!(!is_char(0xC3));
    }
}
