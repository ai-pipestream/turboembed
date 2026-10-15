//! Wallet addresses, `CryptoPiiExtractor`: Bitcoin mainnet addresses
//! (base58check P2PKH and P2SH of 26 to 35 characters, bech32 or bech32m
//! segwit of at most 90 in one case) and Ethereum addresses (`0x` and 40
//! hexadecimal digits, not all zero; in mixed case only when the EIP-55
//! checksum holds).

use turbo_analysis_layers::{PiiConfidence, PiiKind, PiiKinds};

use crate::checksum::{base58, bech32, keccak};
use crate::hits::Hits;
use crate::text::{on_end, on_word_start};

pub(crate) fn scan(text: &str, kinds: PiiKinds, hits: &mut Hits) {
    if kinds.contains(PiiKind::BtcAddress) {
        scan_bitcoin(text, hits);
    }
    if kinds.contains(PiiKind::EthAddress) {
        scan_ethereum(text, hits);
    }
}

fn mixed_case(s: &[u8]) -> bool {
    s.iter().any(u8::is_ascii_uppercase) && s.iter().any(u8::is_ascii_lowercase)
}

fn scan_bitcoin(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let found = match b[i] {
            b'1' | b'3' if on_word_start(text, i) => legacy_end(text, i).map(|end| (end, false)),
            b'b' | b'B' if on_word_start(text, i) => segwit_end(text, i).map(|end| (end, true)),
            _ => None,
        };
        match found {
            Some((end, lower)) => {
                hits.add(i, end, PiiKind::BtcAddress, PiiConfidence::Checksum, |s| {
                    if lower {
                        s.extend(b[i..end].iter().map(|c| c.to_ascii_lowercase() as char));
                    } else {
                        s.push_str(&text[i..end]);
                    }
                });
                i = end;
            }
            None => i += 1,
        }
    }
}

/// A P2PKH or P2SH address at `start`: its end.
fn legacy_end(text: &str, start: usize) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = start;
    while p < b.len() && base58::is_char(b[p]) {
        p += 1;
    }
    if !(26..=35).contains(&(p - start)) || !on_end(text, p) {
        return None;
    }
    matches!(base58::checked_version(&b[start..p]), Some(0x00 | 0x05)).then_some(p)
}

/// A `bc1` segwit address at `start`: its end.
fn segwit_end(text: &str, start: usize) -> Option<usize> {
    let b = text.as_bytes();
    let data = start + 3;
    if data > b.len() || !b[start + 1].eq_ignore_ascii_case(&b'c') || b[data - 1] != b'1' {
        return None;
    }
    let mut p = data;
    while p < b.len() && bech32::is_data_char(b[p]) {
        p += 1;
    }
    if p - start > 90 || !on_end(text, p) || mixed_case(&b[start..p]) {
        return None;
    }
    bech32::checked_witness_version(&b[data..p], b"bc").map(|_| p)
}

fn scan_ethereum(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut lower = [0u8; 40];
    let mut i = 0;
    while i + 42 <= b.len() {
        let digits = &b[i + 2..i + 42];
        let candidate = b[i] == b'0'
            && b[i + 1].eq_ignore_ascii_case(&b'x')
            && on_word_start(text, i)
            && digits.iter().all(u8::is_ascii_hexdigit)
            && !digits.iter().all(|&c| c == b'0')
            && on_end(text, i + 42);
        if !candidate {
            i += 1;
            continue;
        }
        for (l, c) in lower.iter_mut().zip(digits) {
            *l = c.to_ascii_lowercase();
        }
        let checksummed = eip55(&lower);
        let mixed = mixed_case(digits);
        if mixed && digits != checksummed {
            i += 1;
            continue;
        }
        let confidence = if mixed { PiiConfidence::Checksum } else { PiiConfidence::Format };
        hits.add(i, i + 42, PiiKind::EthAddress, confidence, |s| {
            s.push_str("0x");
            s.extend(checksummed.iter().map(|&c| c as char));
        });
        i += 42;
    }
}

/// EIP-55: each letter upper case where the nibble of the Keccak-256 of
/// the lower case address at its place is 8 or more.
fn eip55(lower: &[u8; 40]) -> [u8; 40] {
    let hash = keccak::digest(lower);
    let mut out = *lower;
    for (i, c) in out.iter_mut().enumerate() {
        let nibble = if i % 2 == 0 { hash[i / 2] >> 4 } else { hash[i / 2] & 0xf };
        if nibble >= 8 {
            *c = c.to_ascii_uppercase();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eip_55_vectors() {
        for address in [
            "5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
            "fB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
            "dbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
            "D1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
        ] {
            let lower: [u8; 40] = address.to_ascii_lowercase().as_bytes().try_into().unwrap();
            assert_eq!(&eip55(&lower), address.as_bytes(), "{address}");
        }
    }
}
