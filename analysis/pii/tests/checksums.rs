//! `LuhnTest`, `Keccak256Test` and `ChecksumMutationTest`: the checks,
//! and every single character change of a valid value refused.

mod support;

use support::*;
use turbo_pii::checksum::{keccak, luhn, sha256};
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const DOUBLED: [u32; 10] = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9];

#[test]
fn luhn_candidates_of_every_card_length() {
    for length in [9usize, 13, 14, 15, 16, 17, 18, 19] {
        let mut value = Vec::with_capacity(length);
        let mut sum = 0;
        for index in 0..length - 1 {
            let digit = ((index + 3) % 10) as u32;
            value.push(b'0' + digit as u8);
            sum += if (length - index) % 2 == 0 { DOUBLED[digit as usize] } else { digit };
        }
        let check = (10 - sum % 10) % 10;
        value.push(b'0' + check as u8);
        assert!(luhn::valid(&value));
        for replacement in 0..10u32 {
            value[length - 1] = b'0' + replacement as u8;
            assert_eq!(luhn::valid(&value), replacement == check);
        }
        assert!(luhn::valid(&vec![b'0'; length]));
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn keccak_known_digests() {
    for (message, want) in [
        ("", "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
        ("abc", "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
        (
            "The quick brown fox jumps over the lazy dog",
            "4d741b6f1eb29cb2a9b9911c82f56fa8d73b04959d3d9d222895df6c0b28aa15",
        ),
    ] {
        assert_eq!(hex(&keccak::digest(message.as_bytes())), want);
    }
}

#[test]
fn keccak_around_the_block_size() {
    for (length, want) in [
        (0usize, "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
        (1, "bc36789e7a1e281436464229828f817d6612f7b477d66591ff96a9e064bcc98a"),
        (40, "da227097c39b25f51ebbb255c17b0ee624bc34f0cea142cd9a811b96d3d41f32"),
        (135, "cbdfd9dee5faad3818d6b06f95a219fd290b0e1706f6a82e5a595b9ce9faca62"),
        (136, "7ce759f1ab7f9ce437719970c26b0a66ff11fe3e38e17df89cf5d29c7d7f807e"),
        (137, "ac73d4fae68b8453f764007c1a20ce95994187861f0c3227a3a8e99a73a3b1db"),
        (271, "7c974895b2a88303ff2dc6b58f438ceb0b298cac91099ac0539cc0f477506191"),
        (272, "fdf2ec49e749960d3c8521a0219af8d03e30e2b3bf19bd16150ee0eaf133d66e"),
        (273, "4f707289a9c3ccd0c4a51f2f17339f5dd171d371c04ff7783b735b5b22682eaf"),
        (1000, "aca79e4146e30eb1c733f6d6060d72471c36ea4e01ebf45d7f4916249c2bbd82"),
    ] {
        let message: Vec<u8> = (0..length).map(|i| i as u8).collect();
        assert_eq!(hex(&keccak::digest(&message)), want, "{length}");
    }
    assert_ne!(keccak::digest(b"opennlp"), keccak::digest(b"opennlq"));
    // SHA3-256("abc") differs: Keccak keeps the original padding.
    assert_ne!(hex(&keccak::digest(b"abc")), "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");
    assert_eq!(hex(&sha256::digest(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

const BASE58: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const BECH32: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

fn replaced(value: &str, at: usize, c: u8) -> String {
    let mut b = value.as_bytes().to_vec();
    b[at] = c;
    String::from_utf8(b).unwrap()
}

fn next_digit(c: u8) -> u8 {
    if c == b'9' { b'0' } else { c + 1 }
}

#[track_caller]
fn refused(d: &PiiDetector, kind: PiiKind, text: &str) {
    assert!(!kinds(d, text).contains(&kind.name()), "{text}");
}

fn digit_mutations_refused(d: &PiiDetector, kind: PiiKind, value: &str, prefix: &str) {
    for i in 0..value.len() {
        refused(d, kind, &format!("{prefix}{}", replaced(value, i, next_digit(value.as_bytes()[i]))));
    }
}

#[test]
fn every_card_digit_change_is_refused() {
    digit_mutations_refused(&only(Extractor::Cursor, &[PiiKind::Card]), PiiKind::Card, "4111111111111111", "");
}

#[test]
fn every_iban_character_change_is_refused() {
    let d = only(Extractor::Cursor, &[PiiKind::Iban]);
    let value = "DE89370400440532013000";
    for (i, &c) in value.as_bytes().iter().enumerate() {
        let r = if c.is_ascii_digit() {
            next_digit(c)
        } else if c == b'Z' {
            b'A'
        } else {
            c + 1
        };
        refused(&d, PiiKind::Iban, &replaced(value, i, r));
    }
}

#[test]
fn every_legacy_bitcoin_character_change_is_refused() {
    let d = only(Extractor::Crypto, &[PiiKind::BtcAddress]);
    let value = "1Pjk7c5ipz7ZpbVT3VKwcFNGeqgBFpbQdw";
    assert_eq!(kinds(&d, value), ["btc-address"]);
    for (i, &c) in value.as_bytes().iter().enumerate() {
        let k = BASE58.iter().position(|&x| x == c).unwrap();
        refused(&d, PiiKind::BtcAddress, &replaced(value, i, BASE58[(k + 1) % BASE58.len()]));
    }
}

#[test]
fn every_segwit_data_character_change_is_refused() {
    let d = only(Extractor::Crypto, &[PiiKind::BtcAddress]);
    let value = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4";
    for i in 3..value.len() {
        let k = BECH32.iter().position(|&x| x == value.as_bytes()[i]).unwrap();
        refused(&d, PiiKind::BtcAddress, &replaced(value, i, BECH32[(k + 1) % BECH32.len()]));
    }
}

#[test]
fn every_ethereum_digit_change_is_refused() {
    let d = only(Extractor::Crypto, &[PiiKind::EthAddress]);
    let value = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";
    for i in 2..value.len() {
        let c = value.as_bytes()[i];
        let r = match c {
            b'0'..=b'9' => next_digit(c),
            b'f' => b'a',
            b'F' => b'A',
            _ => c + 1,
        };
        refused(&d, PiiKind::EthAddress, &replaced(value, i, r));
    }
}

#[test]
fn every_national_and_labelled_digit_change_is_refused() {
    digit_mutations_refused(&only(Extractor::EuIdentity, &[PiiKind::UkNhs]), PiiKind::UkNhs, "9434765919", "");
    digit_mutations_refused(
        &only(Extractor::EuIdentity, &[PiiKind::DeSteuerId]),
        PiiKind::DeSteuerId,
        "65929970489",
        "",
    );
    digit_mutations_refused(
        &PiiDetector::new(vec![turbo_pii::Scanner::all(Extractor::Banking)]),
        PiiKind::AbaRouting,
        "021000021",
        "",
    );
    digit_mutations_refused(&pack(Pack::Device), PiiKind::Imei, "490154203237518", "IMEI: ");
    digit_mutations_refused(&pack(Pack::CaIdentity), PiiKind::CaSin, "046454286", "SIN: ");
}
