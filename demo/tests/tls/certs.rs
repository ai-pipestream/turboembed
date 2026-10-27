//! The test CA and the localhost certificate and key, made when the tests
//! run so no private key is committed. ECDSA P-256 on ring, which the
//! demo already links; the X.509 DER is written out by hand.

use std::path::PathBuf;
use std::sync::OnceLock;

use ring::rand::SystemRandom;
use ring::signature::{ECDSA_P256_SHA256_ASN1_SIGNING, EcdsaKeyPair, KeyPair};

/// The directory holding `ca.pem`, `cert.pem` and `key.pem`, written once
/// per test binary under cargo's scratch directory for integration tests.
pub fn dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("demo-tls-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rng = SystemRandom::new();
        let (_, ca) = key(&rng);
        let (leaf_pkcs8, leaf) = key(&rng);
        let ca_name = name("turbo-kserve-demo test CA");
        let ca_cert = cert(&rng, &ca, 1, &ca_name, &ca_name, ca.public_key().as_ref(), &ca_extensions());
        let leaf_cert =
            cert(&rng, &ca, 2, &ca_name, &name("localhost"), leaf.public_key().as_ref(), &leaf_extensions());
        std::fs::write(dir.join("ca.pem"), pem("CERTIFICATE", &ca_cert)).unwrap();
        std::fs::write(dir.join("cert.pem"), pem("CERTIFICATE", &leaf_cert)).unwrap();
        std::fs::write(dir.join("key.pem"), pem("PRIVATE KEY", &leaf_pkcs8)).unwrap();
        dir
    })
}

fn key(rng: &SystemRandom) -> (Vec<u8>, EcdsaKeyPair) {
    let pkcs8 = EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, rng).unwrap();
    let pair = EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, pkcs8.as_ref(), rng).unwrap();
    (pkcs8.as_ref().to_vec(), pair)
}

// Object identifiers, DER encoded with their tag.
const EC_PUBLIC_KEY: &[u8] = &[0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
const P256: &[u8] = &[0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
const ECDSA_SHA256: &[u8] = &[0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
const COMMON_NAME: &[u8] = &[0x06, 0x03, 0x55, 0x04, 0x03];
const KEY_USAGE: &[u8] = &[0x06, 0x03, 0x55, 0x1d, 0x0f];
const SUBJECT_ALT_NAME: &[u8] = &[0x06, 0x03, 0x55, 0x1d, 0x11];
const BASIC_CONSTRAINTS: &[u8] = &[0x06, 0x03, 0x55, 0x1d, 0x13];
const EXT_KEY_USAGE: &[u8] = &[0x06, 0x03, 0x55, 0x1d, 0x25];
const SERVER_AUTH: &[u8] = &[0x06, 0x08, 0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01];

fn tlv(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let n = body.len();
    if n < 0x80 {
        out.push(n as u8);
    } else {
        let bytes: Vec<u8> = n.to_be_bytes().into_iter().skip_while(|b| *b == 0).collect();
        out.push(0x80 | bytes.len() as u8);
        out.extend(bytes);
    }
    out.extend_from_slice(body);
    out
}

fn seq(parts: &[&[u8]]) -> Vec<u8> {
    tlv(0x30, &parts.concat())
}

fn bits(body: &[u8]) -> Vec<u8> {
    tlv(0x03, &[&[0u8][..], body].concat())
}

fn name(cn: &str) -> Vec<u8> {
    seq(&[&tlv(0x31, &seq(&[COMMON_NAME, &tlv(0x0c, cn.as_bytes())]))])
}

fn extension(oid: &[u8], critical: bool, value: &[u8]) -> Vec<u8> {
    let flag: &[u8] = if critical { &[0x01, 0x01, 0xff] } else { &[] };
    seq(&[oid, flag, &tlv(0x04, value)])
}

fn ca_extensions() -> Vec<u8> {
    let ca = seq(&[&[0x01, 0x01, 0xff]]);
    // keyCertSign and cRLSign.
    let usage = tlv(0x03, &[0x01, 0x06]);
    [extension(BASIC_CONSTRAINTS, true, &ca), extension(KEY_USAGE, true, &usage)].concat()
}

fn leaf_extensions() -> Vec<u8> {
    // digitalSignature.
    let usage = tlv(0x03, &[0x07, 0x80]);
    let names = seq(&[&tlv(0x82, b"localhost"), &tlv(0x87, &[127, 0, 0, 1])]);
    [
        extension(BASIC_CONSTRAINTS, true, &seq(&[])),
        extension(KEY_USAGE, true, &usage),
        extension(EXT_KEY_USAGE, false, &seq(&[SERVER_AUTH])),
        extension(SUBJECT_ALT_NAME, false, &names),
    ]
    .concat()
}

fn cert(
    rng: &SystemRandom,
    issuer_key: &EcdsaKeyPair,
    serial: u8,
    issuer: &[u8],
    subject: &[u8],
    public_key: &[u8],
    extensions: &[u8],
) -> Vec<u8> {
    let algorithm = seq(&[ECDSA_SHA256]);
    let validity = seq(&[&tlv(0x17, b"200101000000Z"), &tlv(0x18, b"21260101000000Z")]);
    let spki = seq(&[&seq(&[EC_PUBLIC_KEY, P256]), &bits(public_key)]);
    let tbs = seq(&[
        &tlv(0xa0, &tlv(0x02, &[2])),
        &tlv(0x02, &[serial]),
        &algorithm,
        issuer,
        &validity,
        subject,
        &spki,
        &tlv(0xa3, &seq(&[extensions])),
    ]);
    let signature = issuer_key.sign(rng, &tbs).unwrap();
    seq(&[&tbs, &algorithm, &bits(signature.as_ref())])
}

fn pem(label: &str, der: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::new();
    for chunk in der.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, b)| n | (u32::from(*b) << (16 - 8 * i)));
        for i in 0..4 {
            let c = if i <= chunk.len() { ALPHABET[(n >> (18 - 6 * i)) as usize & 63] } else { b'=' };
            text.push(c as char);
        }
    }
    let mut out = format!("-----BEGIN {label}-----\n");
    for line in text.as_bytes().chunks(64) {
        out.push_str(std::str::from_utf8(line).unwrap());
        out.push('\n');
    }
    out.push_str(&format!("-----END {label}-----\n"));
    out
}
