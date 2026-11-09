//! The FIPS 180-4 and RFC 4231 vectors, and updates split at every
//! offset of messages that cross block borders.

use turbo_sha256::{Sha256, Sha512, hmac_sha256, sha256, sha512, to_hex};

const ABC: &[u8] = b"abc";
const M448: &[u8] = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
const M896: &[u8] =
    b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";

#[test]
fn sha512_fips_vectors() {
    let hex = |d: &[u8]| to_hex(sha512(d));
    assert_eq!(
        hex(b""),
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
    );
    assert_eq!(
        hex(ABC),
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
    );
    assert_eq!(
        hex(M448),
        "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c33596fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445"
    );
    assert_eq!(
        hex(M896),
        "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"
    );
    assert_eq!(
        hex(&vec![b'a'; 1_000_000]),
        "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b"
    );
}

#[test]
fn rfc_4231_vectors() {
    let hex = |k: &[u8], m: &[u8]| hmac_sha256(k, m).to_hex();
    // Test cases 1 to 4, 6 and 7 (5 truncates the output).
    assert_eq!(hex(&[0x0b; 20], b"Hi There"), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    assert_eq!(
        hex(b"Jefe", b"what do ya want for nothing?"),
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    );
    assert_eq!(hex(&[0xaa; 20], &[0xdd; 50]), "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");
    let key4: Vec<u8> = (1..=25).collect();
    assert_eq!(hex(&key4, &[0xcd; 50]), "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b");
    assert_eq!(
        hex(&[0xaa; 131], b"Test Using Larger Than Block-Size Key - Hash Key First"),
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
    );
    assert_eq!(
        hex(
            &[0xaa; 131],
            b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm."
        ),
        "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"
    );
}

#[test]
fn updates_split_at_every_offset_hash_as_one_buffer() {
    // Two blocks of SHA-512 is four of SHA-256, plus a tail that pads
    // into one more block of each.
    let msg: Vec<u8> = (0..300u32).map(|i| (i * 31 + 7) as u8).collect();
    for len in [M896.len(), 2 * 128, msg.len()] {
        let msg = if len == M896.len() { M896 } else { &msg[..len] };
        let (want256, want512) = (sha256(msg), sha512(msg));
        for cut in 0..=msg.len() {
            let mut a = Sha256::new();
            a.update(&msg[..cut]).update(&msg[cut..]);
            assert_eq!(a.finish(), want256, "sha256 of {len} bytes cut at {cut}");
            let mut b = Sha512::new();
            b.update(&msg[..cut]).update(&msg[cut..]);
            assert_eq!(b.finish(), want512, "sha512 of {len} bytes cut at {cut}");
        }
    }
}
