use super::*;
use crate::writer::full_golden;

// ---------------------------------------------------------------------
// Hashes

/// 1,000,000 repetitions of "a", fed in pieces of odd sizes that cycle,
/// so block edges fall in every position of a piece.
fn million_a_in_pieces(sizes: &[usize]) -> (Sha256, Sha512) {
    let (mut a, mut b) = (Sha256::new(), Sha512::new());
    let buf = vec![b'a'; *sizes.iter().max().unwrap()];
    let (mut left, mut i) = (1_000_000usize, 0);
    while left > 0 {
        let n = sizes[i % sizes.len()].min(left);
        a.update(&buf[..n]);
        b.update(&buf[..n]);
        left -= n;
        i += 1;
    }
    (a, b)
}

const MILLION_A_256: &str = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
const MILLION_A_512: &str = "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973eb\
                             de0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b";

#[test]
fn one_million_a_streamed_in_odd_pieces() {
    for sizes in [
        &[1usize][..],
        &[7],
        &[61],
        &[63, 65],
        &[127, 129, 1],
        &[3, 1000, 17, 4093],
        &[55, 56, 57, 111, 112, 113],
        &[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987],
        &[100_000],
        &[999_999, 1],
    ] {
        let (a, b) = million_a_in_pieces(sizes);
        assert_eq!(hex::to_hex(&a.finish()), MILLION_A_256, "sha256 in pieces {sizes:?}");
        assert_eq!(hex::to_hex(&b.finish()), MILLION_A_512, "sha512 in pieces {sizes:?}");
    }
}

#[test]
fn published_vectors_by_streaming_and_in_one_call() {
    let m56 = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    let m112 = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    let v256: [(&[u8], &str); 4] = [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (m56, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
        (m112, "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"),
    ];
    for (msg, want) in v256 {
        assert_eq!(hex::to_hex(&sha256(msg)), want);
        let mut h = Sha256::new();
        for b in msg {
            h.update(std::slice::from_ref(b));
        }
        assert_eq!(hex::to_hex(&h.finish()), want, "byte by byte, {} bytes", msg.len());
    }
    let v512: [(&[u8], &str); 4] = [
        (
            b"",
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
        ),
        (
            b"abc",
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        ),
        (
            m56,
            "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c33596fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445",
        ),
        (
            m112,
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
        ),
    ];
    for (msg, want) in v512 {
        assert_eq!(hex::to_hex(&sha512(msg)), want);
        let mut h = Sha512::new();
        for b in msg {
            h.update(std::slice::from_ref(b));
        }
        assert_eq!(hex::to_hex(&h.finish()), want, "byte by byte, {} bytes", msg.len());
    }
}

#[test]
fn empty_updates_and_defaults_change_nothing() {
    let mut a = Sha256::default();
    a.update(b"");
    a.update(b"ab");
    a.update(b"");
    a.update(b"c");
    a.update(b"");
    assert_eq!(a.finish(), sha256(b"abc"));
    let mut b = Sha512::default();
    b.update(b"");
    b.update(b"abc");
    b.update(b"");
    assert_eq!(b.finish(), sha512(b"abc"));
    assert_eq!(Sha256::new().finish(), sha256(b""));
    assert_eq!(Sha512::new().finish(), sha512(b""));
}

#[test]
fn messages_around_the_padding_edges_agree_with_streaming() {
    // Lengths on both sides of every block and padding edge, hashed whole
    // and as two pieces split at every position.
    let data: Vec<u8> = (0..520u32).map(|i| (i * 31 + 7) as u8).collect();
    for len in (0..=140).chain([191, 192, 193, 255, 256, 257, 383, 384, 385, 511, 512, 519]) {
        let msg = &data[..len];
        for split in [0, 1, len / 2, len.saturating_sub(1), len] {
            let (x, y) = msg.split_at(split.min(len));
            let mut a = Sha256::new();
            a.update(x);
            a.update(y);
            assert_eq!(a.finish(), sha256(msg), "sha256 of {len} bytes split at {split}");
            let mut b = Sha512::new();
            b.update(x);
            b.update(y);
            assert_eq!(b.finish(), sha512(msg), "sha512 of {len} bytes split at {split}");
        }
    }
}

type Edit = Box<dyn Fn(&mut Golden)>;

#[test]
fn what_the_writer_writes_the_reader_reads_for_empty_fields() {
    let cases: Vec<Edit> = vec![
        Box::new(|g| g.header.generator.name = String::new()),
        Box::new(|g| g.header.opennlp.version = String::new()),
        Box::new(|g| g.header.jdk.vendor = String::new()),
        Box::new(|g| g.header.input.set = String::new()),
        Box::new(|g| g.header.input.id = String::new()),
        Box::new(|g| g.header.models[0].file = String::new()),
        Box::new(|g| g.header.annotator.class = String::new()),
        Box::new(|g| g.layers[0].columns[0].name = String::new()),
        Box::new(|g| g.layers[1].anchor = GoldenAnchor::Aligned(String::new())),
    ];
    for (i, change) in cases.iter().enumerate() {
        let mut g = full_golden();
        change(&mut g);
        // Either the writer refuses, or the reader reads what was written.
        if let Ok(text) = g.to_text() {
            assert!(Golden::parse(&text).is_ok(), "case {i}: the writer wrote a file the reader refuses");
        }
    }
}

/// The writer accepts a layer id the reader has no shape for; that is the
/// writer being generic, and the reader is where the id is checked.
#[test]
fn the_writer_does_not_check_layer_ids_but_the_reader_does() {
    let mut g = full_golden();
    g.layers[4].id = "my:layer".into();
    let text = g.to_text().unwrap();
    let e = Golden::parse(&text).unwrap_err();
    assert!(matches!(e, GoldenError::UnknownLayer { .. }), "{e:?}");
}
