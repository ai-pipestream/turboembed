//! Inflates the raw deflate streams java.util.zip.Deflater wrote (see
//! fixtures/README.md) and compares them with their inputs, which are
//! rebuilt here with the generator in fixtures/src/Inputs.java.

use std::path::{Path, PathBuf};

use turbo_maxent::crc32;
use turbo_maxent::inflate::inflate;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inflate")
}

/// The generator of Inputs.java: a 64-bit LCG whose top 31 bits are used.
struct Inputs(u64);

impl Inputs {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

const WORDS: [&str; 19] = [
    "the",
    "model",
    "reads",
    "a",
    "zip",
    "entry",
    "and",
    "inflates",
    "its",
    "bytes",
    "into",
    "memory",
    "while",
    "tokens",
    "sentences",
    "offsets",
    "stay",
    "exact",
    ".",
];

fn text(seed: u64, size: usize) -> Vec<u8> {
    let mut r = Inputs(seed);
    let mut s = String::new();
    while s.len() < size {
        s.push_str(WORDS[r.next() as usize % WORDS.len()]);
        s.push(' ');
    }
    s.truncate(size);
    s.into_bytes()
}

fn random(seed: u64, size: usize) -> Vec<u8> {
    let mut r = Inputs(seed);
    (0..size).map(|_| r.next() as u8).collect()
}

fn runs(seed: u64) -> Vec<u8> {
    let mut r = Inputs(seed);
    let mut out = Vec::new();
    for i in 0..40u8 {
        let len = 1 + r.next() as usize % 300;
        out.extend(std::iter::repeat_n(b'a' + i % 26, len));
    }
    out
}

fn far() -> Vec<u8> {
    let block = random(7, 1000);
    let runs = runs(8);
    let mut out = block.clone();
    out.extend((0..30000).map(|i| runs[i % runs.len()]));
    out.extend_from_slice(&block);
    out
}

fn input(name: &str) -> Vec<u8> {
    match name {
        "empty" => Vec::new(),
        "short" => b"Hello, hello, hello! Deflate me, deflate me.".to_vec(),
        "text" => text(1, 6000),
        "runs" => runs(2),
        "random" => random(3, 500),
        "far" => far(),
        _ => panic!("unknown input {name}"),
    }
}

#[test]
fn every_java_deflater_stream_inflates_to_its_input() {
    let index = std::fs::read_to_string(fixtures().join("index.txt")).unwrap();
    let mut block_types = [false; 3];
    let mut streams = 0;
    for line in index.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split(' ').collect();
        let (file, name, size, crc) =
            (f[0], f[1], f[5].parse::<usize>().unwrap(), u32::from_str_radix(f[6], 16).unwrap());
        let stream = std::fs::read(fixtures().join(file)).unwrap();
        let expected = input(name);
        assert_eq!(expected.len(), size, "{file}: the rebuilt input has the wrong size");
        assert_eq!(crc32(&expected), crc, "{file}: the rebuilt input differs from Java's");
        let got = inflate(&stream, size).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert!(got == expected, "{file}: inflated bytes differ");
        block_types[usize::from((stream[0] >> 1) & 3)] = true;
        if name == "far" {
            // Two copies of a 1000-byte random block cannot fit in 2000 bytes
            // unless the second is coded as a copy from 31000 bytes back.
            assert!(stream.len() < 2000, "{file}: the repeat was not coded as a copy");
        }
        streams += 1;
    }
    assert_eq!(streams, 63);
    assert_eq!(block_types, [true, true, true], "stored, fixed and dynamic first blocks all appear");
}

#[test]
fn a_truncated_java_stream_is_an_error() {
    let stream = std::fs::read(fixtures().join("text-6-default.deflate")).unwrap();
    for cut in [1, stream.len() / 2, stream.len() - 1] {
        assert!(inflate(&stream[..cut], 6000).is_err(), "cut at {cut}");
    }
}
