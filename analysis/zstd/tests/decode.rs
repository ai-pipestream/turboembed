//! Decoding the `zstd` tool's output and the reference project's
//! `decodecorpus` frames back to the bytes they were made from, and the
//! stream-level rules: frames one after another and skippable frames.
//! `tests/data/README.md` has the commands each fixture was made with.

mod common;

use common::build::{self, MAGIC};
use common::{fixture, original};
use turbo_zstd::decompress;

/// Each fixture and the original it was compressed from.
const FIXTURES: [(&str, &str); 12] = [
    ("empty.zst", "empty"),
    ("byte.zst", "byte"),
    ("text-1.zst", "text"),
    ("text-3.zst", "text"),
    ("text-9.zst", "text"),
    ("text-19.zst", "text"),
    ("text-22-pipe.zst", "text"),
    ("small-19-wlog10.zst", "small"),
    ("small-pipe.zst", "small"),
    ("random-wlog10.zst", "random"),
    ("runs-19.zst", "runs"),
    ("long.zst", "long"),
];

#[test]
fn fixtures_decode_to_their_originals() {
    for (name, from) in FIXTURES {
        let want = original(from);
        let got = decompress(&fixture(name), want.len()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(got == want, "{name} does not decode to {from}");
    }
}

/// The parts of a one-frame fixture the tests rely on it having.
#[derive(Debug)]
struct Shape {
    checksum: bool,
    content: Option<u64>,
    /// The window descriptor's window, absent in single-segment mode.
    window: Option<u64>,
    /// Each block's type: 0 raw, 1 RLE, 2 compressed.
    blocks: Vec<u8>,
}

fn shape(f: &[u8]) -> Shape {
    assert_eq!(f[..4], MAGIC);
    let fhd = f[4];
    let single = fhd & 0x20 != 0;
    let fcs = [usize::from(single), 2, 4, 8][usize::from(fhd >> 6)];
    let mut at = 5;
    let window = (!single).then(|| {
        let base = 1u64 << (10 + (f[5] >> 3));
        at += 1;
        base + base / 8 * u64::from(f[5] & 7)
    });
    at += [0, 1, 2, 4][usize::from(fhd & 3)];
    let le = |b: &[u8]| b.iter().rev().fold(0u64, |v, &x| v << 8 | u64::from(x));
    let content = match fcs {
        0 => None,
        2 => Some(le(&f[at..at + 2]) + 256),
        n => Some(le(&f[at..at + n])),
    };
    at += fcs;
    let mut blocks = Vec::new();
    loop {
        let h = le(&f[at..at + 3]) as usize;
        let kind = (h >> 1 & 3) as u8;
        blocks.push(kind);
        at += 3 + if kind == 1 { 1 } else { h >> 3 };
        if h & 1 == 1 {
            break;
        }
    }
    let checksum = fhd & 4 != 0;
    assert_eq!(at + if checksum { 4 } else { 0 }, f.len(), "one frame");
    Shape { checksum, content, window, blocks }
}

/// The fixtures hold the parts of the format they are there for, so a
/// fixture made again with other options cannot quietly stop testing
/// them.
#[test]
fn fixtures_are_shaped_as_made() {
    let s = shape(&fixture("text-1.zst"));
    assert!(!s.checksum && s.content == Some(150_000), "{s:?}");
    assert!(s.blocks.iter().all(|&k| k == 2), "{s:?}");
    let s = shape(&fixture("text-3.zst"));
    assert!(s.checksum && s.content == Some(150_000), "{s:?}");
    // Piped in, the tool does not know the size: no content size, and
    // the window of the level, 2^27 for --ultra -22.
    let s = shape(&fixture("text-22-pipe.zst"));
    assert_eq!((s.content, s.window), (None, Some(1 << 27)), "{s:?}");
    let s = shape(&fixture("small-pipe.zst"));
    assert_eq!(s.content, None, "{s:?}");
    // A 1 KiB window caps blocks at 1 KiB: 40 KB of text in many blocks.
    let s = shape(&fixture("small-19-wlog10.zst"));
    assert_eq!(s.window, Some(1024), "{s:?}");
    assert!(s.blocks.len() >= 40 && s.blocks.iter().all(|&k| k == 2), "{s:?}");
    // Random bytes do not compress: raw blocks only.
    let s = shape(&fixture("random-wlog10.zst"));
    assert!(s.blocks.len() >= 20 && s.blocks.iter().all(|&k| k == 0), "{s:?}");
    // 300,000 zeros: RLE blocks among compressed ones.
    let s = shape(&fixture("runs-19.zst"));
    assert!(s.blocks.contains(&1) && s.blocks.contains(&2), "{s:?}");
    // Single segment: the window is the whole content, 3.3 MB.
    let s = shape(&fixture("long.zst"));
    assert_eq!((s.window, s.content), (None, Some(original("long").len() as u64)), "{s:?}");
    let s = shape(&fixture("empty.zst"));
    assert_eq!((s.content, s.blocks.as_slice()), (Some(0), &[0][..]), "{s:?}");
}

/// The text again, 3 MB later: compressed alone the text takes about
/// 37,600 bytes, and only a match from 3 MB back keeps the second copy
/// from taking as many again.
#[test]
fn long_distance_matches() {
    let f = fixture("long.zst");
    assert!(f.len() < 40_000, "{} bytes", f.len());
    assert_eq!(decompress(&f, original("long").len()).unwrap(), original("long"));
}

/// The stored checksum is the low 32 bits of the content's XXH64 with
/// seed 0, little-endian: XXH64("a") is 0xD24EC4F1A98C6E5B.
#[test]
fn checksum_is_the_low_half_of_xxh64() {
    let f = fixture("byte.zst");
    assert_eq!(f[f.len() - 4..], [0x5b, 0x6e, 0x8c, 0xa9]);
    assert_eq!(decompress(&f, 1).unwrap(), b"a");
}

/// Seven frames from `decodecorpus`, chosen so together they hold what
/// the `zstd` tool's output above does not: Huffman literals in one
/// stream, Huffman weights written directly, sequence tables in RLE
/// mode, sequences sections with no sequences, the 8-byte content size,
/// and windows other than powers of 2.
#[test]
fn decodecorpus_frames() {
    let want = fixture("decodecorpus.bin");
    assert_eq!(decompress(&fixture("decodecorpus.zst"), want.len()).unwrap(), want);
}

#[test]
fn frames_one_after_another() {
    let names = ["text-1.zst", "byte.zst", "empty.zst", "random-wlog10.zst", "text-22-pipe.zst", "text-3.zst"];
    let mut src = Vec::new();
    let mut want = Vec::new();
    for name in names {
        src.extend(fixture(name));
        let from = FIXTURES.iter().find(|(n, _)| *n == name).unwrap().1;
        want.extend(original(from));
    }
    assert!(decompress(&src, want.len()).unwrap() == want);
}

/// A skippable frame: one of the 16 magic numbers, a 4-byte size and
/// that many bytes of anything.
fn skippable(n: u32, data: &[u8]) -> Vec<u8> {
    let mut f = (0x184D_2A50 + n).to_le_bytes().to_vec();
    f.extend((data.len() as u32).to_le_bytes());
    f.extend(data);
    f
}

#[test]
fn skippable_frames_are_stepped_over() {
    let text = fixture("text-1.zst");
    let byte = fixture("byte.zst");
    let mut want = original("text");
    want.extend(b"a");
    for n in 0..16 {
        // Before, between and after frames, empty or holding what looks
        // like a frame.
        let mut src = skippable(n, &byte);
        src.extend(&text);
        src.extend(skippable(n, &[]));
        src.extend(&byte);
        src.extend(skippable(n, &MAGIC));
        assert!(decompress(&src, want.len()).unwrap() == want, "magic {n}");
    }
    // Skippable frames alone are no content.
    assert_eq!(decompress(&skippable(3, b"metadata"), 0).unwrap(), b"");
    // The magic numbers on either side of the range are not skippable.
    for magic in [0x184D_2A4Fu32, 0x184D_2A60] {
        let mut src = magic.to_le_bytes().to_vec();
        src.extend([0; 4]);
        assert_eq!(decompress(&src, 0), Err(turbo_zstd::Error::BadMagic(magic)));
    }
}

/// A hand-built frame next to the tool's: the RLE and raw blocks of one
/// frame come out in order.
#[test]
fn raw_and_rle_blocks() {
    let f = build::frame(
        &build::single(9),
        &[build::raw(b"ab", false), build::rle(b'z', 4, false), build::raw(b"cd", false), build::rle(b'!', 1, true)],
    );
    assert_eq!(decompress(&f, 9).unwrap(), b"abzzzzcd!");
}
