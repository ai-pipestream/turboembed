//! Inputs that are not Zstandard data, or not all of it, or not the size
//! asked for: each is refused with an error, never a panic.

mod common;

use common::build::{compressed, count, frame, raw, raw_literals, rle, windowed};
use common::{Lcg, fixture, original};
use turbo_zstd::{Error, Part, Symbol, decompress};

#[test]
fn empty_input() {
    assert_eq!(decompress(&[], 0), Err(Error::NoFrames));
}

#[test]
fn bad_magic_and_bytes_after_the_last_frame() {
    assert_eq!(decompress(b"PK\x03\x04rest", 0), Err(Error::BadMagic(0x0403_4b50)));
    let mut src = fixture("byte.zst");
    src.extend(b"junk");
    assert_eq!(decompress(&src, 1), Err(Error::BadMagic(u32::from_le_bytes(*b"junk"))));
    src.truncate(src.len() - 2);
    assert_eq!(decompress(&src, 1), Err(Error::Truncated(Part::Magic)));
}

/// Every prefix of the `decodecorpus` frames, which hold every block
/// type and literals and sequences sections of every kind, ends inside
/// some part of a frame or between two frames, where the frames before
/// it come to less than the size asked for.
#[test]
fn truncation_at_every_byte() {
    for (name, want) in [("decodecorpus.zst", fixture("decodecorpus.bin").len()), ("byte.zst", 1), ("empty.zst", 0)] {
        let src = fixture(name);
        assert_eq!(decompress(&src, want).unwrap().len(), want);
        for cut in 0..src.len() {
            match decompress(&src[..cut], want) {
                Err(Error::NoFrames) => assert_eq!(cut, 0),
                Err(Error::Truncated(_) | Error::TooShort { .. } | Error::CannotExpand { .. }) => {}
                r => panic!("{name} cut at {cut}: {r:?}"),
            }
        }
    }
}

#[test]
fn corrupted_checksum() {
    let mut src = fixture("text-3.zst");
    let n = src.len();
    src[n - 1] ^= 0x40;
    match decompress(&src, 150_000) {
        Err(Error::Checksum { stored, computed }) => assert_eq!(stored ^ computed, 0x4000_0000),
        r => panic!("{:?}", r.map(|v| v.len())),
    }
}

/// A byte changed inside a raw block still decodes, to other content,
/// which only the checksum catches.
#[test]
fn checksum_covers_the_content() {
    let mut src = fixture("random-wlog10.zst");
    src[100] ^= 1;
    assert!(matches!(decompress(&src, 20_000), Err(Error::Checksum { .. })));
}

#[test]
fn output_must_be_the_size_asked_for() {
    // A frame with its content size in the header.
    let src = fixture("text-1.zst");
    assert_eq!(decompress(&src, 149_999), Err(Error::TooLong { want: 149_999 }));
    assert_eq!(decompress(&src, 150_001), Err(Error::TooShort { got: 150_000, want: 150_001 }));
    // And one without.
    let src = fixture("small-pipe.zst");
    let n = original("small").len();
    assert_eq!(decompress(&src, 0), Err(Error::TooLong { want: 0 }));
    assert_eq!(decompress(&src, n - 1), Err(Error::TooLong { want: n - 1 }));
    assert_eq!(decompress(&src, n + 1), Err(Error::TooShort { got: n, want: n + 1 }));
}

/// The frame and block headers are read before the output is allocated,
/// and a size the frames cannot come to is refused then: one other than
/// the sum of the content sizes when every frame gives one, or past what
/// the blocks of a frame without one hold. `tests/allocations.rs` checks
/// that nothing is allocated.
#[test]
fn sizes_the_frames_cannot_reach() {
    // The `zstd` tool's frame gives its content size, 1.
    let src = fixture("byte.zst");
    assert_eq!(decompress(&src, 2), Err(Error::TooShort { got: 1, want: 2 }));
    assert_eq!(decompress(&src, usize::MAX), Err(Error::TooShort { got: 1, want: usize::MAX }));
    assert_eq!(decompress(&src, 0), Err(Error::TooLong { want: 0 }));
    // With no content size, a raw block holds its size and an RLE block
    // the bytes it stands for.
    let f = frame(&windowed(0), &[raw(b"abc", false), rle(b'x', 5, true)]);
    assert_eq!(decompress(&f, 8).unwrap(), b"abcxxxxx");
    assert_eq!(decompress(&f, 9), Err(Error::CannotExpand { max: 8, want: 9 }));
    // A compressed block holds at most the frame's block maximum, here
    // the window, 1 KiB.
    let f = frame(&windowed(0), &[raw(b"abc", false), compressed(&raw_literals(b"de"), &count(0), true)]);
    assert_eq!(decompress(&f, 5).unwrap(), b"abcde");
    assert_eq!(decompress(&f, 3 + 1024), Err(Error::TooShort { got: 5, want: 1027 }));
    assert_eq!(decompress(&f, 3 + 1025), Err(Error::CannotExpand { max: 1027, want: 1028 }));
    // The `zstd` tool's output from a pipe: no content size, and 40,000
    // bytes in one compressed block of at most 128 KiB.
    let src = fixture("small-pipe.zst");
    assert_eq!(decompress(&src, usize::MAX), Err(Error::CannotExpand { max: 128 << 10, want: usize::MAX }));
    // Frames one after another: the bounds add up.
    let mut two = frame(&windowed(0), &[rle(b'x', 5, true)]);
    two.extend(&src);
    assert_eq!(decompress(&two, 5 + (128 << 10) + 1), Err(Error::CannotExpand { max: 5 + (128 << 10), want: 131_078 }));
    // A frame whose content size is more than its blocks hold is bounded
    // by its blocks.
    let f = frame(&[0x80, 0, 10, 0, 0, 0], &[raw(b"12345", false), raw(b"6789", true)]);
    assert_eq!(decompress(&f, 10), Err(Error::CannotExpand { max: 9, want: 10 }));
    // 1 MiB of zeros is no frame, whatever size is asked for.
    assert_eq!(decompress(&vec![0; 1 << 20], 1 << 35), Err(Error::BadMagic(0)));
}

/// RLE blocks of 128 KiB, 4 bytes each, expand the most a frame can:
/// about 32,000 to 1 here. With no content size, what the block headers
/// allow is exactly what the blocks decode to.
#[test]
fn rle_blocks_expand_the_most() {
    let blocks: Vec<_> = (0..64).map(|i| rle(b'q', 128 << 10, i == 63)).collect();
    let src = frame(&windowed(7 << 3), &blocks);
    let want = 64 << 17;
    assert!(want / src.len() > 32_000);
    let out = decompress(&src, want).unwrap();
    assert!(out.iter().all(|&b| b == b'q'));
    assert_eq!(decompress(&src, want + 1), Err(Error::CannotExpand { max: want as u64, want: want + 1 }));
}

#[test]
fn messages_say_what_is_wrong() {
    assert_eq!(Error::Truncated(Part::JumpTable).to_string(), "the input ends inside a jump table");
    assert_eq!(Error::BadMagic(0x0403_4b50).to_string(), "bad magic number 0x04034b50");
    assert_eq!(
        Error::CannotExpand { max: 8, want: 9 }.to_string(),
        "the frames decode to at most 8 bytes, fewer than the 9 asked for"
    );
    assert_eq!(
        Error::TooManyLiterals { len: 1025, max: 1024 }.to_string(),
        "a literals section of 1025 bytes, over the frame's block size of 1024"
    );
    assert_eq!(
        Error::WindowTooLarge { window: 1 << 28, limit: 1 << 27 }.to_string(),
        "a window of 268435456 bytes, over the limit of 134217728"
    );
    assert_eq!(
        Error::NoTableToRepeat(Symbol::MatchLength).to_string(),
        "the match length table is in repeat mode with no earlier table"
    );
    assert_eq!(
        Error::OffsetBeforeStart { offset: 9, available: 4 }.to_string(),
        "a match from 9 bytes back, where the frame's content is 4 bytes"
    );
}

/// Seeded single-byte changes to the fixtures: each input is refused
/// or decodes to exactly the size asked for, and none panics.
#[test]
fn changed_bytes_never_panic() {
    let inputs = [
        (fixture("decodecorpus.zst"), fixture("decodecorpus.bin").len()),
        (fixture("small-19-wlog10.zst"), original("small").len()),
        (fixture("small-pipe.zst"), original("small").len()),
        (fixture("byte.zst"), 1),
    ];
    let mut rng = Lcg(429);
    let mut refused = 0;
    for i in 0..4000 {
        let (src, want) = &inputs[i % inputs.len()];
        let mut src = src.clone();
        // Half the changes in the first 64 bytes, where the headers are.
        let at = if rng.below(2) == 0 { rng.below(src.len().min(64)) } else { rng.below(src.len()) };
        src[at] ^= 1 + rng.below(255) as u8;
        let r = std::panic::catch_unwind(|| decompress(&src, *want));
        match r {
            Ok(Ok(out)) => assert_eq!(out.len(), *want),
            Ok(Err(_)) => refused += 1,
            Err(_) => panic!("input {} with byte {at} changed to {:#04x}", i % inputs.len(), src[at]),
        }
    }
    assert!(refused > 3000, "only {refused} refused");
}
