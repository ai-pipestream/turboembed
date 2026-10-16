//! The format's rules one at a time, on frames built by hand: each frame
//! header field, block sizes and types, and the sequences section, from
//! how offsets repeat to where matches may reach.

mod common;

use common::build::{
    Seq, Tables, compressed, count, frame, raw, raw_literals, rle, rle_literals, sequences, single, windowed,
};
use common::{fixture, random};
use turbo_zstd::{Error, Part, Symbol, decompress};

/// A window descriptor for 1 MiB, so blocks may hold 128 KiB.
const MIB: u8 = 10 << 3;

/// `data` as raw blocks of at most `max` bytes, the last one flagged.
fn raw_blocks(data: &[u8], max: usize) -> Vec<Vec<u8>> {
    if data.is_empty() {
        return vec![raw(&[], true)];
    }
    let n = data.len().div_ceil(max);
    data.chunks(max).enumerate().map(|(i, c)| raw(c, i + 1 == n)).collect()
}

/// A compressed block of raw literals and one sequences section whose
/// tables are in RLE mode.
fn block(lits: &[u8], seqs: &[Seq], last: bool) -> Vec<u8> {
    compressed(&raw_literals(lits), &sequences(seqs, Tables::Rle), last)
}

fn seq(ll: u32, ml: u32, offset: u32) -> Seq {
    Seq { ll, ml, offset }
}

/// Appends `n` bytes copied from `offset` back, a byte at a time.
fn copy(out: &mut Vec<u8>, offset: usize, n: usize) {
    for _ in 0..n {
        out.push(out[out.len() - offset]);
    }
}

#[test]
fn content_size_in_each_width() {
    // Single segment: the size in 1 byte, in 2 bytes as the size less
    // 256, then in 4 bytes.
    for n in [0, 1, 255, 256, 65791, 65792, 200_000] {
        let data = random(n, 5);
        let f = frame(&single(n as u64), &raw_blocks(&data, 128 << 10));
        assert_eq!(decompress(&f, n).unwrap(), data, "{n} bytes");
    }
    // With a window descriptor: 2, 4 and 8 bytes, and the 8-byte form in
    // single-segment mode.
    let data = random(300, 6);
    for header in [
        vec![0x40, MIB, 44, 0],
        vec![0x80, MIB, 44, 1, 0, 0],
        vec![0xc0, MIB, 44, 1, 0, 0, 0, 0, 0, 0],
        vec![0xe0, 44, 1, 0, 0, 0, 0, 0, 0],
    ] {
        let f = frame(&header, &raw_blocks(&data, 1000));
        assert_eq!(decompress(&f, 300).unwrap(), data, "{header:?}");
    }
    // No size at all: the caller's size is the only bound.
    let f = frame(&windowed(MIB), &raw_blocks(&data, 1000));
    assert_eq!(decompress(&f, 300).unwrap(), data);
}

#[test]
fn content_must_be_the_size_the_header_says() {
    // A 4-byte content size of 10 and a 1 KiB window.
    let header = [0x80, 0, 10, 0, 0, 0];
    // A compressed block may hold up to 1 KiB, so the headers allow 10.
    let short = frame(&header, &[raw(b"12345", false), compressed(&raw_literals(b"6789"), &count(0), true)]);
    assert_eq!(decompress(&short, 10), Err(Error::ContentTooShort { header: 10, got: 9 }));
    // Raw blocks hold only their size: 9, which is refused before the
    // blocks are decoded.
    let short = frame(&header, &[raw(b"12345", false), raw(b"6789", true)]);
    assert_eq!(decompress(&short, 10), Err(Error::CannotExpand { max: 9, want: 10 }));
    let long = frame(&header, &[raw(b"12345", false), raw(b"678901", true)]);
    assert_eq!(decompress(&long, 10), Err(Error::ContentTooLong { header: 10 }));
    let long = frame(&header, &[block(b"abcd", &[seq(4, 7, 7)], true)]);
    assert_eq!(decompress(&long, 10), Err(Error::ContentTooLong { header: 10 }));
    let exact = frame(&header, &[block(b"abcd", &[seq(4, 6, 7)], true)]);
    assert_eq!(decompress(&exact, 10).unwrap(), b"abcdabcdab");
    // A size other than the header's is refused before the blocks.
    assert_eq!(decompress(&long, 20), Err(Error::TooShort { got: 10, want: 20 }));
    assert_eq!(decompress(&exact, 9), Err(Error::TooLong { want: 9 }));
}

/// The content size is checked even when the last block is empty.
#[test]
fn an_empty_last_block_still_ends_the_content() {
    let f = frame(&[0x80, 0, 10, 0, 0, 0], &[compressed(&raw_literals(b"12345"), &count(0), false), raw(b"", true)]);
    assert_eq!(decompress(&f, 10), Err(Error::ContentTooShort { header: 10, got: 5 }));
}

#[test]
fn output_past_the_size_asked_for() {
    // With a content size the frame is refused before its blocks.
    let f = frame(&single(10), &[raw(b"0123456789", true)]);
    assert_eq!(decompress(&f, 9), Err(Error::TooLong { want: 9 }));
    // Without one, at the block that goes past: raw, RLE or compressed.
    for b in [raw(b"0123456789", true), rle(b'x', 10, true), block(b"abcd", &[seq(4, 6, 7)], true)] {
        let f = frame(&windowed(MIB), &[b]);
        assert_eq!(decompress(&f, 9), Err(Error::TooLong { want: 9 }));
        assert_eq!(decompress(&f, 10).unwrap().len(), 10);
    }
}

#[test]
fn unused_and_reserved_bits() {
    // Bit 4 is unused and ignored; bit 3 is reserved.
    assert_eq!(decompress(&frame(&[0x30, 3], &[raw(b"abc", true)]), 3).unwrap(), b"abc");
    assert_eq!(decompress(&frame(&[0x28, 3], &[raw(b"abc", true)]), 3), Err(Error::ReservedBit));
}

#[test]
fn dictionary_ids() {
    // An id of 0, in 1, 2 or 4 bytes, names no dictionary.
    for header in [vec![0x21, 0, 3], vec![0x22, 0, 0, 3], vec![0x23, 0, 0, 0, 0, 3]] {
        assert_eq!(decompress(&frame(&header, &[raw(b"abc", true)]), 3).unwrap(), b"abc");
    }
    let f = frame(&[0x22, 0x39, 0x30, 3], &[raw(b"abc", true)]);
    let e = decompress(&f, 3).unwrap_err();
    assert_eq!(e, Error::Dictionary(12345));
    assert_eq!(e.to_string(), "the frame needs dictionary 12345; dictionaries are not supported");
}

#[test]
fn window_limit() {
    // 2^27 is taken; the next descriptor, 2^27 + 2^24, is not.
    assert_eq!(decompress(&frame(&windowed(17 << 3), &[raw(b"abc", true)]), 3).unwrap(), b"abc");
    let f = frame(&windowed(17 << 3 | 1), &[raw(b"abc", true)]);
    assert_eq!(decompress(&f, 3), Err(Error::WindowTooLarge { window: (1 << 27) + (1 << 24), limit: 1 << 27 }));
    // The `zstd` tool's --long=28 asks for 2^28.
    let e = decompress(&fixture("window-2-28.zst"), 1).unwrap_err();
    assert_eq!(e, Error::WindowTooLarge { window: 1 << 28, limit: 1 << 27 });
    // In single-segment mode the window is the content size, which the
    // caller's size bounds instead.
    let f = frame(&single(1 << 28), &[rle(0, 128 << 10, true)]);
    assert_eq!(decompress(&f, 1 << 17), Err(Error::TooLong { want: 1 << 17 }));
}

#[test]
fn blocks_are_at_most_the_window_or_128_kib() {
    // A 1 KiB window: raw, RLE and compressed blocks of up to 1024 bytes.
    let kib = random(1025, 7);
    let f = frame(&windowed(0), &[raw(&kib[..1024], true)]);
    assert_eq!(decompress(&f, 1024).unwrap(), &kib[..1024]);
    let too_large = Err(Error::BlockTooLarge { size: 1025, max: 1024 });
    assert_eq!(decompress(&frame(&windowed(0), &[raw(&kib, true)]), 1025), too_large);
    assert_eq!(decompress(&frame(&windowed(0), &[rle(1, 1025, true)]), 1025), too_large);
    // A compressed block's own size counts, whatever it decodes to: a
    // 2-byte literals header, 1022 literals and a 1-byte sequences
    // section.
    let f = frame(&windowed(0), &[compressed(&raw_literals(&kib[..1022]), &count(0), true)]);
    assert_eq!(decompress(&f, 1022), too_large);
    let f = frame(&windowed(0), &[compressed(&raw_literals(&kib[..1021]), &count(0), true)]);
    assert_eq!(decompress(&f, 1021).unwrap(), &kib[..1021]);
    // And so does what it decodes to: its literals, or literals and
    // matches. The headers allow such a block 1 KiB, so a larger size is
    // refused before it is decoded.
    let f = frame(&windowed(0), &[compressed(&rle_literals(b'x', 1025), &count(0), true)]);
    assert_eq!(decompress(&f, 1025), Err(Error::CannotExpand { max: 1024, want: 1025 }));
    assert_eq!(decompress(&f, 1024), Err(Error::TooManyLiterals { len: 1025, max: 1024 }));
    let f = frame(
        &windowed(0),
        &[compressed(&rle_literals(b'x', 1000), &sequences(&[seq(1000, 100, 4)], Tables::Rle), true)],
    );
    assert_eq!(decompress(&f, 1024), Err(Error::BlockTooLarge { size: 1100, max: 1024 }));
    // A larger window still caps blocks at 128 KiB.
    let big = random((128 << 10) + 1, 8);
    let f = frame(&windowed(MIB), &[raw(&big[..128 << 10], true)]);
    assert_eq!(decompress(&f, 128 << 10).unwrap(), &big[..128 << 10]);
    let f = frame(&windowed(MIB), &[raw(&big, true)]);
    assert_eq!(decompress(&f, big.len()), Err(Error::BlockTooLarge { size: 131_073, max: 131_072 }));
    // In single-segment mode the window, and so the cap, is the content
    // size.
    let data = random(301, 9);
    let f = frame(&single(300), &[raw(&data[..150], false), raw(&data[150..300], true)]);
    assert_eq!(decompress(&f, 300).unwrap(), &data[..300]);
    let f = frame(&single(301), &[raw(&data, true)]);
    assert_eq!(decompress(&f, 301).unwrap(), data);
    let f = frame(&single(300), &[raw(&data, true)]);
    assert_eq!(decompress(&f, 300), Err(Error::BlockTooLarge { size: 301, max: 300 }));
}

#[test]
fn the_last_block_ends_the_frame() {
    // With no last block the input ends inside the frame.
    let f = frame(&windowed(MIB), &[raw(b"ab", false)]);
    assert_eq!(decompress(&f, 2), Err(Error::Truncated(Part::BlockHeader)));
    // What follows the last block is read as the next frame.
    let f = frame(&windowed(MIB), &[raw(b"ab", true), raw(b"cd", true)]);
    assert_eq!(decompress(&f, 4), Err(Error::BadMagic(u32::from_le_bytes([0x11, 0, 0, b'c']))));
}

#[test]
fn a_compressed_block_of_size_0_is_refused() {
    // It has no room for a literals section. zstd's streaming decoder
    // takes it as an empty block.
    let f = frame(&windowed(MIB), &[raw(b"ab", false), compressed(&[], &[], true)]);
    assert_eq!(decompress(&f, 2), Err(Error::Truncated(Part::LiteralsHeader)));
}

#[test]
fn reserved_block_type() {
    let f = frame(&windowed(MIB), &[vec![0b111, 0, 0]]);
    assert_eq!(decompress(&f, 0), Err(Error::ReservedBlockType));
}

#[test]
fn literals_header_forms() {
    // Raw literals with 1, 2 and 3-byte headers, then RLE, in sections
    // with no sequences.
    for n in [0, 31, 32, 4095, 4096, 100_000] {
        let lits = random(n, 10);
        let f = frame(&windowed(MIB), &[compressed(&raw_literals(&lits), &count(0), true)]);
        assert_eq!(decompress(&f, n).unwrap(), lits, "{n} raw literals");
    }
    let rle_sections =
        [(vec![5 << 3 | 1, b'r'], 5), (vec![0b0101 | 4 << 4, 2, b'r'], 36), (rle_literals(b'r', 70_000), 70_000)];
    for (section, n) in rle_sections {
        let f = frame(&windowed(MIB), &[compressed(&section, &count(0), true)]);
        assert_eq!(decompress(&f, n).unwrap(), vec![b'r'; n], "{n} RLE literals");
    }
}

#[test]
fn literals_then_matches() {
    let f = frame(&windowed(MIB), &[block(b"abcd", &[seq(4, 8, 4 + 3)], true)]);
    assert_eq!(decompress(&f, 12).unwrap(), b"abcdabcdabcd");
    // Literals left after the last sequence come at the end.
    let f = frame(&windowed(MIB), &[block(b"abcdXYZ", &[seq(4, 4, 4 + 3)], true)]);
    assert_eq!(decompress(&f, 11).unwrap(), b"abcdabcdXYZ");
    // Several sequences, each with its literals, in one block.
    let seqs = [seq(2, 3, 2 + 3), seq(2, 3, 3 + 3), seq(2, 3, 2 + 3)];
    let f = frame(&windowed(MIB), &[block(b"abcdef", &seqs, true)]);
    let mut want = b"ab".to_vec();
    copy(&mut want, 2, 3);
    want.extend(b"cd");
    copy(&mut want, 3, 3);
    want.extend(b"ef");
    copy(&mut want, 2, 3);
    assert_eq!(decompress(&f, want.len()).unwrap(), want);
}

/// Matches longer than their offset repeat the bytes they copy.
#[test]
fn overlapping_matches() {
    let alphabet = b"abcdefghijklmnopqrstuvwxyz";
    for offset in 1..=20 {
        for ml in [3, 17, 100] {
            let f = frame(
                &windowed(MIB),
                &[block(&alphabet[..offset], &[seq(offset as u32, ml as u32, offset as u32 + 3)], true)],
            );
            let mut want = alphabet[..offset].to_vec();
            copy(&mut want, offset, ml);
            assert_eq!(decompress(&f, want.len()).unwrap(), want, "offset {offset} length {ml}");
        }
    }
}

/// Repeated offsets carry from block to block, past a raw block, and a
/// match reaches back into the blocks before its own.
#[test]
fn repeated_offsets_carry_across_blocks() {
    let blocks =
        [block(b"abcdefgh", &[seq(8, 4, 8 + 3)], false), raw(b"XY", false), block(b"Q", &[seq(1, 5, 1)], true)];
    let mut want = b"abcdefgh".to_vec();
    copy(&mut want, 8, 4);
    want.extend(b"XYQ");
    copy(&mut want, 8, 5);
    assert_eq!(decompress(&frame(&windowed(MIB), &blocks), want.len()).unwrap(), want);
}

/// The format's rules for offset values 1 to 3: with literals they pick
/// the first, second or third repeated offset; with none, the second,
/// the third, or the first less 1. One sequence per block, each block's
/// tables in RLE mode.
#[test]
fn repeated_offsets_with_and_without_literals() {
    // (literals, offset value, the offset it stands for) after a first
    // sequence with offset 2, which makes the repeated offsets 2, 1, 4.
    let steps: [(&[u8], u32, usize); 5] = [(b"", 1, 1), (b"", 2, 4), (b"", 3, 3), (b"z", 3, 1), (b"y", 2, 3)];
    let mut blocks = vec![block(b"abcdefgh", &[seq(8, 3, 2 + 3)], false)];
    let mut want = b"abcdefgh".to_vec();
    copy(&mut want, 2, 3);
    for (i, &(lits, value, offset)) in steps.iter().enumerate() {
        blocks.push(block(lits, &[seq(lits.len() as u32, 3, value)], i + 1 == steps.len()));
        want.extend(lits);
        copy(&mut want, offset, 3);
    }
    assert_eq!(decompress(&frame(&windowed(MIB), &blocks), want.len()).unwrap(), want);
}

/// Each frame starts with repeated offsets 1, 4 and 8, whatever the one
/// before left.
#[test]
fn repeated_offsets_start_again_in_each_frame() {
    let first = frame(&windowed(MIB), &[block(b"0123456789", &[seq(10, 4, 10 + 3)], true)]);
    let mut want = b"0123456789".to_vec();
    copy(&mut want, 10, 4);
    for (value, offset) in [(1, 1), (2, 4), (3, 8)] {
        let second = frame(&windowed(MIB), &[block(b"abcdefgh", &[seq(8, 3, value)], true)]);
        let mut src = first.clone();
        src.extend(&second);
        let mut w = b"abcdefgh".to_vec();
        copy(&mut w, offset, 3);
        let w = [want.as_slice(), &w].concat();
        assert_eq!(decompress(&src, w.len()).unwrap(), w, "offset value {value}");
    }
}

#[test]
fn a_repeated_offset_of_zero() {
    // With no literals, value 3 is the first repeated offset less 1.
    let f = frame(&windowed(MIB), &[block(b"a", &[seq(0, 3, 3)], true)]);
    assert_eq!(decompress(&f, 4), Err(Error::ZeroOffset));
}

#[test]
fn matches_stay_inside_the_frame() {
    let f = frame(&windowed(MIB), &[block(b"abc", &[seq(3, 3, 4 + 3)], true)]);
    assert_eq!(decompress(&f, 6), Err(Error::OffsetBeforeStart { offset: 4, available: 3 }));
    // A frame's matches do not reach into the frame before.
    let mut src = frame(&windowed(MIB), &[raw(b"abcdefgh", true)]);
    src.extend(frame(&windowed(MIB), &[block(b"x", &[seq(1, 3, 4 + 3)], true)]));
    assert_eq!(decompress(&src, 12), Err(Error::OffsetBeforeStart { offset: 4, available: 1 }));
}

#[test]
fn matches_stay_inside_the_window() {
    // A 1 KiB window, 1025 bytes back to the start of the frame.
    let data = random(1024, 11);
    for offset in [1024, 1025] {
        let blocks = [raw(&data, false), block(b"x", &[seq(1, 3, offset as u32 + 3)], true)];
        let r = decompress(&frame(&windowed(0), &blocks), 1028);
        if offset == 1024 {
            let mut want = data.clone();
            want.push(b'x');
            copy(&mut want, 1024, 3);
            assert_eq!(r.unwrap(), want);
        } else {
            assert_eq!(r, Err(Error::OffsetPastWindow { offset: 1025, window: 1024 }));
        }
    }
}

#[test]
fn sequences_use_only_the_literals_there_are() {
    let f = frame(&windowed(MIB), &[block(b"ab", &[seq(3, 3, 1)], true)]);
    assert_eq!(decompress(&f, 6), Err(Error::LiteralsOverrun));
}

#[test]
fn the_sequences_stream_ends_where_its_sequences_do() {
    // A byte before the stream's bits is never read.
    let mut s = sequences(&[seq(4, 4, 4 + 3)], Tables::Rle);
    s.insert(5, 0xab);
    let f = frame(&windowed(MIB), &[compressed(&raw_literals(b"abcd"), &s, true)]);
    assert_eq!(decompress(&f, 8), Err(Error::StreamNotConsumed));
    // A stream whose last byte is 0 has no end mark.
    let mut s = sequences(&[seq(4, 4, 4 + 3)], Tables::Rle);
    s.push(0);
    let f = frame(&windowed(MIB), &[compressed(&raw_literals(b"abcd"), &s, true)]);
    assert_eq!(decompress(&f, 8), Err(Error::NoEndMark));
}

/// The number of sequences in 1, 2 and 3 bytes. Each sequence is one
/// literal and a match of 3 from offset 1, so the tables in RLE mode
/// read no bits and the stream is its end mark alone.
#[test]
fn sequence_counts_in_each_form() {
    for n in [100, 300, 32_600] {
        let lits = random(n, 12);
        let seqs = vec![seq(1, 3, 1); n];
        let s = sequences(&seqs, Tables::Rle);
        let header = [1, 2, 3][usize::from(n >= 128) + usize::from(n >= 0x7f00)];
        assert_eq!(s.len(), header + 4 + 1);
        let f = frame(&single(4 * n as u64), &[compressed(&raw_literals(&lits), &s, true)]);
        let mut want = Vec::new();
        for &b in &lits {
            want.push(b);
            copy(&mut want, 1, 3);
        }
        assert_eq!(decompress(&f, 4 * n).unwrap(), want, "{n} sequences");
    }
}

#[test]
fn no_sequences() {
    let f = frame(&windowed(MIB), &[compressed(&raw_literals(b"hello"), &count(0), true)]);
    assert_eq!(decompress(&f, 5).unwrap(), b"hello");
    let f = frame(&windowed(MIB), &[compressed(&raw_literals(b"hello"), &[0, 0], true)]);
    assert_eq!(decompress(&f, 5), Err(Error::BytesAfterNoSequences));
}

/// A count of 0 in two bytes is not the 1-byte 0: the modes and tables
/// follow it and replace the frame's tables, and the bytes after them are
/// not read.
#[test]
fn a_count_of_zero_in_two_bytes() {
    let first = [seq(4, 4, 4 + 3)];
    let then = [seq(2, 3, 2 + 3)];
    // The modes and RLE codes of `then`, and a byte of 0 after them, which
    // as a bitstream would have no end mark.
    let tables = &sequences(&then, Tables::Rle)[1..5];
    let zero = [&[0x80, 0][..], tables, &[0]].concat();
    let blocks = [
        block(b"abcd", &first, false),
        compressed(&raw_literals(b"-"), &zero, false),
        compressed(&raw_literals(b"ef"), &sequences(&then, Tables::Repeat), true),
    ];
    let mut want = b"abcd".to_vec();
    copy(&mut want, 4, 4);
    want.extend(b"-ef");
    copy(&mut want, 2, 3);
    let f = frame(&windowed(MIB), &blocks);
    assert_eq!(decompress(&f, want.len()).unwrap(), want);
    // Without the modes the section ends inside its header.
    let f = frame(&windowed(MIB), &[compressed(&raw_literals(b"abcd"), &[0x80, 0], true)]);
    assert_eq!(decompress(&f, 4), Err(Error::Truncated(Part::SequencesHeader)));
}

/// Repeat mode takes the tables of the last block that set them, across
/// blocks with no sequences, but not across frames.
#[test]
fn repeat_mode() {
    let seqs = [seq(4, 4, 4 + 3)];
    let blocks = [
        block(b"abcd", &seqs, false),
        compressed(&raw_literals(b"-"), &count(0), false),
        compressed(&raw_literals(b"efgh"), &sequences(&seqs, Tables::Repeat), true),
    ];
    let want = b"abcdabcd-efghefgh";
    assert_eq!(decompress(&frame(&windowed(MIB), &blocks), want.len()).unwrap(), want);
    // In a frame's first block there is nothing to repeat, even after
    // another frame.
    let repeat = frame(&windowed(MIB), &[compressed(&raw_literals(b"efgh"), &sequences(&seqs, Tables::Repeat), true)]);
    assert_eq!(decompress(&repeat, 8), Err(Error::NoTableToRepeat(Symbol::LiteralLength)));
    let mut src = frame(&windowed(MIB), &blocks);
    src.extend(&repeat);
    assert_eq!(decompress(&src, want.len() + 8), Err(Error::NoTableToRepeat(Symbol::LiteralLength)));
    // Each table on its own: literal lengths in RLE mode, offsets in
    // repeat mode.
    let s = [1, 1 << 6 | 3 << 4 | 1 << 2, 4, 1, 1];
    let f = frame(&windowed(MIB), &[compressed(&raw_literals(b"abcd"), &s, true)]);
    assert_eq!(decompress(&f, 8), Err(Error::NoTableToRepeat(Symbol::Offset)));
}

#[test]
fn modes_and_codes_past_the_last() {
    let section = |s: &[u8]| frame(&windowed(MIB), &[compressed(&raw_literals(b"abcd"), s, true)]);
    assert_eq!(decompress(&section(&[1, 0x55, 4, 2, 1, 1]), 8), Err(Error::ReservedModeBits));
    let rle = 0x54;
    let past = [
        ([36, 2, 1], Symbol::LiteralLength, 36),
        ([4, 32, 1], Symbol::Offset, 32),
        ([4, 2, 53], Symbol::MatchLength, 53),
    ];
    for (codes, symbol, code) in past {
        let s = [&[1, rle][..], &codes, &[1]].concat();
        assert_eq!(decompress(&section(&s), 8), Err(Error::CodePastMax { symbol, code }));
    }
}

/// Treeless literals need a Huffman table from an earlier block of the
/// same frame.
#[test]
fn treeless_literals_need_a_table() {
    // Treeless, one stream, 3 literals from 1 byte.
    let h: u32 = 0b11 | 3 << 4 | 1 << 14;
    let lits = [&h.to_le_bytes()[..3], &[0b1000_0001]].concat();
    let treeless = frame(&windowed(MIB), &[compressed(&lits, &count(0), true)]);
    assert_eq!(decompress(&treeless, 3), Err(Error::NoHuffmanTable));
    // text-1.zst's literals are Huffman coded, and the next frame starts
    // with no table.
    let mut src = fixture("text-1.zst");
    src.extend(&treeless);
    assert_eq!(decompress(&src, 150_003), Err(Error::NoHuffmanTable));
}
