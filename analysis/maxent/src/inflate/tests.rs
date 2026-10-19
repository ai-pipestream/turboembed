use super::*;

fn message(result: Result<Vec<u8>>) -> &'static str {
    match result {
        Err(Error::Inflate(message)) => message,
        other => panic!("expected an inflate error, got {other:?}"),
    }
}

/// Writes bits least significant first, as deflate packs them, for
/// building small streams by hand.
struct BitWriter {
    bytes: Vec<u8>,
    bit: u32,
}

impl BitWriter {
    fn new() -> BitWriter {
        BitWriter { bytes: Vec::new(), bit: 0 }
    }

    fn put(&mut self, value: u32, n: u32) {
        for i in 0..n {
            if self.bit == 0 {
                self.bytes.push(0);
            }
            let b = (value >> i) & 1;
            *self.bytes.last_mut().unwrap() |= (b as u8) << self.bit;
            self.bit = (self.bit + 1) % 8;
        }
    }

    /// Writes a Huffman code most significant bit first.
    fn code(&mut self, code: u32, len: u32) {
        for i in (0..len).rev() {
            self.put((code >> i) & 1, 1);
        }
    }

    /// A fixed Huffman literal or length symbol.
    fn fixed(&mut self, symbol: u32) {
        match symbol {
            0..=143 => self.code(0x30 + symbol, 8),
            144..=255 => self.code(0x190 + symbol - 144, 9),
            256..=279 => self.code(symbol - 256, 7),
            _ => self.code(0xC0 + symbol - 280, 8),
        }
    }
}

/// Starts a final dynamic block declaring `257 + hlit` length codes
/// and `1 + hdist` distance codes, with a code length code over the
/// symbols 0 and 8 (two bits each), 16 (two bits), and 17 and 18
/// (three bits each). Write code lengths with [`code_length`].
fn dynamic_header(w: &mut BitWriter, hlit: u32, hdist: u32) {
    w.put(1, 1);
    w.put(2, 2);
    w.put(hlit, 5);
    w.put(hdist, 5);
    w.put(1, 4); // five code length lengths: symbols 16, 17, 18, 0, 8
    for len in [2, 3, 3, 2, 2] {
        w.put(len, 3);
    }
}

/// One code length symbol under the code `dynamic_header` declares,
/// with its extra bits.
fn code_length(w: &mut BitWriter, symbol: u32, extra: u32) {
    match symbol {
        0 => w.code(0b00, 2),
        8 => w.code(0b01, 2),
        16 => {
            w.code(0b10, 2);
            w.put(extra, 2);
        }
        17 => {
            w.code(0b110, 3);
            w.put(extra, 3);
        }
        18 => {
            w.code(0b111, 3);
            w.put(extra, 7);
        }
        _ => unreachable!("the code has no symbol {symbol}"),
    }
}

#[test]
fn a_huge_stated_size_is_refused_before_allocating() {
    // 0xFF is a final block of the reserved type, but the stated size
    // is checked first: 4 GiB must not be allocated for it.
    assert_eq!(message(inflate(&vec![0xFF; 4_200_000], 0xFFFF_FFFE)), "the stated size is above the 1 GiB limit");
    assert_eq!(message(inflate(&[0xFF], MAX_SIZE + 1)), "the stated size is above the 1 GiB limit");
}

#[test]
fn a_stated_size_the_stream_does_not_produce_allocates_little() {
    // The size is under the limit and the input ratio, so only the
    // initial capacity, 64 bytes per input byte, is reserved before the
    // reserved block type is found.
    assert_eq!(message(inflate(&vec![0xFF; 1 << 20], 1 << 30)), "a block has the reserved type 3");
}

#[test]
fn a_dynamic_block_without_an_end_of_block_code() {
    let mut w = BitWriter::new();
    dynamic_header(&mut w, 0, 0);
    code_length(&mut w, 18, 127); // 138 zeros
    code_length(&mut w, 18, 109); // 120 zeros: 258 lengths, all zero
    assert_eq!(message(inflate(&w.bytes, 0)), "a dynamic block has no end-of-block code");
}

#[test]
fn a_repeat_code_with_no_previous_length() {
    let mut w = BitWriter::new();
    dynamic_header(&mut w, 0, 0);
    code_length(&mut w, 16, 0);
    assert_eq!(message(inflate(&w.bytes, 0)), "a repeat code has no previous length");
}

#[test]
fn code_lengths_that_repeat_past_the_end() {
    let mut w = BitWriter::new();
    dynamic_header(&mut w, 0, 0);
    code_length(&mut w, 8, 0);
    code_length(&mut w, 18, 127); // 139 lengths
    code_length(&mut w, 18, 127); // 277, past the 258 declared
    assert_eq!(message(inflate(&w.bytes, 0)), "code lengths repeat past the end of the list");
}

#[test]
fn too_many_length_or_distance_codes() {
    let mut w = BitWriter::new();
    dynamic_header(&mut w, 30, 0); // 287 length codes
    assert_eq!(message(inflate(&w.bytes, 0)), "a dynamic block declares more than 286 length or 30 distance codes");
    let mut w = BitWriter::new();
    dynamic_header(&mut w, 0, 30); // 31 distance codes
    assert_eq!(message(inflate(&w.bytes, 0)), "a dynamic block declares more than 286 length or 30 distance codes");
}

#[test]
fn length_symbols_286_and_287() {
    for symbol in [286, 287] {
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(1, 2);
        w.fixed(u32::from(b'a'));
        w.fixed(symbol);
        w.fixed(256);
        assert_eq!(message(inflate(&w.bytes, 10)), "a length symbol is 286 or 287", "symbol {symbol}");
    }
}

#[test]
fn distance_symbols_30_and_31() {
    for symbol in [30, 31] {
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(1, 2);
        w.fixed(u32::from(b'a'));
        w.fixed(257); // length 3
        w.code(symbol, 5);
        w.fixed(256);
        assert_eq!(message(inflate(&w.bytes, 4)), "a distance symbol is 30 or 31", "symbol {symbol}");
    }
}

#[test]
fn a_fixed_block_with_an_overlapping_copy() {
    // "ab" then a copy of length 6 at distance 2: "abababab".
    let mut w = BitWriter::new();
    w.put(1, 1);
    w.put(1, 2);
    w.fixed(u32::from(b'a'));
    w.fixed(u32::from(b'b'));
    w.fixed(260); // length 6
    w.code(1, 5); // distance 2
    w.fixed(256);
    assert_eq!(inflate(&w.bytes, 8).unwrap(), b"abababab");
}

#[test]
fn a_run_at_distance_one() {
    let mut w = BitWriter::new();
    w.put(1, 1);
    w.put(1, 2);
    w.fixed(u32::from(b'z'));
    w.fixed(285); // length 258
    w.code(0, 5); // distance 1
    w.fixed(256);
    assert_eq!(inflate(&w.bytes, 259).unwrap(), vec![b'z'; 259]);
}

#[test]
fn stored_blocks_including_an_empty_one() {
    let mut bytes = vec![0b000, 3, 0, !3, 0xFF, b'a', b'b', b'c'];
    bytes.extend_from_slice(&[0b001, 0, 0, 0xFF, 0xFF]);
    assert_eq!(inflate(&bytes, 3).unwrap(), b"abc");
}

#[test]
fn a_distance_before_the_start_of_the_output() {
    let mut w = BitWriter::new();
    w.put(1, 1);
    w.put(1, 2);
    w.fixed(u32::from(b'a'));
    w.fixed(257); // length 3
    w.code(1, 5); // distance 2, but only one byte is out
    w.fixed(256);
    assert_eq!(message(inflate(&w.bytes, 4)), "a distance reaches before the start of the output");
}

#[test]
fn a_stored_length_that_disagrees_with_its_complement() {
    let bytes = [0b001, 3, 0, 0, 0, b'a', b'b', b'c'];
    assert_eq!(message(inflate(&bytes, 3)), "a stored block's length and its complement disagree");
}

#[test]
fn the_reserved_block_type() {
    assert_eq!(message(inflate(&[0b111], 0)), "a block has the reserved type 3");
}

#[test]
fn truncated_input() {
    let mut w = BitWriter::new();
    w.put(1, 1);
    w.put(1, 2);
    w.fixed(u32::from(b'a'));
    // No end-of-block code follows; the remaining bits of the byte are
    // zero, which is the start of a seven-bit code with no input left.
    assert_eq!(message(inflate(&w.bytes, 1)), "the input ends inside a block");
    assert_eq!(message(inflate(&[0b000, 3, 0], 3)), "the input ends inside a stored block header");
    assert_eq!(message(inflate(&[0b001, 3, 0, !3, 0xFF, b'a'], 3)), "the input ends inside a stored block");
}

#[test]
fn output_longer_or_shorter_than_stated() {
    let bytes = [0b001, 3, 0, !3, 0xFF, b'a', b'b', b'c'];
    assert_eq!(message(inflate(&bytes, 2)), "the stream produces more than the stated size");
    assert_eq!(message(inflate(&bytes, 4)), "the stream ends before the stated size");
    assert_eq!(message(inflate(&bytes, 1 << 30)), "the stated size is more than the input can produce");
}

#[test]
fn an_over_subscribed_code_length_code() {
    // A dynamic block whose code length code gives four symbols one bit.
    let mut w = BitWriter::new();
    w.put(1, 1);
    w.put(2, 2);
    w.put(0, 5);
    w.put(0, 5);
    w.put(0, 4); // four code length lengths follow: symbols 16, 17, 18, 0
    for _ in 0..4 {
        w.put(1, 3);
    }
    assert_eq!(message(inflate(&w.bytes, 0)), "a Huffman code is over-subscribed");
}

#[test]
fn an_incomplete_code_length_code() {
    let mut w = BitWriter::new();
    w.put(1, 1);
    w.put(2, 2);
    w.put(0, 5);
    w.put(0, 5);
    w.put(0, 4);
    w.put(2, 3); // symbol 16 gets two bits, nothing else: incomplete
    w.put(0, 3);
    w.put(0, 3);
    w.put(0, 3);
    assert_eq!(message(inflate(&w.bytes, 0)), "a Huffman code is incomplete");
}

#[test]
fn codes_longer_than_the_table_use_the_canonical_loop() {
    // A code where symbol 0 has one bit, then 1, 2, ... down to two
    // fifteen-bit codes: lengths 1..=14 for symbols 0..=13 and 15 for
    // symbols 14 and 15. That is complete.
    let mut lengths = [0u8; 16];
    for (i, len) in lengths.iter_mut().enumerate() {
        *len = (i as u8 + 1).min(15);
    }
    let mut h = Huffman::new();
    h.build(&lengths, CodeKind::LiteralsOrDistances).unwrap();
    // Canonical codes: symbol k < 14 is k ones then a zero; symbols 14
    // and 15 are fourteen ones then 0 and 1.
    for symbol in 0..16u32 {
        let len = u32::from(lengths[symbol as usize]);
        let code = match symbol {
            0..=13 => ((1u32 << symbol) - 1) << 1,
            14 => ((1u32 << 14) - 1) << 1,
            _ => (1u32 << 15) - 1,
        };
        let mut w = BitWriter::new();
        w.code(code, len);
        let mut bits = Bits { data: &w.bytes, pos: 0, buf: 0, count: 0 };
        assert_eq!(u32::from(h.decode(&mut bits).unwrap()), symbol, "symbol {symbol}");
    }
}
