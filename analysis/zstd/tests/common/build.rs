//! Frames built by hand, for the rules of the format the `zstd` tool's
//! output does not reach or cannot break on purpose: each header field,
//! each block type, and sequences sections whose three tables are in RLE
//! or repeat mode, so every sequence's codes are fixed and only its
//! extra bits are written.

pub const MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

/// A frame: the magic number, `header` and the blocks.
pub fn frame(header: &[u8], blocks: &[Vec<u8>]) -> Vec<u8> {
    let mut f = MAGIC.to_vec();
    f.extend(header);
    for b in blocks {
        f.extend(b);
    }
    f
}

/// A single-segment header for `content` bytes, in the shortest form.
pub fn single(content: u64) -> Vec<u8> {
    let mut h = Vec::new();
    match content {
        0..256 => h.extend([0x20, content as u8]),
        256..65792 => {
            h.push(0x60);
            h.extend(((content - 256) as u16).to_le_bytes());
        }
        65792..0x1_0000_0000 => {
            h.push(0xa0);
            h.extend((content as u32).to_le_bytes());
        }
        _ => {
            h.push(0xe0);
            h.extend(content.to_le_bytes());
        }
    }
    h
}

/// A header with the window descriptor `window` and no content size.
pub fn windowed(window: u8) -> Vec<u8> {
    vec![0x00, window]
}

fn block_header(kind: u32, size: usize, last: bool) -> Vec<u8> {
    let h = u32::from(last) | kind << 1 | (size as u32) << 3;
    h.to_le_bytes()[..3].to_vec()
}

pub fn raw(data: &[u8], last: bool) -> Vec<u8> {
    let mut b = block_header(0, data.len(), last);
    b.extend(data);
    b
}

pub fn rle(byte: u8, n: usize, last: bool) -> Vec<u8> {
    let mut b = block_header(1, n, last);
    b.push(byte);
    b
}

/// A compressed block of a literals section and a sequences section.
pub fn compressed(literals: &[u8], sequences: &[u8], last: bool) -> Vec<u8> {
    let mut b = block_header(2, literals.len() + sequences.len(), last);
    b.extend(literals);
    b.extend(sequences);
    b
}

/// A raw literals section, its header in the shortest form.
pub fn raw_literals(lits: &[u8]) -> Vec<u8> {
    let n = lits.len();
    let mut s = match n {
        0..32 => vec![(n << 3) as u8],
        32..4096 => vec![0b0100 | (n << 4) as u8, (n >> 4) as u8],
        _ => vec![0b1100 | (n << 4) as u8, (n >> 4) as u8, (n >> 12) as u8],
    };
    s.extend(lits);
    s
}

/// An RLE literals section of `n` copies of `byte`.
pub fn rle_literals(byte: u8, n: usize) -> Vec<u8> {
    vec![0b1101 | (n << 4) as u8, (n >> 4) as u8, (n >> 12) as u8, byte]
}

/// The format's literal length and match length codes: each code's
/// baseline and the bits added to it.
const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
];
const LL_BITS: [u32; 36] =
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539,
];
const ML_BITS: [u32; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
    3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

/// A value as its code, the extra bits and their number.
fn code(base: &[u32], bits: &[u32], v: u32) -> (u8, u32, u32) {
    let c = base.iter().rposition(|&b| b <= v).unwrap();
    assert!(v - base[c] < 1 << bits[c], "{v} past the last code");
    (c as u8, v - base[c], bits[c])
}

/// One sequence: its literal length, match length and offset value (an
/// offset plus 3, or 1 to 3 for a repeated offset).
#[derive(Clone, Copy)]
pub struct Seq {
    pub ll: u32,
    pub ml: u32,
    pub offset: u32,
}

impl Seq {
    /// The three codes, extra bits and widths: literal length, offset,
    /// match length.
    fn codes(&self) -> [(u8, u32, u32); 3] {
        let of = 31 - self.offset.leading_zeros();
        [code(&LL_BASE, &LL_BITS, self.ll), (of as u8, self.offset - (1 << of), of), code(&ML_BASE, &ML_BITS, self.ml)]
    }
}

/// How a sequences section sets its tables.
pub enum Tables {
    /// RLE mode, the codes of the first sequence written out.
    Rle,
    /// Repeat mode: the tables of the block before.
    Repeat,
}

/// Bits written forward, low bit first, as a backward stream is laid out.
#[derive(Default)]
pub struct Bits {
    bits: Vec<bool>,
}

impl Bits {
    pub fn put(&mut self, v: u32, n: u32) {
        self.bits.extend((0..n).map(|i| v >> i & 1 == 1));
    }

    /// The stream with its end mark.
    pub fn finish(mut self) -> Vec<u8> {
        self.bits.push(true);
        self.bits.chunks(8).map(|c| c.iter().rev().fold(0, |b, &x| b << 1 | u8::from(x))).collect()
    }
}

/// The number of sequences in the shortest of the three forms.
pub fn count(n: usize) -> Vec<u8> {
    match n {
        0..128 => vec![n as u8],
        128..0x7f00 => vec![0x80 | (n >> 8) as u8, n as u8],
        _ => vec![0xff, (n - 0x7f00) as u8, ((n - 0x7f00) >> 8) as u8],
    }
}

/// A sequences section of `seqs`, which must all have the same codes.
/// The tables read no bits, so the stream holds only the extra bits:
/// per sequence the offset's, the match length's and the literal
/// length's in that order, read from the end.
pub fn sequences(seqs: &[Seq], tables: Tables) -> Vec<u8> {
    let codes = seqs[0].codes();
    let mut s = count(seqs.len());
    match tables {
        Tables::Rle => s.extend([0x54, codes[0].0, codes[1].0, codes[2].0]),
        Tables::Repeat => s.push(0xfc),
    }
    let mut bits = Bits::default();
    for q in seqs.iter().rev() {
        let c = q.codes();
        assert!(c.iter().zip(&codes).all(|(a, b)| a.0 == b.0), "sequences with different codes");
        let [ll, of, ml] = c;
        bits.put(ll.1, ll.2);
        bits.put(ml.1, ml.2);
        bits.put(of.1, of.2);
    }
    s.extend(bits.finish());
    s
}
