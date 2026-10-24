//! ICU4X's bidirectional LSTM word segmenter for complex scripts.
//!
//! Each `.lstm` file holds a code-point alphabet and the `f32` weights of a
//! single-layer bidirectional LSTM (`embed_dim = 40`, `hunits = 27`, 4 output
//! classes `B`, `I`, `E`, `S`), as stored in ICU4X's
//! `segmenter_lstm_auto_v1.rs.data` (`ModelType::Codepoints`).
//!
//! The file format (`.lstm`), little-endian:
//!
//! | Bytes | Contents |
//! |---|---|
//! | 4 | the magic `TZL1` |
//! | 4 | the number of alphabet ranges `R`, at most 255 |
//! | `8 R` | each range: its first code point and the code point after its last |
//! | 4 | `embed_dim` (`40`) |
//! | 4 | `hunits` (`27`) |
//! | rest | the 9 `f32` weight arrays (`embedding`, `fw_w`, `fw_u`, `fw_b`, `bw_w`, `bw_u`, `bw_b`, `time_w`, `time_b`) |
//!
//! The ranges are sorted and disjoint and hold `V <= 255` code points in all;
//! a character's cluster id is the number of alphabet characters below it, or
//! `V` when it lies outside the alphabet. When [`Lstm::from_bytes`] loads a
//! model, it precomputes `b + W * embedding[id]` for each of the `V + 1`
//! cluster ids (`fw_bx` and `bw_bx`), eliminating the `[108, 40]` input
//! matrix-vector product at every character step while keeping bit-for-bit
//! IEEE-754 `f32` parity with ICU4X's `compute_hc`.
//!
//! [`Lstm::segment`] is ICU4X's `LstmSegmenterIterator` (ICU4X,
//! `components/segmenter/src/complex/lstm/mod.rs`, Unicode-3.0), with the same
//! grapheme-cluster invariant as [`Dictionary::segment`](super::Dictionary::segment):
//! any boundary that would fall inside an extended grapheme cluster moves to
//! the end of that cluster.

use std::cell::RefCell;
use std::fmt;

use turbo_unicode::grapheme_boundaries;

/// The first four bytes of an LSTM file.
pub const MAGIC: [u8; 4] = *b"TZL1";

/// The embedding dimension of the bundled ICU4X LSTM models.
pub const EMBED_DIM: usize = 40;

/// The hidden state dimension of the bundled ICU4X LSTM models.
pub const HUNITS: usize = 27;

/// Number of LSTM gates (`i`, `f`, `c`, `o`) times [`HUNITS`].
const GATES_HUNITS: usize = 4 * HUNITS;

/// The most characters an alphabet can hold (`V <= 255`, so `V` fits in `u8`).
const ALPHABET_MAX: u32 = 255;

/// Why an `.lstm` file was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LstmError {
    /// The file does not start with [`MAGIC`].
    Magic,
    /// The file ends inside its header.
    Truncated,
    /// The alphabet has more than 255 ranges or characters, or its ranges
    /// are empty, unsorted, overlapping or outside the code space.
    Alphabet,
    /// `embed_dim` or `hunits` does not match [`EMBED_DIM`] and [`HUNITS`].
    Dimensions,
    /// The weight bytes do not have the exact size required by the header.
    Weights,
}

impl fmt::Display for LstmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LstmError::Magic => "not an LSTM file: the magic TZL1 is missing",
            LstmError::Truncated => "the LSTM file ends inside its header",
            LstmError::Alphabet => "the LSTM alphabet is malformed",
            LstmError::Dimensions => "the LSTM dimensions are not 40 and 27",
            LstmError::Weights => "the LSTM weight byte length does not match its dimensions",
        })
    }
}

impl std::error::Error for LstmError {}

/// A bidirectional LSTM segmenter read from a `.lstm` file.
#[derive(Clone, Debug)]
pub struct Lstm<'a> {
    /// Marker borrowing the source slice so [`Lstm<'a>`] has the same lifetime
    /// signature as [`Dictionary<'a>`](super::Dictionary).
    _marker: std::marker::PhantomData<&'a [u8]>,
    /// The alphabet ranges, as (first, after last, position of first).
    ranges: Vec<(u32, u32, u8)>,
    /// The cluster id assigned to a character outside the alphabet (`V`).
    outside: u8,
    /// Precomputed `fw_b + fw_w * embedding[id]` for each `id in 0..=V`.
    fw_bx: Vec<[f32; GATES_HUNITS]>,
    /// Precomputed `bw_b + bw_w * embedding[id]` for each `id in 0..=V`.
    bw_bx: Vec<[f32; GATES_HUNITS]>,
    /// Forward recurrent weights `[4 * HUNITS][HUNITS]`.
    fw_u: Box<[[f32; HUNITS]; GATES_HUNITS]>,
    /// Backward recurrent weights `[4 * HUNITS][HUNITS]`.
    bw_u: Box<[[f32; HUNITS]; GATES_HUNITS]>,
    /// Time-distributed output weights `[2][4][HUNITS]`.
    time_w: Box<[[[f32; HUNITS]; 4]; 2]>,
    /// Time-distributed output bias `[4]`.
    time_b: [f32; 4],
}

/// Reusable scratch buffers for [`Lstm::segment_with`], so repeated calls
/// allocate nothing once the vectors have grown to the longest complex-script
/// run.
#[derive(Clone, Debug, Default)]
pub struct LstmScratch {
    chars: Vec<(u8, usize)>,
    h_fw: Vec<[f32; HUNITS]>,
    h_bw: Vec<[f32; HUNITS]>,
}

thread_local! {
    static SCRATCH: RefCell<LstmScratch> = RefCell::new(LstmScratch::default());
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, LstmError> {
    let word = bytes.get(at..at + 4).ok_or(LstmError::Truncated)?;
    Ok(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
}

#[inline]
fn f32_at(bytes: &[u8], i: usize) -> f32 {
    let at = 4 * i;
    f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// ICU4X's 8-way unrolled dot product (`unrolled_dot_2` in `matrix.rs`).
#[inline]
fn dot_bytes(a_bytes: &[u8], b_bytes: &[u8], len: usize) -> f32 {
    let mut p = [0.0f32; 8];
    let chunks = len / 8;
    let mut sum = 0.0f32;
    for i in chunks * 8..len {
        sum += f32_at(a_bytes, i) * f32_at(b_bytes, i);
    }
    for c in 0..chunks {
        let base = c * 8;
        for (j, slot) in p.iter_mut().enumerate() {
            *slot += f32_at(a_bytes, base + j) * f32_at(b_bytes, base + j);
        }
    }
    sum + (p[0] + p[4]) + (p[1] + p[5]) + (p[2] + p[6]) + (p[3] + p[7])
}

/// ICU4X's 8-way unrolled dot product over `[f32; 27]` (`unrolled_dot_1` in `matrix.rs`).
#[inline]
fn dot_hunits(a: &[f32; HUNITS], b: &[f32; HUNITS]) -> f32 {
    let sum = a[24] * b[24] + a[25] * b[25] + a[26] * b[26];
    let p0 = (a[0] * b[0] + a[8] * b[8] + a[16] * b[16]) + (a[4] * b[4] + a[12] * b[12] + a[20] * b[20]);
    let p1 = (a[1] * b[1] + a[9] * b[9] + a[17] * b[17]) + (a[5] * b[5] + a[13] * b[13] + a[21] * b[21]);
    let p2 = (a[2] * b[2] + a[10] * b[10] + a[18] * b[18]) + (a[6] * b[6] + a[14] * b[14] + a[22] * b[22]);
    let p3 = (a[3] * b[3] + a[11] * b[11] + a[19] * b[19]) + (a[7] * b[7] + a[15] * b[15] + a[23] * b[23]);
    sum + p0 + p1 + p2 + p3
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// One recurrent step of ICU4X's `compute_hc` using precomputed `bx = b + W * x`.
#[inline]
fn step(bx: &[f32; GATES_HUNITS], h: &mut [f32; HUNITS], c: &mut [f32; HUNITS], u: &[[f32; HUNITS]; GATES_HUNITS]) {
    let mut s = *bx;
    for (s_row, u_row) in s.iter_mut().zip(u.iter()) {
        *s_row += dot_hunits(h, u_row);
    }
    for i in 0..HUNITS {
        let i_t = sigmoid(s[i]);
        let f_t = sigmoid(s[HUNITS + i]);
        let c_tilde = s[2 * HUNITS + i].tanh();
        let o_t = sigmoid(s[3 * HUNITS + i]);
        let c_new = i_t * c_tilde + c[i] * f_t;
        c[i] = c_new;
        h[i] = o_t * c_new.tanh();
    }
}

fn precompute_bx(vocab: usize, embedding: &[u8], w_bytes: &[u8], b_bytes: &[u8]) -> Vec<[f32; GATES_HUNITS]> {
    let mut out = Vec::with_capacity(vocab);
    for id in 0..vocab {
        let x_bytes = &embedding[id * EMBED_DIM * 4..(id + 1) * EMBED_DIM * 4];
        let mut row_vals = [0.0f32; GATES_HUNITS];
        for (row, dst) in row_vals.iter_mut().enumerate() {
            let b_val = f32_at(b_bytes, row);
            let wx = dot_bytes(x_bytes, &w_bytes[row * EMBED_DIM * 4..(row + 1) * EMBED_DIM * 4], EMBED_DIM);
            *dst = b_val + wx;
        }
        out.push(row_vals);
    }
    out
}

fn unpack_u(u_bytes: &[u8]) -> Box<[[f32; HUNITS]; GATES_HUNITS]> {
    let mut u = Box::new([[0.0f32; HUNITS]; GATES_HUNITS]);
    for (row, dst_row) in u.iter_mut().enumerate() {
        for (col, dst) in dst_row.iter_mut().enumerate() {
            *dst = f32_at(u_bytes, row * HUNITS + col);
        }
    }
    u
}

impl<'a> Lstm<'a> {
    /// Reads an LSTM model from the bytes of a `.lstm` file.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Lstm<'a>, LstmError> {
        if bytes.get(..4) != Some(&MAGIC[..]) {
            return Err(LstmError::Magic);
        }
        let count = read_u32(bytes, 4)?;
        if count > ALPHABET_MAX {
            return Err(LstmError::Alphabet);
        }
        let mut ranges = Vec::with_capacity(count as usize);
        let mut size = 0u32;
        let mut previous_end = 0u32;
        for i in 0..count as usize {
            let first = read_u32(bytes, 8 + 8 * i)?;
            let end = read_u32(bytes, 12 + 8 * i)?;
            if first >= end || end > 0x11_0000 || (i > 0 && first < previous_end) {
                return Err(LstmError::Alphabet);
            }
            if size + (end - first) > ALPHABET_MAX {
                return Err(LstmError::Alphabet);
            }
            ranges.push((first, end, size as u8));
            size += end - first;
            previous_end = end;
        }
        let dims_at = 8 + 8 * count as usize;
        let embed_dim = read_u32(bytes, dims_at)?;
        let hunits = read_u32(bytes, dims_at + 4)?;
        if embed_dim != EMBED_DIM as u32 || hunits != HUNITS as u32 {
            return Err(LstmError::Dimensions);
        }
        let rest = bytes.get(dims_at + 8..).ok_or(LstmError::Truncated)?;
        let vocab = size as usize + 1;
        let emb_bytes = vocab * EMBED_DIM * 4;
        let w_bytes = GATES_HUNITS * EMBED_DIM * 4;
        let u_bytes = GATES_HUNITS * HUNITS * 4;
        let b_bytes = GATES_HUNITS * 4;
        let tw_bytes = 2 * 4 * HUNITS * 4;
        let tb_bytes = 4 * 4;
        let expected = emb_bytes + 2 * (w_bytes + u_bytes + b_bytes) + tw_bytes + tb_bytes;
        if rest.len() != expected {
            return Err(LstmError::Weights);
        }
        let (embedding, rest) = rest.split_at(emb_bytes);
        let (fw_w, rest) = rest.split_at(w_bytes);
        let (fw_u_raw, rest) = rest.split_at(u_bytes);
        let (fw_b, rest) = rest.split_at(b_bytes);
        let (bw_w, rest) = rest.split_at(w_bytes);
        let (bw_u_raw, rest) = rest.split_at(u_bytes);
        let (bw_b, rest) = rest.split_at(b_bytes);
        let (time_w_raw, time_b_raw) = rest.split_at(tw_bytes);

        let fw_bx = precompute_bx(vocab, embedding, fw_w, fw_b);
        let bw_bx = precompute_bx(vocab, embedding, bw_w, bw_b);
        let fw_u = unpack_u(fw_u_raw);
        let bw_u = unpack_u(bw_u_raw);
        let mut time_w = Box::new([[[0.0f32; HUNITS]; 4]; 2]);
        for (d, dir) in time_w.iter_mut().enumerate() {
            for (k, row) in dir.iter_mut().enumerate() {
                for (i, dst) in row.iter_mut().enumerate() {
                    *dst = f32_at(time_w_raw, (d * 4 + k) * HUNITS + i);
                }
            }
        }
        let mut time_b = [0.0f32; 4];
        for (k, dst) in time_b.iter_mut().enumerate() {
            *dst = f32_at(time_b_raw, k);
        }

        Ok(Lstm {
            _marker: std::marker::PhantomData,
            ranges,
            outside: size as u8,
            fw_bx,
            bw_bx,
            fw_u,
            bw_u,
            time_w,
            time_b,
        })
    }

    /// The number of characters in the alphabet (`V`; the embedding table has
    /// `V + 1` rows).
    pub fn alphabet_len(&self) -> usize {
        usize::from(self.outside)
    }

    /// The cluster id of `c` in `0..=V`.
    #[inline]
    fn cluster_id(&self, c: char) -> u8 {
        let cp = c as u32;
        for &(first, end, base) in &self.ranges {
            if cp < first {
                break;
            }
            if cp < end {
                return base + (cp - first) as u8;
            }
        }
        self.outside
    }

    /// The word boundaries of `text` that the LSTM finds, as byte offsets
    /// after each word, in increasing order; the last is the end of the text.
    /// Nothing is reported for an empty text.
    pub fn segment(&self, text: &str, boundary: impl FnMut(usize)) {
        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut scratch) => self.segment_with(text, &mut scratch, boundary),
            Err(_) => self.segment_with(text, &mut LstmScratch::default(), boundary),
        });
    }

    /// Runs [`Lstm::segment`] reusing `scratch`, so repeated calls allocate
    /// nothing once `scratch` has grown to the longest complex-script run.
    pub fn segment_with(&self, text: &str, scratch: &mut LstmScratch, mut boundary: impl FnMut(usize)) {
        if text.is_empty() {
            return;
        }
        scratch.chars.clear();
        for (offset, c) in text.char_indices() {
            scratch.chars.push((self.cluster_id(c), offset + c.len_utf8()));
        }
        let seq_len = scratch.chars.len();
        scratch.h_fw.clear();
        scratch.h_bw.clear();
        scratch.h_fw.reserve(seq_len);
        scratch.h_bw.reserve(seq_len);

        let mut h_fw = [0.0f32; HUNITS];
        let mut c_fw = [0.0f32; HUNITS];
        let mut h_bw = [0.0f32; HUNITS];
        let mut c_bw = [0.0f32; HUNITS];
        for i in 0..seq_len {
            let id_fw = usize::from(scratch.chars[i].0);
            step(&self.fw_bx[id_fw], &mut h_fw, &mut c_fw, &self.fw_u);
            scratch.h_fw.push(h_fw);

            let id_bw = usize::from(scratch.chars[seq_len - 1 - i].0);
            step(&self.bw_bx[id_bw], &mut h_bw, &mut c_bw, &self.bw_u);
            scratch.h_bw.push(h_bw);
        }

        let mut clusters = grapheme_boundaries(text);
        let mut cluster_end = 0usize;
        let mut emit = |end: usize| {
            if end <= cluster_end {
                return;
            }
            while cluster_end < end {
                cluster_end = clusters.next().unwrap_or(text.len());
            }
            boundary(cluster_end);
        };

        for i in 0..seq_len - 1 {
            let fw = &scratch.h_fw[i];
            let bw = &scratch.h_bw[seq_len - 1 - i];
            let mut weights = [0.0f32; 4];
            for (k, w) in weights.iter_mut().enumerate() {
                let fw_k = dot_hunits(fw, &self.time_w[0][k]);
                let bw_k = dot_hunits(bw, &self.time_w[1][k]);
                *w = (fw_k + bw_k) + self.time_b[k];
            }
            if weights[2] > weights[0] && weights[2] > weights[1] && weights[2] > weights[3] {
                emit(scratch.chars[i].1);
            }
        }
        emit(text.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::{ComplexScript, bundled_lstm};

    fn segments(lstm: &Lstm<'_>, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut start = 0;
        lstm.segment(text, |end| {
            out.push(text[start..end].to_owned());
            start = end;
        });
        out
    }

    #[test]
    fn header_and_weights_are_checked() {
        assert_eq!(Lstm::from_bytes(b"XZL1").unwrap_err(), LstmError::Magic);
        assert_eq!(Lstm::from_bytes(b"TZL1\x01\0\0\0").unwrap_err(), LstmError::Truncated);
        let mut bad_alpha = MAGIC.to_vec();
        for v in [2u32, 10, 20, 15, 30] {
            bad_alpha.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(Lstm::from_bytes(&bad_alpha).unwrap_err(), LstmError::Alphabet);
        let mut bad_dims = MAGIC.to_vec();
        for v in [1u32, 10, 12, 32, 27] {
            bad_dims.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(Lstm::from_bytes(&bad_dims).unwrap_err(), LstmError::Dimensions);
        let mut bad_weights = MAGIC.to_vec();
        for v in [1u32, 10, 12, EMBED_DIM as u32, HUNITS as u32] {
            bad_weights.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(Lstm::from_bytes(&bad_weights).unwrap_err(), LstmError::Weights);
    }

    #[test]
    fn thai_lstm_segments_words_and_reuses_scratch() {
        let thai = Lstm::from_bytes(bundled_lstm(ComplexScript::Thai)).unwrap();
        assert_eq!(thai.alphabet_len(), 73);
        assert_eq!(segments(&thai, "รถไฟฟ้า"), ["รถ", "ไฟ", "ฟ้า"]);
        assert_eq!(segments(&thai, "โรงพยาบาล"), ["โรง", "พยาบาล"]);
        assert_eq!(segments(&thai, "กรุงเทพมหานคร"), ["กรุงเทพมหาน", "คร"]);
        assert_eq!(segments(&thai, ""), Vec::<String>::new());

        let mut scratch = LstmScratch::default();
        let mut first = Vec::new();
        thai.segment_with("มหาวิทยาลัยเชียงใหม่", &mut scratch, |b| {
            first.push(b)
        });
        let cap = (scratch.chars.capacity(), scratch.h_fw.capacity(), scratch.h_bw.capacity());
        let mut second = Vec::new();
        thai.segment_with("รถไฟฟ้า", &mut scratch, |b| second.push(b));
        assert_eq!((scratch.chars.capacity(), scratch.h_fw.capacity(), scratch.h_bw.capacity()), cap);
        assert_eq!(second, [6, 12, 21]);
    }
}
