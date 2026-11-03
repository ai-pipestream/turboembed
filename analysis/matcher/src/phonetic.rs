//! Phonetic keys of a single word, allocation free: American Soundex and
//! Double Metaphone, after Philips' "The Double Metaphone Search Algorithm"
//! (C/C++ Users Journal, 2000) and the Apache Commons Codec implementation
//! (Apache-2.0).
//!
//! # Dialect
//!
//! Both keys are computed over Unicode scalars with ASCII case folding, so
//! [`soundex`] and [`double_metaphone`] take `&str` and return fixed-size
//! keys without allocating:
//!
//! - [`soundex`] returns the four-byte American Soundex code (US Census
//!   rules, as in Knuth TAOCP 6 and Commons Codec `US_ENGLISH`): the first
//!   letter is kept, consonants map to `1`..=`6`, `H` and `W` are ignored
//!   and do not separate equal codes, vowels and `Y` give no code but break
//!   a run of equal codes, and the code is padded with `'0'`. Non-ASCII
//!   letters and other non-letters are skipped; a word with no ASCII letter
//!   encodes as `*b"0000"`. (Commons Codec throws on a non-ASCII letter
//!   instead; skipping is this dialect's panic-free choice.)
//! - [`double_metaphone`] returns the `(primary, alternate)` keys at length
//!   8, NUL-padded. The port follows Commons Codec's rule order exactly, on
//!   scalars rather than UTF-16 code units: a supplementary character is
//!   skipped as one unit where the reference skips two halves, which never
//!   changes either key, and `Ç`/`ç` and `Ñ`/`ñ` match their reference
//!   branches while other non-ASCII letters (and `ß`, which Java would
//!   expand to `SS`) are skipped. Input is not trimmed and not uppercased as
//!   a whole, so internal spaces keep the `VAN ` and `SAN ` prefix rules.
//!   Empty input yields two all-NUL keys, and at most the first
//!   [`MAX_WORD`] scalars of a word are read, far past where both keys are
//!   complete.

/// The most scalars of a word [`double_metaphone`] reads; keys are complete
/// long before this, and it keeps the working buffer on the stack.
const MAX_WORD: usize = 64;

/// The Soundex code of `word`: the first letter then up to three digit
/// codes, `b`/`f`/`p`/`v` = 1, `c`/`g`/`j`/`k`/`q`/`s`/`x`/`z` = 2,
/// `d`/`t` = 3, `l` = 4, `m`/`n` = 5, `r` = 6, padded with `'0'`.
pub fn soundex(word: &str) -> [u8; 4] {
    const MAP: [u8; 26] = *b"01230120022455012623010202";
    let mut out = [b'0'; 4];
    let mut count = 0;
    let mut last = b'0';
    for c in word.chars() {
        if !c.is_ascii_alphabetic() {
            continue;
        }
        let up = c.to_ascii_uppercase() as usize;
        let digit = MAP[up - b'A' as usize];
        if count == 0 {
            out[0] = up as u8;
            count = 1;
            last = digit;
            continue;
        }
        // H and W are ignored and do not separate equal codes.
        if up == b'H' as usize || up == b'W' as usize {
            continue;
        }
        if digit != b'0' && digit != last {
            out[count] = digit;
            count += 1;
            if count == 4 {
                break;
            }
        }
        last = digit;
    }
    out
}

/// The Double Metaphone `(primary, alternate)` keys of `word`, each eight
/// bytes NUL-padded; equal keys mean the alternates never diverged.
pub fn double_metaphone(word: &str) -> ([u8; 8], [u8; 8]) {
    let mut buf = ['\0'; MAX_WORD];
    let mut n = 0;
    for c in word.chars() {
        if n == MAX_WORD {
            break;
        }
        buf[n] = if c.is_ascii_lowercase() { c.to_ascii_uppercase() } else { c };
        n += 1;
    }
    let w = &buf[..n];
    let slavo = is_slavo_germanic(w);
    let mut index = if silent_start(w) { 1 } else { 0 };
    let mut keys = Keys::default();
    while !keys.complete() && index < n {
        let i = index as isize;
        index = match w[index] {
            'A' | 'E' | 'I' | 'O' | 'U' | 'Y' => {
                if index == 0 {
                    keys.push(b"A");
                }
                index + 1
            }
            'B' => {
                keys.push(b"P");
                index + usize::from(at(w, i + 1) == 'B') + 1
            }
            // C with a cedilla, upper or lower (only ASCII case is folded).
            '\u{C7}' | '\u{E7}' => {
                keys.push(b"S");
                index + 1
            }
            'C' => handle_c(w, &mut keys, index),
            'D' => handle_d(w, &mut keys, index),
            'F' => {
                keys.push(b"F");
                index + usize::from(at(w, i + 1) == 'F') + 1
            }
            'G' => handle_g(w, &mut keys, index, slavo),
            'H' => handle_h(w, &mut keys, index),
            'J' => handle_j(w, &mut keys, index, slavo),
            'K' => {
                keys.push(b"K");
                index + usize::from(at(w, i + 1) == 'K') + 1
            }
            'L' => handle_l(w, &mut keys, index),
            'M' => {
                keys.push(b"M");
                index + usize::from(condition_m0(w, index)) + 1
            }
            'N' => {
                keys.push(b"N");
                index + usize::from(at(w, i + 1) == 'N') + 1
            }
            // N with a tilde, upper or lower (only ASCII case is folded).
            '\u{D1}' | '\u{F1}' => {
                keys.push(b"N");
                index + 1
            }
            'P' => handle_p(w, &mut keys, index),
            'Q' => {
                keys.push(b"K");
                index + usize::from(at(w, i + 1) == 'Q') + 1
            }
            'R' => handle_r(w, &mut keys, index, slavo),
            'S' => handle_s(w, &mut keys, index, slavo),
            'T' => handle_t(w, &mut keys, index),
            'V' => {
                keys.push(b"F");
                index + usize::from(at(w, i + 1) == 'V') + 1
            }
            'W' => handle_w(w, &mut keys, index),
            'X' => handle_x(w, &mut keys, index),
            'Z' => handle_z(w, &mut keys, index, slavo),
            _ => index + 1,
        };
    }
    (keys.pri, keys.alt)
}

/// The two keys under construction; each stops growing at eight bytes.
#[derive(Default)]
struct Keys {
    pri: [u8; 8],
    alt: [u8; 8],
    npri: usize,
    nalt: usize,
}

impl Keys {
    fn push(&mut self, s: &[u8]) {
        self.push_pri(s);
        self.push_alt(s);
    }

    fn push2(&mut self, pri: &[u8], alt: &[u8]) {
        self.push_pri(pri);
        self.push_alt(alt);
    }

    fn push_pri(&mut self, s: &[u8]) {
        let take = s.len().min(8 - self.npri);
        self.pri[self.npri..self.npri + take].copy_from_slice(&s[..take]);
        self.npri += take;
    }

    fn push_alt(&mut self, s: &[u8]) {
        let take = s.len().min(8 - self.nalt);
        self.alt[self.nalt..self.nalt + take].copy_from_slice(&s[..take]);
        self.nalt += take;
    }

    fn complete(&self) -> bool {
        self.npri >= 8 && self.nalt >= 8
    }
}

/// The scalar at `i`, or NUL out of bounds, as the reference's `charAt`.
fn at(w: &[char], i: isize) -> char {
    if i < 0 {
        return '\0';
    }
    w.get(i as usize).copied().unwrap_or('\0')
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'A' | 'E' | 'I' | 'O' | 'U' | 'Y')
}

/// Whether `w[start..start + len]` equals one of the ASCII `opts`.
fn contains(w: &[char], start: isize, len: usize, opts: &[&str]) -> bool {
    if start < 0 || start as usize + len > w.len() {
        return false;
    }
    let start = start as usize;
    opts.iter().any(|opt| opt.bytes().enumerate().all(|(k, b)| k < len && w[start + k] == b as char))
}

fn silent_start(w: &[char]) -> bool {
    ["GN", "KN", "PN", "WR", "PS"].iter().any(|p| contains(w, 0, 2, &[*p]))
}

fn is_slavo_germanic(w: &[char]) -> bool {
    // Contains W, K, CZ, or WITZ.
    w.contains(&'W')
        || w.contains(&'K')
        || w.windows(2).any(|x| x == ['C', 'Z'])
        || w.windows(4).any(|x| x == ['W', 'I', 'T', 'Z'])
}

mod metaphone;

use metaphone::{
    condition_m0, handle_c, handle_d, handle_g, handle_h, handle_j, handle_l, handle_p, handle_r, handle_s, handle_t,
    handle_w, handle_x, handle_z,
};
