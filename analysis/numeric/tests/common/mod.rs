//! What the tests compare: each mention's covered text and value.

#![allow(dead_code)]

use turbo_analysis::ViewId;
use turbo_numeric::{NumberNotation, NumberScanner, Numbers};

/// Every mention in `text` as its covered text and its value, written as
/// `BigDecimal.toPlainString` writes it.
pub fn found(text: &str, notation: NumberNotation, apply_scale: bool) -> Vec<(String, String)> {
    found_with_parity(text, notation, apply_scale, false)
}

/// Every mention in `text` with `opennlp_parity` configured.
pub fn found_with_parity(
    text: &str,
    notation: NumberNotation,
    apply_scale: bool,
    opennlp_parity: bool,
) -> Vec<(String, String)> {
    let mut out = Numbers::new(ViewId::SOURCE);
    NumberScanner { notation, apply_scale, opennlp_parity }.scan(text, &mut out);
    out.spans.iter().map(|(i, s)| (text[s.range()].to_string(), out.value(i).to_plain_string())).collect()
}

/// `found` with scale markers read, in the US notation.
pub fn us(text: &str) -> Vec<(String, String)> {
    found(text, NumberNotation::LatinUs, true)
}

/// `found` with scale markers read, in the European notation.
pub fn eu(text: &str) -> Vec<(String, String)> {
    found(text, NumberNotation::LatinEu, true)
}

/// Pairs of string slices, for comparing with `found`.
pub fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
    expected.iter().map(|&(t, v)| (t.to_string(), v.to_string())).collect()
}

/// xorshift64*: deterministic and dependency free, so a failure
/// reproduces from its seed.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
