//! The exact score table: 128-bit scaled integers and promotion to word sums.

use crate::neural::NeuralError;

use super::words::WORD_SCALE;

/// Stores exact sums of finite `f64` values for CRF path scoring.
#[derive(Clone, Debug)]
pub struct CrfScoreTable {
    high: Vec<i64>,
    low: Vec<i64>,
    words: Option<Vec<Vec<i32>>>,
    scale: i32,
    filled: usize,
    index: Option<usize>,
}

impl CrfScoreTable {
    /// Creates a table holding `size` exact scores.
    pub fn new(size: usize) -> Self {
        Self { high: vec![0; size], low: vec![0; size], words: None, scale: i32::MIN, filled: 0, index: None }
    }

    /// Resets the table for reuse without reallocating `high` and `low`.
    pub fn reset(&mut self, used: usize) {
        for i in 0..used.min(self.high.len()) {
            self.high[i] = 0;
            self.low[i] = 0;
        }
        self.words = None;
        self.scale = i32::MIN;
        self.filled = 0;
        self.index = None;
    }

    /// Copies the exact score at `source` from `other` into `target`.
    pub fn copy_from(&mut self, target: usize, other: &CrfScoreTable, source: usize) {
        if self.words.is_none() && other.words.is_none() && (self.scale == other.scale || self.scale == i32::MIN) {
            if self.scale == i32::MIN {
                self.scale = other.scale;
            }
            self.high[target] = other.high[source];
            self.low[target] = other.low[source];
            if target >= self.filled {
                self.filled = target + 1;
            }
            return;
        }
        self.to_words();
        let word_value = if let Some(ref other_words) = other.words {
            other_words[source].clone()
        } else {
            Self::words_from_128(other.high[source], other.low[source], other.scale - WORD_SCALE)
        };
        if let Some(ref mut words) = self.words {
            words[target] = word_value;
        }
        if target >= self.filled {
            self.filled = target + 1;
        }
    }

    /// Decomposes a finite `f64` into `mantissa * 2^exponent` and appends it to the term arrays.
    pub fn add_term(
        &mut self,
        exponents: &mut [i32],
        mantissas: &mut [i64],
        count: usize,
        index: usize,
        value: f64,
    ) -> Result<usize, NeuralError> {
        if !value.is_finite() {
            return Err(NeuralError::NonFiniteScore(format!("CRF score is not finite: {value}")));
        }
        if value == 0.0 {
            return Ok(count);
        }
        let bits = value.to_bits() as i64;
        let raw = ((bits as u64 >> 52) & 0x7ff) as i32;
        let mut significand = bits & 0x000f_ffff_ffff_ffff;
        let mut exponent = if raw == 0 {
            -1074
        } else {
            significand |= 1i64 << 52;
            raw - 1075
        };
        let zeros = significand.trailing_zeros() as i32;
        significand = ((significand as u64) >> zeros) as i64;
        exponent += zeros;
        exponents[count] = exponent;
        mantissas[count] = if bits < 0 { -significand } else { significand };
        self.index = Some(index);
        Ok(count + 1)
    }

    /// Stores the exact sum of the `count` decomposed terms at `index`.
    pub fn store(&mut self, index: usize, exponents: &[i32], mantissas: &[i64], count: usize) {
        if count == 0 {
            if let Some(ref mut words) = self.words {
                words[index] = Vec::new();
            } else {
                self.high[index] = 0;
                self.low[index] = 0;
            }
            if index >= self.filled {
                self.filled = index + 1;
            }
            return;
        }
        let mut smallest = exponents[0];
        let mut largest = exponents[0] + 64 - mantissas[0].unsigned_abs().leading_zeros() as i32 - 1;
        for i in 1..count {
            if exponents[i] < smallest {
                smallest = exponents[i];
            }
            let top = exponents[i] + 64 - mantissas[i].unsigned_abs().leading_zeros() as i32 - 1;
            if top > largest {
                largest = top;
            }
        }
        let sum_bits = 32 - (count as u32).leading_zeros() as i32;
        if self.words.is_none()
            && self.scale != i32::MIN
            && smallest >= self.scale
            && largest - self.scale + 1 + sum_bits < 128
        {
            self.store_at_scale(index, exponents, mantissas, count);
            return;
        }
        if self.words.is_none() && (count != 1 || self.scale != i32::MIN) {
            let mut candidate = smallest;
            for j in 0..self.filled {
                if self.high[j] != 0 || self.low[j] != 0 {
                    let low_bit = self.scale + Self::trailing_zeros_128(self.high[j], self.low[j]);
                    if low_bit < candidate {
                        candidate = low_bit;
                    }
                }
            }
            let mut max_bit = largest;
            for j in 0..self.filled {
                if self.high[j] != 0 || self.low[j] != 0 {
                    let high_bit = self.scale + Self::highest_bit_128(self.high[j], self.low[j]);
                    if high_bit > max_bit {
                        max_bit = high_bit;
                    }
                }
            }
            if max_bit - candidate + 1 + sum_bits < 128 {
                self.rescale_existing(candidate);
                self.scale = candidate;
                self.store_at_scale(index, exponents, mantissas, count);
                return;
            }
        }
        if self.words.is_none() && self.scale == i32::MIN && count == 1 {
            self.scale = exponents[0];
            let m = mantissas[0];
            self.high[index] = m >> 63;
            self.low[index] = m;
            if index >= self.filled {
                self.filled = index + 1;
            }
            return;
        }
        self.to_words();
        let mut sum: Vec<i32> = Vec::new();
        for i in 0..count {
            let term = Self::words_from_term(mantissas[i], exponents[i] - WORD_SCALE);
            sum = Self::add_words(&sum, &term);
        }
        if let Some(ref mut words) = self.words {
            words[index] = sum;
        }
        if index >= self.filled {
            self.filled = index + 1;
        }
    }

    /// Stores the sum of the table entry at `base_index` and the `count - 1` decomposed terms in `1..count`.
    pub fn store_with_base(
        &mut self,
        index: usize,
        base_index: usize,
        exponents: &[i32],
        mantissas: &[i64],
        count: usize,
    ) {
        if count <= 1 {
            if index != base_index {
                if let Some(ref mut words) = self.words {
                    words[index] = words[base_index].clone();
                } else {
                    self.high[index] = self.high[base_index];
                    self.low[index] = self.low[base_index];
                }
                if index >= self.filled {
                    self.filled = index + 1;
                }
            }
            return;
        }
        if self.words.is_none() && self.scale != i32::MIN {
            let mut smallest = exponents[1];
            let mut largest = exponents[1] + 64 - mantissas[1].unsigned_abs().leading_zeros() as i32 - 1;
            for i in 2..count {
                if exponents[i] < smallest {
                    smallest = exponents[i];
                }
                let top = exponents[i] + 64 - mantissas[i].unsigned_abs().leading_zeros() as i32 - 1;
                if top > largest {
                    largest = top;
                }
            }
            let base_high = self.high[base_index];
            let base_low = self.low[base_index];
            if base_high != 0 || base_low != 0 {
                let base_top = self.scale + Self::highest_bit_128(base_high, base_low);
                if base_top > largest {
                    largest = base_top;
                }
                let base_bottom = self.scale + Self::trailing_zeros_128(base_high, base_low);
                if base_bottom < smallest {
                    smallest = base_bottom;
                }
            }
            let sum_bits = 32 - (count as u32).leading_zeros() as i32;
            if smallest >= self.scale && largest - self.scale + 1 + sum_bits < 128 {
                let mut sum_high = base_high;
                let mut sum_low = base_low;
                for i in 1..count {
                    let shift = exponents[i] - self.scale;
                    let m = mantissas[i];
                    let (term_high, term_low) = if shift == 0 {
                        (m >> 63, m)
                    } else if shift < 64 {
                        ((m >> (64 - shift)), m << shift)
                    } else {
                        (m << (shift - 64), 0)
                    };
                    let next_low = sum_low.wrapping_add(term_low);
                    let carry = if (next_low as u64) < (sum_low as u64) { 1i64 } else { 0i64 };
                    sum_low = next_low;
                    sum_high = sum_high.wrapping_add(term_high).wrapping_add(carry);
                }
                self.high[index] = sum_high;
                self.low[index] = sum_low;
                if index >= self.filled {
                    self.filled = index + 1;
                }
                return;
            }
        }
        self.to_words();
        let mut sum = self.words.as_ref().unwrap()[base_index].clone();
        for i in 1..count {
            let term = Self::words_from_term(mantissas[i], exponents[i] - WORD_SCALE);
            sum = Self::add_words(&sum, &term);
        }
        if let Some(ref mut words) = self.words {
            words[index] = sum;
        }
        if index >= self.filled {
            self.filled = index + 1;
        }
    }

    /// Returns the round-half-to-even `f64` value of `table[first] - table[second]`.
    pub fn difference(&self, first: usize, second: usize) -> f64 {
        if let Some(ref words) = self.words {
            let neg_second = Self::negate(&words[second]);
            let delta = Self::add_words(&words[first], &neg_second);
            return Self::words_to_double(&delta);
        }
        if self.scale == i32::MIN {
            return 0.0;
        }
        let a_high = self.high[first];
        let a_low = self.low[first];
        let b_high = self.high[second];
        let b_low = self.low[second];
        if a_high == b_high && a_low == b_low {
            return 0.0;
        }
        let borrow = if (a_low as u64) < (b_low as u64) { 1i64 } else { 0i64 };
        let mut diff_low = a_low.wrapping_sub(b_low);
        let mut diff_high = a_high.wrapping_sub(b_high).wrapping_sub(borrow);
        let negative = diff_high < 0;
        if negative {
            diff_low = diff_low.wrapping_neg();
            diff_high = (!diff_high).wrapping_add(if diff_low == 0 { 1 } else { 0 });
        }
        let bit_length = if diff_high != 0 {
            128 - (diff_high as u64).leading_zeros() as i32
        } else {
            64 - (diff_low as u64).leading_zeros() as i32
        };
        let shift = bit_length - 53;
        let mut significand: u64;
        let mut exponent = self.scale;
        if shift > 0 {
            exponent += shift;
            significand = Self::shift_right_unsigned(diff_high, diff_low, shift) as u64;
            let half = Self::test_bit(diff_high, diff_low, shift - 1);
            let sticky = Self::has_low_bits(diff_high, diff_low, shift - 1);
            if half && (sticky || (significand & 1) != 0) {
                significand += 1;
                if significand == (1u64 << 53) {
                    significand >>= 1;
                    exponent += 1;
                }
            }
        } else {
            significand = diff_low as u64;
        }
        let magnitude = Self::scalb(significand as f64, exponent);
        if negative { -magnitude } else { magnitude }
    }

    /// Multiplies `value` by `2^scale_factor`, matching `Math.scalb(double, int)`.
    #[inline]
    pub fn scalb(mut value: f64, mut scale_factor: i32) -> f64 {
        const MAX_SCALE: i32 = 1023;
        const MIN_SCALE: i32 = -1022;
        if scale_factor < 0 {
            scale_factor = scale_factor.max(-2099);
            while scale_factor < MIN_SCALE {
                value *= f64::from_bits(0x0010_0000_0000_0000); // 2^-1022
                scale_factor += 1022;
            }
            let factor = f64::from_bits(((1023 + scale_factor) as u64) << 52);
            value * factor
        } else {
            scale_factor = scale_factor.min(2099);
            while scale_factor > MAX_SCALE {
                value *= f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
                scale_factor -= 1023;
            }
            let factor = f64::from_bits(((1023 + scale_factor) as u64) << 52);
            value * factor
        }
    }

    fn store_at_scale(&mut self, index: usize, exponents: &[i32], mantissas: &[i64], count: usize) {
        let mut sum_high = 0i64;
        let mut sum_low = 0i64;
        for i in 0..count {
            let shift = exponents[i] - self.scale;
            let m = mantissas[i];
            let (term_high, term_low) = if shift == 0 {
                (m >> 63, m)
            } else if shift < 64 {
                (m >> (64 - shift), m << shift)
            } else {
                (m << (shift - 64), 0)
            };
            let next_low = sum_low.wrapping_add(term_low);
            let carry = if (next_low as u64) < (sum_low as u64) { 1i64 } else { 0i64 };
            sum_low = next_low;
            sum_high = sum_high.wrapping_add(term_high).wrapping_add(carry);
        }
        self.high[index] = sum_high;
        self.low[index] = sum_low;
        if index >= self.filled {
            self.filled = index + 1;
        }
    }

    fn rescale_existing(&mut self, next_scale: i32) {
        let shift = self.scale - next_scale;
        if shift == 0 {
            return;
        }
        for i in 0..self.filled {
            let h = self.high[i];
            let l = self.low[i];
            if h == 0 && l == 0 {
                continue;
            }
            if shift > 0 {
                if shift < 64 {
                    self.high[i] = (h << shift) | ((l as u64 >> (64 - shift)) as i64);
                    self.low[i] = l << shift;
                } else {
                    self.high[i] = l << (shift - 64);
                    self.low[i] = 0;
                }
            } else {
                let right = -shift;
                if right < 64 {
                    self.low[i] = ((l as u64 >> right) as i64) | (h << (64 - right));
                    self.high[i] = h >> right;
                } else {
                    self.low[i] = h >> (right - 64);
                    self.high[i] = h >> 63;
                }
            }
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_words(&mut self) {
        if self.words.is_some() {
            return;
        }
        let mut word_list = vec![Vec::new(); self.high.len()];
        if self.scale != i32::MIN {
            let shift = self.scale - WORD_SCALE;
            for (i, slot) in word_list.iter_mut().enumerate().take(self.filled) {
                *slot = Self::words_from_128(self.high[i], self.low[i], shift);
            }
        }
        self.words = Some(word_list);
    }

    fn trailing_zeros_128(high_word: i64, low_word: i64) -> i32 {
        if low_word != 0 {
            (low_word as u64).trailing_zeros() as i32
        } else {
            64 + (high_word as u64).trailing_zeros() as i32
        }
    }

    fn highest_bit_128(mut high_word: i64, mut low_word: i64) -> i32 {
        if high_word < 0 {
            low_word = low_word.wrapping_neg();
            high_word = (!high_word).wrapping_add(if low_word == 0 { 1 } else { 0 });
        }
        if high_word != 0 {
            127 - (high_word as u64).leading_zeros() as i32
        } else {
            63 - (low_word as u64).leading_zeros() as i32
        }
    }

    fn shift_right_unsigned(high_word: i64, low_word: i64, shift: i32) -> i64 {
        if shift < 64 {
            ((low_word as u64 >> shift) as i64) | (high_word << (64 - shift))
        } else {
            (high_word as u64 >> (shift - 64)) as i64
        }
    }

    fn test_bit(high_word: i64, low_word: i64, bit: i32) -> bool {
        if bit < 64 { ((low_word as u64 >> bit) & 1) != 0 } else { ((high_word as u64 >> (bit - 64)) & 1) != 0 }
    }

    fn has_low_bits(high_word: i64, low_word: i64, count: i32) -> bool {
        if count <= 0 {
            return false;
        }
        if count < 64 {
            (low_word & ((1i64 << count) - 1)) != 0
        } else if low_word != 0 {
            true
        } else {
            let high_bits = count - 64;
            high_bits > 0 && (high_word & ((1i64 << high_bits) - 1)) != 0
        }
    }
}
