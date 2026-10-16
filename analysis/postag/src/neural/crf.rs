//! Exact linear-chain CRF scoring (`CrfScoreTable` and `CrfScorer`).
//!
//! `CrfScoreTable` stores exact sums of finite IEEE-754 `f64` values as
//! scaled 128-bit two's-complement integers when all stored values share a
//! common binary exponent window of under 128 bits, and promotes to a
//! two's-complement `i32`-word representation anchored at `2^-1074` when
//! dynamic range exceeds 128 bits. `difference(a, b)` returns the
//! round-half-to-even `f64` value of `table[a] - table[b]`.
//!
//! `CrfScorer` uses `CrfScoreTable` for exact Viterbi path comparison and
//! backward-forward marginal computation so that large path scores that share
//! high-order bits do not lose low-order transition or emission differences.

use crate::neural::NeuralError;

const WORD_SCALE: i32 = -1074;
const WORD_MASK: i64 = 0xffff_ffff;

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

    fn words_from_128(high_word: i64, low_word: i64, shift: i32) -> Vec<i32> {
        if high_word == 0 && low_word == 0 {
            return Vec::new();
        }
        let raw = [(high_word >> 32) as i32, high_word as i32, (low_word >> 32) as i32, low_word as i32];
        Self::shift_left(&Self::trim(&raw), shift)
    }

    fn words_from_term(mantissa: i64, shift: i32) -> Vec<i32> {
        if mantissa == 0 {
            return Vec::new();
        }
        let raw = [(mantissa >> 32) as i32, mantissa as i32];
        Self::shift_left(&Self::trim(&raw), shift)
    }

    fn shift_left(source: &[i32], shift: i32) -> Vec<i32> {
        if source.is_empty() || shift <= 0 {
            return source.to_vec();
        }
        let word_shift = (shift >> 5) as usize;
        let bit_shift = shift & 31;
        if bit_shift == 0 {
            let mut result = vec![0i32; source.len() + word_shift];
            result[..source.len()].copy_from_slice(source);
            return Self::trim(&result);
        }
        let mut result = vec![0i32; source.len() + word_shift + 1];
        result[0] = source[0] >> (32 - bit_shift);
        for i in 0..source.len() - 1 {
            result[i + 1] = (source[i] << bit_shift) | ((source[i + 1] as u32 >> (32 - bit_shift)) as i32);
        }
        result[source.len()] = source[source.len() - 1] << bit_shift;
        Self::trim(&result)
    }

    fn add_words(first: &[i32], second: &[i32]) -> Vec<i32> {
        if first.is_empty() {
            return second.to_vec();
        }
        if second.is_empty() {
            return first.to_vec();
        }
        let length = first.len().max(second.len()) + 1;
        let mut result = vec![0i32; length];
        let mut carry = 0i64;
        let first_sign = if first[0] < 0 { WORD_MASK } else { 0 };
        let second_sign = if second[0] < 0 { WORD_MASK } else { 0 };
        for i in 0..length {
            let a = if i < first.len() { (first[first.len() - 1 - i] as i64) & WORD_MASK } else { first_sign };
            let b = if i < second.len() { (second[second.len() - 1 - i] as i64) & WORD_MASK } else { second_sign };
            let sum = a + b + carry;
            result[length - 1 - i] = sum as i32;
            carry = (sum as u64 >> 32) as i64;
        }
        Self::trim(&result)
    }

    fn negate(source: &[i32]) -> Vec<i32> {
        if source.is_empty() {
            return Vec::new();
        }
        let mut inverted = vec![0i32; source.len() + 1];
        inverted[0] = if source[0] < 0 { 0 } else { -1 };
        for i in 0..source.len() {
            inverted[i + 1] = !source[i];
        }
        let mut carry = 1i64;
        for i in (0..inverted.len()).rev() {
            if carry == 0 {
                break;
            }
            let sum = ((inverted[i] as i64) & WORD_MASK) + carry;
            inverted[i] = sum as i32;
            carry = (sum as u64 >> 32) as i64;
        }
        Self::trim(&inverted)
    }

    fn trim(value: &[i32]) -> Vec<i32> {
        let mut start = 0;
        while start < value.len() - 1 {
            if (value[start] == 0 && value[start + 1] >= 0) || (value[start] == -1 && value[start + 1] < 0) {
                start += 1;
            } else {
                break;
            }
        }
        if start == value.len() - 1 && value[start] == 0 {
            return Vec::new();
        }
        value[start..].to_vec()
    }

    fn words_to_double(value: &[i32]) -> f64 {
        if value.is_empty() {
            return 0.0;
        }
        let negative = value[0] < 0;
        let magnitude = if negative { Self::negate(value) } else { value.to_vec() };
        let bit_length = (magnitude.len() as i32 - 1) * 32 + (32 - (magnitude[0] as u32).leading_zeros() as i32);
        let shift = bit_length - 53;
        let mut significand: u64;
        let mut exponent = WORD_SCALE;
        if shift > 0 {
            exponent += shift;
            significand = Self::round_shift_right(&magnitude, shift) as u64;
            if significand == (1u64 << 53) {
                significand >>= 1;
                exponent += 1;
            }
        } else {
            significand = 0;
            for &word in &magnitude {
                significand = (significand << 32) | ((word as i64 & WORD_MASK) as u64);
            }
        }
        let scaled = Self::scalb(significand as f64, exponent);
        if negative { -scaled } else { scaled }
    }

    fn round_shift_right(magnitude: &[i32], shift: i32) -> i64 {
        let mut significand = 0i64;
        let total_bits = (magnitude.len() as i32) * 32;
        for bit in (shift..total_bits).rev() {
            significand = (significand << 1) | if Self::test_word_bit(magnitude, bit) { 1 } else { 0 };
        }
        let half = Self::test_word_bit(magnitude, shift - 1);
        let mut sticky = false;
        for bit in 0..(shift - 1) {
            if Self::test_word_bit(magnitude, bit) {
                sticky = true;
                break;
            }
        }
        if half && (sticky || (significand & 1) != 0) {
            significand += 1;
        }
        significand
    }

    fn test_word_bit(magnitude: &[i32], bit: i32) -> bool {
        let word_index = magnitude.len() as i32 - 1 - (bit >> 5);
        if word_index < 0 || word_index >= magnitude.len() as i32 {
            return false;
        }
        ((magnitude[word_index as usize] as u32 >> (bit & 31)) & 1) != 0
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

/// Linear-chain CRF decoding, marginal probabilities, and negative log-likelihood gradients.
pub struct CrfScorer;

impl CrfScorer {
    /// Finds the highest-scoring tag index sequence by Viterbi decoding.
    pub fn viterbi(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
    ) -> Result<Vec<usize>, NeuralError> {
        let steps = emissions.len();
        if steps == 0 {
            return Ok(Vec::new());
        }
        let num_tags = start.len();
        let mut backpointers = vec![vec![0usize; num_tags]; steps];
        let mut previous = CrfScoreTable::new(num_tags + 1);
        let mut current = CrfScoreTable::new(num_tags + 1);
        let mut candidate = CrfScoreTable::new(1);
        let mut exponents = [0i32; 4];
        let mut mantissas = [0i64; 4];

        for j in 0..num_tags {
            let mut count = 0;
            count = previous.add_term(&mut exponents, &mut mantissas, count, j, start[j])?;
            count = previous.add_term(&mut exponents, &mut mantissas, count, j, emissions[0][j])?;
            if steps == 1 {
                count = previous.add_term(&mut exponents, &mut mantissas, count, j, end[j])?;
            }
            previous.store(j, &exponents, &mantissas, count);
        }

        for t in 1..steps {
            let is_last = t == steps - 1;
            current.reset(num_tags + 1);
            for k in 0..num_tags {
                let mut best_prev = 0usize;
                for j in 0..num_tags {
                    candidate.reset(1);
                    candidate.copy_from(0, &previous, j);
                    let mut count = 1;
                    count = candidate.add_term(&mut exponents, &mut mantissas, count, 0, transitions[j][k])?;
                    count = candidate.add_term(&mut exponents, &mut mantissas, count, 0, emissions[t][k])?;
                    if is_last {
                        count = candidate.add_term(&mut exponents, &mut mantissas, count, 0, end[k])?;
                    }
                    candidate.store_with_base(0, 0, &exponents, &mantissas, count);
                    if j == 0 {
                        current.copy_from(k, &candidate, 0);
                    } else {
                        current.copy_from(num_tags, &candidate, 0);
                        if current.difference(num_tags, k) > 0.0 {
                            current.copy_from(k, &candidate, 0);
                            best_prev = j;
                        }
                    }
                }
                backpointers[t][k] = best_prev;
            }
            std::mem::swap(&mut previous, &mut current);
        }

        let mut best_last = 0usize;
        for j in 1..num_tags {
            if previous.difference(j, best_last) > 0.0 {
                best_last = j;
            }
        }
        let mut path = vec![0usize; steps];
        path[steps - 1] = best_last;
        for t in (1..steps).rev() {
            path[t - 1] = backpointers[t][path[t]];
        }
        Ok(path)
    }

    /// Computes the marginal posterior probability `P(y_t = k | x)` for every token and tag.
    pub fn marginals(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
    ) -> Result<Vec<Vec<f64>>, NeuralError> {
        let (_, marginals) = Self::infer(emissions, transitions, start, end, None, None)?;
        Ok(marginals)
    }

    /// Computes negative log-likelihood loss and accumulates gradients into `d_emissions`, `d_transitions`, `d_start`, and `d_end`.
    #[allow(clippy::too_many_arguments)]
    pub fn loss_and_gradients(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
        gold: &[usize],
        d_emissions: &mut [Vec<f64>],
        d_transitions: &mut [Vec<f64>],
        d_start: &mut [f64],
        d_end: &mut [f64],
    ) -> Result<f64, NeuralError> {
        let steps = emissions.len();
        if steps == 0 {
            return Ok(0.0);
        }
        let num_tags = start.len();
        let (loss, marginals) = Self::infer(emissions, transitions, start, end, Some(gold), Some(d_transitions))?;
        for t in 0..steps {
            for k in 0..num_tags {
                d_emissions[t][k] += marginals[t][k];
            }
            d_emissions[t][gold[t]] -= 1.0;
            if t > 0 {
                d_transitions[gold[t - 1]][gold[t]] -= 1.0;
            }
        }
        for k in 0..num_tags {
            d_start[k] += marginals[0][k];
            d_end[k] += marginals[steps - 1][k];
        }
        d_start[gold[0]] -= 1.0;
        d_end[gold[steps - 1]] -= 1.0;
        Ok(loss)
    }

    fn infer(
        emissions: &[Vec<f64>],
        transitions: &[Vec<f64>],
        start: &[f64],
        end: &[f64],
        gold: Option<&[usize]>,
        mut transition_grads: Option<&mut [Vec<f64>]>,
    ) -> Result<(f64, Vec<Vec<f64>>), NeuralError> {
        let steps = emissions.len();
        if steps == 0 {
            return Ok((0.0, Vec::new()));
        }
        let num_tags = start.len();
        let mut best: Vec<CrfScoreTable> = (0..steps).map(|_| CrfScoreTable::new(num_tags)).collect();
        let mut log_mass = vec![vec![0.0f64; num_tags]; steps];
        let mut exponents = [0i32; 4];
        let mut mantissas = [0i64; 4];

        for k in 0..num_tags {
            let count = best[steps - 1].add_term(&mut exponents, &mut mantissas, 0, k, end[k])?;
            best[steps - 1].store(k, &exponents, &mantissas, count);
        }

        let mut next = CrfScoreTable::new(num_tags);
        let mut candidates = CrfScoreTable::new(num_tags);
        let mut weights = vec![0.0f64; num_tags];
        for t in (0..steps - 1).rev() {
            next.reset(num_tags);
            for k in 0..num_tags {
                next.copy_from(k, &best[t + 1], k);
                let count = next.add_term(&mut exponents, &mut mantissas, 1, k, emissions[t + 1][k])?;
                next.store_with_base(k, k, &exponents, &mantissas, count);
            }
            let mut largest_mass = f64::NEG_INFINITY;
            for j in 0..num_tags {
                candidates.reset(num_tags);
                let mut best_next = 0usize;
                for k in 0..num_tags {
                    candidates.copy_from(k, &next, k);
                    let count = candidates.add_term(&mut exponents, &mut mantissas, 1, k, transitions[j][k])?;
                    candidates.store_with_base(k, k, &exponents, &mantissas, count);
                    if k > 0 && candidates.difference(k, best_next) > 0.0 {
                        best_next = k;
                    }
                }
                best[t].copy_from(j, &candidates, best_next);
                for k in 0..num_tags {
                    weights[k] = candidates.difference(k, best_next) + log_mass[t + 1][k];
                }
                log_mass[t][j] = Self::log_sum_exp(&weights);
                if log_mass[t][j] > largest_mass {
                    largest_mass = log_mass[t][j];
                }
            }
            for j in 0..num_tags {
                log_mass[t][j] -= largest_mass;
            }
        }

        let mut probabilities = vec![vec![0.0f64; num_tags]; steps];
        candidates.reset(num_tags);
        let mut best_first = 0usize;
        for k in 0..num_tags {
            candidates.copy_from(k, &best[0], k);
            let mut count = 1;
            count = candidates.add_term(&mut exponents, &mut mantissas, count, k, start[k])?;
            count = candidates.add_term(&mut exponents, &mut mantissas, count, k, emissions[0][k])?;
            candidates.store_with_base(k, k, &exponents, &mantissas, count);
            if k > 0 && candidates.difference(k, best_first) > 0.0 {
                best_first = k;
            }
        }
        for k in 0..num_tags {
            probabilities[0][k] = candidates.difference(k, best_first) + log_mass[0][k];
        }
        let mut loss = Self::normalize_in_place(&mut probabilities[0], gold.map(|g| g[0]));
        let mut conditional = vec![0.0f64; num_tags];
        let mut candidate = CrfScoreTable::new(2);
        for t in 0..steps - 1 {
            candidates.reset(num_tags);
            for k in 0..num_tags {
                candidates.copy_from(k, &best[t + 1], k);
                let count = candidates.add_term(&mut exponents, &mut mantissas, 1, k, emissions[t + 1][k])?;
                candidates.store_with_base(k, k, &exponents, &mantissas, count);
            }
            for j in 0..num_tags {
                for k in 0..num_tags {
                    candidate.reset(2);
                    candidate.copy_from(0, &candidates, k);
                    let count = candidate.add_term(&mut exponents, &mut mantissas, 1, 0, transitions[j][k])?;
                    candidate.store_with_base(0, 0, &exponents, &mantissas, count);
                    candidate.copy_from(1, &best[t], j);
                    conditional[k] = candidate.difference(0, 1) + log_mass[t + 1][k];
                }
                let selected_gold = gold.and_then(|g| if g[t] == j { Some(g[t + 1]) } else { None });
                loss += Self::normalize_in_place(&mut conditional, selected_gold);
                for k in 0..num_tags {
                    let probability = probabilities[t][j] * conditional[k];
                    probabilities[t + 1][k] += probability;
                    if let Some(tg) = transition_grads.as_deref_mut() {
                        tg[j][k] += probability;
                    }
                }
            }
            let mut total = 0.0f64;
            for &value in &probabilities[t + 1] {
                total += value;
            }
            for k in 0..num_tags {
                probabilities[t + 1][k] /= total;
            }
        }
        Ok((loss, probabilities))
    }

    fn normalize_in_place(values: &mut [f64], gold: Option<usize>) -> f64 {
        let mut max = f64::NEG_INFINITY;
        for &v in values.iter() {
            if v > max {
                max = v;
            }
        }
        let selected = gold.map_or(max, |g| values[g]);
        let mut total = 0.0f64;
        for v in values.iter_mut() {
            *v = (*v - max).exp();
            total += *v;
        }
        for v in values.iter_mut() {
            *v /= total;
        }
        if gold.is_some() { (max - selected) + total.ln() } else { 0.0 }
    }

    fn log_sum_exp(xs: &[f64]) -> f64 {
        let mut max = f64::NEG_INFINITY;
        for &x in xs {
            if x > max {
                max = x;
            }
        }
        let mut sum = 0.0f64;
        for &x in xs {
            sum += (x - max).exp();
        }
        max + sum.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_table_word_promotion_and_differences() {
        let mut table = CrfScoreTable::new(3);
        let mut exponents = [0i32; 4];
        let mut mantissas = [0i64; 4];

        let mut count = table.add_term(&mut exponents, &mut mantissas, 0, 0, 1e100).unwrap();
        count = table.add_term(&mut exponents, &mut mantissas, count, 0, 1e-100).unwrap();
        table.store(0, &exponents, &mantissas, count);

        count = table.add_term(&mut exponents, &mut mantissas, 0, 1, 1e100).unwrap();
        table.store(1, &exponents, &mantissas, count);

        assert_eq!(table.difference(0, 1), 1e-100);
        assert_eq!(table.difference(1, 0), -1e-100);
        assert_eq!(table.difference(0, 0), 0.0);

        table.copy_from(2, &table.clone(), 1);
        count = table.add_term(&mut exponents, &mut mantissas, 1, 2, -2.5e-100).unwrap();
        table.store_with_base(2, 2, &exponents, &mantissas, count);
        assert_eq!(table.difference(0, 2), 3.5e-100);
    }

    #[test]
    fn score_table_rescaling_and_subnormal_cancel() {
        let mut first = CrfScoreTable::new(3);
        let mut exponents = [0i32; 4];
        let mut mantissas = [0i64; 4];

        let count = first.add_term(&mut exponents, &mut mantissas, 0, 0, 8.0).unwrap();
        first.store(0, &exponents, &mantissas, count);
        let count = first.add_term(&mut exponents, &mut mantissas, 0, 1, 0.25).unwrap();
        first.store(1, &exponents, &mantissas, count);
        assert_eq!(first.difference(0, 1), 7.75);

        let mut second = CrfScoreTable::new(1);
        let count = second.add_term(&mut exponents, &mut mantissas, 0, 0, 1e-200).unwrap();
        second.store(0, &exponents, &mantissas, count);
        first.copy_from(2, &second, 0);
        assert_eq!(first.difference(0, 2), 8.0);
    }

    #[test]
    fn viterbi_matches_brute_force_and_marginals_sum_to_one() {
        let emissions = vec![vec![0.4, -0.3, 0.1], vec![-0.2, 0.5, 0.2], vec![0.3, 0.1, -0.4], vec![-0.1, 0.2, 0.6]];
        let transitions = vec![vec![0.1, -0.5, 0.3], vec![0.4, 0.2, -0.2], vec![-0.3, 0.5, 0.1]];
        let start = vec![0.2, -0.1, 0.05];
        let end = vec![-0.15, 0.25, 0.1];

        let best = CrfScorer::viterbi(&emissions, &transitions, &start, &end).unwrap();
        let mut best_brute = vec![0usize; 4];
        let mut best_score = f64::NEG_INFINITY;
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        let s = start[a]
                            + emissions[0][a]
                            + transitions[a][b]
                            + emissions[1][b]
                            + transitions[b][c]
                            + emissions[2][c]
                            + transitions[c][d]
                            + emissions[3][d]
                            + end[d];
                        if s > best_score {
                            best_score = s;
                            best_brute = vec![a, b, c, d];
                        }
                    }
                }
            }
        }
        assert_eq!(best, best_brute);

        let marginals = CrfScorer::marginals(&emissions, &transitions, &start, &end).unwrap();
        assert_eq!(marginals.len(), 4);
        for row in &marginals {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn precision_cancellation_preserves_small_delta() {
        let large = CrfScoreTable::scalb(1.0, 100);
        let delta = CrfScoreTable::scalb(1.0, -20);
        let emissions = vec![vec![large, large], vec![delta, 0.0]];
        let transitions = vec![vec![0.0, -large], vec![-large, 0.0]];
        let start = vec![0.0, 0.0];
        let end = vec![0.0, 0.0];

        let path = CrfScorer::viterbi(&emissions, &transitions, &start, &end).unwrap();
        assert_eq!(path, vec![0, 0]);
        let marginals = CrfScorer::marginals(&emissions, &transitions, &start, &end).unwrap();
        let expected_first = 1.0 / (1.0 + (-delta).exp());
        assert!((marginals[0][0] - expected_first).abs() <= 2.0 * f64::EPSILON);
        assert!((marginals[1][0] - expected_first).abs() <= 2.0 * f64::EPSILON);
    }
}
