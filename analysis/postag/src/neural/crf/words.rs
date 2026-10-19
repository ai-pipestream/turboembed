//! Two's-complement `i32`-word arithmetic anchored at `2^-1074`.

use super::CrfScoreTable;

pub(super) const WORD_SCALE: i32 = -1074;
pub(super) const WORD_MASK: i64 = 0xffff_ffff;

impl CrfScoreTable {
    pub(super) fn words_from_128(high_word: i64, low_word: i64, shift: i32) -> Vec<i32> {
        if high_word == 0 && low_word == 0 {
            return Vec::new();
        }
        let raw = [(high_word >> 32) as i32, high_word as i32, (low_word >> 32) as i32, low_word as i32];
        Self::shift_left(&Self::trim(&raw), shift)
    }

    pub(super) fn words_from_term(mantissa: i64, shift: i32) -> Vec<i32> {
        if mantissa == 0 {
            return Vec::new();
        }
        let raw = [(mantissa >> 32) as i32, mantissa as i32];
        Self::shift_left(&Self::trim(&raw), shift)
    }

    pub(super) fn shift_left(source: &[i32], shift: i32) -> Vec<i32> {
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

    pub(super) fn add_words(first: &[i32], second: &[i32]) -> Vec<i32> {
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

    pub(super) fn negate(source: &[i32]) -> Vec<i32> {
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

    pub(super) fn trim(value: &[i32]) -> Vec<i32> {
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

    pub(super) fn words_to_double(value: &[i32]) -> f64 {
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

    pub(super) fn round_shift_right(magnitude: &[i32], shift: i32) -> i64 {
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

    pub(super) fn test_word_bit(magnitude: &[i32], bit: i32) -> bool {
        let word_index = magnitude.len() as i32 - 1 - (bit >> 5);
        if word_index < 0 || word_index >= magnitude.len() as i32 {
            return false;
        }
        ((magnitude[word_index as usize] as u32 >> (bit & 31)) & 1) != 0
    }
}
