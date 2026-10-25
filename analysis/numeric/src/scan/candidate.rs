//! Wide candidate prefilter. Unicode membership remains `extended::is_nd_at`.

use std::arch::x86_64::*;

use super::extended::{EXTENDED_START_BYTES, is_nd_at};

// Keep feature dispatch off the original ASCII/Latin candidate loop.
#[cold]
#[inline(never)]
pub(super) fn try_wide(text: &str, start: usize) -> Option<Option<usize>> {
    if text.len() - start >= 64 && std::arch::is_x86_feature_detected!("avx2") {
        // AVX2 was detected; next_avx2 bounds each load against the slice.
        Some(unsafe { next_avx2(text, start) })
    } else {
        None
    }
}

#[target_feature(enable = "avx2")]
unsafe fn eq(x: __m256i, byte: u8) -> __m256i {
    _mm256_cmpeq_epi8(x, _mm256_set1_epi8(byte as i8))
}

// Each interval is wholly below or above 0x80, so signed byte ordering agrees.
#[target_feature(enable = "avx2")]
unsafe fn range(x: __m256i, lo: u8, hi: u8) -> __m256i {
    let above = if lo == 0x80 {
        _mm256_cmpgt_epi8(_mm256_setzero_si256(), x)
    } else {
        _mm256_cmpgt_epi8(x, _mm256_set1_epi8(lo.wrapping_sub(1) as i8))
    };
    _mm256_and_si256(above, _mm256_cmpgt_epi8(_mm256_set1_epi8(hi.wrapping_add(1) as i8), x))
}

/// Caller checks AVX2 support. Loads cover 32 candidate bytes plus two bytes
/// of lookahead; every possible non-ASCII candidate is confirmed by the shared
/// Unicode classifier before it is returned.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn next_avx2(text: &str, start: usize) -> Option<usize> {
    let b = text.as_bytes();
    let mut i = start;
    while b.len() - i >= 34 {
        // All three 32-byte loads fit inside b by the loop condition.
        let (a, n, t) = unsafe {
            (
                _mm256_loadu_si256(b.as_ptr().add(i).cast()),
                _mm256_loadu_si256(b.as_ptr().add(i + 1).cast()),
                _mm256_loadu_si256(b.as_ptr().add(i + 2).cast()),
            )
        };
        let candidates = unsafe {
            let ascii = _mm256_or_si256(eq(a, b'-'), range(a, b'0', b'9'));
            let arabic = _mm256_or_si256(
                _mm256_and_si256(eq(a, 0xD9), range(n, 0xA0, 0xA9)),
                _mm256_and_si256(eq(a, 0xDB), range(n, 0xB0, 0xB9)),
            );
            let nko = _mm256_and_si256(eq(a, 0xDF), range(n, 0x80, 0x89));
            let indic = _mm256_and_si256(eq(a, 0xE0), _mm256_and_si256(range(n, 0xA5, 0xBC), range(t, 0x90, 0xAF)));
            let fullwidth = _mm256_and_si256(eq(a, 0xEF), _mm256_and_si256(eq(n, 0xBC), range(t, 0x90, 0x99)));
            let rare = _mm256_or_si256(_mm256_or_si256(eq(a, 0xE1), eq(a, 0xEA)), eq(a, 0xF0));
            _mm256_or_si256(
                _mm256_or_si256(ascii, arabic),
                _mm256_or_si256(_mm256_or_si256(nko, indic), _mm256_or_si256(fullwidth, rare)),
            )
        };
        let mut mask = _mm256_movemask_epi8(candidates) as u32;
        while mask != 0 {
            let pos = i + mask.trailing_zeros() as usize;
            if b[pos] < 0x80 || is_nd_at(text, pos) {
                return Some(pos);
            }
            mask &= mask - 1;
        }
        i += 32;
    }
    // Fewer than 34 bytes remain. A candidate lead byte is always a character
    // boundary, even when the last wide block ended inside another character.
    b[i..].iter().enumerate().find_map(|(skip, &lead)| {
        let pos = i + skip;
        (EXTENDED_START_BYTES[usize::from(lead)] && (lead < 0x80 || is_nd_at(text, pos))).then_some(pos)
    })
}
