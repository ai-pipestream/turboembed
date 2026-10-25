//! Cheap rejection before walking possible table cells. The input is the
//! parser's exact content slice, after container matching and header trim.

#[inline]
pub(crate) fn candidate(text: &str) -> bool {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        use std::arch::x86_64::{_mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8};
        let (chunks, tail) = text.as_bytes().as_chunks::<16>();
        let mut space = false;
        // SAFETY: SSE2 is enabled for this target. Each array is exactly
        // 16 initialized bytes; the unaligned load reads only that array.
        // No vector load is made for the shorter tail.
        unsafe {
            let blank = _mm_set1_epi8(b' ' as i8);
            let tab = _mm_set1_epi8(b'\t' as i8);
            for chunk in chunks {
                let word = _mm_loadu_si128(chunk.as_ptr().cast());
                let spaces = _mm_movemask_epi8(_mm_cmpeq_epi8(word, blank)) as u32;
                if _mm_movemask_epi8(_mm_cmpeq_epi8(word, tab)) != 0
                    || spaces & (spaces << 1) != 0
                    || (space && spaces & 1 != 0)
                {
                    return true;
                }
                space = spaces & 0x8000 != 0;
            }
        }
        for &byte in tail {
            if byte == b'\t' || (space && byte == b' ') {
                return true;
            }
            space = byte == b' ';
        }
        false
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    candidate_scalar(text)
}

#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "sse2"))))]
fn candidate_scalar(text: &str) -> bool {
    let zero = |x: u64| x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080;
    let (words, tail) = text.as_bytes().as_chunks::<8>();
    let mut space = false;
    for bytes in words {
        let word = u64::from_le_bytes(*bytes);
        let spaces = zero(word ^ 0x2020_2020_2020_2020);
        if (space && bytes[0] == b' ')
            || zero(word ^ 0x0909_0909_0909_0909) != 0
            || (spaces & (spaces >> 8) != 0 && bytes.windows(2).any(|b| b == b"  "))
        {
            return true;
        }
        space = bytes[7] == b' ';
    }
    for &byte in tail {
        if byte == b'\t' || (space && byte == b' ') {
            return true;
        }
        space = byte == b' ';
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{candidate, candidate_scalar};

    fn check(text: &str) {
        let expected = text.contains("  ") || text.contains('\t');
        assert_eq!(candidate(text), expected, "vector: {text:?}");
        assert_eq!(candidate_scalar(text), expected, "scalar: {text:?}");
    }

    #[test]
    fn both_paths_match_the_original_check_on_document_lines() {
        for line in include_str!("../../../../docs/analysis.md").lines() {
            for padding in 0..32 {
                let text = "x".repeat(padding) + line;
                check(&text);
                check(&text[padding..]);
            }
        }
    }

    #[test]
    fn every_tail_and_vector_boundary_keeps_tabs_spaces_and_utf8() {
        for padding in 0..48 {
            for middle in ["  ", "\t", " !", "é", "茶", "😊", " \r\n", "", " "] {
                for tail in ["", "x", " ", "\t", "  ", "é"] {
                    let text = "x".repeat(padding) + middle + tail;
                    for end in padding..=text.len() {
                        if text.is_char_boundary(end) {
                            check(&text[padding..end]);
                        }
                    }
                }
            }
            for byte in 0u8..=127 {
                check(&("x".repeat(padding) + &(byte as char).to_string() + " ! tail"));
            }
        }
    }
}
