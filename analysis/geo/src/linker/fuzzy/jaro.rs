//! The ASCII `u128` bitmask fast paths for Jaro and
//! Jaro-Winkler, split from `fuzzy` for the file size cap.

use super::{JARO_WINKLER_BOOST_THRESHOLD, JARO_WINKLER_MAX_PREFIX, JARO_WINKLER_PREFIX_SCALE};

/// Jaro over ASCII bytes with `u128` match bitmasks: the same
/// result as the char path for strings of at most 128 bytes,
/// with no allocation.
pub(super) fn jaro_ascii(a: &[u8], b: &[u8]) -> f64 {
    let (m, n) = (a.len(), b.len());
    if m == 0 || n == 0 {
        return 0.0;
    }
    let match_dist = (m.max(n) / 2).saturating_sub(1);
    let (mut a_matched, mut b_matched) = (0u128, 0u128);
    let mut matches = 0usize;
    for (i, &ca) in a.iter().enumerate() {
        let start = i.saturating_sub(match_dist);
        let end = (i + match_dist + 1).min(n);
        for (j, &cb) in b.iter().enumerate().take(end).skip(start) {
            let bit = 1u128 << j;
            if (b_matched & bit) == 0 && ca == cb {
                a_matched |= 1u128 << i;
                b_matched |= bit;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let mut transpositions = 0usize;
    let mut rem_b = b_matched;
    for (i, &ca) in a.iter().enumerate() {
        if (a_matched & (1u128 << i)) == 0 {
            continue;
        }
        let k = rem_b.trailing_zeros() as usize;
        rem_b &= rem_b - 1;
        if ca != b[k] {
            transpositions += 1;
        }
    }
    let m_f = matches as f64;
    let t_f = (transpositions as f64) / 2.0;
    ((m_f / m as f64) + (m_f / n as f64) + ((m_f - t_f) / m_f)) / 3.0
}

/// Jaro-Winkler over ASCII bytes: prefix boost over
/// [`jaro_ascii`], gated on the boost threshold.
pub(super) fn jw_ascii(a: &[u8], b: &[u8]) -> f64 {
    let jaro = jaro_ascii(a, b);
    if jaro < JARO_WINKLER_BOOST_THRESHOLD {
        return jaro;
    }
    let prefix = a.iter().zip(b.iter()).take(JARO_WINKLER_MAX_PREFIX).take_while(|(ca, cb)| ca == cb).count();
    (jaro + (prefix as f64) * JARO_WINKLER_PREFIX_SCALE * (1.0 - jaro)).clamp(0.0, 1.0)
}

pub(super) fn ascii_pair<'a>(a: &'a str, b: &'a str) -> Option<(&'a [u8], &'a [u8])> {
    (a.is_ascii() && b.is_ascii() && a.len() <= 128 && b.len() <= 128).then_some((a.as_bytes(), b.as_bytes()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn ascii_fast_path_agrees_with_known_values() {
        let near = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(near(crate::linker::jaro_winkler("MARTHA", "MARHTA"), 0.9611111111111111));
        assert!(near(crate::linker::jaro_similarity("MARTHA", "MARHTA"), 0.9444444444444444));
        assert!(near(crate::linker::jaro_winkler("DIXON", "DICKSONX"), 0.8133333333333332));
        assert_eq!(crate::linker::jaro_winkler("abc", "xyz"), 0.0);
        // Below the 0.7 threshold there is no prefix boost.
        let (j, jw) =
            (crate::linker::jaro_similarity("aaaaab", "aaaccc"), crate::linker::jaro_winkler("aaaaab", "aaaccc"));
        assert!(j < 0.7 && near(j, jw), "j={j} jw={jw}");
    }
}
