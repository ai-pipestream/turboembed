//! Alias normalization, string similarity (Jaro-Winkler and token Dice),
//! and context term extraction for entity candidate generation and scoring.

use turbo_unicode::casefold;

/// Common English stop words excluded from context overlap scoring so
/// disambiguation focuses on content-bearing terms.
const STOP_WORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "between", "both", "but", "by", "can", "could", "did", "do", "does", "during", "each", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "more", "most", "my", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your",
];

/// Folds a surface mention or KB alias into a canonical lookup key:
/// accent-folds with the shared `turbo_normalize::AccentFold` (marks
/// of Latin, Greek, and Cyrillic bases removed, other scripts' marks
/// kept, stroke letters mapped), applies full Unicode case folding,
/// collapses dotted initialisms
/// (`"U.S.A."` -> `"usa"`), splits on punctuation and whitespace, and
/// joins alphanumeric tokens with a single ASCII space.
pub fn fold_alias(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }
    let mut stripped = String::with_capacity(input.len());
    thread_local! {
        static FOLD_SCRATCH: std::cell::RefCell<turbo_normalize::FoldScratch> =
            std::cell::RefCell::new(turbo_normalize::FoldScratch::default());
    }
    FOLD_SCRATCH.with(|scratch| {
        turbo_normalize::AccentFold::new().fold_str(input, &mut scratch.borrow_mut(), &mut stripped);
    });

    let mut folded = String::with_capacity(stripped.len());
    casefold::fold(&stripped, &mut folded);

    // Collapse dotted single-letter initialisms like "u.s.a." -> "usa"
    let chars: Vec<char> = folded.chars().collect();
    let mut collapsed = String::with_capacity(folded.len());
    let mut i = 0;
    while i < chars.len() {
        if i + 3 <= chars.len()
            && chars[i].is_alphanumeric()
            && chars[i + 1] == '.'
            && (i == 0 || !chars[i - 1].is_alphanumeric())
        {
            // Check if this starts a dotted initialism of >= 2 single letters (e.g., "u.s." or "u.s.a")
            let mut j = i;
            let mut letters = Vec::new();
            while j < chars.len() && chars[j].is_alphanumeric() {
                if j + 1 < chars.len() && chars[j + 1].is_alphanumeric() {
                    break;
                }
                letters.push(chars[j]);
                if j + 1 < chars.len() && chars[j + 1] == '.' {
                    j += 2;
                } else if !letters.is_empty() && (j + 1 == chars.len() || !chars[j + 1].is_alphanumeric()) {
                    j += 1;
                    break;
                } else {
                    break;
                }
            }
            if letters.len() >= 2 && j > i + 2 {
                for ch in letters {
                    collapsed.push(ch);
                }
                collapsed.push(' ');
                i = j;
                continue;
            }
        }
        collapsed.push(chars[i]);
        i += 1;
    }

    let mut out = String::with_capacity(collapsed.len());
    let mut in_word = false;
    for ch in collapsed.chars() {
        if ch.is_alphanumeric() {
            out.push(ch);
            in_word = true;
        } else if in_word {
            out.push(' ');
            in_word = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Returns `true` if `word` (assumed already folded to lowercase) is a stop word.
pub fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.binary_search(&word).is_ok()
}

/// Extracts normalized content terms from `text`, excluding stop words and
/// single-character non-numeric tokens.
pub fn extract_context_terms(text: &str) -> Vec<String> {
    let folded = fold_alias(text);
    folded
        .split_whitespace()
        .filter(|w| (w.chars().count() >= 2 || w.chars().all(|c| c.is_ascii_digit())) && !is_stop_word(w))
        .map(str::to_owned)
        .collect()
}

/// Computes the Jaro similarity in `[0.0, 1.0]` between two strings
/// over `char`s: the shared `turbo_geo` implementation (ASCII bitmask
/// fast path, thread-local scratch otherwise).
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    turbo_geo::jaro_similarity(a, b)
}

/// Computes the Jaro-Winkler similarity in `[0.0, 1.0]` between two
/// strings: the shared `turbo_geo` implementation, with the standard
/// 0.7 boost threshold.
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    turbo_geo::jaro_winkler(a, b)
}

/// Computes the Sørensen-Dice coefficient over space-separated tokens of two
/// already-folded strings.
pub fn token_dice_similarity(folded_a: &str, folded_b: &str) -> f64 {
    if folded_a == folded_b {
        return if folded_a.is_empty() { 0.0 } else { 1.0 };
    }
    if !folded_a.contains(char::is_whitespace) && !folded_b.contains(char::is_whitespace) {
        return 0.0;
    }
    let mut a_tokens: Vec<&str> = folded_a.split_whitespace().collect();
    let mut b_tokens: Vec<&str> = folded_b.split_whitespace().collect();
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return 0.0;
    }
    a_tokens.sort_unstable();
    a_tokens.dedup();
    b_tokens.sort_unstable();
    b_tokens.dedup();

    let mut shared = 0usize;
    let (mut i, mut j) = (0, 0);
    while i < a_tokens.len() && j < b_tokens.len() {
        match a_tokens[i].cmp(b_tokens[j]) {
            std::cmp::Ordering::Equal => {
                shared += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    ((2 * shared) as f64 / ((a_tokens.len() + b_tokens.len()) as f64)).clamp(0.0, 1.0)
}

/// Combined similarity between two folded aliases in `[0.0, 1.0]`.
pub fn alias_similarity(query_folded: &str, candidate_folded: &str) -> f64 {
    if query_folded.is_empty() || candidate_folded.is_empty() {
        return 0.0;
    }
    if query_folded == candidate_folded {
        return 1.0;
    }
    let jw = jaro_winkler(query_folded, candidate_folded);
    if jw >= 0.95 {
        return jw.clamp(0.0, 1.0);
    }
    let dice = token_dice_similarity(query_folded, candidate_folded) * 0.95;
    jw.max(dice).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Default)]
pub(crate) struct KeySketch {
    pub char_len: u16,
    pub k_idx: u32,
    pub prefix4: [char; 4],
    pub mask: [u64; 2],
}

pub(crate) struct QueryFilterCtx {
    pub q_len: usize,
    pub q_prefix4: [char; 4],
    pub q_mask: [u64; 2],
}

pub(crate) struct LenFilterBounds {
    pub k_len: u16,
    pub cutoff_bits: u64,
    pub max_missing_q_p4: i32,
    pub max_missing_q: [i32; 5],
    pub max_missing_k: [i32; 5],
}

impl LenFilterBounds {
    #[inline]
    pub(crate) fn compute(q_len: usize, k_len: u16, cutoff: f64) -> Self {
        let k_len_usize = k_len as usize;
        let max_p = 4.min(q_len).min(k_len_usize);
        let mut max_missing_q = [-1i32; 5];
        let mut max_missing_k = [-1i32; 5];
        for p in 0..=max_p {
            let m_req = min_jaro_matches(q_len, k_len_usize, p, cutoff);
            if m_req <= q_len.min(k_len_usize) {
                max_missing_q[p] = (q_len - m_req) as i32;
                max_missing_k[p] = (k_len_usize - m_req) as i32;
            }
        }
        Self {
            k_len,
            cutoff_bits: cutoff.to_bits(),
            max_missing_q_p4: max_missing_q[max_p],
            max_missing_q,
            max_missing_k,
        }
    }
}

#[inline]
pub(crate) fn passes_sketch_prefilter(sk: &KeySketch, ctx: &QueryFilterCtx, cutoff: f64) -> bool {
    let k_len = sk.char_len as usize;
    let max_p_len = 4.min(ctx.q_len).min(k_len);
    let mut p = 0usize;
    while p < max_p_len && ctx.q_prefix4[p] == sk.prefix4[p] {
        p += 1;
    }
    let m_req = min_jaro_matches(ctx.q_len, k_len, p, cutoff);
    if m_req > ctx.q_len.min(k_len) {
        return false;
    }
    let missing_q = ((ctx.q_mask[0] & !sk.mask[0]).count_ones() + (ctx.q_mask[1] & !sk.mask[1]).count_ones()) as usize;
    if missing_q > ctx.q_len - m_req {
        return false;
    }
    let missing_k = ((!ctx.q_mask[0] & sk.mask[0]).count_ones() + (!ctx.q_mask[1] & sk.mask[1]).count_ones()) as usize;
    missing_k <= k_len - m_req
}

#[inline]
pub(crate) fn length_bounds(q_len: usize, cutoff: f64) -> (u16, u16) {
    if cutoff > 0.80 {
        let ratio = (((cutoff - 0.8) / 0.2) - 1e-9).max(0.0);
        let min_len = ((q_len as f64) * ratio).floor().max(1.0) as u16;
        let max_len = if ratio > 1e-6 {
            (((q_len as f64) / ratio).ceil() as usize).min(u16::MAX as usize) as u16
        } else {
            u16::MAX
        };
        (min_len, max_len)
    } else {
        (1, u16::MAX)
    }
}

pub(crate) fn compute_char_mask(chars: &[char], counts: &mut Vec<(char, u16)>) -> [u64; 2] {
    counts.clear();
    let mut mask = [0u64; 2];
    for &c in chars {
        let occ = if let Some(entry) = counts.iter_mut().find(|(ch, _)| *ch == c) {
            entry.1 = entry.1.saturating_add(1);
            entry.1
        } else {
            counts.push((c, 1));
            1
        };
        let bit = occ_char_bit(c, occ);
        mask[(bit >> 6) as usize] |= 1u64 << (bit & 63);
    }
    mask
}

#[inline]
fn occ_char_bit(c: char, occ: u16) -> u32 {
    if c.is_ascii_lowercase() && (1..=3).contains(&occ) {
        ((c as u32 - b'a' as u32) * 3) + (occ as u32 - 1)
    } else if c.is_ascii_digit() && (1..=4).contains(&occ) {
        78 + ((c as u32 - b'0' as u32) * 4) + (occ as u32 - 1)
    } else if c == ' ' && (1..=4).contains(&occ) {
        118 + (occ as u32 - 1)
    } else {
        let h = (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((occ as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        122 + (((h ^ (h >> 32)) as u32) % 6)
    }
}

#[inline]
pub(crate) fn min_jaro_matches(q_len: usize, k_len: usize, prefix_len: usize, threshold: f64) -> usize {
    if threshold <= 0.0 {
        return 0;
    }
    let p = prefix_len.min(4).min(q_len).min(k_len) as f64;
    let min_jaro = ((threshold - 0.1 * p) / (1.0 - 0.1 * p)) - 1e-9;
    if min_jaro <= 0.0 {
        return 1;
    }
    let rhs = 3.0 * min_jaro - 1.0;
    if rhs <= 0.0 {
        return 1;
    }
    let numer = rhs * (q_len as f64) * (k_len as f64);
    let denom = (q_len + k_len) as f64;
    ((numer / denom) - 1e-9).ceil().max(1.0) as usize
}

#[cfg(test)]
mod fold_alias_accent_tests {
    #[test]
    fn marks_outside_folded_scripts_survive_and_thorn_maps() {
        // Devanagari anusvara (Mn) is kept: the two words stay distinct.
        assert_ne!(super::fold_alias("\u{0938}\u{0902}\u{0924}"), super::fold_alias("\u{0938}\u{0924}"));
        assert_eq!(super::fold_alias("\u{00DE}ingvellir"), "thingvellir");
        assert_eq!(super::fold_alias("\u{0126}amrun"), "hamrun");
        assert_eq!(super::fold_alias("S\u{00E3}o Paulo"), "sao paulo");
    }
}

#[cfg(test)]
mod char_count_tests {
    #[test]
    fn single_character_terms_drop_by_character_count() {
        let terms = super::extract_context_terms("\u{4EAC} \u{00E9} a tokyo paris");
        assert_eq!(terms, vec!["tokyo".to_string(), "paris".to_string()]);
    }
}
