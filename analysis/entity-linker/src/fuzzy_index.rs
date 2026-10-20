//! Sublinear inverted trigram, bigram, 1-delete, token, and prefix index over
//! folded alias keys for fuzzy candidate generation in [`KnowledgeBase`](crate::kb::KnowledgeBase).

use std::cmp::Reverse;
use std::collections::HashMap;

/// Maximum number of postings scanned from a single high-frequency feature list.
const MAX_FEATURE_SCAN: usize = 1024;
/// Maximum number of candidate folded keys returned for full Jaro-Winkler / Dice verification.
const MAX_FUZZY_VERIFY: usize = 512;
/// Maximum key length (in characters) indexed in the short-key 1-delete / anagram table.
const SHORT_KEY_MAX_CHARS: usize = 8;

/// Inverted index over `KnowledgeBase::folded_keys` supporting sublinear fuzzy lookup.
#[derive(Clone, Default)]
pub(crate) struct FuzzyAliasIndex {
    key_char_lens: Vec<u16>,
    token_postings: HashMap<String, Vec<(u16, u32)>>,
    trigram_postings: HashMap<[char; 3], Vec<(u16, u32)>>,
    bigram_postings: HashMap<[char; 2], Vec<(u16, u32)>>,
    short_delete_postings: HashMap<String, Vec<(u16, u32)>>,
    short1_postings: HashMap<char, Vec<u32>>,
}

impl FuzzyAliasIndex {
    /// Builds a [`FuzzyAliasIndex`] over a slice of sorted `folded_keys`.
    pub(crate) fn build(folded_keys: &[String]) -> Self {
        let mut key_char_lens = Vec::with_capacity(folded_keys.len());
        let mut token_postings: HashMap<String, Vec<(u16, u32)>> = HashMap::new();
        let mut trigram_postings: HashMap<[char; 3], Vec<(u16, u32)>> = HashMap::new();
        let mut bigram_postings: HashMap<[char; 2], Vec<(u16, u32)>> = HashMap::new();
        let mut short_delete_postings: HashMap<String, Vec<(u16, u32)>> = HashMap::new();
        let mut short1_postings: HashMap<char, Vec<u32>> = HashMap::new();

        let mut grams3_buf = Vec::new();
        let mut grams2_buf = Vec::new();
        let mut deletes_buf = Vec::new();
        let mut toks_buf: Vec<&str> = Vec::new();

        for (idx, key) in folded_keys.iter().enumerate() {
            let k_idx = idx as u32;
            let chars: Vec<char> = key.chars().collect();
            let char_len = chars.len().min(u16::MAX as usize) as u16;
            key_char_lens.push(char_len);

            if chars.is_empty() {
                continue;
            }
            if chars.len() == 1 {
                short1_postings.entry(chars[0]).or_default().push(k_idx);
            }

            toks_buf.clear();
            for tok in key.split_whitespace() {
                if !tok.is_empty() && !toks_buf.contains(&tok) {
                    toks_buf.push(tok);
                    token_postings.entry(tok.to_owned()).or_default().push((char_len, k_idx));
                }
            }

            extract_padded_trigrams(&chars, &mut grams3_buf);
            for &gram in &grams3_buf {
                trigram_postings.entry(gram).or_default().push((char_len, k_idx));
            }

            extract_padded_bigrams(&chars, &mut grams2_buf);
            for &gram in &grams2_buf {
                bigram_postings.entry(gram).or_default().push((char_len, k_idx));
            }

            if chars.len() <= SHORT_KEY_MAX_CHARS {
                extract_short_signatures(&chars, &mut deletes_buf);
                for sig in &deletes_buf {
                    short_delete_postings.entry(sig.clone()).or_default().push((char_len, k_idx));
                }
            }
        }

        for list in token_postings.values_mut() {
            list.sort_unstable();
        }
        for list in trigram_postings.values_mut() {
            list.sort_unstable();
        }
        for list in bigram_postings.values_mut() {
            list.sort_unstable();
        }
        for list in short_delete_postings.values_mut() {
            list.sort_unstable();
        }

        Self {
            key_char_lens,
            token_postings,
            trigram_postings,
            bigram_postings,
            short_delete_postings,
            short1_postings,
        }
    }

    /// Returns up to `MAX_FUZZY_VERIFY` candidate `folded_keys` indices ranked by shared
    /// token/trigram/bigram/delete overlap and character-length proximity.
    pub(crate) fn candidate_indices(&self, query_folded: &str, threshold: f64) -> Vec<u32> {
        let q_chars: Vec<char> = query_folded.chars().collect();
        let q_len = q_chars.len();
        if q_len == 0 || self.key_char_lens.is_empty() {
            return Vec::new();
        }
        let q_len_u16 = q_len.min(u16::MAX as usize) as u16;

        let mut counts: HashMap<u32, u16> = HashMap::new();

        if q_len == 1
            && let Some(list) = self.short1_postings.get(&q_chars[0])
        {
            for &k_idx in list.iter().take(MAX_FEATURE_SCAN) {
                *counts.entry(k_idx).or_insert(0) += 4;
            }
        }

        // Compute character-length window for Jaro-Winkler / edit-similarity candidates.
        let ratio = if threshold > 0.8 { ((threshold - 0.8) / 0.2).clamp(0.25, 0.9) } else { 0.25 };
        let min_len = ((q_len as f64) * ratio * 0.75).floor().max(1.0) as u16;
        let max_len = (((q_len as f64) / (ratio * 0.75)).ceil() as usize).min(u16::MAX as usize) as u16;

        // Collect slices of postings (with weight) so we can process rarest features first.
        let mut slices: Vec<(&[(u16, u32)], u16)> = Vec::new();

        for tok in query_folded.split_whitespace() {
            if !tok.is_empty()
                && let Some(list) = self.token_postings.get(tok)
            {
                let slice = length_window_slice(list, 1, u16::MAX);
                if !slice.is_empty() {
                    slices.push((slice, 4));
                }
            }
        }

        if q_len <= SHORT_KEY_MAX_CHARS + 1 {
            let mut sigs = Vec::new();
            extract_short_signatures(&q_chars, &mut sigs);
            for sig in &sigs {
                if let Some(list) = self.short_delete_postings.get(sig) {
                    let slice = length_window_slice(list, min_len, max_len);
                    if !slice.is_empty() {
                        slices.push((slice, 5));
                    }
                }
            }
        }

        let mut grams3 = Vec::new();
        extract_padded_trigrams(&q_chars, &mut grams3);
        for gram in &grams3 {
            if let Some(list) = self.trigram_postings.get(gram) {
                let slice = length_window_slice(list, min_len, max_len);
                if !slice.is_empty() {
                    slices.push((slice, 2));
                }
            }
        }

        let mut grams2 = Vec::new();
        extract_padded_bigrams(&q_chars, &mut grams2);
        for gram in &grams2 {
            if let Some(list) = self.bigram_postings.get(gram) {
                let slice = length_window_slice(list, min_len, max_len);
                if !slice.is_empty() {
                    slices.push((slice, 1));
                }
            }
        }

        // Process rarest posting lists first (ascending slice length).
        slices.sort_unstable_by_key(|(s, _)| s.len());

        for (slice, weight) in slices {
            if slice.len() <= MAX_FEATURE_SCAN {
                for &(_, k_idx) in slice {
                    *counts.entry(k_idx).or_insert(0) += weight;
                }
            } else if counts.len() < 128 {
                // High-frequency feature and few rare candidates so far: scan a window around q_len.
                let mid = slice.partition_point(|&(len, _)| len < q_len_u16);
                let half = MAX_FEATURE_SCAN / 2;
                let start = mid.saturating_sub(half);
                let end = (start + MAX_FEATURE_SCAN).min(slice.len());
                let start = end.saturating_sub(MAX_FEATURE_SCAN);
                for &(_, k_idx) in &slice[start..end] {
                    *counts.entry(k_idx).or_insert(0) += weight;
                }
            }
        }

        let mut ranked: Vec<(u16, Reverse<u16>, Reverse<u32>)> = counts
            .into_iter()
            .map(|(k_idx, score)| {
                let k_len = self.key_char_lens[k_idx as usize];
                let len_diff = k_len.abs_diff(q_len_u16);
                (score, Reverse(len_diff), Reverse(k_idx))
            })
            .collect();

        ranked.sort_unstable_by(|a, b| b.cmp(a));
        ranked.into_iter().take(MAX_FUZZY_VERIFY).map(|(_, _, Reverse(k_idx))| k_idx).collect()
    }
}

fn length_window_slice(list: &[(u16, u32)], min_len: u16, max_len: u16) -> &[(u16, u32)] {
    let lo = list.partition_point(|&(len, _)| len < min_len);
    let hi = list.partition_point(|&(len, _)| len <= max_len);
    if lo < hi { &list[lo..hi] } else { &[] }
}

fn extract_padded_trigrams(chars: &[char], out: &mut Vec<[char; 3]>) {
    out.clear();
    if chars.is_empty() {
        return;
    }
    let mut padded = Vec::with_capacity(chars.len() + 2);
    padded.push('\x02');
    padded.extend_from_slice(chars);
    padded.push('\x03');
    for w in padded.windows(3) {
        let gram = [w[0], w[1], w[2]];
        if !out.contains(&gram) {
            out.push(gram);
        }
    }
}

fn extract_padded_bigrams(chars: &[char], out: &mut Vec<[char; 2]>) {
    out.clear();
    if chars.is_empty() {
        return;
    }
    let mut padded = Vec::with_capacity(chars.len() + 2);
    padded.push('\x02');
    padded.extend_from_slice(chars);
    padded.push('\x03');
    for w in padded.windows(2) {
        let gram = [w[0], w[1]];
        if !out.contains(&gram) {
            out.push(gram);
        }
    }
}

fn extract_short_signatures(chars: &[char], out: &mut Vec<String>) {
    out.clear();
    let mut sorted = chars.to_vec();
    sorted.sort_unstable();
    let mut anagram = String::with_capacity(sorted.len() + 2);
    anagram.push('#');
    for c in sorted {
        anagram.push(c);
    }
    out.push(anagram);

    let full: String = chars.iter().collect();
    out.push(full);
    for skip in 0..chars.len() {
        let del: String = chars.iter().enumerate().filter(|&(i, _)| i != skip).map(|(_, &c)| c).collect();
        if !del.is_empty() && !out.contains(&del) {
            out.push(del);
        }
    }
}
