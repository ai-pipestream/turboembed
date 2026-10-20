//! Sublinear inverted trigram, bigram, 1-delete, token, and positional-character
//! index over folded alias keys for fuzzy candidate generation in
//! [`KnowledgeBase`](crate::kb::KnowledgeBase).

use std::cmp::Reverse;
use std::collections::HashMap;

/// Base number of postings scanned from a single feature list when rarer features exist.
const BASE_FEATURE_SCAN: usize = 1024;
/// Total posting scan budget across rarest-first feature lists.
const TOTAL_POSTING_BUDGET: usize = 32_768;
/// Base number of candidate folded keys returned for full Jaro-Winkler / Dice verification.
const BASE_FUZZY_VERIFY: usize = 512;
/// Maximum key length (in characters) indexed in the short-key 1-delete / anagram table.
const SHORT_KEY_MAX_CHARS: usize = 8;

type PostingList = Vec<(u16, u32)>;
type TrigramKey = ([char; 3], u8);
type BigramKey = ([char; 2], u8);

struct QueryFilterCtx {
    q_len: usize,
    q_prefix4: [char; 4],
    q_mask: [u64; 2],
    q_mask_bits: usize,
    threshold: f64,
}

/// Inverted index over `KnowledgeBase::folded_keys` supporting sublinear fuzzy lookup.
#[derive(Clone, Default)]
pub(crate) struct FuzzyAliasIndex {
    key_char_lens: Vec<u16>,
    key_token_lens: Vec<u8>,
    key_prefix4: Vec<[char; 4]>,
    key_char_masks: Vec<[u64; 2]>,
    token_postings: HashMap<String, PostingList>,
    trigram_postings: HashMap<TrigramKey, PostingList>,
    bigram_postings: HashMap<BigramKey, PostingList>,
    pos_char_postings: HashMap<(u8, char), PostingList>,
    char_postings: HashMap<char, PostingList>,
    short_delete_postings: HashMap<String, PostingList>,
    short1_postings: HashMap<char, Vec<u32>>,
}

impl FuzzyAliasIndex {
    /// Builds a [`FuzzyAliasIndex`] over a slice of sorted `folded_keys`.
    pub(crate) fn build(folded_keys: &[String]) -> Self {
        let n = folded_keys.len();
        let mut key_char_lens = Vec::with_capacity(n);
        let mut key_token_lens = Vec::with_capacity(n);
        let mut key_prefix4 = Vec::with_capacity(n);
        let mut key_char_masks = Vec::with_capacity(n);
        let mut token_postings: HashMap<String, PostingList> = HashMap::new();
        let mut trigram_postings: HashMap<TrigramKey, PostingList> = HashMap::new();
        let mut bigram_postings: HashMap<BigramKey, PostingList> = HashMap::new();
        let mut pos_char_postings: HashMap<(u8, char), PostingList> = HashMap::new();
        let mut char_postings: HashMap<char, PostingList> = HashMap::new();
        let mut short_delete_postings: HashMap<String, PostingList> = HashMap::new();
        let mut short1_postings: HashMap<char, Vec<u32>> = HashMap::new();

        let mut grams3_buf = Vec::new();
        let mut grams2_buf = Vec::new();
        let mut pos_chars_buf: Vec<(u8, char)> = Vec::new();
        let mut unique_chars_buf: Vec<char> = Vec::new();
        let mut deletes_buf = Vec::new();
        let mut toks_buf: Vec<&str> = Vec::new();
        let mut char_counts: Vec<(char, u16)> = Vec::new();

        for (idx, key) in folded_keys.iter().enumerate() {
            let k_idx = idx as u32;
            let chars: Vec<char> = key.chars().collect();
            let char_len = chars.len().min(u16::MAX as usize) as u16;
            key_char_lens.push(char_len);

            let mut p4 = ['\0'; 4];
            for (i, &c) in chars.iter().take(4).enumerate() {
                p4[i] = c;
            }
            key_prefix4.push(p4);
            key_char_masks.push(compute_char_mask(&chars, &mut char_counts));

            toks_buf.clear();
            for tok in key.split_whitespace() {
                if !tok.is_empty() && !toks_buf.contains(&tok) {
                    toks_buf.push(tok);
                    token_postings.entry(tok.to_owned()).or_default().push((char_len, k_idx));
                }
            }
            key_token_lens.push(toks_buf.len().min(u8::MAX as usize) as u8);

            if chars.is_empty() {
                continue;
            }
            if chars.len() == 1 {
                short1_postings.entry(chars[0]).or_default().push(k_idx);
            }

            extract_padded_trigrams(&chars, &mut grams3_buf);
            for &gram in &grams3_buf {
                trigram_postings.entry(gram).or_default().push((char_len, k_idx));
            }
            extract_padded_bigrams(&chars, &mut grams2_buf);
            for &gram in &grams2_buf {
                bigram_postings.entry(gram).or_default().push((char_len, k_idx));
            }

            pos_chars_buf.clear();
            unique_chars_buf.clear();
            for (pos, &c) in chars.iter().enumerate() {
                if c.is_whitespace() {
                    continue;
                }
                let bucket = (pos / 4).min(127) as u8;
                if !pos_chars_buf.contains(&(bucket, c)) {
                    pos_chars_buf.push((bucket, c));
                    pos_char_postings.entry((bucket, c)).or_default().push((char_len, k_idx));
                }
                let occ = chars[..=pos].iter().filter(|&&x| x == c).count();
                if (2..=8).contains(&occ) {
                    let occ_tag = 128u8 + (occ as u8);
                    if !pos_chars_buf.contains(&(occ_tag, c)) {
                        pos_chars_buf.push((occ_tag, c));
                        pos_char_postings.entry((occ_tag, c)).or_default().push((char_len, k_idx));
                    }
                }
                if !unique_chars_buf.contains(&c) {
                    unique_chars_buf.push(c);
                    char_postings.entry(c).or_default().push((char_len, k_idx));
                }
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
        for list in pos_char_postings.values_mut() {
            list.sort_unstable();
        }
        for list in char_postings.values_mut() {
            list.sort_unstable();
        }
        for list in short_delete_postings.values_mut() {
            list.sort_unstable();
        }

        Self {
            key_char_lens,
            key_token_lens,
            key_prefix4,
            key_char_masks,
            token_postings,
            trigram_postings,
            bigram_postings,
            pos_char_postings,
            char_postings,
            short_delete_postings,
            short1_postings,
        }
    }

    /// Returns candidate `folded_keys` indices ranked by shared occurrence-tagged
    /// token/trigram/bigram/positional-char/delete overlap and filtered by exact
    /// Jaro-Winkler / Token-Dice threshold bounds.
    pub(crate) fn candidate_indices(&self, query_folded: &str, threshold: f64) -> Vec<u32> {
        let q_chars: Vec<char> = query_folded.chars().collect();
        let q_len = q_chars.len();
        if q_len == 0 || self.key_char_lens.is_empty() {
            return Vec::new();
        }
        let q_len_u16 = q_len.min(u16::MAX as usize) as u16;

        let mut q_prefix4 = ['\0'; 4];
        for (i, &c) in q_chars.iter().take(4).enumerate() {
            q_prefix4[i] = c;
        }
        let mut char_counts = Vec::new();
        let q_mask = compute_char_mask(&q_chars, &mut char_counts);
        let q_mask_bits = (q_mask[0].count_ones() + q_mask[1].count_ones()) as usize;
        let q_ctx = QueryFilterCtx { q_len, q_prefix4, q_mask, q_mask_bits, threshold };

        let mut counts: HashMap<u32, u16> = HashMap::new();
        if q_len == 1
            && let Some(list) = self.short1_postings.get(&q_chars[0])
        {
            for &k_idx in list.iter().take(BASE_FEATURE_SCAN) {
                *counts.entry(k_idx).or_insert(0) += 4;
            }
        }

        let ratio = if threshold > 0.8 { ((threshold - 0.8) / 0.2).clamp(0.25, 0.9) } else { 0.25 };
        let min_len = ((q_len as f64) * ratio * 0.75).floor().max(1.0) as u16;
        let max_len = (((q_len as f64) / (ratio * 0.75)).ceil() as usize).min(u16::MAX as usize) as u16;

        let mut slices: Vec<(&[(u16, u32)], u16)> = Vec::new();
        let mut q_tokens: Vec<&str> = query_folded.split_whitespace().collect();
        q_tokens.sort_unstable();
        q_tokens.dedup();
        let t_q = q_tokens.len();

        let mut dice_shared: HashMap<u32, u8> = HashMap::new();
        for &tok in &q_tokens {
            if let Some(list) = self.token_postings.get(tok) {
                let slice = length_window_slice(list, 1, u16::MAX);
                if !slice.is_empty() {
                    slices.push((slice, 4));
                    if threshold <= 0.95 && slice.len() <= TOTAL_POSTING_BUDGET {
                        for &(_, k_idx) in slice {
                            *dice_shared.entry(k_idx).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        if threshold <= 0.95 && t_q > 0 {
            for (k_idx, shared) in dice_shared {
                let t_k = self.key_token_lens[k_idx as usize] as usize;
                let dice_scaled = 0.95 * (2.0 * (shared as f64)) / ((t_q + t_k) as f64);
                if dice_scaled + 1e-9 >= threshold {
                    *counts.entry(k_idx).or_insert(0) += (shared as u16) * 4;
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
                    slices.push((slice, if gram.1 >= 2 { 4 } else { 2 }));
                }
            }
        }

        let mut grams2 = Vec::new();
        extract_padded_bigrams(&q_chars, &mut grams2);
        for gram in &grams2 {
            if let Some(list) = self.bigram_postings.get(gram) {
                let slice = length_window_slice(list, min_len, max_len);
                if !slice.is_empty() {
                    slices.push((slice, if gram.1 >= 2 { 2 } else { 1 }));
                }
            }
        }

        let min_gram_slice_len = slices.iter().map(|(s, _)| s.len()).min().unwrap_or(usize::MAX);
        let char_slice_cap = if min_gram_slice_len > 128 { TOTAL_POSTING_BUDGET } else { 64 };

        let mut q_pos_chars: Vec<(u8, char)> = Vec::new();
        let mut q_unique_chars: Vec<char> = Vec::new();
        for (pos, &c) in q_chars.iter().enumerate() {
            if c.is_whitespace() {
                continue;
            }
            let bucket = (pos / 4).min(127) as u8;
            let b_lo = if min_gram_slice_len > 128 { bucket.saturating_sub(1) } else { bucket };
            let b_hi = if min_gram_slice_len > 128 { (bucket + 1).min(127) } else { bucket };
            for b in b_lo..=b_hi {
                if !q_pos_chars.contains(&(b, c)) {
                    q_pos_chars.push((b, c));
                    if let Some(list) = self.pos_char_postings.get(&(b, c)) {
                        let slice = length_window_slice(list, min_len, max_len);
                        if !slice.is_empty() && slice.len() <= char_slice_cap {
                            slices.push((slice, 2));
                        }
                    }
                }
            }
            let occ = q_chars[..=pos].iter().filter(|&&x| x == c).count();
            if (2..=8).contains(&occ) {
                let occ_tag = 128u8 + (occ as u8);
                if !q_pos_chars.contains(&(occ_tag, c)) {
                    q_pos_chars.push((occ_tag, c));
                    if let Some(list) = self.pos_char_postings.get(&(occ_tag, c)) {
                        let slice = length_window_slice(list, min_len, max_len);
                        if !slice.is_empty() && slice.len() <= char_slice_cap {
                            slices.push((slice, 2));
                        }
                    }
                }
            }
            if !q_unique_chars.contains(&c) {
                q_unique_chars.push(c);
                if let Some(list) = self.char_postings.get(&c) {
                    let slice = length_window_slice(list, min_len, max_len);
                    if !slice.is_empty() && slice.len() <= char_slice_cap {
                        slices.push((slice, 1));
                    }
                }
            }
        }

        slices.sort_unstable_by_key(|(s, _)| s.len());

        let mut scanned_postings = 0usize;
        for (slice, weight) in slices {
            if slice.len() <= BASE_FEATURE_SCAN || scanned_postings.saturating_add(slice.len()) <= TOTAL_POSTING_BUDGET
            {
                scanned_postings = scanned_postings.saturating_add(slice.len());
                for &(k_len_u16, k_idx) in slice {
                    if self.passes_jaro_prefilter(k_idx, k_len_u16 as usize, &q_ctx) {
                        *counts.entry(k_idx).or_insert(0) += weight;
                    }
                }
            } else if counts.len() < 128 {
                let mid = slice.partition_point(|&(len, _)| len < q_len_u16);
                let start = mid.saturating_sub(BASE_FEATURE_SCAN / 2);
                let end = (start + BASE_FEATURE_SCAN).min(slice.len());
                let start = end.saturating_sub(BASE_FEATURE_SCAN);
                for &(k_len_u16, k_idx) in &slice[start..end] {
                    if self.passes_jaro_prefilter(k_idx, k_len_u16 as usize, &q_ctx) {
                        counts.entry(k_idx).or_insert(1);
                    }
                }
            }
        }

        let mut ranked: Vec<(u16, Reverse<u16>, Reverse<u32>)> = counts
            .into_iter()
            .map(|(k_idx, score)| {
                let k_len = self.key_char_lens[k_idx as usize];
                (score, Reverse(k_len.abs_diff(q_len_u16)), Reverse(k_idx))
            })
            .collect();

        ranked.sort_unstable_by(|a, b| b.cmp(a));
        let top_score = ranked.first().map_or(0, |r| r.0);
        ranked
            .into_iter()
            .enumerate()
            .take_while(|(idx, (score, _, _))| *idx < BASE_FUZZY_VERIFY || *score == top_score)
            .map(|(_, (_, _, Reverse(k_idx)))| k_idx)
            .collect()
    }

    #[inline]
    fn passes_jaro_prefilter(&self, k_idx: u32, k_len: usize, ctx: &QueryFilterCtx) -> bool {
        let max_p_len = 4.min(ctx.q_len).min(k_len);
        let k_p4 = self.key_prefix4[k_idx as usize];
        let mut p = 0usize;
        while p < max_p_len && ctx.q_prefix4[p] == k_p4[p] {
            p += 1;
        }
        let m_req = min_jaro_matches(ctx.q_len, k_len, p, ctx.threshold);
        if m_req > ctx.q_len.min(k_len) {
            return false;
        }
        let k_mask = self.key_char_masks[k_idx as usize];
        let shared = ((ctx.q_mask[0] & k_mask[0]).count_ones() + (ctx.q_mask[1] & k_mask[1]).count_ones()) as usize;
        shared + (ctx.q_len - m_req) >= ctx.q_mask_bits
    }
}

fn compute_char_mask(chars: &[char], counts: &mut Vec<(char, u16)>) -> [u64; 2] {
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
    let h = (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((occ as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((h ^ (h >> 32)) as u32) & 127
}

#[inline]
fn min_jaro_matches(q_len: usize, k_len: usize, prefix_len: usize, threshold: f64) -> usize {
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

fn length_window_slice(list: &[(u16, u32)], min_len: u16, max_len: u16) -> &[(u16, u32)] {
    let lo = list.partition_point(|&(len, _)| len < min_len);
    let hi = list.partition_point(|&(len, _)| len <= max_len);
    if lo < hi { &list[lo..hi] } else { &[] }
}

fn extract_padded_trigrams(chars: &[char], out: &mut Vec<TrigramKey>) {
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
        let occ = out.iter().filter(|(g, _)| *g == gram).count().saturating_add(1).min(255) as u8;
        out.push((gram, occ));
    }
}

fn extract_padded_bigrams(chars: &[char], out: &mut Vec<BigramKey>) {
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
        let occ = out.iter().filter(|(g, _)| *g == gram).count().saturating_add(1).min(255) as u8;
        out.push((gram, occ));
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
