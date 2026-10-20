//! Sublinear inverted trigram, bigram, 1-delete, token, and positional-character
//! index over folded alias keys with a lossless length-window sketch fallback
//! for fuzzy candidate generation in [`KnowledgeBase`](crate::kb::KnowledgeBase).

use std::collections::HashMap;

use crate::similarity::{
    KeySketch, LenFilterBounds, QueryFilterCtx, compute_char_mask, length_bounds, passes_sketch_prefilter,
};

/// Maximum postings scanned from a single feature list before triggering the lossless window scan.
const BASE_FEATURE_SCAN: usize = 2_048;
/// Total posting scan budget across rarest-first feature lists before triggering the lossless window scan.
const TOTAL_POSTING_BUDGET: usize = 12_288;
/// Maximum candidates verified from the posting path before triggering the lossless window scan.
const BASE_FUZZY_VERIFY: usize = 256;
/// Maximum key length (in characters) indexed in the short-key 1-delete / anagram table.
const SHORT_KEY_MAX_CHARS: usize = 8;

type PostingList = Vec<(u16, u32)>;
type TrigramKey = ([char; 3], u8);
type BigramKey = ([char; 2], u8);

/// Inverted index over `KnowledgeBase::folded_keys` supporting sublinear fuzzy lookup
/// with an exact length-window sketch fallback when posting lists saturate.
#[derive(Clone, Default)]
pub(crate) struct FuzzyAliasIndex {
    sketches_by_idx: Vec<KeySketch>,
    sorted_sketches: Vec<KeySketch>,
    key_token_lens: Vec<u8>,
    token_postings: HashMap<String, PostingList>,
    trigram_postings: HashMap<TrigramKey, PostingList>,
    bigram_postings: HashMap<BigramKey, PostingList>,
    pos_char_postings: HashMap<(u8, char), PostingList>,
    short_delete_postings: HashMap<String, PostingList>,
    short1_postings: HashMap<char, Vec<u32>>,
}

impl FuzzyAliasIndex {
    /// Builds a [`FuzzyAliasIndex`] over a slice of sorted `folded_keys`.
    pub(crate) fn build(folded_keys: &[String]) -> Self {
        let n = folded_keys.len();
        let mut sketches_by_idx = Vec::with_capacity(n);
        let mut key_token_lens = Vec::with_capacity(n);
        let mut token_postings: HashMap<String, PostingList> = HashMap::new();
        let mut trigram_postings: HashMap<TrigramKey, PostingList> = HashMap::new();
        let mut bigram_postings: HashMap<BigramKey, PostingList> = HashMap::new();
        let mut pos_char_postings: HashMap<(u8, char), PostingList> = HashMap::new();
        let mut short_delete_postings: HashMap<String, PostingList> = HashMap::new();
        let mut short1_postings: HashMap<char, Vec<u32>> = HashMap::new();

        let mut grams3_buf = Vec::new();
        let mut grams2_buf = Vec::new();
        let mut pos_chars_buf: Vec<(u8, char)> = Vec::new();
        let mut deletes_buf = Vec::new();
        let mut toks_buf: Vec<&str> = Vec::new();
        let mut char_counts: Vec<(char, u16)> = Vec::new();

        for (idx, key) in folded_keys.iter().enumerate() {
            let k_idx = idx as u32;
            let chars: Vec<char> = key.chars().collect();
            let char_len = chars.len().min(u16::MAX as usize) as u16;

            let mut prefix4 = ['\0'; 4];
            for (i, &c) in chars.iter().take(4).enumerate() {
                prefix4[i] = c;
            }
            let mask = compute_char_mask(&chars, &mut char_counts);
            sketches_by_idx.push(KeySketch { char_len, k_idx, prefix4, mask });

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
        for list in short_delete_postings.values_mut() {
            list.sort_unstable();
        }

        let mut sorted_sketches = sketches_by_idx.clone();
        sorted_sketches.sort_unstable_by_key(|sk| (sk.char_len, sk.k_idx));

        Self {
            sketches_by_idx,
            sorted_sketches,
            key_token_lens,
            token_postings,
            trigram_postings,
            bigram_postings,
            pos_char_postings,
            short_delete_postings,
            short1_postings,
        }
    }

    /// Visits candidate `folded_keys` indices using rarest-first posting lists and,
    /// whenever any posting list or verification cap saturates, falls back to scanning
    /// the lossless `[min_len, max_len]` sketch window with an $O(1)$ Jaro-Winkler upper bound.
    pub(crate) fn for_each_candidate(&self, query_folded: &str, threshold: f64, mut visit: impl FnMut(u32, &mut f64)) {
        let q_chars: Vec<char> = query_folded.chars().collect();
        let q_len = q_chars.len();
        let n_keys = self.sketches_by_idx.len();
        if q_len == 0 || n_keys == 0 {
            return;
        }

        let mut q_prefix4 = ['\0'; 4];
        for (i, &c) in q_chars.iter().take(4).enumerate() {
            q_prefix4[i] = c;
        }
        let mut char_counts = Vec::new();
        let q_mask = compute_char_mask(&q_chars, &mut char_counts);
        let q_ctx = QueryFilterCtx { q_len, q_prefix4, q_mask };

        let mut running_cutoff = threshold;
        let mut visited = vec![0u64; n_keys.div_ceil(64)];

        let mut q_tokens: Vec<&str> = query_folded.split_whitespace().collect();
        q_tokens.sort_unstable();
        q_tokens.dedup();
        let t_q = q_tokens.len();

        let (min_len, max_len) = length_bounds(q_len, threshold);
        let mut slices: Vec<&[(u16, u32)]> = Vec::new();
        let mut any_slice_truncated = threshold < 0.67;

        if t_q > 0 {
            q_tokens.sort_unstable_by_key(|&tok| self.token_postings.get(tok).map_or(0, Vec::len));
            for &tok in &q_tokens {
                if let Some(list) = self.token_postings.get(tok) {
                    let slice = length_window_slice(list, min_len, max_len);
                    if !slice.is_empty() {
                        slices.push(slice);
                    }
                }
            }
            if threshold <= 0.95 {
                let min_shared = if threshold <= 0.0 {
                    1
                } else {
                    (((threshold * (t_q as f64)) / (1.90 - threshold)) - 1e-9).ceil().max(1.0) as usize
                };
                if min_shared <= t_q {
                    let seed_count = t_q - min_shared + 1;
                    let mut dice_shared: HashMap<u32, u8> = HashMap::new();
                    for (idx, &tok) in q_tokens.iter().enumerate() {
                        if let Some(list) = self.token_postings.get(tok) {
                            if idx < seed_count {
                                for &(_, k_idx) in list {
                                    *dice_shared.entry(k_idx).or_insert(0) += 1;
                                }
                            } else if !dice_shared.is_empty() {
                                for &(_, k_idx) in list {
                                    if let Some(c) = dice_shared.get_mut(&k_idx) {
                                        *c += 1;
                                    }
                                }
                            }
                        }
                    }
                    for (k_idx, shared) in dice_shared {
                        if (shared as usize) >= min_shared {
                            let t_k = self.key_token_lens[k_idx as usize] as usize;
                            let dice_scaled = 0.95 * (2.0 * (shared as f64)) / ((t_q + t_k) as f64);
                            if dice_scaled + 1e-9 >= running_cutoff && mark_visited(&mut visited, k_idx) {
                                visit(k_idx, &mut running_cutoff);
                            }
                        }
                    }
                }
            }
        }

        if q_len == 1
            && let Some(list) = self.short1_postings.get(&q_chars[0])
        {
            for &k_idx in list {
                if mark_visited(&mut visited, k_idx) {
                    visit(k_idx, &mut running_cutoff);
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
                        slices.push(slice);
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
                    slices.push(slice);
                }
            }
        }

        let mut grams2 = Vec::new();
        extract_padded_bigrams(&q_chars, &mut grams2);
        for gram in &grams2 {
            if let Some(list) = self.bigram_postings.get(gram) {
                let slice = length_window_slice(list, min_len, max_len);
                if !slice.is_empty() {
                    slices.push(slice);
                }
            }
        }

        let w_max = q_len.max(max_len as usize) / 2;
        if w_max > 16 {
            any_slice_truncated = true;
        }
        let w_clamped = w_max.min(16);
        let mut q_pos_chars: Vec<(u8, char)> = Vec::new();
        for (pos, &c) in q_chars.iter().enumerate() {
            if c.is_whitespace() {
                continue;
            }
            let b_lo = (pos.saturating_sub(w_clamped) / 4).min(127) as u8;
            let b_hi = ((pos + w_clamped) / 4).min(127) as u8;
            for b in b_lo..=b_hi {
                if !q_pos_chars.contains(&(b, c)) {
                    q_pos_chars.push((b, c));
                    if let Some(list) = self.pos_char_postings.get(&(b, c)) {
                        let slice = length_window_slice(list, min_len, max_len);
                        if !slice.is_empty() {
                            slices.push(slice);
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
                        if !slice.is_empty() {
                            slices.push(slice);
                        }
                    }
                }
            }
        }

        slices.sort_unstable_by_key(|s| s.len());

        let mut scanned_postings = 0usize;
        let mut verified_count = 0usize;
        for slice in slices {
            if slice.len() <= BASE_FEATURE_SCAN
                && scanned_postings.saturating_add(slice.len()) <= TOTAL_POSTING_BUDGET
                && verified_count < BASE_FUZZY_VERIFY
            {
                scanned_postings = scanned_postings.saturating_add(slice.len());
                for &(_, k_idx) in slice {
                    let word = (k_idx as usize) >> 6;
                    let bit = 1u64 << (k_idx & 63);
                    if (visited[word] & bit) != 0 {
                        continue;
                    }
                    let sk = &self.sketches_by_idx[k_idx as usize];
                    if passes_sketch_prefilter(sk, &q_ctx, running_cutoff) {
                        visited[word] |= bit;
                        verified_count += 1;
                        visit(k_idx, &mut running_cutoff);
                    }
                }
            } else {
                any_slice_truncated = true;
            }
        }

        if any_slice_truncated {
            let (cur_min, cur_max) = length_bounds(q_len, running_cutoff);
            let win_min = cur_min.max(min_len);
            let win_max = cur_max.min(max_len);
            let lo = self.sorted_sketches.partition_point(|sk| sk.char_len < win_min);
            let hi = self.sorted_sketches.partition_point(|sk| sk.char_len <= win_max);
            if lo < hi {
                let mut bounds = LenFilterBounds::compute(q_len, self.sorted_sketches[lo].char_len, running_cutoff);
                for sk in &self.sorted_sketches[lo..hi] {
                    if sk.char_len != bounds.k_len || running_cutoff.to_bits() != bounds.cutoff_bits {
                        bounds = LenFilterBounds::compute(q_len, sk.char_len, running_cutoff);
                    }
                    let missing_q = ((q_ctx.q_mask[0] & !sk.mask[0]).count_ones()
                        + (q_ctx.q_mask[1] & !sk.mask[1]).count_ones()) as i32;
                    if missing_q > bounds.max_missing_q_p4 {
                        continue;
                    }
                    let word = (sk.k_idx as usize) >> 6;
                    let bit = 1u64 << (sk.k_idx & 63);
                    if (visited[word] & bit) != 0 {
                        continue;
                    }
                    let max_p = 4.min(q_len).min(sk.char_len as usize);
                    let mut p = 0usize;
                    while p < max_p && q_ctx.q_prefix4[p] == sk.prefix4[p] {
                        p += 1;
                    }
                    if missing_q <= bounds.max_missing_q[p] {
                        let missing_k = ((!q_ctx.q_mask[0] & sk.mask[0]).count_ones()
                            + (!q_ctx.q_mask[1] & sk.mask[1]).count_ones())
                            as i32;
                        if missing_k <= bounds.max_missing_k[p] {
                            visited[word] |= bit;
                            visit(sk.k_idx, &mut running_cutoff);
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn mark_visited(visited: &mut [u64], k_idx: u32) -> bool {
    let word = (k_idx as usize) >> 6;
    let bit = 1u64 << (k_idx & 63);
    if (visited[word] & bit) == 0 {
        visited[word] |= bit;
        true
    } else {
        false
    }
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
