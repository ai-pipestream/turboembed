//! Sentence tokenization and Viterbi trellis decoding over `NGramLanguageModel`
//! for [`super::ContextSpellCorrector`].

use std::ops::Range;

use turbo_analysis::Span;
use turbo_ngram::NGramLanguageModel;

use super::compound::tokens;
use super::guards::core;

/// Stupid Backoff factor matching `NGramLanguageModel`.
const STUPID_BACKOFF_ALPHA: f64 = 0.4;

#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub word: String,
    pub trellis_edit_cost: f64,
    pub report_distance: usize,
    pub is_original: bool,
}

#[derive(Clone, Debug)]
struct TrellisState {
    history: Vec<usize>,
    cand_idx: usize,
    cost: f64,
    backptr: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct TokenSlot {
    pub full: Range<usize>,
    pub core: Range<usize>,
    pub protected: bool,
}

pub(crate) fn collect_tokens(text: &str, protected: Option<&[Span]>) -> Vec<TokenSlot> {
    let mut out = Vec::new();
    let spans = protected.unwrap_or(&[]);
    let mut p_idx = 0;
    // Whitespace tokenization keeps unspaced CJK text in one
    // token; split after unambiguous non-ASCII sentence
    // terminators (STerm, not the ambiguous full stop) so
    // sentence grouping and trellis context stay per sentence.
    // ASCII terminators do not split: a `?` inside a URL or a
    // `!` glued to a word must reach the guards as one token,
    // as it did before this split existed.
    let is_sterm = |c: char| {
        !c.is_ascii() && turbo_unicode::sentence_break::sentence_break(c as u32) == turbo_unicode::SentenceBreak::STerm
    };
    let tokens = tokens(text).into_iter().flat_map(|tok| {
        let mut pieces = Vec::new();
        let mut start = tok.start;
        for (off, c) in text[tok.clone()].char_indices() {
            let end = tok.start + off + c.len_utf8();
            if is_sterm(c) && end < tok.end {
                pieces.push(start..end);
                start = end;
            }
        }
        pieces.push(start..tok.end);
        pieces
    });
    for tok in tokens {
        while p_idx < spans.len() && (spans[p_idx].end as usize) <= tok.start {
            p_idx += 1;
        }
        let is_protected = spans[p_idx..]
            .iter()
            .take_while(|s| (s.start as usize) < tok.end)
            .any(|s| !s.is_empty() && (s.end as usize) > tok.start);
        let (c_start, c_end) = core(&text[tok.clone()]);
        let core_range = tok.start + c_start..tok.start + c_end;
        if !core_range.is_empty() {
            out.push(TokenSlot { full: tok, core: core_range, protected: is_protected });
        }
    }
    out
}

pub(crate) fn group_into_sentences<'a>(
    text: &str,
    all_tokens: &'a [TokenSlot],
    sentence_spans: Option<&[Span]>,
) -> Vec<&'a [TokenSlot]> {
    if let Some(sents) = sentence_spans.filter(|s| !s.is_empty()) {
        let mut groups = Vec::new();
        let mut start = 0;
        for s in sents {
            let gap_start = start;
            while start < all_tokens.len() && all_tokens[start].full.end <= s.start as usize {
                start += 1;
            }
            if gap_start < start {
                groups.push(&all_tokens[gap_start..start]);
            }
            let mut end = start;
            while end < all_tokens.len() && all_tokens[end].full.start < s.end as usize {
                end += 1;
            }
            if start < end {
                groups.push(&all_tokens[start..end]);
                start = end;
            }
        }
        if start < all_tokens.len() {
            groups.push(&all_tokens[start..]);
        }
        if !groups.is_empty() {
            return groups;
        }
    }

    let mut groups = Vec::new();
    let mut start = 0;
    for (i, tok) in all_tokens.iter().enumerate() {
        let ends_sent =
            text[tok.core.end..tok.full.end].chars().any(|c| {
                matches!(
                    turbo_unicode::sentence_break::sentence_break(c as u32),
                    turbo_unicode::SentenceBreak::STerm | turbo_unicode::SentenceBreak::ATerm
                )
            }) || all_tokens.get(i + 1).is_some_and(|next| text[tok.full.end..next.full.start].contains('\n'));
        if ends_sent {
            groups.push(&all_tokens[start..=i]);
            start = i + 1;
        }
    }
    if start < all_tokens.len() {
        groups.push(&all_tokens[start..]);
    }
    groups
}

/// Computes a deterministic 64-bit content fingerprint over `lm`'s order,
/// n-grams, and counts, independent of internal hash-table iteration order.
pub(crate) fn lm_fingerprint(lm: &NGramLanguageModel) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mix_bytes = |mut h: u64, bytes: &[u8]| -> u64 {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    };
    let splitmix64 = |mut z: u64| -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };

    let mut sum = 0u64;
    let mut xor = 0u64;
    for ngram in lm.iter() {
        let mut eh = mix_bytes(FNV_OFFSET, &(ngram.len() as u64).to_le_bytes());
        for token in ngram {
            eh = mix_bytes(eh, &(token.len() as u64).to_le_bytes());
            eh = mix_bytes(eh, token.as_bytes());
        }
        let count = lm.get_count(ngram);
        eh = splitmix64(mix_bytes(eh, &count.to_le_bytes()));
        sum = sum.wrapping_add(eh);
        xor ^= eh.rotate_left(23);
    }

    let mut hash = FNV_OFFSET;
    hash = mix_bytes(hash, &(lm.order() as u64).to_le_bytes());
    hash = mix_bytes(hash, &(lm.size() as u64).to_le_bytes());
    hash = mix_bytes(hash, &sum.to_le_bytes());
    hash = mix_bytes(hash, &xor.to_le_bytes());
    splitmix64(hash)
}

/// Decodes the minimum-cost candidate sequence across `cand_sets` via Viterbi
/// with beam width `max_beam_width`.
pub(crate) fn viterbi_decode(
    lm: &NGramLanguageModel,
    edit_weight: f64,
    max_beam_width: usize,
    cand_sets: Vec<Vec<Candidate>>,
) -> Vec<Candidate> {
    if cand_sets.iter().all(|cs| cs.len() <= 1 && cs.first().is_some_and(|c| c.is_original)) {
        return cand_sets.into_iter().map(|mut cs| cs.remove(0)).collect();
    }
    let beam_width = max_beam_width.max(1);
    let m = cand_sets.len();
    let order = lm.order().max(1);
    let history_len = order - 1;
    let use_bos = order > 1 && lm.contains(&["<s>"]);
    let use_eos = order > 1 && lm.contains(&["</s>"]);
    let total = (lm.number_of_grams() as f64).max(lm.size() as f64).max(1.0);

    let mut layers: Vec<Vec<TrellisState>> = Vec::with_capacity(m);
    for (i, cands) in cand_sets.iter().enumerate() {
        let mut next_states: Vec<TrellisState> = Vec::new();
        if i == 0 {
            for (c_idx, cand) in cands.iter().enumerate() {
                let lm_cost = if use_bos {
                    -stupid_backoff_prob(lm, &["<s>", cand.word.as_str()], total).ln()
                } else {
                    -unigram_prob(lm, cand.word.as_str(), total).ln()
                };
                let cost = edit_weight * cand.trellis_edit_cost + lm_cost;
                next_states.push(TrellisState { history: vec![c_idx], cand_idx: c_idx, cost, backptr: 0 });
            }
        } else {
            let prev_states = &layers[i - 1];
            let cand_unigram_probs: Vec<f64> = cands.iter().map(|c| unigram_prob(lm, c.word.as_str(), total)).collect();
            let cand_edit_costs: Vec<f64> = cands.iter().map(|c| edit_weight * c.trellis_edit_cost).collect();

            let mut state_index: std::collections::HashMap<Vec<usize>, usize> =
                std::collections::HashMap::with_capacity(prev_states.len().saturating_mul(cands.len()).min(2048));
            let buf_cap = history_len.min(m).saturating_add(2);
            let mut ngram_buf: Vec<&str> = Vec::with_capacity(buf_cap);
            let mut prefix_counts: Vec<i32> = Vec::with_capacity(buf_cap);

            for (prev_idx, prev) in prev_states.iter().enumerate() {
                ngram_buf.clear();
                let start_pos = i.saturating_sub(history_len);
                if use_bos && i < history_len {
                    ngram_buf.push("<s>");
                }
                for (offset, &hist_c_idx) in prev.history.iter().enumerate() {
                    let tok_pos = i - prev.history.len() + offset;
                    if tok_pos >= start_pos {
                        ngram_buf.push(cand_sets[tok_pos][hist_c_idx].word.as_str());
                    }
                }
                let hist_len = ngram_buf.len();
                prefix_counts.clear();
                for s in 0..hist_len {
                    prefix_counts.push(lm.get_count(&ngram_buf[s..hist_len]));
                }
                ngram_buf.push("");

                let base_hist: Vec<usize> = prev
                    .history
                    .iter()
                    .copied()
                    .skip(usize::from(prev.history.len() >= history_len && history_len > 0))
                    .collect();

                for (c_idx, cand) in cands.iter().enumerate() {
                    ngram_buf[hist_len] = cand.word.as_str();
                    let p = backoff_prob_with_prefix_counts(lm, &ngram_buf, &prefix_counts, cand_unigram_probs[c_idx]);
                    let total_cost = prev.cost + cand_edit_costs[c_idx] - p.ln();

                    let mut new_hist = if history_len == 0 { Vec::with_capacity(1) } else { base_hist.clone() };
                    new_hist.push(c_idx);

                    if let Some(&existing_idx) = state_index.get(&new_hist) {
                        let existing = &mut next_states[existing_idx];
                        if total_cost < existing.cost {
                            existing.cost = total_cost;
                            existing.backptr = prev_idx;
                        }
                    } else {
                        let idx = next_states.len();
                        state_index.insert(new_hist.clone(), idx);
                        next_states.push(TrellisState {
                            history: new_hist,
                            cand_idx: c_idx,
                            cost: total_cost,
                            backptr: prev_idx,
                        });
                    }
                }
            }
        }
        if next_states.len() > beam_width {
            next_states.sort_by(|a, b| {
                a.cost
                    .total_cmp(&b.cost)
                    .then_with(|| a.cand_idx.cmp(&b.cand_idx))
                    .then_with(|| a.history.cmp(&b.history))
            });
            next_states.truncate(beam_width);
        }
        layers.push(next_states);
    }

    let last_layer = layers.last().expect("sentence is non-empty");
    let mut best_idx = 0;
    let mut best_cost = f64::INFINITY;
    for (idx, state) in last_layer.iter().enumerate() {
        let mut cost = state.cost;
        if use_eos {
            let mut ngram_buf: Vec<&str> = Vec::with_capacity(history_len.min(m).saturating_add(1));
            let start_pos = m.saturating_sub(history_len);
            for (offset, &hist_c_idx) in state.history.iter().enumerate() {
                let tok_pos = m - state.history.len() + offset;
                if tok_pos >= start_pos {
                    ngram_buf.push(cand_sets[tok_pos][hist_c_idx].word.as_str());
                }
            }
            ngram_buf.push("</s>");
            cost -= stupid_backoff_prob(lm, &ngram_buf, total).ln();
        }
        if cost < best_cost {
            best_cost = cost;
            best_idx = idx;
        }
    }

    let mut chosen_indices = vec![0usize; m];
    let mut cur = best_idx;
    for i in (0..m).rev() {
        let st = &layers[i][cur];
        chosen_indices[i] = st.cand_idx;
        cur = st.backptr;
    }
    cand_sets.into_iter().zip(chosen_indices).map(|(mut cs, idx)| cs.swap_remove(idx)).collect()
}

#[inline]
fn unigram_prob(lm: &NGramLanguageModel, word: &str, total: f64) -> f64 {
    let count = lm.get_count(&[word]);
    if count > 0 { (count as f64 / total).min(1.0) } else { STUPID_BACKOFF_ALPHA / (total * 10.0) }
}

#[inline]
fn backoff_prob_with_prefix_counts(
    lm: &NGramLanguageModel,
    ngram: &[&str],
    prefix_counts: &[i32],
    unigram_p: f64,
) -> f64 {
    let mut alpha_scale = 1.0;
    for (s, &pref_cnt) in prefix_counts.iter().enumerate() {
        if pref_cnt > 0 {
            let cnt = lm.get_count(&ngram[s..]);
            if cnt > 0 {
                return alpha_scale * (cnt as f64 / pref_cnt as f64).min(1.0);
            }
        }
        alpha_scale *= STUPID_BACKOFF_ALPHA;
    }
    alpha_scale * unigram_p
}

fn stupid_backoff_prob(lm: &NGramLanguageModel, ngram: &[&str], total: f64) -> f64 {
    if ngram.is_empty() {
        return 1.0 / total;
    }
    if ngram.len() > 1 {
        let prefix = &ngram[..ngram.len() - 1];
        let prefix_count = lm.get_count(prefix);
        if prefix_count > 0 {
            let count = lm.get_count(ngram);
            if count > 0 {
                return (count as f64 / prefix_count as f64).min(1.0);
            }
        }
        STUPID_BACKOFF_ALPHA * stupid_backoff_prob(lm, &ngram[1..], total)
    } else {
        unigram_prob(lm, ngram[0], total)
    }
}

#[cfg(test)]
mod grouping_tests {
    #[test]
    fn fallback_grouping_ends_at_unicode_terminators() {
        for text in [
            "\u{4ECA}\u{65E5}\u{306F}\u{96E8}\u{3002}\u{660E}\u{65E5}\u{306F}\u{6674}\u{308C}\u{3002}",
            "\u{092F}\u{0939} \u{0939}\u{0948}\u{0964} \u{0935}\u{0939} \u{0939}\u{0948}\u{0964}",
        ] {
            let toks = super::collect_tokens(text, None);
            let groups = super::group_into_sentences(text, &toks, None);
            assert_eq!(groups.len(), 2, "{text:?} grouped as {}", groups.len());
        }
    }
}
