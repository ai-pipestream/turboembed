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
    for tok in tokens(text) {
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
            while start < all_tokens.len() && all_tokens[start].full.end <= s.start as usize {
                start += 1;
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
        let ends_sent = text[tok.core.end..tok.full.end].chars().any(|c| matches!(c, '.' | '!' | '?'))
            || all_tokens.get(i + 1).is_some_and(|next| text[tok.full.end..next.full.start].contains('\n'));
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

/// Computes a deterministic 64-bit FNV-1a content fingerprint over `lm`'s
/// order, n-grams, and counts.
pub(crate) fn lm_fingerprint(lm: &NGramLanguageModel) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    mix(&(lm.order() as u64).to_le_bytes());
    mix(&(lm.size() as u64).to_le_bytes());
    for ngram in lm.iter() {
        mix(&(ngram.len() as u64).to_le_bytes());
        for token in ngram {
            mix(&(token.len() as u64).to_le_bytes());
            mix(token.as_bytes());
        }
        let count = lm.get_count(ngram);
        mix(&count.to_le_bytes());
    }
    hash
}

/// Decodes the minimum-cost candidate sequence across `cand_sets` via Viterbi.
pub(crate) fn viterbi_decode(
    lm: &NGramLanguageModel,
    edit_weight: f64,
    cand_sets: Vec<Vec<Candidate>>,
) -> Vec<Candidate> {
    if cand_sets.iter().all(|cs| cs.len() <= 1 && cs.first().is_some_and(|c| c.is_original)) {
        return cand_sets.into_iter().map(|mut cs| cs.remove(0)).collect();
    }
    let m = cand_sets.len();
    let order = lm.order().max(1);
    let history_len = order - 1;
    let use_bos = order > 1 && lm.contains(&["<s>"]);
    let use_eos = order > 1 && lm.contains(&["</s>"]);

    let mut layers: Vec<Vec<TrellisState>> = Vec::with_capacity(m);
    for (i, cands) in cand_sets.iter().enumerate() {
        let mut next_states: Vec<TrellisState> = Vec::new();
        if i == 0 {
            for (c_idx, cand) in cands.iter().enumerate() {
                let lm_cost = if use_bos {
                    neg_log_prob(lm, &["<s>", cand.word.as_str()])
                } else {
                    neg_log_prob(lm, &[cand.word.as_str()])
                };
                let cost = edit_weight * cand.trellis_edit_cost + lm_cost;
                next_states.push(TrellisState { history: vec![c_idx], cand_idx: c_idx, cost, backptr: 0 });
            }
        } else {
            let prev_states = &layers[i - 1];
            let mut ngram_buf: Vec<&str> = Vec::with_capacity(history_len + 2);
            for (c_idx, cand) in cands.iter().enumerate() {
                let edit_cost = edit_weight * cand.trellis_edit_cost;
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
                    ngram_buf.push(cand.word.as_str());
                    let total = prev.cost + edit_cost + neg_log_prob(lm, &ngram_buf);

                    let mut new_hist: Vec<usize> = prev
                        .history
                        .iter()
                        .copied()
                        .skip(usize::from(prev.history.len() >= history_len && history_len > 0))
                        .collect();
                    if history_len == 0 {
                        new_hist.clear();
                    }
                    new_hist.push(c_idx);

                    if let Some(existing) =
                        next_states.iter_mut().find(|s| s.cand_idx == c_idx && s.history == new_hist)
                    {
                        if total < existing.cost {
                            existing.cost = total;
                            existing.backptr = prev_idx;
                        }
                    } else {
                        next_states.push(TrellisState {
                            history: new_hist,
                            cand_idx: c_idx,
                            cost: total,
                            backptr: prev_idx,
                        });
                    }
                }
            }
        }
        layers.push(next_states);
    }

    let last_layer = layers.last().expect("sentence is non-empty");
    let mut best_idx = 0;
    let mut best_cost = f64::INFINITY;
    for (idx, state) in last_layer.iter().enumerate() {
        let mut cost = state.cost;
        if use_eos {
            let mut ngram_buf: Vec<&str> = Vec::with_capacity(history_len + 1);
            let start_pos = m.saturating_sub(history_len);
            for (offset, &hist_c_idx) in state.history.iter().enumerate() {
                let tok_pos = m - state.history.len() + offset;
                if tok_pos >= start_pos {
                    ngram_buf.push(cand_sets[tok_pos][hist_c_idx].word.as_str());
                }
            }
            ngram_buf.push("</s>");
            cost += neg_log_prob(lm, &ngram_buf);
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

/// Computes $-\ln P_{\text{LM}}(w_k \mid w_1 \dots w_{k-1})$ using Stupid Backoff
/// with a smooth unseen floor so every candidate receives a finite log-probability.
fn neg_log_prob(lm: &NGramLanguageModel, ngram: &[&str]) -> f64 {
    let total = (lm.number_of_grams() as f64).max(lm.size() as f64).max(1.0);
    let p = stupid_backoff_prob(lm, ngram, total);
    -p.ln()
}

fn stupid_backoff_prob(lm: &NGramLanguageModel, ngram: &[&str], total: f64) -> f64 {
    if ngram.is_empty() {
        return 1.0 / total;
    }
    let count = lm.get_count(ngram);
    if ngram.len() > 1 {
        let prefix = &ngram[..ngram.len() - 1];
        let prefix_count = lm.get_count(prefix) as f64;
        if count > 0 && prefix_count > 0.0 {
            (count as f64 / prefix_count).min(1.0)
        } else {
            STUPID_BACKOFF_ALPHA * stupid_backoff_prob(lm, &ngram[1..], total)
        }
    } else if count > 0 {
        (count as f64 / total).min(1.0)
    } else {
        STUPID_BACKOFF_ALPHA / (total * 10.0)
    }
}
