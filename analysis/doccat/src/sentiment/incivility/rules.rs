use super::*;

pub(super) fn scan_lexicon(
    lexicon: &IncivilityLexicon,
    token_lower: &[String],
    tok_start: usize,
    tok_end: usize,
) -> (IncivilityScores, Vec<(IncivilityLabel, IxRange<Tokens>)>) {
    let mut scores = IncivilityScores::default();
    let mut evidence: Vec<(IncivilityLabel, IxRange<Tokens>)> = Vec::new();

    // Walk the unit once: at each token, try only the patterns starting with
    // that token (the lexicon's first-token index). A pattern's matches stay
    // non-overlapping: after a match at `i`, `next_from` skips it forward past
    // the matched run. Matched hits are then sorted by `(entry_idx, start)` so
    // probabilistic folding and evidence order match the entry-major reference
    // scan bit-for-bit.
    let mut hits: Vec<(u32, u32, u32, IncivilityLabel, f64)> = Vec::new();
    let mut next_from = vec![0usize; lexicon.entries.len()];
    for i in tok_start..tok_end {
        let w = token_lower[i].as_str();
        let Ok(group) = lexicon.first_token.binary_search_by(|(t, _, _)| t.as_str().cmp(w)) else { continue };
        let (_, lo, hi) = lexicon.first_token[group];
        let entry_run = &lexicon.entries[lo as usize..hi as usize];
        let next_run = &mut next_from[lo as usize..hi as usize];
        for (offset, ((pat, label, weight), next)) in entry_run.iter().zip(next_run.iter_mut()).enumerate() {
            if i < *next {
                continue;
            }
            let pat_len = pat.len();
            if i + pat_len > tok_end {
                continue;
            }
            let mut matched = true;
            for k in 0..pat_len {
                if token_lower[i + k] != pat[k] {
                    matched = false;
                    break;
                }
            }
            if matched {
                // Guard against negated benign adjectives like `"not stupid"` or `"not dumb"`.
                let negated_mild = pat_len == 1
                    && matches!(pat[0].as_str(), "stupid" | "dumb" | "hell" | "damn")
                    && i > tok_start
                    && matches!(token_lower[i - 1].as_str(), "not" | "never" | "no");
                if !negated_mild {
                    let entry_idx = lo + offset as u32;
                    hits.push((entry_idx, i as u32, (i + pat_len) as u32, *label, *weight));
                }
                *next = i + pat_len;
            }
        }
    }

    hits.sort_unstable_by_key(|&(entry_idx, start, _, _, _)| (entry_idx, start));
    for (_, start, end, label, weight) in hits {
        let prev = scores.get(label);
        let combined = if prev > 0.0 { (1.0 - (1.0 - prev) * (1.0 - weight)).clamp(0.0, 1.0) } else { weight };
        scores.set(label, combined);
        evidence.push((label, IxRange::new(start, end)));
    }

    (scores, evidence)
}

/// Opinion-conflict patterns (Zhang et al., 2018, arXiv:1809.00289) that become
/// uncivil when directed at a parent/quoted post in a reply context.
const REPLY_CONFLICT_PATTERNS: &[(&[&str], IncivilityLabel, f64)] = &[
    (&["you", "have", "no", "clue"], IncivilityLabel::Insult, 0.78),
    (&["you", "know", "nothing"], IncivilityLabel::Insult, 0.76),
    (&["you", "are", "delusional"], IncivilityLabel::Insult, 0.82),
    (&["you", "are", "lying"], IncivilityLabel::Insult, 0.76),
    (&["you", "always", "lie"], IncivilityLabel::Insult, 0.78),
    (&["your", "argument", "is", "garbage"], IncivilityLabel::Insult, 0.80),
    (&["stop", "spreading", "lies"], IncivilityLabel::Insult, 0.75),
    (&["hypocrite"], IncivilityLabel::Insult, 0.76),
    (&["hypocrites"], IncivilityLabel::Insult, 0.76),
    (&["people", "like", "you", "do", "not", "belong"], IncivilityLabel::Intolerance, 0.84),
    (&["your", "kind", "should", "be", "silenced"], IncivilityLabel::Intolerance, 0.86),
];

pub(super) fn score_reply_conflict(
    token_lower: &[String],
    parent_start: usize,
    parent_end: usize,
    reply_start: usize,
    reply_end: usize,
) -> (IncivilityScores, Vec<(IncivilityLabel, IxRange<Tokens>)>) {
    let mut scores = IncivilityScores::default();
    let mut evidence = Vec::new();
    let parent_has_content = (parent_start..parent_end).any(|t| !token_lower[t].is_empty());
    if !parent_has_content {
        return (scores, evidence);
    }

    for &(pat, label, weight) in REPLY_CONFLICT_PATTERNS {
        let m = pat.len();
        if m == 0 || reply_start + m > reply_end {
            continue;
        }
        for i in reply_start..=(reply_end - m) {
            if (0..m).all(|k| token_lower[i + k] == pat[k]) {
                if weight > scores.get(label) {
                    scores.set(label, weight);
                }
                evidence.push((label, IxRange::new(i as u32, (i + m) as u32)));
            }
        }
    }

    (scores, evidence)
}

pub(super) fn find_parent_or_quote_token_range(
    structure: &Structure,
    unit_start_byte: u32,
    unit_end_byte: u32,
    tok_starts: &[u32],
    tok_ends: &[u32],
) -> Option<(usize, usize)> {
    let (b_starts, b_ends) = structure.spans.columns();
    // Find the innermost non-Quote block containing `unit_start_byte..unit_end_byte`.
    let mut enclosing_block: Option<usize> = None;
    for b_idx in 0..structure.len() {
        if b_starts[b_idx] <= unit_start_byte && unit_end_byte <= b_ends[b_idx] {
            if structure.kind[b_idx] == Block::Quote {
                // Inside the quote itself, not the reply to the quote.
                return None;
            }
            enclosing_block = Some(b_idx);
        }
    }
    let b_idx = enclosing_block?;
    find_block_parent_or_quote_token_range(structure, b_idx, tok_starts, tok_ends)
}

pub(super) fn find_block_parent_or_quote_token_range(
    structure: &Structure,
    b_idx: usize,
    tok_starts: &[u32],
    tok_ends: &[u32],
) -> Option<(usize, usize)> {
    if structure.kind[b_idx] == Block::Quote {
        return None;
    }
    let (b_starts, b_ends) = structure.spans.columns();
    // 1. Check if the immediately preceding block is a `Block::Quote`.
    if b_idx > 0 && structure.kind[b_idx - 1] == Block::Quote {
        let q_start = b_starts[b_idx - 1];
        let q_end = b_ends[b_idx - 1];
        let ts = tok_ends.partition_point(|&e| e <= q_start);
        let te = tok_starts.partition_point(|&s| s < q_end);
        if ts < te {
            return Some((ts, te));
        }
    }
    // 2. Check if the enclosing parent block has an earlier child `Block::Quote` or earlier sibling paragraph.
    if let Some(parent_ix) = structure.parent[b_idx] {
        let p_idx = parent_ix.get();
        for prev in (p_idx + 1)..b_idx {
            if structure.parent[prev] == Some(parent_ix)
                && matches!(structure.kind[prev], Block::Quote | Block::Paragraph)
            {
                let q_start = b_starts[prev];
                let q_end = b_ends[prev];
                let ts = tok_ends.partition_point(|&e| e <= q_start);
                let te = tok_starts.partition_point(|&s| s < q_end);
                if ts < te {
                    return Some((ts, te));
                }
            }
        }
    }
    None
}

pub(super) fn select_salient_token_span(token_lower: &[String], tok_start: usize, tok_end: usize) -> IxRange<Tokens> {
    let mut best_idx = tok_start;
    let mut best_len = 0usize;
    for (idx, w) in token_lower[tok_start..tok_end].iter().enumerate() {
        let tok_idx = tok_start + idx;
        if w.is_empty() || is_stopword(w) {
            continue;
        }
        if w.len() > best_len {
            best_len = w.len();
            best_idx = tok_idx;
        }
    }
    IxRange::new(best_idx as u32, (best_idx + 1) as u32)
}

pub(super) fn is_stopword(w: &str) -> bool {
    matches!(
        w,
        "the"
            | "a"
            | "an"
            | "and"
            | "or"
            | "but"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "to"
            | "of"
            | "in"
            | "on"
            | "for"
            | "with"
            | "at"
            | "by"
            | "from"
            | "this"
            | "that"
            | "these"
            | "those"
            | "it"
            | "its"
            | "i"
            | "we"
            | "you"
            | "they"
            | "he"
            | "she"
    )
}
