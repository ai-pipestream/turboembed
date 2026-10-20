//! Syntactic clause and phrase boundary detection from dependency trees,
//! constituency phrases, shallow phrase chunks, POS tags, and surface text.

use std::ops::Range;

use turbo_analysis::{Ix, Span, View};
use turbo_analysis_layers::{Dependencies, PhraseChunks, Phrases, PosTags, Tokens};

use super::{ClauseBreak, ClauseCosts, SentenceClauseCuts};

fn base_rel(rel: &str) -> &str {
    rel.split(':').next().unwrap_or(rel)
}

fn token_text(view: Option<&View>, span: Span) -> Option<&str> {
    view.and_then(|v| v.text().get(span.range())).map(str::trim)
}

fn pos_label(pos: Option<&PosTags>, tok_ix: usize) -> Option<&str> {
    pos.and_then(|p| p.tags.get(tok_ix).map(|&t| p.tagset.label(t)))
}

fn dep_label(deps: Option<&Dependencies<Tokens>>, tok_ix: usize) -> Option<&str> {
    deps.and_then(|d| d.relation.get(tok_ix).map(|&t| d.relations.label(t)))
}

fn is_opening_punct(text: Option<&str>, pos: Option<&str>) -> bool {
    matches!(text, Some("(" | "[" | "{" | "``" | "\u{201c}" | "\u{2018}")) || matches!(pos, Some("-LRB-" | "``"))
}

fn is_closing_or_sep_punct(
    tok_ix: usize,
    span: Span,
    deps: Option<&Dependencies<Tokens>>,
    pos: Option<&PosTags>,
    view: Option<&View>,
) -> bool {
    let txt = token_text(view, span);
    let p = pos_label(pos, tok_ix);
    if is_opening_punct(txt, p) {
        return false;
    }
    if matches!(txt, Some("," | ";" | ":" | "." | "!" | "?" | ")" | "]" | "}" | "--" | "-")) {
        return true;
    }
    if matches!(p, Some("," | ":" | "." | "-RRB-" | "PUNCT")) {
        return true;
    }
    matches!(dep_label(deps, tok_ix).map(base_rel), Some("punct"))
}

fn is_hard_clause_punct(tok_ix: usize, span: Span, pos: Option<&PosTags>, view: Option<&View>) -> bool {
    let txt = token_text(view, span);
    let p = pos_label(pos, tok_ix);
    matches!(txt, Some(";" | ":" | "--")) || matches!(p, Some(":"))
}

fn is_comma(tok_ix: usize, span: Span, pos: Option<&PosTags>, view: Option<&View>) -> bool {
    matches!(token_text(view, span), Some(",")) || matches!(pos_label(pos, tok_ix), Some(","))
}

fn is_coordinator_or_subordinator(
    tok_ix: usize,
    span: Span,
    deps: Option<&Dependencies<Tokens>>,
    pos: Option<&PosTags>,
    view: Option<&View>,
) -> bool {
    if matches!(dep_label(deps, tok_ix).map(base_rel), Some("cc" | "mark")) {
        return true;
    }
    if matches!(pos_label(pos, tok_ix), Some("CC" | "CCONJ" | "SCONJ" | "WDT" | "WP" | "WP$" | "WRB")) {
        return true;
    }
    let Some(txt) = token_text(view, span) else {
        return false;
    };
    matches!(
        txt.to_ascii_lowercase().as_str(),
        "and"
            | "but"
            | "or"
            | "nor"
            | "yet"
            | "so"
            | "because"
            | "although"
            | "though"
            | "while"
            | "whereas"
            | "when"
            | "whenever"
            | "where"
            | "wherever"
            | "if"
            | "unless"
            | "until"
            | "since"
            | "after"
            | "before"
            | "which"
            | "who"
            | "whom"
            | "whose"
    )
}

fn is_clause_rel(base: &str) -> bool {
    matches!(base, "parataxis" | "list" | "remnant" | "advcl" | "ccomp" | "csubj" | "acl" | "xcomp" | "conj")
}

/// Precomputes in $O(m)$ time for all sentence-local tokens `0..m` in
/// `tok_range`:
/// 1. `bounds[k] = (lo, hi)`: the minimum and maximum sentence-local token
///    indices in the dependency subtree rooted at `k`.
/// 2. `verbal[k]`: whether token `tok_range.start + k` has a verbal POS tag
///    or an in-sentence verbal dependent (`nsubj`, `csubj`, `obj`, `iobj`,
///    `obl`, `aux`, `cop`, `mark`, `advmod`).
fn precompute_dep_subtrees(
    tok_range: &Range<usize>,
    deps: &Dependencies<Tokens>,
    pos: Option<&PosTags>,
) -> (Vec<(usize, usize)>, Vec<bool>) {
    let m = tok_range.len();
    let mut first_child = vec![u32::MAX; m];
    let mut next_sibling = vec![u32::MAX; m];
    let mut verbal: Vec<bool> = (0..m)
        .map(|k| {
            pos_label(pos, tok_range.start + k)
                .is_some_and(|p| p.starts_with("VB") || matches!(p, "VERB" | "AUX" | "MD"))
        })
        .collect();

    for (k, sibling) in next_sibling.iter_mut().enumerate() {
        let t = tok_range.start + k;
        let Some(Some(h)) = deps.head.get(t) else {
            continue;
        };
        let h_ix = h.get();
        if h_ix < tok_range.start || h_ix >= tok_range.end || h_ix == t {
            continue;
        }
        let p = h_ix - tok_range.start;
        *sibling = first_child[p];
        first_child[p] = k as u32;
        if deps.relation.get(t).is_some_and(|&r| {
            matches!(
                base_rel(deps.relations.label(r)),
                "nsubj" | "csubj" | "obj" | "iobj" | "obl" | "aux" | "cop" | "mark" | "advmod"
            )
        }) {
            verbal[p] = true;
        }
    }

    let mut bounds: Vec<(usize, usize)> = (0..m).map(|k| (k, k)).collect();
    let mut state = vec![0u8; m];
    let mut stack: Vec<(usize, u32)> = Vec::with_capacity(m);
    for start_k in 0..m {
        if state[start_k] != 0 {
            continue;
        }
        state[start_k] = 1;
        stack.push((start_k, first_child[start_k]));
        while let Some((u, next_c)) = stack.last_mut() {
            let c = *next_c;
            if c != u32::MAX {
                let c_idx = c as usize;
                *next_c = next_sibling[c_idx];
                if state[c_idx] == 0 {
                    state[c_idx] = 1;
                    stack.push((c_idx, first_child[c_idx]));
                } else if state[c_idx] == 2 {
                    let (c_lo, c_hi) = bounds[c_idx];
                    bounds[*u].0 = bounds[*u].0.min(c_lo);
                    bounds[*u].1 = bounds[*u].1.max(c_hi);
                }
            } else {
                let (u_idx, _) = stack.pop().expect("non-empty stack");
                state[u_idx] = 2;
                if let Some(&(parent_u, _)) = stack.last() {
                    let (u_lo, u_hi) = bounds[u_idx];
                    bounds[parent_u].0 = bounds[parent_u].0.min(u_lo);
                    bounds[parent_u].1 = bounds[parent_u].1.max(u_hi);
                }
            }
        }
    }

    (bounds, verbal)
}

/// Adjusts a raw cut before sentence-local token `cut_before` (`0 < cut_before < m`)
/// so closing/separating punctuation stays attached to the left clause and a
/// preceding coordinating/subordinating conjunction starts the right clause.
fn adjust_cut(
    mut cut_before: usize,
    tok_range: Range<usize>,
    tok_spans: &[Span],
    breaks: &[ClauseBreak],
    deps: Option<&Dependencies<Tokens>>,
    pos: Option<&PosTags>,
    view: Option<&View>,
) -> Option<usize> {
    let m = tok_range.len();
    while cut_before < m
        && is_closing_or_sep_punct(tok_range.start + cut_before, tok_spans[cut_before], deps, pos, view)
    {
        cut_before += 1;
    }
    if cut_before > 1
        && is_coordinator_or_subordinator(tok_range.start + cut_before - 1, tok_spans[cut_before - 1], deps, pos, view)
        && !is_closing_or_sep_punct(tok_range.start + cut_before - 1, tok_spans[cut_before - 1], deps, pos, view)
    {
        cut_before -= 1;
        while cut_before < m
            && is_closing_or_sep_punct(tok_range.start + cut_before, tok_spans[cut_before], deps, pos, view)
        {
            cut_before += 1;
        }
    }
    if cut_before == 0 || cut_before >= m {
        return None;
    }
    let gap = cut_before - 1;
    (breaks[gap] != ClauseBreak::Held).then_some(gap)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sentence_cuts(
    sent_span: Span,
    tok_range: Range<usize>,
    tokens: &Tokens,
    deps: Option<&Dependencies<Tokens>>,
    phrases: Option<&Phrases>,
    phrase_chunks: Option<&PhraseChunks>,
    pos: Option<&PosTags>,
    view: Option<&View>,
    costs: ClauseCosts,
) -> SentenceClauseCuts {
    let mut tok_spans = Vec::with_capacity(tok_range.len());
    for t in tok_range.clone() {
        let raw = tokens.spans.get(Ix::new(t as u32));
        let start = raw.start.clamp(sent_span.start, sent_span.end);
        let end = raw.end.clamp(start, sent_span.end);
        tok_spans.push(Span::new(start, end));
    }
    let m = tok_spans.len();
    if m <= 1 {
        return SentenceClauseCuts { tokens: tok_spans, breaks: Vec::new() };
    }
    let mut breaks: Vec<ClauseBreak> = (0..m - 1)
        .map(|k| {
            if tok_spans[k].is_empty() || tok_spans[k + 1].is_empty() || tok_spans[k].end > tok_spans[k + 1].start {
                ClauseBreak::Held
            } else {
                ClauseBreak::Token(costs.token)
            }
        })
        .collect();

    if let Some(d) = deps {
        let has_any_clause_rel =
            (0..m).any(|k| dep_label(Some(d), tok_range.start + k).map(base_rel).is_some_and(is_clause_rel));
        if has_any_clause_rel {
            let (bounds, verbal) = precompute_dep_subtrees(&tok_range, d, pos);
            for k in 0..m {
                let t = tok_range.start + k;
                let Some(rel) = dep_label(Some(d), t) else {
                    continue;
                };
                let base = base_rel(rel);
                if !is_clause_rel(base) {
                    continue;
                }
                let (lo, hi) = bounds[k];
                let span_len = hi + 1 - lo;
                let kind = match base {
                    "parataxis" | "list" | "remnant" => Some(ClauseBreak::Clause(costs.clause)),
                    "advcl" | "ccomp" | "csubj" => Some(ClauseBreak::Clause(costs.subordinate)),
                    "acl" if rel == "acl:relcl" || span_len >= 2 => Some(ClauseBreak::Clause(costs.subordinate)),
                    "xcomp" if span_len >= 2 => Some(ClauseBreak::Clause(costs.subordinate)),
                    "conj" => {
                        let head_verbal = d.head.get(t).and_then(|&h| h).is_some_and(|h| {
                            let h_ix = h.get();
                            h_ix >= tok_range.start && h_ix < tok_range.end && verbal[h_ix - tok_range.start]
                        });
                        let preceded_by_punct = lo > 0
                            && is_closing_or_sep_punct(tok_range.start + lo - 1, tok_spans[lo - 1], Some(d), pos, view);
                        if span_len >= 2 || verbal[k] || head_verbal || preceded_by_punct {
                            Some(ClauseBreak::Clause(costs.clause))
                        } else {
                            Some(ClauseBreak::Phrase(costs.phrase))
                        }
                    }
                    _ => None,
                };
                if let Some(brk) = kind {
                    if lo > 0
                        && let Some(g) = adjust_cut(lo, tok_range.clone(), &tok_spans, &breaks, Some(d), pos, view)
                    {
                        breaks[g].upgrade(brk);
                    }
                    if hi + 1 < m
                        && let Some(g) = adjust_cut(hi + 1, tok_range.clone(), &tok_spans, &breaks, Some(d), pos, view)
                    {
                        breaks[g].upgrade(brk);
                    }
                }
            }
        }
    }

    if let Some(phr) = phrases {
        for i in 0..phr.spans.len() {
            let r = phr.tokens[i];
            let (start, end) = (r.start.get(), r.end.get());
            if start < tok_range.start || end > tok_range.end || start >= end {
                continue;
            }
            let (lo, hi) = (start - tok_range.start, end - tok_range.start);
            let label = phr.kinds.label(phr.kind[i]);
            let brk = match label {
                "S" | "SINV" | "SQ" | "SBARQ" => ClauseBreak::Clause(costs.clause),
                "SBAR" | "RRC" | "PRN" | "FRAG" | "UCP" => ClauseBreak::Clause(costs.subordinate),
                _ => ClauseBreak::Phrase(costs.phrase),
            };
            if lo > 0
                && let Some(g) = adjust_cut(lo, tok_range.clone(), &tok_spans, &breaks, deps, pos, view)
            {
                breaks[g].upgrade(brk);
            }
            if hi < m
                && let Some(g) = adjust_cut(hi, tok_range.clone(), &tok_spans, &breaks, deps, pos, view)
            {
                breaks[g].upgrade(brk);
            }
        }
    }

    if let Some(pc) = phrase_chunks {
        let mut seen_vp = false;
        let mut prev_np_start: Option<usize> = None;
        for i in 0..pc.spans.len() {
            let r = pc.tokens[i];
            let (start, end) = (r.start.get(), r.end.get());
            if start < tok_range.start || end > tok_range.end || start >= end {
                continue;
            }
            let (lo, hi) = (start - tok_range.start, end - tok_range.start);
            let label = pc.kinds.label(pc.kind[i]);
            match label {
                "SBAR" | "CONJP" => {
                    if lo > 0
                        && let Some(g) = adjust_cut(lo, tok_range.clone(), &tok_spans, &breaks, deps, pos, view)
                    {
                        breaks[g].upgrade(ClauseBreak::Clause(costs.subordinate));
                    }
                }
                "NP" if seen_vp && prev_np_start.is_none() => prev_np_start = Some(lo),
                "VP" => {
                    if seen_vp {
                        let cut_at = prev_np_start.unwrap_or(lo);
                        if cut_at > 0
                            && let Some(g) = adjust_cut(cut_at, tok_range.clone(), &tok_spans, &breaks, deps, pos, view)
                        {
                            breaks[g].upgrade(ClauseBreak::Clause(costs.clause));
                        }
                    }
                    seen_vp = true;
                    prev_np_start = None;
                }
                _ => {}
            }
            if lo > 0
                && let Some(g) = adjust_cut(lo, tok_range.clone(), &tok_spans, &breaks, deps, pos, view)
            {
                breaks[g].upgrade(ClauseBreak::Phrase(costs.phrase));
            }
            if hi < m
                && let Some(g) = adjust_cut(hi, tok_range.clone(), &tok_spans, &breaks, deps, pos, view)
            {
                breaks[g].upgrade(ClauseBreak::Phrase(costs.phrase));
            }
        }
    }

    for k in 0..m - 1 {
        let t = tok_range.start + k;
        if is_hard_clause_punct(t, tok_spans[k], pos, view) {
            if let Some(g) = adjust_cut(k + 1, tok_range.clone(), &tok_spans, &breaks, deps, pos, view) {
                breaks[g].upgrade(ClauseBreak::Clause(costs.clause));
            }
        } else if is_comma(t, tok_spans[k], pos, view) {
            let next_coord = is_coordinator_or_subordinator(t + 1, tok_spans[k + 1], deps, pos, view);
            let brk =
                if next_coord { ClauseBreak::Clause(costs.subordinate) } else { ClauseBreak::Phrase(costs.phrase) };
            if let Some(g) = adjust_cut(k + 1, tok_range.clone(), &tok_spans, &breaks, deps, pos, view) {
                breaks[g].upgrade(brk);
            }
        }
    }

    SentenceClauseCuts { tokens: tok_spans, breaks }
}
