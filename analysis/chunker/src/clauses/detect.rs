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

fn is_verbal(tok_ix: usize, deps: Option<&Dependencies<Tokens>>, pos: Option<&PosTags>) -> bool {
    if let Some(p) = pos_label(pos, tok_ix)
        && (p.starts_with("VB") || matches!(p, "VERB" | "AUX" | "MD"))
    {
        return true;
    }
    let Some(d) = deps else {
        return false;
    };
    d.head.iter().enumerate().any(|(child, &h)| {
        h.is_some_and(|ix| ix.get() == tok_ix)
            && d.relation.get(child).is_some_and(|&r| {
                matches!(
                    base_rel(d.relations.label(r)),
                    "nsubj" | "csubj" | "obj" | "iobj" | "obl" | "aux" | "cop" | "mark" | "advmod"
                )
            })
    })
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

/// Computes the sentence-local token subtree range `[lo, hi]` (`0 <= lo <= hi < m`)
/// of sentence-local token `root_k` under `deps`.
fn subtree_bounds(root_k: usize, tok_range: &Range<usize>, deps: &Dependencies<Tokens>) -> (usize, usize) {
    let m = tok_range.len();
    let (mut lo, mut hi) = (root_k, root_k);
    for k in 0..m {
        let mut cur = tok_range.start + k;
        let mut steps = 0usize;
        while steps <= m {
            if cur == tok_range.start + root_k {
                lo = lo.min(k);
                hi = hi.max(k);
                break;
            }
            let Some(Some(h)) = deps.head.get(cur) else {
                break;
            };
            let h_ix = h.get();
            if h_ix < tok_range.start || h_ix >= tok_range.end || h_ix == cur {
                break;
            }
            cur = h_ix;
            steps += 1;
        }
    }
    (lo, hi)
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
        for k in 0..m {
            let t = tok_range.start + k;
            let Some(rel) = dep_label(Some(d), t) else {
                continue;
            };
            let base = base_rel(rel);
            let (lo, hi) = subtree_bounds(k, &tok_range, d);
            let span_len = hi + 1 - lo;
            let kind = match base {
                "parataxis" | "list" | "remnant" => Some(ClauseBreak::Clause(costs.clause)),
                "advcl" | "ccomp" | "csubj" => Some(ClauseBreak::Clause(costs.subordinate)),
                "acl" if rel == "acl:relcl" || span_len >= 2 => Some(ClauseBreak::Clause(costs.subordinate)),
                "xcomp" if span_len >= 2 => Some(ClauseBreak::Clause(costs.subordinate)),
                "conj" => {
                    let head_verbal = d.head.get(t).and_then(|&h| h).is_some_and(|h| is_verbal(h.get(), Some(d), pos));
                    let preceded_by_punct = lo > 0
                        && is_closing_or_sep_punct(tok_range.start + lo - 1, tok_spans[lo - 1], Some(d), pos, view);
                    if span_len >= 2 || is_verbal(t, Some(d), pos) || head_verbal || preceded_by_punct {
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
