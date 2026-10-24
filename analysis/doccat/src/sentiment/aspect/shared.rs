//! ABSA structural helpers shared by the rule-based aspect
//! extractor in this crate and the learned aspect decoder in
//! `turbo-analysis-embed`: clause segmentation, negation scope,
//! and opinion-to-aspect linking.
//!
//! The two callers have drifted in policy, not in structure, so
//! the policies stay with the callers: clause segmentation takes
//! the caller's adversative list, negation scope takes the
//! caller's negation predicate, and linking takes a
//! [`LinkPolicy`]. Sharing the code without sharing the policy
//! keeps every existing output unchanged.

use turbo_analysis::{IxRange, Layer};
use turbo_analysis_layers::{Dependencies, Tokens};

/// Assigns each token in `[sent_start, sent_end)` a clause id,
/// advancing at `;` and `,` tokens, at `and`, and at any word in
/// `adversatives` (compared against the lowered token text).
pub fn assign_clauses(
    token_lower: &[String],
    text: &str,
    tok_starts: &[u32],
    tok_ends: &[u32],
    sent_start: usize,
    sent_end: usize,
    adversatives: &[&str],
) -> Vec<usize> {
    let mut clauses = Vec::with_capacity(sent_end - sent_start);
    let mut current_clause = 0usize;
    for tok in sent_start..sent_end {
        let w = token_lower[tok].as_str();
        let raw = text.get(tok_starts[tok] as usize..tok_ends[tok] as usize).unwrap_or("");
        if tok > sent_start && (adversatives.contains(&w) || raw == ";" || raw == "," || w == "and") {
            current_clause += 1;
        }
        clauses.push(current_clause);
    }
    clauses
}

/// Whether token `tok` is negated: a negation word depending on
/// `tok` or on its head, or one within the three preceding
/// tokens of the same clause.
pub fn is_token_negated(
    tok: usize,
    sent_start: usize,
    sent_end: usize,
    token_lower: &[String],
    clause_ids: &[usize],
    deps: Option<&Dependencies>,
    is_negation_word: impl Fn(&str) -> bool,
) -> bool {
    if let Some(d) = deps
        && d.len() >= sent_end
    {
        for i in sent_start..sent_end {
            if !is_negation_word(token_lower[i].as_str()) {
                continue;
            }
            let h = d.head[i].map(|ix| ix.get());
            if h == Some(tok) {
                return true;
            }
            if let Some(tok_head) = d.head[tok].map(|ix| ix.get())
                && h == Some(tok_head)
                && i < tok
                && clause_ids[i - sent_start] == clause_ids[tok - sent_start]
            {
                return true;
            }
        }
    }

    let min_i = sent_start.max(tok.saturating_sub(3));
    let tok_clause = clause_ids[tok - sent_start];
    (min_i..tok).any(|i| clause_ids[i - sent_start] == tok_clause && is_negation_word(token_lower[i].as_str()))
}

/// The gap between two token ranges, in tokens: 0 when they
/// overlap or touch.
pub fn token_distance(a: IxRange<Tokens>, b: IxRange<Tokens>) -> usize {
    if a.end.get() <= b.start.get() {
        b.start.get() - a.end.get()
    } else if b.end.get() <= a.start.get() {
        a.start.get() - b.end.get()
    } else {
        0
    }
}

/// Whether `rel` conjoins two spans (`conj` or `appos`).
pub fn is_conj_rel(rel: &str) -> bool {
    rel.eq_ignore_ascii_case("conj") || rel.eq_ignore_ascii_case("appos")
}

/// An aspect or opinion span as linking sees it: its token
/// range, head token, and clause.
pub trait LinkedSpan {
    fn tokens(&self) -> IxRange<Tokens>;
    fn head_token(&self) -> usize;
    fn clause_id(&self) -> usize;
}

/// The caller-specific parts of dependency linking: adversative
/// words veto a shared-head link when one sits between the
/// aspect and opinion heads (only consulted when non-empty),
/// and conjunction chains extend a dependency link across a
/// conjoined head.
#[derive(Clone, Copy, Debug)]
pub struct LinkPolicy<'a> {
    pub adversatives: &'a [&'a str],
    pub conjunction_chains: bool,
}

/// Whether any of `adversatives` sits between tokens `a` and `b` (inclusive).
pub fn has_adversative_between(a: usize, b: usize, token_lower: &[String], adversatives: &[&str]) -> bool {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    token_lower[lo..=hi].iter().any(|w| adversatives.contains(&w.as_str()))
}

/// Links opinion expressions in `sent_opinions` to `asp` (at
/// position `asp_pos` in `sent_aspects`) using:
/// 1. Direct or shared-governor dependency arcs (`deps`) when
///    available, subject to `policy`;
/// 2. Opinions in the same clause whose nearest aspect in that
///    clause is `asp`;
/// 3. The nearest opinion in the same clause, then in the
///    sentence.
#[allow(clippy::too_many_arguments)]
pub fn link_opinions_to_aspect<A: LinkedSpan, O: LinkedSpan>(
    asp: &A,
    asp_pos: usize,
    sent_aspects: &[A],
    sent_opinions: &[O],
    deps: Option<&Dependencies>,
    token_lower: &[String],
    sent_end: usize,
    policy: LinkPolicy<'_>,
) -> Vec<usize> {
    if sent_opinions.is_empty() {
        return Vec::new();
    }
    let mut linked = Vec::new();

    if let Some(d) = deps
        && d.len() >= sent_end
    {
        let t_tok = asp.head_token();
        let t_head = d.head[t_tok].map(|ix| ix.get());
        for (o_idx, op) in sent_opinions.iter().enumerate() {
            let o_tok = op.head_token();
            let o_head = d.head[o_tok].map(|ix| ix.get());
            let direct = o_head == Some(t_tok) || t_head == Some(o_tok);
            let shared_head = t_head.is_some()
                && t_head == o_head
                && (policy.adversatives.is_empty()
                    || !has_adversative_between(t_tok, o_tok, token_lower, policy.adversatives));
            let conj_target = if policy.conjunction_chains
                && let Some(th) = t_head
                && is_conj_rel(d.relations.label(d.relation[t_tok]))
            {
                o_head == Some(th) || d.head[th].map(|ix| ix.get()) == Some(o_tok)
            } else {
                false
            };
            let conj_opinion = if policy.conjunction_chains
                && let Some(oh) = o_head
                && is_conj_rel(d.relations.label(d.relation[o_tok]))
            {
                d.head[oh].map(|ix| ix.get()) == Some(t_tok) || t_head == Some(oh)
            } else {
                false
            };
            if direct || shared_head || conj_target || conj_opinion {
                linked.push(o_idx);
            }
        }
        if !linked.is_empty() {
            return linked;
        }
    }

    let same_clause_aspects: Vec<(usize, &A)> =
        sent_aspects.iter().enumerate().filter(|(_, a)| a.clause_id() == asp.clause_id()).collect();
    for (o_idx, op) in sent_opinions.iter().enumerate() {
        if op.clause_id() != asp.clause_id() {
            continue;
        }
        let best_asp = same_clause_aspects
            .iter()
            .min_by_key(|(_, a)| token_distance(a.tokens(), op.tokens()))
            .map(|(idx, _)| *idx);
        if best_asp == Some(asp_pos) {
            linked.push(o_idx);
        }
    }
    if !linked.is_empty() {
        return linked;
    }

    if let Some((o_idx, _)) = sent_opinions
        .iter()
        .enumerate()
        .filter(|(_, op)| op.clause_id() == asp.clause_id())
        .min_by_key(|(_, op)| token_distance(asp.tokens(), op.tokens()))
    {
        linked.push(o_idx);
        return linked;
    }

    if let Some((o_idx, _)) =
        sent_opinions.iter().enumerate().min_by_key(|(_, op)| token_distance(asp.tokens(), op.tokens()))
    {
        linked.push(o_idx);
    }
    linked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clause_boundaries_follow_the_callers_adversatives() {
        let text = "a even b except c and d";
        let lower: Vec<String> = text.split(' ').map(str::to_string).collect();
        let starts = [0u32, 2, 7, 9, 16, 18, 22];
        let ends = [1u32, 6, 8, 15, 17, 21, 23];
        // With "even" adversative: boundaries at even and at and.
        assert_eq!(assign_clauses(&lower, text, &starts, &ends, 0, 7, &["even"]), [0, 1, 1, 1, 1, 2, 2]);
        // With "except" instead: boundaries at except and at and.
        assert_eq!(assign_clauses(&lower, text, &starts, &ends, 0, 7, &["except"]), [0, 0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn negation_scope_uses_the_callers_predicate() {
        let lower: Vec<String> = ["it", "lacks", "shine"].iter().map(|s| s.to_string()).collect();
        let clauses = [0usize, 0, 0];
        assert!(is_token_negated(2, 0, 3, &lower, &clauses, None, |w| w == "lacks"));
        assert!(!is_token_negated(2, 0, 3, &lower, &clauses, None, |w| w == "not"));
    }
}
