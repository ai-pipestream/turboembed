use turbo_analysis::{IxRange, Layer};
use turbo_analysis_layers::{Dependencies, PolarityScores, Tokens};

use super::propagation::{ExtractedAspect, ExtractedOpinion, is_conj_rel};
use super::tables::ADVERSATIVE_CONJUNCTIONS;

pub(super) fn span_text<'a>(text: &'a str, tok_starts: &[u32], tok_ends: &[u32], range: IxRange<Tokens>) -> &'a str {
    let s = tok_starts[range.start.get()] as usize;
    let e = tok_ends[range.end.get() - 1] as usize;
    text.get(s..e).unwrap_or("")
}

pub(super) fn has_adversative_between(a: usize, b: usize, token_lower: &[String]) -> bool {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    token_lower[lo..=hi].iter().any(|w| ADVERSATIVE_CONJUNCTIONS.contains(&w.as_str()))
}

pub(super) fn range_contains(r: IxRange<Tokens>, tok: usize) -> bool {
    r.start.get() <= tok && tok < r.end.get()
}

pub(super) fn ranges_overlap(a: IxRange<Tokens>, b: IxRange<Tokens>) -> bool {
    a.start.get() < b.end.get() && b.start.get() < a.end.get()
}

pub(super) fn dedup_opinions(ops: &mut Vec<ExtractedOpinion>) {
    let mut out: Vec<ExtractedOpinion> = Vec::with_capacity(ops.len());
    for op in ops.drain(..) {
        if let Some(prev) = out.last()
            && ranges_overlap(prev.tokens, op.tokens)
        {
            continue;
        }
        out.push(op);
    }
    *ops = out;
}

pub(super) fn dedup_aspects(aspects: &mut Vec<ExtractedAspect>) {
    let mut out: Vec<ExtractedAspect> = Vec::with_capacity(aspects.len());
    for asp in aspects.drain(..) {
        if let Some(prev) = out.last_mut()
            && ranges_overlap(prev.tokens, asp.tokens)
        {
            if asp.tokens.len() > prev.tokens.len() {
                *prev = asp;
            }
            continue;
        }
        out.push(asp);
    }
    *aspects = out;
}

/// Links opinion expressions in `sent_opinions` to `asp` using:
/// 1. Direct or shared-governor dependency arcs (`deps`) when available;
/// 2. Opinions in the same clause whose nearest aspect in that clause is `asp`;
/// 3. Nearest opinion in the sentence when the aspect's clause has no opinion.
#[allow(clippy::too_many_arguments)]
pub(super) fn link_opinions_to_aspect(
    asp: &ExtractedAspect,
    asp_pos: usize,
    sent_aspects: &[ExtractedAspect],
    sent_opinions: &[ExtractedOpinion],
    deps: Option<&Dependencies>,
    token_lower: &[String],
    sent_start: usize,
    sent_end: usize,
) -> Vec<usize> {
    if sent_opinions.is_empty() {
        return Vec::new();
    }
    let mut linked = Vec::new();

    // 1. Direct or copular/shared-head dependency link.
    if let Some(d) = deps
        && d.len() >= sent_end
    {
        let t_tok = asp.head_token;
        let t_head = d.head[t_tok].map(|ix| ix.get());
        for (o_idx, op) in sent_opinions.iter().enumerate() {
            let o_tok = op.head_token;
            let o_head = d.head[o_tok].map(|ix| ix.get());
            let direct = o_head == Some(t_tok) || t_head == Some(o_tok);
            let shared_head =
                t_head.is_some() && t_head == o_head && !has_adversative_between(t_tok, o_tok, token_lower);
            // Also check conjunction chain (`t_tok` conjoined with another aspect linked to `o_tok`,
            // or `o_tok` conjoined with another opinion linked to `t_tok`).
            let conj_target = if let Some(th) = t_head
                && is_conj_rel(d.relations.label(d.relation[t_tok]))
            {
                o_head == Some(th) || d.head[th].map(|ix| ix.get()) == Some(o_tok)
            } else {
                false
            };
            let conj_opinion = if let Some(oh) = o_head
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

    // 2. Clause-bounded nearest-aspect assignment:
    // Every opinion in `asp.clause_id` links to whichever aspect in `asp.clause_id` is closest to it.
    let same_clause_aspects: Vec<(usize, &ExtractedAspect)> =
        sent_aspects.iter().enumerate().filter(|(_, a)| a.clause_id == asp.clause_id).collect();

    for (o_idx, op) in sent_opinions.iter().enumerate() {
        if op.clause_id != asp.clause_id {
            continue;
        }
        let best_asp =
            same_clause_aspects.iter().min_by_key(|(_, a)| token_distance(a.tokens, op.tokens)).map(|(idx, _)| *idx);
        if best_asp == Some(asp_pos) {
            linked.push(o_idx);
        }
    }
    if !linked.is_empty() {
        return linked;
    }

    // 3. If `asp` is in a clause with other opinions (already assigned to a sibling aspect in the same clause),
    // attach the closest opinion in the same clause.
    if let Some((o_idx, _)) = sent_opinions
        .iter()
        .enumerate()
        .filter(|(_, op)| op.clause_id == asp.clause_id)
        .min_by_key(|(_, op)| token_distance(asp.tokens, op.tokens))
    {
        linked.push(o_idx);
        return linked;
    }

    // 4. Fallback: closest opinion in the sentence, provided no adversative conjunction separates them.
    let _ = sent_start;
    if let Some((o_idx, _)) =
        sent_opinions.iter().enumerate().min_by_key(|(_, op)| token_distance(asp.tokens, op.tokens))
    {
        linked.push(o_idx);
    }
    linked
}

fn token_distance(a: IxRange<Tokens>, b: IxRange<Tokens>) -> usize {
    if a.end.get() <= b.start.get() {
        b.start.get() - a.end.get()
    } else if b.end.get() <= a.start.get() {
        a.start.get() - b.end.get()
    } else {
        0
    }
}

pub(super) fn compute_aspect_polarity(linked_ops: &[usize], sent_opinions: &[ExtractedOpinion]) -> PolarityScores {
    if linked_ops.is_empty() {
        return PolarityScores::new(0.0, 0.0, 1.0);
    }
    let mut pos_sum = 0.0f64;
    let mut neg_sum = 0.0f64;
    for &idx in linked_ops {
        let eff = sent_opinions[idx].effective_polarity();
        if eff > 0.0 {
            pos_sum += eff;
        } else if eff < 0.0 {
            neg_sum += -eff;
        }
    }
    let total = pos_sum + neg_sum;
    if total <= 1e-9 {
        return PolarityScores::new(0.0, 0.0, 1.0);
    }
    // Convert raw valence magnitudes into normalized three-way probabilities in `0.0..=1.0`.
    let confidence = (total / (total + 1.2)).clamp(0.55, 0.96);
    let neu = (1.0 - confidence).max(0.0);
    let pos = confidence * (pos_sum / total);
    let neg = confidence * (neg_sum / total);
    PolarityScores::new(pos, neg, neu)
}
