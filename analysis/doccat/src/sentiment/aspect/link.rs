use turbo_analysis::IxRange;
use turbo_analysis_layers::{Dependencies, PolarityScores, Tokens};

use super::propagation::{ExtractedAspect, ExtractedOpinion};
use super::shared::{LinkPolicy, LinkedSpan};
use super::tables::ASPECT_ADVERSATIVES;

pub(super) fn span_text<'a>(text: &'a str, tok_starts: &[u32], tok_ends: &[u32], range: IxRange<Tokens>) -> &'a str {
    let s = tok_starts[range.start.get()] as usize;
    let e = tok_ends[range.end.get() - 1] as usize;
    text.get(s..e).unwrap_or("")
}

pub(super) fn has_adversative_between(a: usize, b: usize, token_lower: &[String]) -> bool {
    super::shared::has_adversative_between(a, b, token_lower, ASPECT_ADVERSATIVES)
}

impl LinkedSpan for ExtractedAspect {
    fn tokens(&self) -> IxRange<Tokens> {
        self.tokens
    }
    fn head_token(&self) -> usize {
        self.head_token
    }
    fn clause_id(&self) -> usize {
        self.clause_id
    }
}

impl LinkedSpan for ExtractedOpinion {
    fn tokens(&self) -> IxRange<Tokens> {
        self.tokens
    }
    fn head_token(&self) -> usize {
        self.head_token
    }
    fn clause_id(&self) -> usize {
        self.clause_id
    }
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
    let _ = sent_start;
    super::shared::link_opinions_to_aspect(
        asp,
        asp_pos,
        sent_aspects,
        sent_opinions,
        deps,
        token_lower,
        sent_end,
        LinkPolicy { adversatives: ASPECT_ADVERSATIVES, conjunction_chains: true },
    )
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
