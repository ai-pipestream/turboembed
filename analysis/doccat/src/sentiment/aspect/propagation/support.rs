use super::*;

impl RuleAspectSentiment {
    pub(super) fn can_be_aspect_target(
        &self,
        tok: usize,
        pos_tags: Option<&PosTags>,
        token_lower: &[String],
        opinions: &[ExtractedOpinion],
        aspects: &[ExtractedAspect],
    ) -> bool {
        let w = token_lower[tok].as_str();
        if w.is_empty() || LEADING_DETERMINERS.contains(&w) || is_aspect_stopword(w) {
            return false;
        }
        if opinions.iter().any(|o| range_contains(o.tokens, tok))
            || aspects.iter().any(|a| range_contains(a.tokens, tok))
        {
            return false;
        }
        if self.token_opinion_valence(w).is_some() {
            return false;
        }
        if let Some(pos) = pos_tags {
            let tag = pos.tagset.label(pos.tags[tok]);
            if !is_noun_pos(tag) {
                return false;
            }
        }
        true
    }
}

pub(in super::super) fn assign_clauses(
    token_lower: &[String],
    text: &str,
    tok_starts: &[u32],
    tok_ends: &[u32],
    sent_start: usize,
    sent_end: usize,
) -> Vec<usize> {
    super::super::shared::assign_clauses(
        token_lower,
        text,
        tok_starts,
        tok_ends,
        sent_start,
        sent_end,
        ASPECT_ADVERSATIVES,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn expand_noun_compound(
    tok: usize,
    sent_start: usize,
    sent_end: usize,
    token_lower: &[String],
    covered: &[bool],
    deps: Option<&Dependencies>,
    pos_tags: Option<&PosTags>,
    extractor: &RuleAspectSentiment,
) -> (usize, usize) {
    let mut s = tok;
    let mut e = tok + 1;

    if let Some(d) = deps
        && d.len() >= sent_end
    {
        while s > sent_start && !covered[s - 1 - sent_start] {
            let prev = s - 1;
            let rel = d.relations.label(d.relation[prev]);
            let w = token_lower[prev].as_str();
            if d.head[prev].map(|ix| ix.get()) == Some(tok)
                && is_compound_rel(rel)
                && !w.is_empty()
                && !LEADING_DETERMINERS.contains(&w)
                && extractor.token_opinion_valence(w).is_none()
            {
                s -= 1;
            } else {
                break;
            }
        }
        if e < sent_end && !covered[e - sent_start] {
            let rel = d.relations.label(d.relation[tok]);
            let w = token_lower[e].as_str();
            if d.head[tok].map(|ix| ix.get()) == Some(e)
                && is_compound_rel(rel)
                && !w.is_empty()
                && !is_aspect_stopword(w)
                && extractor.token_opinion_valence(w).is_none()
            {
                e += 1;
            }
        }
    } else if let Some(pos) = pos_tags {
        while s > sent_start && !covered[s - 1 - sent_start] {
            let prev = s - 1;
            let ptag = pos.tagset.label(pos.tags[prev]);
            let w = token_lower[prev].as_str();
            if is_noun_pos(ptag)
                && !w.is_empty()
                && !LEADING_DETERMINERS.contains(&w)
                && !is_aspect_stopword(w)
                && extractor.token_opinion_valence(w).is_none()
            {
                s -= 1;
            } else {
                break;
            }
        }
    }

    (s, e)
}

pub(super) fn expand_dep_noun_span(
    head: usize,
    sent_start: usize,
    _sent_end: usize,
    deps: &Dependencies,
    token_lower: &[String],
    opinions: &[ExtractedOpinion],
) -> (usize, usize) {
    let mut s = head;
    let e = head + 1;
    while s > sent_start {
        let prev = s - 1;
        let rel = deps.relations.label(deps.relation[prev]);
        let w = token_lower[prev].as_str();
        if deps.head[prev].map(|ix| ix.get()) == Some(head)
            && is_compound_rel(rel)
            && !w.is_empty()
            && !LEADING_DETERMINERS.contains(&w)
            && !opinions.iter().any(|o| range_contains(o.tokens, prev))
        {
            s -= 1;
        } else {
            break;
        }
    }
    (s, e)
}

pub(super) fn is_noun_pos(tag: &str) -> bool {
    matches!(tag, "NN" | "NNS" | "NNP" | "NNPS" | "NOUN" | "PROPN")
}

pub(super) fn is_adj_pos(tag: &str) -> bool {
    matches!(tag, "JJ" | "JJR" | "JJS" | "ADJ")
}

pub(super) fn is_compound_rel(rel: &str) -> bool {
    rel.eq_ignore_ascii_case("compound") || rel.eq_ignore_ascii_case("nn") || rel.eq_ignore_ascii_case("flat")
}

pub(in super::super) use super::super::shared::is_conj_rel;

pub(super) fn is_mr_opinion_to_head_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "amod" | "acl" | "advmod" | "nmod" | "dep" | "xcomp" | "acomp" | "ccomp")
}

pub(super) fn is_mr_target_child_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "nsubj" | "obj" | "dobj" | "iobj" | "obl" | "nmod")
}

pub(super) fn is_mr_complement_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "acomp" | "xcomp" | "ccomp" | "attr" | "advmod" | "amod" | "obj" | "dobj")
}

pub(super) fn is_token_negated(
    tok: usize,
    sent_start: usize,
    sent_end: usize,
    token_lower: &[String],
    clause_ids: &[usize],
    deps: Option<&Dependencies>,
) -> bool {
    super::super::shared::is_token_negated(tok, sent_start, sent_end, token_lower, clause_ids, deps, is_negation_word)
}

pub(super) fn compute_token_intensity(
    tok: usize,
    sent_start: usize,
    token_lower: &[String],
    text: &str,
    tok_starts: &[u32],
    tok_ends: &[u32],
) -> f64 {
    let mut mult = 1.0f64;
    if tok > sent_start {
        let prev1 = token_lower[tok - 1].as_str();
        for &(deg, delta) in DEGREE_MODIFIERS {
            if deg == prev1 {
                mult += delta;
                break;
            }
        }
    }
    if tok >= sent_start + 2 {
        let prev2 = token_lower[tok - 2].as_str();
        for &(deg, delta) in DEGREE_MODIFIERS {
            if deg == prev2 {
                mult += delta * 0.85;
                break;
            }
        }
    }
    let raw = text.get(tok_starts[tok] as usize..tok_ends[tok] as usize).unwrap_or("");
    let has_alpha = raw.chars().any(|c| c.is_ascii_alphabetic());
    let all_caps = has_alpha && raw.chars().filter(|c| c.is_ascii_alphabetic()).all(|c| c.is_ascii_uppercase());
    if all_caps && raw.len() >= 2 {
        mult += 0.25;
    }
    mult.clamp(0.25, 2.0)
}
