use turbo_analysis::{IxRange, Layer};
use turbo_analysis_layers::{Dependencies, PhraseChunks, PosTags, Tokens};

use super::RuleAspectSentiment;
use super::link::{has_adversative_between, range_contains};
use super::tables::{
    ADVERSATIVE_CONJUNCTIONS, DEGREE_MODIFIERS, LEADING_DETERMINERS, MULTIWORD_ASPECT_TERMS, is_aspect_stopword,
    is_negation_word, lookup_category_hint, single_word_aspect_category,
};

#[derive(Clone, Debug)]
pub(super) struct ExtractedOpinion {
    pub(super) tokens: IxRange<Tokens>,
    pub(super) head_token: usize,
    pub(super) polarity: f64,
    pub(super) intensity: f64,
    pub(super) negated: bool,
    pub(super) clause_id: usize,
}

impl ExtractedOpinion {
    pub(super) fn effective_polarity(&self) -> f64 {
        let sign = if self.negated { -1.0 } else { 1.0 };
        self.polarity * self.intensity * sign
    }
}

#[derive(Clone, Debug)]
pub(super) struct ExtractedAspect {
    pub(super) tokens: IxRange<Tokens>,
    pub(super) head_token: usize,
    pub(super) clause_id: usize,
    pub(super) category_hint: Option<&'static str>,
}

impl RuleAspectSentiment {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn extract_sentence_candidates(
        &self,
        text: &str,
        tok_starts: &[u32],
        tok_ends: &[u32],
        token_lower: &[String],
        sent_start: usize,
        sent_end: usize,
        clause_ids: &[usize],
        deps: Option<&Dependencies>,
        phrase_chunks: Option<&PhraseChunks>,
        pos_tags: Option<&PosTags>,
    ) -> (Vec<ExtractedOpinion>, Vec<ExtractedAspect>) {
        let mut opinions = Vec::new();
        let mut aspects = Vec::new();
        let mut covered_aspect_tokens = vec![false; sent_end - sent_start];

        // 1a. Match multi-word aspect terms first (longest match wins).
        let mut t = sent_start;
        while t < sent_end {
            let rem = sent_end - t;
            let mut matched_mw: Option<(usize, &'static str)> = None;
            for &(pat, cat) in MULTIWORD_ASPECT_TERMS {
                if pat.len() <= rem
                    && pat.iter().enumerate().all(|(k, &w)| token_lower[t + k] == w)
                    && (matched_mw.is_none() || pat.len() > matched_mw.unwrap().0)
                {
                    matched_mw = Some((pat.len(), cat));
                }
            }
            if let Some((len, cat)) = matched_mw {
                let head = t + len - 1;
                for flag in &mut covered_aspect_tokens[(t - sent_start)..(t + len - sent_start)] {
                    *flag = true;
                }
                aspects.push(ExtractedAspect {
                    tokens: IxRange::new(t as u32, (t + len) as u32),
                    head_token: head,
                    clause_id: clause_ids[head - sent_start],
                    category_hint: Some(cat),
                });
                t += len;
            } else {
                t += 1;
            }
        }

        // 1b. Match single-word aspect lexicon terms not already inside a multi-word aspect.
        for tok in sent_start..sent_end {
            if covered_aspect_tokens[tok - sent_start] {
                continue;
            }
            let w = token_lower[tok].as_str();
            if let Some(cat) = single_word_aspect_category(w) {
                // Expand left/right if attached to adjacent noun compound in POS or Dependencies.
                let (span_start, span_end) = expand_noun_compound(
                    tok,
                    sent_start,
                    sent_end,
                    token_lower,
                    &covered_aspect_tokens,
                    deps,
                    pos_tags,
                    self,
                );
                for flag in &mut covered_aspect_tokens[(span_start - sent_start)..(span_end - sent_start)] {
                    *flag = true;
                }
                aspects.push(ExtractedAspect {
                    tokens: IxRange::new(span_start as u32, span_end as u32),
                    head_token: tok,
                    clause_id: clause_ids[tok - sent_start],
                    category_hint: Some(cat),
                });
            }
        }

        // 1c. Extract NP candidates from `PhraseChunks` (`opennlp:chunks`) when present.
        if let Some(chunks) = phrase_chunks {
            for (c_idx, range) in chunks.tokens.iter().enumerate() {
                let c_start = range.start.get();
                let c_end = range.end.get();
                if c_start < sent_start || c_end > sent_end || c_start >= c_end {
                    continue;
                }
                let kind_label = chunks.kinds.label(chunks.kind[c_idx]);
                if !kind_label.eq_ignore_ascii_case("NP") {
                    continue;
                }
                // Trim leading determiners and leading opinion adjectives from the NP.
                let mut a_start = c_start;
                while a_start < c_end {
                    let w = token_lower[a_start].as_str();
                    if LEADING_DETERMINERS.contains(&w) || self.token_opinion_valence(w).is_some() || w.is_empty() {
                        a_start += 1;
                    } else {
                        break;
                    }
                }
                if a_start >= c_end {
                    continue;
                }
                let head = c_end - 1;
                let head_w = token_lower[head].as_str();
                if head_w.is_empty() || is_aspect_stopword(head_w) || self.token_opinion_valence(head_w).is_some() {
                    continue;
                }
                if (a_start..c_end).any(|i| covered_aspect_tokens[i - sent_start]) {
                    continue;
                }
                for flag in &mut covered_aspect_tokens[(a_start - sent_start)..(c_end - sent_start)] {
                    *flag = true;
                }
                let cat_hint = lookup_category_hint(&token_lower[a_start..c_end]);
                aspects.push(ExtractedAspect {
                    tokens: IxRange::new(a_start as u32, c_end as u32),
                    head_token: head,
                    clause_id: clause_ids[head - sent_start],
                    category_hint: cat_hint,
                });
            }
        }

        // 1d. Extract POS-tagged nouns (`NN`, `NNS`, `NNP`, `NNPS`, `NOUN`, `PROPN`) when `PosTags` is present.
        if let Some(pos) = pos_tags {
            let mut tok = sent_start;
            while tok < sent_end {
                if covered_aspect_tokens[tok - sent_start] {
                    tok += 1;
                    continue;
                }
                let tag_str = pos.tagset.label(pos.tags[tok]);
                let w = token_lower[tok].as_str();
                if is_noun_pos(tag_str)
                    && !w.is_empty()
                    && !is_aspect_stopword(w)
                    && self.token_opinion_valence(w).is_none()
                {
                    let mut end = tok + 1;
                    while end < sent_end && !covered_aspect_tokens[end - sent_start] {
                        let next_tag = pos.tagset.label(pos.tags[end]);
                        let next_w = token_lower[end].as_str();
                        if is_noun_pos(next_tag)
                            && !next_w.is_empty()
                            && !is_aspect_stopword(next_w)
                            && self.token_opinion_valence(next_w).is_none()
                        {
                            end += 1;
                        } else {
                            break;
                        }
                    }
                    let head = end - 1;
                    for flag in &mut covered_aspect_tokens[(tok - sent_start)..(end - sent_start)] {
                        *flag = true;
                    }
                    let cat_hint = lookup_category_hint(&token_lower[tok..end]);
                    aspects.push(ExtractedAspect {
                        tokens: IxRange::new(tok as u32, end as u32),
                        head_token: head,
                        clause_id: clause_ids[head - sent_start],
                        category_hint: cat_hint,
                    });
                    tok = end;
                } else {
                    tok += 1;
                }
            }
        }

        // 2. Seed opinion expressions from tokens not covered by explicit aspect terms.
        for tok in sent_start..sent_end {
            if covered_aspect_tokens[tok - sent_start] {
                continue;
            }
            let w = token_lower[tok].as_str();
            if w.is_empty() {
                continue;
            }
            if let Some(prior) = self.token_opinion_valence(w) {
                let negated = is_token_negated(tok, sent_start, sent_end, token_lower, clause_ids, deps);
                let intensity = compute_token_intensity(tok, sent_start, token_lower, text, tok_starts, tok_ends);
                opinions.push(ExtractedOpinion {
                    tokens: IxRange::new(tok as u32, (tok + 1) as u32),
                    head_token: tok,
                    polarity: prior,
                    intensity,
                    negated,
                    clause_id: clause_ids[tok - sent_start],
                });
            }
        }

        (opinions, aspects)
    }

    /// Qiu, Liu, Bu, and Chen (2011) iterative Double Propagation over
    /// dependency arcs (`R1`, `R2`, `R3`, `R4`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_double_propagation(
        &self,
        deps: &Dependencies,
        pos_tags: Option<&PosTags>,
        token_lower: &[String],
        sent_start: usize,
        sent_end: usize,
        clause_ids: &[usize],
        opinions: &mut Vec<ExtractedOpinion>,
        aspects: &mut Vec<ExtractedAspect>,
    ) {
        if deps.len() < sent_end {
            return;
        }
        // Run up to 3 propagation passes until fixed point.
        for _pass in 0..3 {
            let mut added_any = false;

            // R1 (O -> T): Known opinion words extract modified or governing noun targets.
            let current_ops = opinions.clone();
            for op in &current_ops {
                let o_tok = op.head_token;
                let o_head = deps.head[o_tok].map(|ix| ix.get());
                let o_rel = deps.relations.label(deps.relation[o_tok]);

                // R11a: Opinion `o_tok` is a dependent of target `t_tok` (e.g., `amod(screen, blurry)` or `acl(pizza, cooked)`),
                // or `o_tok` is a predicate head whose subject/object is `t_tok` (`nsubj(great, battery)` or `nsubj(scratches, screen)`).
                if let Some(h) = o_head
                    && h >= sent_start
                    && h < sent_end
                    && is_mr_opinion_to_head_rel(o_rel)
                    && self.can_be_aspect_target(h, pos_tags, token_lower, opinions, aspects)
                {
                    let (s, e) = expand_dep_noun_span(h, sent_start, sent_end, deps, token_lower, opinions);
                    aspects.push(ExtractedAspect {
                        tokens: IxRange::new(s as u32, e as u32),
                        head_token: h,
                        clause_id: clause_ids[h - sent_start],
                        category_hint: lookup_category_hint(&token_lower[s..e]),
                    });
                    added_any = true;
                }

                // R11b: Target `t_tok` is a child of opinion predicate `o_tok` (`nsubj`, `obj`, `dobj`, `obl`).
                for cand in sent_start..sent_end {
                    if deps.head[cand].map(|ix| ix.get()) == Some(o_tok) {
                        let rel = deps.relations.label(deps.relation[cand]);
                        if is_mr_target_child_rel(rel)
                            && self.can_be_aspect_target(cand, pos_tags, token_lower, opinions, aspects)
                        {
                            let (s, e) = expand_dep_noun_span(cand, sent_start, sent_end, deps, token_lower, opinions);
                            aspects.push(ExtractedAspect {
                                tokens: IxRange::new(s as u32, e as u32),
                                head_token: cand,
                                clause_id: clause_ids[cand - sent_start],
                                category_hint: lookup_category_hint(&token_lower[s..e]),
                            });
                            added_any = true;
                        }
                    }
                }

                // R12: Opinion `o_tok` and Target `cand` share a common governor `h` (`o_tok <- h -> cand`,
                // e.g., copular / verb-complement `nsubj(is, battery)` + `acomp(is, great)`).
                if let Some(h) = o_head
                    && h >= sent_start
                    && h < sent_end
                    && is_mr_complement_rel(o_rel)
                {
                    for cand in sent_start..sent_end {
                        if cand != o_tok && deps.head[cand].map(|ix| ix.get()) == Some(h) {
                            let rel = deps.relations.label(deps.relation[cand]);
                            if is_mr_target_child_rel(rel)
                                && self.can_be_aspect_target(cand, pos_tags, token_lower, opinions, aspects)
                            {
                                let (s, e) =
                                    expand_dep_noun_span(cand, sent_start, sent_end, deps, token_lower, opinions);
                                aspects.push(ExtractedAspect {
                                    tokens: IxRange::new(s as u32, e as u32),
                                    head_token: cand,
                                    clause_id: clause_ids[cand - sent_start],
                                    category_hint: lookup_category_hint(&token_lower[s..e]),
                                });
                                added_any = true;
                            }
                        }
                    }
                }
            }

            // R3 (T -> T'): Known targets extract conjoined targets (`conj`, `appos`).
            let current_aspects = aspects.clone();
            for asp in &current_aspects {
                let t_tok = asp.head_token;
                let t_head = deps.head[t_tok].map(|ix| ix.get());
                let t_rel = deps.relations.label(deps.relation[t_tok]);

                // Child conj of `t_tok`
                for cand in sent_start..sent_end {
                    if deps.head[cand].map(|ix| ix.get()) == Some(t_tok) {
                        let rel = deps.relations.label(deps.relation[cand]);
                        if is_conj_rel(rel) && self.can_be_aspect_target(cand, pos_tags, token_lower, opinions, aspects)
                        {
                            let (s, e) = expand_dep_noun_span(cand, sent_start, sent_end, deps, token_lower, opinions);
                            aspects.push(ExtractedAspect {
                                tokens: IxRange::new(s as u32, e as u32),
                                head_token: cand,
                                clause_id: clause_ids[cand - sent_start],
                                category_hint: lookup_category_hint(&token_lower[s..e]),
                            });
                            added_any = true;
                        }
                    }
                }
                // Parent conj when `t_tok` is the second conjunct
                if let Some(h) = t_head
                    && h >= sent_start
                    && h < sent_end
                    && is_conj_rel(t_rel)
                    && self.can_be_aspect_target(h, pos_tags, token_lower, opinions, aspects)
                {
                    let (s, e) = expand_dep_noun_span(h, sent_start, sent_end, deps, token_lower, opinions);
                    aspects.push(ExtractedAspect {
                        tokens: IxRange::new(s as u32, e as u32),
                        head_token: h,
                        clause_id: clause_ids[h - sent_start],
                        category_hint: lookup_category_hint(&token_lower[s..e]),
                    });
                    added_any = true;
                }
            }

            // R2 (T -> O): Extracted targets extract adjectival/predicate modifiers (`amod` or governing predicate)
            // that weren't in the seed opinion list if POS-tagged as `JJ*`.
            if let Some(pos) = pos_tags {
                let current_aspects = aspects.clone();
                for asp in &current_aspects {
                    let t_tok = asp.head_token;
                    for cand in sent_start..sent_end {
                        if deps.head[cand].map(|ix| ix.get()) == Some(t_tok) {
                            let rel = deps.relations.label(deps.relation[cand]);
                            let ptag = pos.tagset.label(pos.tags[cand]);
                            let w = token_lower[cand].as_str();
                            if rel.eq_ignore_ascii_case("amod")
                                && is_adj_pos(ptag)
                                && !w.is_empty()
                                && !LEADING_DETERMINERS.contains(&w)
                                && !opinions.iter().any(|o| range_contains(o.tokens, cand))
                                && !aspects.iter().any(|a| range_contains(a.tokens, cand))
                            {
                                let negated =
                                    is_token_negated(cand, sent_start, sent_end, token_lower, clause_ids, Some(deps));
                                let prior = self.lexicon.get(w).unwrap_or(1.0);
                                if prior.abs() > 0.0 {
                                    opinions.push(ExtractedOpinion {
                                        tokens: IxRange::new(cand as u32, (cand + 1) as u32),
                                        head_token: cand,
                                        polarity: prior,
                                        intensity: 1.0,
                                        negated,
                                        clause_id: clause_ids[cand - sent_start],
                                    });
                                    added_any = true;
                                }
                            }
                        }
                    }
                }
            }

            // R4 (O -> O'): Known opinions propagate across `conj` to conjoined adjectives/predicates.
            let current_ops = opinions.clone();
            for op in &current_ops {
                let o_tok = op.head_token;
                for cand in sent_start..sent_end {
                    if deps.head[cand].map(|ix| ix.get()) == Some(o_tok) {
                        let rel = deps.relations.label(deps.relation[cand]);
                        let w = token_lower[cand].as_str();
                        if is_conj_rel(rel)
                            && !w.is_empty()
                            && !is_aspect_stopword(w)
                            && !LEADING_DETERMINERS.contains(&w)
                            && !opinions.iter().any(|o| range_contains(o.tokens, cand))
                            && !aspects.iter().any(|a| range_contains(a.tokens, cand))
                        {
                            let adversative = has_adversative_between(o_tok, cand, token_lower);
                            let negated =
                                is_token_negated(cand, sent_start, sent_end, token_lower, clause_ids, Some(deps));
                            let base_pol = self
                                .token_opinion_valence(w)
                                .unwrap_or_else(|| if adversative { -op.polarity } else { op.polarity });
                            opinions.push(ExtractedOpinion {
                                tokens: IxRange::new(cand as u32, (cand + 1) as u32),
                                head_token: cand,
                                polarity: base_pol,
                                intensity: 1.0,
                                negated,
                                clause_id: clause_ids[cand - sent_start],
                            });
                            added_any = true;
                        }
                    }
                }
            }

            if !added_any {
                break;
            }
        }
    }

    fn can_be_aspect_target(
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

pub(super) fn assign_clauses(
    token_lower: &[String],
    text: &str,
    tok_starts: &[u32],
    tok_ends: &[u32],
    sent_start: usize,
    sent_end: usize,
) -> Vec<usize> {
    let mut clauses = Vec::with_capacity(sent_end - sent_start);
    let mut current_clause = 0usize;
    for tok in sent_start..sent_end {
        let w = token_lower[tok].as_str();
        let raw = text.get(tok_starts[tok] as usize..tok_ends[tok] as usize).unwrap_or("");
        if tok > sent_start
            && (ADVERSATIVE_CONJUNCTIONS.contains(&w) || raw == ";" || raw == "," || w == "and" || w == "while")
        {
            // Advance clause ID on adversative conjunctions, semicolons, or commas/coordinators
            // that separate clauses.
            current_clause += 1;
        }
        clauses.push(current_clause);
    }
    clauses
}

#[allow(clippy::too_many_arguments)]
fn expand_noun_compound(
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

fn expand_dep_noun_span(
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

fn is_noun_pos(tag: &str) -> bool {
    matches!(tag, "NN" | "NNS" | "NNP" | "NNPS" | "NOUN" | "PROPN")
}

fn is_adj_pos(tag: &str) -> bool {
    matches!(tag, "JJ" | "JJR" | "JJS" | "ADJ")
}

fn is_compound_rel(rel: &str) -> bool {
    rel.eq_ignore_ascii_case("compound") || rel.eq_ignore_ascii_case("nn") || rel.eq_ignore_ascii_case("flat")
}

pub(super) fn is_conj_rel(rel: &str) -> bool {
    rel.eq_ignore_ascii_case("conj") || rel.eq_ignore_ascii_case("appos")
}

fn is_mr_opinion_to_head_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "amod" | "acl" | "advmod" | "nmod" | "dep" | "xcomp" | "acomp" | "ccomp")
}

fn is_mr_target_child_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "nsubj" | "obj" | "dobj" | "iobj" | "obl" | "nmod")
}

fn is_mr_complement_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "acomp" | "xcomp" | "ccomp" | "attr" | "advmod" | "amod" | "obj" | "dobj")
}

fn is_token_negated(
    tok: usize,
    sent_start: usize,
    sent_end: usize,
    token_lower: &[String],
    clause_ids: &[usize],
    deps: Option<&Dependencies>,
) -> bool {
    // 1. Check dependency negation (`neg` or `advmod`/`det` child of `tok` or its copular head).
    if let Some(d) = deps
        && d.len() >= sent_end
    {
        for i in sent_start..sent_end {
            let w = token_lower[i].as_str();
            if !is_negation_word(w) {
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

    // 2. Check up to 3 preceding tokens within the same clause.
    let min_i = sent_start.max(tok.saturating_sub(3));
    let tok_clause = clause_ids[tok - sent_start];
    for i in min_i..tok {
        if clause_ids[i - sent_start] == tok_clause && is_negation_word(token_lower[i].as_str()) {
            return true;
        }
    }
    false
}

fn compute_token_intensity(
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
