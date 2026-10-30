//! Syntactic and morphological agreement rules (`SVA-001` and `DET-001`) and
//! sentence rule orchestration.

use turbo_analysis::{Ix, IxRange, Layer, Span};

use crate::{Dependencies, Lemmas, MorphFeatures, PosTags, Sentences, Tokens};

use super::clause::{
    check_articles, check_auxiliaries, check_case_government, check_determiners, check_punctuation, check_repetitions,
    is_bare_infinitive_modal,
};
use super::grammar::{GrammarCategory, GrammarChecker, GrammarDiagnostic};
use super::morphology::{SubjectPersonNumber, check_and_repair_sva_verb, classify_subject_token};

const INTERVENING_ADVERBS: &[&str] = &[
    "not",
    "n't",
    "never",
    "always",
    "also",
    "only",
    "even",
    "still",
    "just",
    "already",
    "often",
    "rarely",
    "seldom",
    "really",
    "certainly",
    "definitely",
    "probably",
    "possibly",
];

const PREPOSITIONS: &[&str] = &[
    "of",
    "in",
    "on",
    "at",
    "from",
    "with",
    "by",
    "for",
    "between",
    "among",
    "under",
    "over",
    "through",
    "across",
    "into",
    "without",
    "within",
    "about",
    "during",
    "including",
    "near",
    "despite",
    "towards",
    "toward",
    "upon",
    "against",
    "behind",
    "beyond",
];

const SUBORDINATING_CONJUNCTIONS: &[&str] =
    &["that", "if", "whether", "because", "although", "though", "while", "since", "unless", "whereas"];

pub(crate) const SVA_AUX_VERBS: &[&str] = &[
    "is", "are", "am", "was", "were", "has", "have", "does", "do", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "doesn't", "don't",
];

pub(crate) struct SentCtx<'a> {
    pub(crate) checker: &'a GrammarChecker,
    pub(crate) text: &'a str,
    pub(crate) tokens: &'a Tokens,
    pub(crate) pos: &'a PosTags,
    pub(crate) lemmas: Option<&'a Lemmas>,
    pub(crate) deps: Option<&'a Dependencies>,
    pub(crate) morph: Option<&'a MorphFeatures<Tokens>>,
    pub(crate) dep_children: Vec<Vec<usize>>,
    pub(crate) range: IxRange<Tokens>,
}

impl<'a> SentCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        checker: &'a GrammarChecker,
        text: &'a str,
        tokens: &'a Tokens,
        pos: &'a PosTags,
        lemmas: Option<&'a Lemmas>,
        deps: Option<&'a Dependencies>,
        morph: Option<&'a MorphFeatures<Tokens>>,
        range: IxRange<Tokens>,
    ) -> Self {
        let start = range.start.get();
        let end = range.end.get();
        let mut dep_children = Vec::new();
        if let Some(d) = deps {
            dep_children.resize_with(end - start, Vec::new);
            for t in start..end {
                if let Some(h) = d.head[t].map(Ix::get)
                    && (start..end).contains(&h)
                {
                    dep_children[h - start].push(t);
                }
            }
        }
        Self { checker, text, tokens, pos, lemmas, deps, morph, dep_children, range }
    }

    pub(crate) fn span(&self, i: usize) -> Span {
        self.tokens.spans.get(Ix::new(i as u32))
    }

    pub(crate) fn surface(&self, i: usize) -> &'a str {
        &self.text[self.span(i).range()]
    }

    pub(crate) fn pos_str(&self, i: usize) -> &'a str {
        self.pos.tagset.label(self.pos.tags[i])
    }

    pub(crate) fn lemma_str(&self, i: usize) -> Option<&'a str> {
        self.lemmas.map(|l| l.lemma.get(i))
    }

    pub(crate) fn morph_str(&self, i: usize) -> Option<&'a str> {
        self.morph.map(|m| m.feats.get(i))
    }

    pub(crate) fn dep_head(&self, i: usize) -> Option<usize> {
        self.deps.and_then(|d| d.head[i].map(Ix::get))
    }

    pub(crate) fn dep_rel(&self, i: usize) -> Option<&'a str> {
        self.deps.map(|d| d.relations.label(d.relation[i]))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn check_sentence_rules(
    checker: &GrammarChecker,
    text: &str,
    sentences: &Sentences,
    tokens: &Tokens,
    pos: &PosTags,
    lemmas: Option<&Lemmas>,
    deps: Option<&Dependencies>,
    morph: Option<&MorphFeatures<Tokens>>,
) -> Vec<GrammarDiagnostic> {
    let mut out = Vec::new();
    let sent_count = sentences.len();
    if sent_count == 0 && tokens.len() > 0 {
        let range = IxRange::new(0, tokens.len() as u32);
        let cx = SentCtx::new(checker, text, tokens, pos, lemmas, deps, morph, range);
        run_on_sentence(&cx, &mut out);
        return out;
    }
    for s_ix in 0..sent_count {
        let range = tokens.in_sentence(sentences, Ix::new(s_ix as u32));
        if !range.is_empty() {
            let cx = SentCtx::new(checker, text, tokens, pos, lemmas, deps, morph, range);
            run_on_sentence(&cx, &mut out);
        }
    }
    out
}

fn run_on_sentence(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    if cx.checker.check_article_phonetics {
        check_articles(cx, out);
    }
    if cx.checker.check_repeated_tokens {
        check_repetitions(cx, out);
    }
    if cx.checker.check_auxiliary_verb_form {
        check_auxiliaries(cx, out);
    }
    if cx.checker.check_determiner_noun_agreement {
        check_determiners(cx, out);
    }
    if cx.checker.check_subject_verb_agreement {
        check_sva(cx, out);
    }
    if cx.checker.check_case_government {
        check_case_government(cx, out);
    }
    if cx.checker.check_clause_punctuation {
        check_punctuation(cx, out);
    }
}

pub(crate) fn is_intervening_adverb(surface: &str, pos_tag: &str) -> bool {
    matches!(pos_tag, "RB" | "RBR" | "RBS" | "ADV")
        || INTERVENING_ADVERBS.contains(&surface.to_ascii_lowercase().as_str())
}

pub(crate) fn is_preposition(surface: &str, pos_tag: &str) -> bool {
    let lower = surface.to_ascii_lowercase();
    if matches!(pos_tag, "IN" | "ADP") && !SUBORDINATING_CONJUNCTIONS.contains(&lower.as_str()) {
        return true;
    }
    PREPOSITIONS.contains(&lower.as_str())
}

fn check_sva(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    for v in start..end {
        let v_surf = cx.surface(v);
        let v_pos = cx.pos_str(v);
        let v_lower = v_surf.to_ascii_lowercase();
        if (!SVA_AUX_VERBS.contains(&v_lower.as_str()) && !matches!(v_pos, "VBZ" | "VBP" | "VERB"))
            || is_governed_by_preceding_aux_or_to(cx, v)
            || is_subjunctive_were(cx, v, &v_lower)
        {
            continue;
        }
        let Some((subj_ix, target)) = resolve_clause_subject(cx, v) else {
            continue;
        };
        if let Some(rep) = check_and_repair_sva_verb(v_surf, v_pos, cx.lemma_str(v), target) {
            out.push(GrammarDiagnostic::new(
                cx.span(v),
                "SVA-001",
                GrammarCategory::SubjectVerbAgreement,
                format!(
                    "Subject {:?} disagrees in number/person with finite verb {:?}; expected {:?}",
                    cx.surface(subj_ix),
                    v_surf,
                    rep
                ),
                [rep],
                0.92,
            ));
        }
    }
}

fn is_subjunctive_were(cx: &SentCtx<'_>, v: usize, v_lower: &str) -> bool {
    if !matches!(v_lower, "were" | "weren't") {
        return false;
    }
    if cx.morph_str(v).is_some_and(|m| m.contains("Mood=Sub")) {
        return true;
    }
    let start = cx.range.start.get();
    let window_start = v.saturating_sub(6).max(start);
    (window_start..v).any(|i| {
        matches!(
            cx.surface(i).to_ascii_lowercase().as_str(),
            "if" | "though" | "although" | "unless" | "whether" | "wish" | "wishes" | "wished" | "wishing" | "suppose"
        )
    })
}

fn is_governed_by_preceding_aux_or_to(cx: &SentCtx<'_>, v: usize) -> bool {
    let start = cx.range.start.get();
    let mut k = v;
    while k > start && v - k < 3 {
        k -= 1;
        let prev_lower = cx.surface(k).to_ascii_lowercase();
        if is_intervening_adverb(cx.surface(k), cx.pos_str(k)) {
            continue;
        }
        return prev_lower == "to"
            || is_bare_infinitive_modal(&prev_lower)
            || SVA_AUX_VERBS.contains(&prev_lower.as_str())
            || matches!(prev_lower.as_str(), "had" | "be" | "been" | "being");
    }
    false
}

fn resolve_clause_subject(cx: &SentCtx<'_>, v: usize) -> Option<(usize, SubjectPersonNumber)> {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    if cx.deps.is_some() {
        let head_for_subj = match cx.dep_rel(v) {
            Some("cop" | "aux" | "aux:pass" | "auxpass") => cx.dep_head(v).unwrap_or(v),
            _ => v,
        };
        if (start..end).contains(&head_for_subj) {
            let children = &cx.dep_children[head_for_subj - start];
            if children.iter().any(|&t| cx.dep_rel(t) == Some("expl") || cx.surface(t).eq_ignore_ascii_case("there")) {
                return None;
            }
            let subj_before = children
                .iter()
                .rev()
                .copied()
                .find(|&t| t < v && matches!(cx.dep_rel(t), Some("nsubj" | "nsubj:pass" | "nsubjpass")));
            let subj_after = children
                .iter()
                .copied()
                .find(|&t| t > v && matches!(cx.dep_rel(t), Some("nsubj" | "nsubj:pass" | "nsubjpass")));
            if let Some(subj) = subj_before.or(subj_after) {
                let eff_subj = if is_relative_pronoun(cx.surface(subj), cx.pos_str(subj)) && subj > start {
                    find_antecedent_before(cx, subj).unwrap_or(subj)
                } else {
                    subj
                };
                if (start..end).contains(&eff_subj) {
                    let has_conj = cx.dep_children[eff_subj - start].iter().any(|&t| {
                        matches!(cx.dep_rel(t), Some("conj" | "cc")) || cx.surface(t).eq_ignore_ascii_case("and")
                    });
                    if has_conj {
                        return Some((eff_subj, SubjectPersonNumber::SecondOrPlural));
                    }
                }
                if let Some(spn) = classify_subject_token(cx.surface(eff_subj), cx.pos_str(eff_subj)) {
                    return Some((eff_subj, spn));
                }
            }
        }
    }

    if v == start {
        return None;
    }
    let window_floor = v.saturating_sub(24).max(start);
    let punct_start = (window_floor..v)
        .rev()
        .find(|&i| matches!(cx.surface(i), ";" | ":" | "," | "(" | ")"))
        .map_or(window_floor, |i| i + 1);
    if punct_start >= v {
        return None;
    }

    let mut back = v;
    while back > punct_start && is_intervening_adverb(cx.surface(back - 1), cx.pos_str(back - 1)) {
        back -= 1;
    }
    if back > punct_start && is_relative_marker(cx, back - 1) {
        let ant = find_antecedent_before(cx, back - 1)?;
        return Some((ant, classify_subject_token(cx.surface(ant), cx.pos_str(ant))?));
    }

    let (clause_start, rel_bound) = narrow_clause_bounds(cx, punct_start, v);
    if clause_start >= v || cx.surface(clause_start).eq_ignore_ascii_case("there") || cx.pos_str(clause_start) == "EX" {
        return None;
    }

    let np_end = (clause_start + 1..rel_bound)
        .find(|&i| is_relative_marker(cx, i) || is_preposition(cx.surface(i), cx.pos_str(i)))
        .unwrap_or(rel_bound);

    let mut heads = Vec::new();
    let mut saw_and = false;
    for i in clause_start..np_end {
        let s = cx.surface(i);
        if s.eq_ignore_ascii_case("and") {
            saw_and = true;
        } else if classify_subject_token(s, cx.pos_str(i)).is_some() {
            heads.push(i);
        }
    }
    let &head_ix = heads.last()?;
    if saw_and && heads.len() >= 2 {
        return Some((head_ix, SubjectPersonNumber::SecondOrPlural));
    }
    Some((head_ix, classify_subject_token(cx.surface(head_ix), cx.pos_str(head_ix))?))
}

fn narrow_clause_bounds(cx: &SentCtx<'_>, punct_start: usize, v: usize) -> (usize, usize) {
    if let Some(rel_ix) = (punct_start + 1..v).find(|&i| is_relative_marker(cx, i)) {
        let rel_verbs: Vec<usize> = (rel_ix + 1..v).filter(|&i| is_clause_finite_verb(cx, i)).collect();
        if rel_verbs.is_empty() {
            return (rel_ix + 1, v);
        }
        let last_rv = *rel_verbs.last().unwrap();
        let has_coord_after = (last_rv + 1..v)
            .any(|k| matches!(cx.surface(k).to_ascii_lowercase().as_str(), "and" | "or" | "but" | "yet" | "so"));
        if !has_coord_after {
            return (punct_start, rel_ix);
        }
    }
    if let Some(last_fv) = (punct_start..v).rev().find(|&i| is_clause_finite_verb(cx, i)) {
        let after_conj = (last_fv + 1..v)
            .find(|&k| {
                let low = cx.surface(k).to_ascii_lowercase();
                matches!(low.as_str(), "and" | "or" | "but" | "yet" | "so" | "nor")
                    || SUBORDINATING_CONJUNCTIONS.contains(&low.as_str())
            })
            .map_or(last_fv + 1, |k| k + 1);
        return (after_conj, v);
    }
    (punct_start, v)
}

fn is_clause_finite_verb(cx: &SentCtx<'_>, i: usize) -> bool {
    let pos = cx.pos_str(i);
    let low = cx.surface(i).to_ascii_lowercase();
    matches!(pos, "VBD" | "VBP" | "VBZ" | "MD" | "VERB" | "AUX")
        || SVA_AUX_VERBS.contains(&low.as_str())
        || is_bare_infinitive_modal(&low)
}

fn find_antecedent_before(cx: &SentCtx<'_>, rel_ix: usize) -> Option<usize> {
    let floor = rel_ix.saturating_sub(12).max(cx.range.start.get());
    (floor..rel_ix).rev().find(|&i| classify_subject_token(cx.surface(i), cx.pos_str(i)).is_some())
}

fn is_relative_marker(cx: &SentCtx<'_>, i: usize) -> bool {
    is_relative_pronoun(cx.surface(i), cx.pos_str(i))
        || (cx.surface(i).eq_ignore_ascii_case("that")
            && i > cx.range.start.get()
            && matches!(cx.pos_str(i - 1), "NN" | "NNS" | "NNP" | "NNPS" | "NOUN" | "PROPN"))
}

fn is_relative_pronoun(surface: &str, pos_tag: &str) -> bool {
    matches!(pos_tag, "WP" | "WDT") || matches!(surface.to_ascii_lowercase().as_str(), "who" | "whom" | "which")
}
