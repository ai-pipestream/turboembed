//! Syntactic and morphological agreement rules (`SVA-001` and `DET-001`) and
//! sentence rule orchestration.

use turbo_analysis::{Ix, IxRange, Layer, Span};

use crate::{Dependencies, Lemmas, PosTags, Sentences, Tokens};

use super::clause::{
    check_articles, check_auxiliaries, check_case_government, check_punctuation, check_repetitions,
    is_bare_infinitive_modal,
};
use super::grammar::{GrammarCategory, GrammarChecker, GrammarDiagnostic};
use super::morphology::{
    NounNumber, SubjectPersonNumber, check_and_repair_sva_verb, classify_noun_number, classify_subject_token,
    match_case, pluralize_noun, singularize_noun,
};

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

const PARTITIVE_DETERMINERS: &[&str] = &["the", "these", "those", "my", "our", "their", "his", "her", "its"];

const ARTICLE_QUANTIFIER_NOUNS: &[&str] = &[
    "few", "lot", "couple", "dozen", "hundred", "thousand", "million", "billion", "number", "pair", "variety", "group",
    "set",
];

const SVA_AUX_VERBS: &[&str] = &[
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
    pub(crate) range: IxRange<Tokens>,
}

impl<'a> SentCtx<'a> {
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

    pub(crate) fn dep_head(&self, i: usize) -> Option<usize> {
        self.deps.and_then(|d| d.head[i].map(Ix::get))
    }

    pub(crate) fn dep_rel(&self, i: usize) -> Option<&'a str> {
        self.deps.map(|d| d.relations.label(d.relation[i]))
    }
}

pub(crate) fn check_sentence_rules(
    checker: &GrammarChecker,
    text: &str,
    sentences: &Sentences,
    tokens: &Tokens,
    pos: &PosTags,
    lemmas: Option<&Lemmas>,
    deps: Option<&Dependencies>,
) -> Vec<GrammarDiagnostic> {
    let mut out = Vec::new();
    let sent_count = sentences.len();
    if sent_count == 0 && tokens.len() > 0 {
        let cx = SentCtx { checker, text, tokens, pos, lemmas, deps, range: IxRange::new(0, tokens.len() as u32) };
        run_on_sentence(&cx, &mut out);
        return out;
    }
    for s_ix in 0..sent_count {
        let range = tokens.in_sentence(sentences, Ix::new(s_ix as u32));
        if !range.is_empty() {
            let cx = SentCtx { checker, text, tokens, pos, lemmas, deps, range };
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

fn check_determiners(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();

    for i in start..end.saturating_sub(3) {
        if cx.surface(i).eq_ignore_ascii_case("one")
            && cx.surface(i + 1).eq_ignore_ascii_case("of")
            && PARTITIVE_DETERMINERS.contains(&cx.surface(i + 2).to_ascii_lowercase().as_str())
            && let Some((head_ix, NounNumber::Singular)) = find_np_head_noun(cx, i + 3, end)
        {
            let plural = pluralize_noun(cx.surface(head_ix));
            out.push(GrammarDiagnostic::new(
                cx.span(head_ix),
                "DET-001",
                GrammarCategory::DeterminerNounAgreement,
                format!("Partitive \"one of {}\" requires plural noun {:?}", cx.surface(i + 2), plural),
                [plural],
                0.93,
            ));
        }
    }

    for i in start..end.saturating_sub(1) {
        let det_surf = cx.surface(i);
        let det_lower = det_surf.to_ascii_lowercase();
        if !matches!(
            det_lower.as_str(),
            "this" | "that" | "these" | "those" | "many" | "several" | "both" | "fewer" | "much" | "less" | "a" | "an"
        ) {
            continue;
        }
        if det_lower == "that" && !matches!(cx.pos_str(i), "DT" | "DET") && is_preceded_by_verb_or_noun(cx, i) {
            continue;
        }
        let next_lower = cx.surface(i + 1).to_ascii_lowercase();
        if matches!(det_lower.as_str(), "a" | "an") && ARTICLE_QUANTIFIER_NOUNS.contains(&next_lower.as_str()) {
            continue;
        }
        if matches!(det_lower.as_str(), "both" | "this" | "that" | "these" | "those")
            && (next_lower == "of" || next_lower == "and" || SVA_AUX_VERBS.contains(&next_lower.as_str()))
        {
            continue;
        }
        if i >= start + 2
            && cx.surface(i - 2).eq_ignore_ascii_case("one")
            && cx.surface(i - 1).eq_ignore_ascii_case("of")
        {
            continue;
        }
        let Some((head_ix, number)) = find_np_head_noun(cx, i + 1, end) else {
            continue;
        };
        let head_surf = cx.surface(head_ix);
        let diag = match (det_lower.as_str(), number) {
            ("this", NounNumber::Plural) => Some((
                cx.span(i),
                match_case(det_surf, "these"),
                format!("Singular determiner {:?} disagrees with plural noun {:?}", det_surf, head_surf),
            )),
            ("that", NounNumber::Plural) => Some((
                cx.span(i),
                match_case(det_surf, "those"),
                format!("Singular determiner {:?} disagrees with plural noun {:?}", det_surf, head_surf),
            )),
            ("these", NounNumber::Singular) => Some((
                cx.span(i),
                match_case(det_surf, "this"),
                format!("Plural determiner {:?} disagrees with singular noun {:?}", det_surf, head_surf),
            )),
            ("those", NounNumber::Singular) => Some((
                cx.span(i),
                match_case(det_surf, "that"),
                format!("Plural determiner {:?} disagrees with singular noun {:?}", det_surf, head_surf),
            )),
            ("many" | "several" | "both" | "fewer", NounNumber::Singular) => {
                let rep = pluralize_noun(head_surf);
                let msg = format!("Plural quantifier {:?} requires plural noun {:?}", det_surf, rep);
                Some((cx.span(head_ix), rep, msg))
            }
            ("much", NounNumber::Plural) => Some((
                cx.span(i),
                match_case(det_surf, "many"),
                format!("Mass quantifier {:?} disagrees with plural count noun {:?}", det_surf, head_surf),
            )),
            ("less", NounNumber::Plural) => {
                let rep = match_case(det_surf, "fewer");
                let msg = format!("Use {:?} with plural count noun {:?}", rep, head_surf);
                Some((cx.span(i), rep, msg))
            }
            ("a" | "an", NounNumber::Plural) => {
                let rep = singularize_noun(head_surf);
                let msg = format!("Indefinite article {:?} requires singular noun {:?}", det_surf, rep);
                Some((cx.span(head_ix), rep, msg))
            }
            _ => None,
        };
        if let Some((span, rep, msg)) = diag {
            out.push(GrammarDiagnostic::new(
                span,
                "DET-001",
                GrammarCategory::DeterminerNounAgreement,
                msg,
                [rep],
                0.93,
            ));
        }
    }
}

fn is_preceded_by_verb_or_noun(cx: &SentCtx<'_>, i: usize) -> bool {
    i > cx.range.start.get()
        && matches!(
            cx.pos_str(i - 1),
            "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" | "VERB" | "NN" | "NNS" | "NNP" | "NNPS" | "NOUN" | "PROPN"
        )
}

fn find_np_head_noun(cx: &SentCtx<'_>, from: usize, end: usize) -> Option<(usize, NounNumber)> {
    let mut last_noun = None;
    let mut j = from;
    while j < end && j <= from + 4 {
        let p = cx.pos_str(j);
        let lower = cx.surface(j).to_ascii_lowercase();
        if matches!(p, "JJ" | "JJR" | "JJS" | "ADJ" | "RB" | "ADV" | "VBN" | "VBG") && last_noun.is_none() {
            j += 1;
            continue;
        }
        if let Some(num) = classify_noun_number(&lower, p)
            && !matches!(p, "VBZ" | "VBP" | "VBD" | "MD" | "AUX")
            && !SVA_AUX_VERBS.contains(&lower.as_str())
        {
            last_noun = Some((j, num));
            j += 1;
            continue;
        }
        break;
    }
    last_noun
}

fn check_sva(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    for v in start..end {
        let v_surf = cx.surface(v);
        let v_pos = cx.pos_str(v);
        if (!SVA_AUX_VERBS.contains(&v_surf.to_ascii_lowercase().as_str()) && !matches!(v_pos, "VBZ" | "VBP" | "VERB"))
            || is_governed_by_preceding_aux_or_to(cx, v)
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
        if (start..end).any(|t| {
            cx.dep_head(t) == Some(head_for_subj)
                && (cx.dep_rel(t) == Some("expl") || cx.surface(t).eq_ignore_ascii_case("there"))
        }) {
            return None;
        }
        if let Some(subj) = (start..v).rev().find(|&t| {
            cx.dep_head(t) == Some(head_for_subj) && matches!(cx.dep_rel(t), Some("nsubj" | "nsubj:pass" | "nsubjpass"))
        }) {
            let eff_subj = if is_relative_pronoun(cx.surface(subj), cx.pos_str(subj)) && subj > start {
                find_antecedent_before(cx, subj).unwrap_or(subj)
            } else {
                subj
            };
            let has_conj = (eff_subj + 1..v).any(|t| {
                (cx.dep_head(t) == Some(eff_subj) && matches!(cx.dep_rel(t), Some("conj" | "cc")))
                    || (cx.surface(t).eq_ignore_ascii_case("and") && cx.dep_head(t) == Some(eff_subj))
            });
            if has_conj {
                return Some((eff_subj, SubjectPersonNumber::SecondOrPlural));
            }
            if let Some(spn) = classify_subject_token(cx.surface(eff_subj), cx.pos_str(eff_subj)) {
                return Some((eff_subj, spn));
            }
        }
    }

    if v == start {
        return None;
    }
    let clause_start =
        (start..v).rev().find(|&i| matches!(cx.surface(i), ";" | ":" | "," | "(" | ")")).map_or(start, |i| i + 1);
    if clause_start >= v || cx.surface(clause_start).eq_ignore_ascii_case("there") || cx.pos_str(clause_start) == "EX" {
        return None;
    }

    let mut back = v;
    while back > clause_start && is_intervening_adverb(cx.surface(back - 1), cx.pos_str(back - 1)) {
        back -= 1;
    }
    if back > clause_start && is_relative_pronoun(cx.surface(back - 1), cx.pos_str(back - 1)) {
        let ant = find_antecedent_before(cx, back - 1)?;
        return Some((ant, classify_subject_token(cx.surface(ant), cx.pos_str(ant))?));
    }

    let np_end = (clause_start + 1..v)
        .find(|&i| is_relative_pronoun(cx.surface(i), cx.pos_str(i)) || is_preposition(cx.surface(i), cx.pos_str(i)))
        .unwrap_or(v);

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

fn find_antecedent_before(cx: &SentCtx<'_>, rel_ix: usize) -> Option<usize> {
    (cx.range.start.get()..rel_ix).rev().find(|&i| classify_subject_token(cx.surface(i), cx.pos_str(i)).is_some())
}

fn is_relative_pronoun(surface: &str, pos_tag: &str) -> bool {
    matches!(pos_tag, "WP" | "WDT") || matches!(surface.to_ascii_lowercase().as_str(), "who" | "whom" | "which")
}
