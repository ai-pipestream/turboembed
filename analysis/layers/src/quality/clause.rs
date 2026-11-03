//! Clause-level phonetic, lexical repetition, auxiliary verb government,
//! prepositional case government, and punctuation rules (`ART-001`, `REP-001`,
//! `AUX-001`, `CASE-001`, `PUNCT-001`).

use turbo_analysis::Span;

use super::agreement::{SVA_AUX_VERBS, SentCtx, is_intervening_adverb, is_preposition};
use super::grammar::{GrammarCategory, GrammarDiagnostic};
use super::morphology::{
    NounNumber, classify_noun_number, inflected_verb_to_base, irregular_verb_base, match_case,
    nominative_to_objective_pronoun, pluralize_noun, singularize_noun,
};
use super::phonetics::{replacement_article, requires_an};

const PARTITIVE_DETERMINERS: &[&str] = &["the", "these", "those", "my", "our", "their", "his", "her", "its"];

const ARTICLE_QUANTIFIER_NOUNS: &[&str] = &[
    "few", "lot", "couple", "dozen", "hundred", "thousand", "million", "billion", "number", "pair", "variety", "group",
    "set",
];

const ALLOWED_REDUPLICATIONS: &[&str] =
    &["had", "that", "so", "blah", "very", "ha", "bye", "no", "yes", "tut", "tsk", "there", "now", "well", "hear"];

const PREPOSITIONAL_TO_GOVERNORS: &[&str] = &[
    "according",
    "due",
    "prior",
    "pursuant",
    "subsequent",
    "subject",
    "thanks",
    "contrary",
    "relative",
    "compared",
    "objected",
    "opposed",
    "committed",
    "dedicated",
    "devoted",
    "used",
    "accustomed",
];

pub(crate) fn is_bare_infinitive_modal(lower: &str) -> bool {
    matches!(
        lower,
        "can" | "could" | "may" | "might" | "must" | "shall" | "should" | "will" | "would" | "did" | "does" | "do"
    ) || matches!(
        lower,
        "didn't"
            | "doesn't"
            | "don't"
            | "cannot"
            | "can't"
            | "won't"
            | "shouldn't"
            | "wouldn't"
            | "couldn't"
            | "mustn't"
            | "shan't"
            | "mightn't"
    )
}

pub(crate) fn check_articles(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    for i in start..end.saturating_sub(1) {
        let surf = cx.surface(i);
        if !matches!(surf, "a" | "A" | "an" | "An" | "AN") {
            continue;
        }
        let gap = &cx.text[cx.span(i).end as usize..cx.span(i + 1).start as usize];
        if !gap.chars().all(char::is_whitespace) {
            continue;
        }
        let next_surf = cx.surface(i + 1);
        let Some(want_an) = requires_an(next_surf, cx.checker.american_silent_h_herb) else {
            continue;
        };
        if let Some(rep) = replacement_article(surf, want_an) {
            let msg = if want_an {
                format!("Use {:?} before vowel-sound onset {:?}", rep, next_surf)
            } else {
                format!("Use {:?} before consonant-sound onset {:?}", rep, next_surf)
            };
            out.push(GrammarDiagnostic::new(
                cx.span(i),
                "ART-001",
                GrammarCategory::ArticlePhonetics,
                msg,
                [rep],
                0.96,
            ));
        }
    }
}

pub(crate) fn check_repetitions(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    let mut i = start;
    while i + 1 < end {
        let s1 = cx.surface(i);
        let s2 = cx.surface(i + 1);
        if !s1.is_empty()
            && s1.chars().all(char::is_alphabetic)
            && s1.eq_ignore_ascii_case(s2)
            && cx.text[cx.span(i).end as usize..cx.span(i + 1).start as usize].chars().all(char::is_whitespace)
        {
            let lower = s1.to_ascii_lowercase();
            if !ALLOWED_REDUPLICATIONS.contains(&lower.as_str()) {
                let combined = Span::new(cx.span(i).start, cx.span(i + 1).end);
                out.push(GrammarDiagnostic::new(
                    combined,
                    "REP-001",
                    GrammarCategory::RepeatedToken,
                    format!("Repeated adjacent token {:?}", s2),
                    [s1],
                    0.94,
                ));
                i += 2;
                continue;
            }
        }
        i += 1;
    }
}

pub(crate) fn check_determiners(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
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

pub(crate) fn check_auxiliaries(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    for i in start..end.saturating_sub(1) {
        let i_pos = cx.pos_str(i);
        if matches!(i_pos, "NN" | "NNS" | "NNP" | "NNPS" | "NOUN" | "PROPN") {
            continue;
        }
        let modal_lower = cx.surface(i).to_ascii_lowercase();
        let is_modal = is_bare_infinitive_modal(&modal_lower);
        let is_inf_to = modal_lower == "to"
            && matches!(i_pos, "TO" | "PART")
            && i > start
            && !PREPOSITIONAL_TO_GOVERNORS.contains(&cx.surface(i - 1).to_ascii_lowercase().as_str());
        if !is_modal && !is_inf_to {
            continue;
        }
        let mut j = i + 1;
        while j < end && j <= i + 3 && is_intervening_adverb(cx.surface(j), cx.pos_str(j)) {
            j += 1;
        }
        if j >= end {
            continue;
        }
        if is_inf_to && cx.deps.is_some() && !(cx.dep_rel(i) == Some("mark") && cx.dep_head(i) == Some(j)) {
            continue;
        }
        let v_surf = cx.surface(j);
        let v_pos = cx.pos_str(j);
        let v_lower = v_surf.to_ascii_lowercase();
        if is_inf_to && !matches!(v_pos, "VBZ" | "VBD") {
            continue;
        }
        let is_do_aux = matches!(modal_lower.as_str(), "do" | "does" | "did" | "don't" | "doesn't" | "didn't");
        if matches!(
            v_pos,
            "DT" | "DET"
                | "PRP"
                | "PRP$"
                | "PRON"
                | "IN"
                | "ADP"
                | "JJ"
                | "JJR"
                | "JJS"
                | "ADJ"
                | "NN"
                | "NNS"
                | "NNP"
                | "NNPS"
                | "NOUN"
                | "PROPN"
        ) && (is_do_aux || !irregular_verb_base(&v_lower).is_some_and(|b| b != v_lower))
        {
            continue;
        }
        let kind = if is_inf_to { "Infinitive marker" } else { "Auxiliary" };
        if let Some(base) = inflected_verb_to_base(v_surf, v_pos, cx.lemma_str(j))
            && !base.eq_ignore_ascii_case(v_surf)
        {
            out.push(GrammarDiagnostic::new(
                cx.span(j),
                "AUX-001",
                GrammarCategory::AuxiliaryVerbForm,
                format!("{kind} {:?} governs bare infinitive {:?}, found {:?}", cx.surface(i), base, v_surf),
                [base],
                0.95,
            ));
        } else if matches!(v_pos, "VBZ" | "VBD" | "VBN" | "VBG")
            && (v_lower.ends_with("ed") || v_lower.ends_with("ing") || v_pos == "VBZ")
        {
            out.push(GrammarDiagnostic::new(
                cx.span(j),
                "AUX-001",
                GrammarCategory::AuxiliaryVerbForm,
                format!("{kind} {:?} requires a bare infinitive verb, found {:?}", cx.surface(i), v_surf),
                Vec::<String>::new(),
                0.95,
            ));
        }
    }
}

pub(crate) fn check_case_government(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    for i in start..end.saturating_sub(1) {
        let surf = cx.surface(i);
        if matches!(surf.to_ascii_lowercase().as_str(), "than" | "as") {
            continue;
        }
        if !surf.eq_ignore_ascii_case("to") && !is_preposition(surf, cx.pos_str(i)) {
            continue;
        }
        let p1 = i + 1;
        if let Some(obj) = nominative_to_objective_pronoun(cx.surface(p1))
            && !is_followed_by_clause_verb(cx, p1, end)
        {
            out.push(GrammarDiagnostic::new(
                cx.span(p1),
                "CASE-001",
                GrammarCategory::CaseGovernment,
                format!(
                    "Preposition {:?} governs objective case {:?}, found nominative {:?}",
                    surf,
                    obj,
                    cx.surface(p1)
                ),
                [obj],
                0.93,
            ));
        }
        if i + 3 < end
            && matches!(cx.surface(i + 2).to_ascii_lowercase().as_str(), "and" | "or")
            && let Some(obj) = nominative_to_objective_pronoun(cx.surface(i + 3))
            && !is_followed_by_clause_verb(cx, i + 3, end)
        {
            out.push(GrammarDiagnostic::new(
                cx.span(i + 3),
                "CASE-001",
                GrammarCategory::CaseGovernment,
                format!(
                    "Preposition {:?} governs objective case {:?} in coordinated object, found {:?}",
                    surf,
                    obj,
                    cx.surface(i + 3)
                ),
                [obj],
                0.94,
            ));
        }
    }
}

fn is_followed_by_clause_verb(cx: &SentCtx<'_>, pron_ix: usize, end: usize) -> bool {
    if pron_ix + 1 >= end {
        return false;
    }
    let next_surf = cx.surface(pron_ix + 1).to_ascii_lowercase();
    let next_pos = cx.pos_str(pron_ix + 1);
    matches!(next_pos, "VBD" | "VBP" | "VBZ" | "MD" | "VERB" | "AUX")
        || is_bare_infinitive_modal(&next_surf)
        || matches!(next_surf.as_str(), "is" | "are" | "am" | "was" | "were" | "has" | "have" | "had")
}

pub(crate) fn check_punctuation(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    let mut i = start;
    while i < end {
        let surf = cx.surface(i);
        if matches!(surf, ".." | ",," | ";;" | "!!" | "??") {
            let single = &surf[..1];
            out.push(GrammarDiagnostic::new(
                cx.span(i),
                "PUNCT-001",
                GrammarCategory::ClausePunctuation,
                format!("Repeated clause punctuation {:?}; expected {:?}", surf, single),
                [single],
                0.95,
            ));
            i += 1;
            continue;
        }
        if i + 1 < end && matches!(surf, "," | ";" | "." | "!" | "?") && cx.surface(i + 1) == surf {
            let combined = Span::new(cx.span(i).start, cx.span(i + 1).end);
            out.push(GrammarDiagnostic::new(
                combined,
                "PUNCT-001",
                GrammarCategory::ClausePunctuation,
                format!("Repeated adjacent punctuation {:?}", surf),
                [surf],
                0.95,
            ));
            i += 2;
            continue;
        }
        i += 1;
    }
}
