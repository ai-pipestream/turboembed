//! Clause-level phonetic, lexical repetition, auxiliary verb government,
//! prepositional case government, and punctuation rules (`ART-001`, `REP-001`,
//! `AUX-001`, `CASE-001`, `PUNCT-001`).

use turbo_analysis::Span;

use super::agreement::{SentCtx, is_intervening_adverb, is_preposition};
use super::grammar::{GrammarCategory, GrammarDiagnostic};
use super::morphology::{inflected_verb_to_base, irregular_verb_base, nominative_to_objective_pronoun};
use super::phonetics::{replacement_article, requires_an};

const ALLOWED_REDUPLICATIONS: &[&str] =
    &["had", "that", "so", "blah", "very", "ha", "bye", "no", "yes", "tut", "tsk", "there", "now", "well", "hear"];

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

pub(crate) fn check_auxiliaries(cx: &SentCtx<'_>, out: &mut Vec<GrammarDiagnostic>) {
    let start = cx.range.start.get();
    let end = cx.range.end.get();
    for i in start..end.saturating_sub(1) {
        let modal_lower = cx.surface(i).to_ascii_lowercase();
        if !is_bare_infinitive_modal(&modal_lower) {
            continue;
        }
        let mut j = i + 1;
        while j < end && j <= i + 3 && is_intervening_adverb(cx.surface(j), cx.pos_str(j)) {
            j += 1;
        }
        if j >= end {
            continue;
        }
        let v_surf = cx.surface(j);
        let v_pos = cx.pos_str(j);
        let v_lower = v_surf.to_ascii_lowercase();
        if matches!(v_pos, "DT" | "DET" | "PRP" | "PRON" | "IN" | "ADP" | "JJ" | "ADJ" | "NN" | "NNP")
            && !irregular_verb_base(&v_lower).is_some_and(|b| b != v_lower)
        {
            continue;
        }
        if let Some(base) = inflected_verb_to_base(v_surf, v_pos, cx.lemma_str(j))
            && !base.eq_ignore_ascii_case(v_surf)
        {
            out.push(GrammarDiagnostic::new(
                cx.span(j),
                "AUX-001",
                GrammarCategory::AuxiliaryVerbForm,
                format!("Auxiliary {:?} governs bare infinitive {:?}, found {:?}", cx.surface(i), base, v_surf),
                [base],
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
