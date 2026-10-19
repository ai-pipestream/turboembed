use super::*;

/// Checks whether token `i` (at document token index `sent_start + i`) is under
/// negation scope in `deps`:
///
/// 1. Any dependent `d` in `[sent_start, sent_end)` with `head[d] == tok` that
///    either has relation `"neg"` or is a negation token (`is_negation_word`,
///    `"no"`, or `"least"` not preceded by `"at"`/`"very"`).
/// 2. If `tok` is an adjectival, clausal, or open complement or conjunction
///    (`amod`, `acomp`, `xcomp`, `ccomp`, `conj`, `advmod`, `dep`, `attr`) of
///    a head `h` in `[sent_start, sent_end)` (and `tok` is not separated from
///    `h` by a contrastive `"but"`), checks whether `h` itself has a direct
///    negation dependent (for example, `"The movie was not good"` where `"good"`
///    is head with `"not"` attached, or copular/verbal constructions where
///    `"was"` governs `"not"` and `"good"`).
pub(super) fn is_negated_in_dependencies(
    i: usize,
    sent_start: usize,
    sent_end: usize,
    lower_words: &[String],
    lexicon: &VaderLexicon,
    deps: &Dependencies,
) -> bool {
    let tok_global = sent_start + i;
    if has_direct_negation_dependent(tok_global, sent_start, sent_end, lower_words, lexicon, deps) {
        return true;
    }

    let Some(Some(head_ix)) = deps.head.get(tok_global) else {
        return false;
    };
    let head_global = head_ix.get();
    if head_global < sent_start || head_global >= sent_end || head_global == tok_global {
        return false;
    }

    // Do not propagate negation from a head across a contrastive "but" between head and token.
    let lo = head_global.min(tok_global) - sent_start;
    let hi = head_global.max(tok_global) - sent_start;
    if lower_words[lo..=hi].iter().any(|w| w == "but") {
        return false;
    }

    let rel = deps.relation.get(tok_global).map(|&t| deps.relations.label(t)).unwrap_or("");
    let base_rel = rel.split(':').next().unwrap_or(rel);
    if matches!(base_rel, "acomp" | "xcomp" | "ccomp" | "amod" | "conj" | "advmod" | "attr" | "oprd" | "dep")
        && has_direct_negation_dependent(head_global, sent_start, sent_end, lower_words, lexicon, deps)
    {
        return true;
    }

    false
}

pub(super) fn has_direct_negation_dependent(
    target_global: usize,
    sent_start: usize,
    sent_end: usize,
    lower_words: &[String],
    lexicon: &VaderLexicon,
    deps: &Dependencies,
) -> bool {
    for dep_global in sent_start..sent_end {
        if dep_global == target_global {
            continue;
        }
        let Some(Some(h)) = deps.head.get(dep_global) else {
            continue;
        };
        if h.get() != target_global {
            continue;
        }
        let local_idx = dep_global - sent_start;
        let w = lower_words[local_idx].as_str();
        let rel = deps.relation.get(dep_global).map(|&t| deps.relations.label(t)).unwrap_or("");
        let base_rel = rel.split(':').next().unwrap_or(rel);
        if base_rel == "neg" || is_negation_word(w) || w == "no" {
            return true;
        }
        if w == "least" && !lexicon.contains(w) {
            let preceded_by_at_or_very = local_idx > 0 && matches!(lower_words[local_idx - 1].as_str(), "at" | "very");
            if !preceded_by_at_or_very {
                return true;
            }
        }
    }
    false
}

pub(super) fn lookup_emoji_prior(
    surface: &str,
    tok_span: Span,
    view_text: &str,
    emoji_layer: Option<&Emoji>,
) -> Option<f64> {
    if let Some(a) = opennlp_annotation(surface)
        && a.sentiment != 0
    {
        return Some(a.sentiment as f64);
    }
    if let Some(emoji) = emoji_layer {
        for (_, esp) in emoji.spans.iter() {
            if esp.start >= tok_span.start
                && esp.end <= tok_span.end
                && let Some(a) = opennlp_annotation(&view_text[esp.range()])
                && a.sentiment != 0
            {
                return Some(a.sentiment as f64);
            }
        }
    }
    None
}
