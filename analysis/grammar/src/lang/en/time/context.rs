//! Context for the ambiguous adverbial phrase `to date`.

use super::{Ctx, Kind};

// Verbs licensing a to-infinitive, rather than a retrospective adjunct.
const INFINITIVE: &[&str] = &[
    "allow",
    "be",
    "encourage",
    "force",
    "have",
    "invite",
    "ought",
    "permit",
    "require",
    "teach",
    "tell",
    "urge",
    "afford",
    "agree",
    "aim",
    "arrange",
    "ask",
    "attempt",
    "beg",
    "begin",
    "choose",
    "claim",
    "come",
    "continue",
    "dare",
    "decide",
    "deserve",
    "desire",
    "expect",
    "fail",
    "forget",
    "go",
    "happen",
    "help",
    "hesitate",
    "hope",
    "intend",
    "learn",
    "like",
    "long",
    "love",
    "manage",
    "mean",
    "need",
    "offer",
    "plan",
    "prefer",
    "prepare",
    "pretend",
    "promise",
    "refuse",
    "remember",
    "seek",
    "start",
    "strive",
    "try",
    "use",
    "volunteer",
    "want",
    "wish",
];

fn inflected(word: &str, base: &str) -> bool {
    if word == base || ["s", "es", "d", "ed", "ing"].into_iter().any(|s| word.strip_suffix(s) == Some(base)) {
        return true;
    }
    if let Some(stem) = base.strip_suffix('e')
        && word.strip_suffix("ing") == Some(stem)
    {
        return true;
    }
    if let Some(stem) = base.strip_suffix('y')
        && ["ies", "ied"].into_iter().any(|s| word.strip_suffix(s) == Some(stem))
    {
        return true;
    }
    // Doubled final consonants: planned, preferred, beginning.
    ["ed", "ing"]
        .into_iter()
        .filter_map(|s| word.strip_suffix(s))
        .any(|stem| stem.strip_prefix(base).is_some_and(|tail| tail.len() == 1 && base.ends_with(tail)))
}

/// Admit an adjunct at a clause boundary, or before a finite predicate
/// (`results to date exceed ...`). A following object or noun modifier
/// (`to date him`, `to date night`) is not this construction. The lexical
/// governor check also excludes intransitive infinitives at clause end.
/// Check while building both point and modified forms: filtering only
/// the selected node lets rejected points leak into larger expressions.
pub(super) fn adverbial_to_date(c: &Ctx<'_>) -> bool {
    if c.before().is_some_and(|p| {
        let w = c.word(p);
        INFINITIVE.iter().any(|base| inflected(w, base))
            || matches!(
                w,
                "up" | "is"
                    | "are"
                    | "was"
                    | "were"
                    | "been"
                    | "being"
                    | "has"
                    | "had"
                    | "free"
                    | "supposed"
                    | "allowed"
                    | "able"
                    | "afraid"
                    | "eager"
                    | "easy"
                    | "hard"
                    | "difficult"
                    | "impossible"
                    | "possible"
                    | "ready"
                    | "reluctant"
                    | "safe"
                    | "willing"
                    | "went"
                    | "gone"
                    | "came"
                    | "chose"
                    | "chosen"
                    | "sought"
            )
    }) {
        return false;
    }
    c.after().is_none_or(|a| {
        if c.lexeme(a).kind == Kind::Punct {
            return matches!(c.punct(a), Some('.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '"' | '\''));
        }
        c.before().is_some()
            && matches!(
                c.word(a),
                "is" | "are"
                    | "was"
                    | "were"
                    | "has"
                    | "have"
                    | "had"
                    | "will"
                    | "would"
                    | "can"
                    | "could"
                    | "exceed"
                    | "exceeds"
                    | "exceeded"
                    | "remain"
                    | "remains"
                    | "remained"
                    | "show"
                    | "shows"
                    | "showed"
                    | "suggest"
                    | "suggests"
                    | "suggested"
                    | "include"
                    | "includes"
                    | "included"
                    | "indicate"
                    | "indicates"
                    | "indicated"
                    | "support"
                    | "supports"
                    | "supported"
                    | "total"
                    | "totals"
                    | "totaled"
            )
    })
}
