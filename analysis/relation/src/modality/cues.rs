//! The modality cue lexicon, compiled into one `turbo-matcher` phrase
//! automaton over lowercase tokens.

use turbo_analysis_layers::Tokens;
use turbo_matcher::{Attr, Field, Input, MatchError, PhraseMatcher};

/// What a cue contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Class {
    Neg,
    /// A negative hedge (`impossible`, `unlikely`): a negation that also
    /// hedges, so `not impossible` is affirmed and hedged.
    NegHedge,
    Hedge,
    Reported,
    Cond,
    Factive,
    Oblig,
    Permit,
    Prohib,
    /// `may`: a permission when its subject is a party, else a hedge.
    May,
}

const CLASSES: [(Class, &str); 10] = [
    (Class::Neg, "neg"),
    (Class::NegHedge, "neg-hedge"),
    (Class::Hedge, "hedge"),
    (Class::Reported, "reported"),
    (Class::Cond, "cond"),
    (Class::Factive, "factive"),
    (Class::Oblig, "oblig"),
    (Class::Permit, "permit"),
    (Class::Prohib, "prohib"),
    (Class::May, "may"),
];

/// The built-in cues, pre-tokenized on spaces.
const LEXICON: &[(Class, &[&str])] = &[
    (
        Class::Neg,
        &[
            "not",
            "n't",
            "no",
            "never",
            "neither",
            "nor",
            "without",
            "cannot",
            "none",
            "nothing",
            "nobody",
            "didn't",
            "don't",
            "doesn't",
            "won't",
            "can't",
            "isn't",
            "wasn't",
            "aren't",
            "weren't",
            "hasn't",
            "haven't",
            "hadn't",
            "couldn't",
            "wouldn't",
            "shouldn't",
            "fail to",
            "fails to",
            "failed to",
            "failing to",
            "decline to",
            "declines to",
            "declined to",
            "refuse to",
            "refuses to",
            "refused to",
            "lack",
            "lacks",
            "lacked",
        ],
    ),
    (Class::NegHedge, &["impossible", "unlikely", "improbable", "doubtful"]),
    (
        Class::Hedge,
        &[
            "might",
            "could",
            "possibly",
            "perhaps",
            "conceivably",
            "probably",
            "likely",
            "potentially",
            "estimate",
            "estimates",
            "estimated",
            "rumored",
            "rumoured",
            "suggest",
            "suggests",
            "suggested",
            "appears to",
            "appeared to",
            "seems to",
            "seemed to",
            "is expected to",
            "are expected to",
            "was expected to",
            "possible",
        ],
    ),
    (
        Class::Reported,
        &["allegedly", "reportedly", "purportedly", "supposedly", "according to", "is said to", "are said to"],
    ),
    (Class::Cond, &["if", "unless", "provided that", "in the event that", "assuming", "as long as"]),
    (
        Class::Factive,
        &[
            "realized that",
            "realised that",
            "confirmed that",
            "acknowledged that",
            "knew that",
            "revealed that",
            "proved that",
            "admitted that",
        ],
    ),
    (
        Class::Oblig,
        &[
            "must",
            "shall",
            "should",
            "required to",
            "obligated to",
            "obliged to",
            "has to",
            "have to",
            "had to",
            "need to",
            "needs to",
        ],
    ),
    (Class::Permit, &["permitted to", "allowed to", "entitled to", "authorized to"]),
    (
        Class::Prohib,
        &["prohibited from", "forbidden to", "forbidden from", "banned from", "barred from", "mustn't", "shan't"],
    ),
    (Class::May, &["may"]),
];

/// One cue found in the text: tokens `start..end`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Cue {
    pub start: usize,
    pub end: usize,
    pub class: Class,
}

pub(super) struct Lexicon {
    matcher: PhraseMatcher,
}

impl Lexicon {
    pub(super) fn new() -> Lexicon {
        let mut matcher = PhraseMatcher::new(Field::Attr(Attr::Lower));
        for (class, phrases) in LEXICON {
            let label = CLASSES.iter().find(|c| c.0 == *class).expect("every class is named").1;
            for p in *phrases {
                let tokens: Vec<&str> = p.split(' ').collect();
                matcher.add(label, &tokens).expect("the built-in cues are valid phrases");
            }
        }
        Lexicon { matcher }
    }

    /// The phrase count, for the config hash.
    pub(super) fn phrases(&self) -> usize {
        self.matcher.phrases()
    }

    /// Cues in token order, the longest at each start, none overlapping.
    pub(super) fn find(&self, text: &str, tokens: &Tokens) -> Result<Vec<Cue>, MatchError> {
        let input = Input::new(text, tokens)?;
        let mut found: Vec<Cue> = self
            .matcher
            .find(&input)?
            .into_iter()
            .map(|m| {
                let name = self.matcher.label_name(m.label);
                let class = CLASSES.iter().find(|c| c.1 == name).expect("labels are classes").0;
                Cue { start: m.tokens.start.get(), end: m.tokens.end.get(), class }
            })
            .collect();
        found.sort_by_key(|c| (c.start, std::cmp::Reverse(c.end)));
        let mut out: Vec<Cue> = Vec::new();
        for c in found {
            if out.last().is_none_or(|l| c.start >= l.end) {
                out.push(c);
            }
        }
        Ok(out)
    }
}
