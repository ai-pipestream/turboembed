//! Sentiment layers for aspect-based sentiment, opinion expressions,
//! multi-label incivility, and sentence pragmatics (sarcasm and irony).
//!
//! Every span is an [`Ix`] or [`IxRange`] into an existing layer
//! ([`Tokens`], [`Sentences`](crate::Sentences), [`Chunks`](crate::Chunks), or [`Structure`](crate::Structure)), so byte
//! offsets follow the document's view alignment without copying spans.
//! All scores are `f64` so `1e-9` parity checks hold.

use std::fmt;

use turbo_analysis::{Anchor, Ix, IxRange, Key, KeyId, Layer, Span};

use crate::Tokens;

mod aspects;
mod incivility;
mod pragmatics;
#[cfg(test)]
mod tests;

pub use aspects::{Aspect, AspectTarget, Aspects, PolarityScores};
pub use incivility::{Incivility, IncivilityItem, IncivilityTarget};
pub use pragmatics::{Incongruity, Pragmatics, PragmaticsItem};

/// Coarse polarity sign derived from a signed polarity score or a
/// three-way polarity distribution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Polarity {
    Positive,
    Negative,
    Neutral,
}

impl Polarity {
    /// Classifies a signed polarity score (`> 0.0` positive, `< 0.0`
    /// negative, `0.0` neutral).
    pub fn from_score(score: f64) -> Polarity {
        if score > 0.0 {
            Polarity::Positive
        } else if score < 0.0 {
            Polarity::Negative
        } else {
            Polarity::Neutral
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Polarity::Positive => "positive",
            Polarity::Negative => "negative",
            Polarity::Neutral => "neutral",
        }
    }
}

impl fmt::Display for Polarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One opinion expression in [`Opinions`] (`turbo:opinions`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Opinion {
    /// The non-empty token span (`start < end`) of the opinion expression.
    pub tokens: IxRange<Tokens>,
    /// Prior polarity score in `f64` (positive `> 0.0`, negative `< 0.0`,
    /// neutral `0.0`).
    pub polarity: f64,
    /// Intensity multiplier or weight (`>= 0.0`, `1.0` unmodified).
    pub intensity: f64,
    /// Whether the opinion expression falls under negation scope.
    pub negated: bool,
}

impl Opinion {
    /// Effective polarity after applying `intensity` and `negated`.
    pub fn effective_polarity(self) -> f64 {
        let sign = if self.negated { -1.0 } else { 1.0 };
        self.polarity * self.intensity * sign
    }
}

/// Opinion expressions (`turbo:opinions`), each over a non-empty token
/// range with its prior polarity, intensity, and negation/scope flag.
#[derive(Clone, Debug)]
pub struct Opinions {
    pub tokens: Vec<IxRange<Tokens>>,
    pub polarity: Vec<f64>,
    pub intensity: Vec<f64>,
    pub negated: Vec<bool>,
    pub tokens_key: Key<Tokens>,
}

impl Default for Opinions {
    fn default() -> Self {
        Self::new()
    }
}

impl Opinions {
    /// An empty `Opinions` layer indexing the default [`Tokens`] instance.
    pub fn new() -> Opinions {
        Self::on(Key::DEFAULT)
    }

    /// An empty `Opinions` layer indexing the named `tokens_key` instance.
    pub fn on(tokens_key: Key<Tokens>) -> Opinions {
        Opinions { tokens: Vec::new(), polarity: Vec::new(), intensity: Vec::new(), negated: Vec::new(), tokens_key }
    }

    /// Builds an `Opinions` layer from `items`, bounds-checking every
    /// token range against `tokens.len()` and checking all layer invariants.
    pub fn from_items(tokens: &Tokens, items: impl IntoIterator<Item = Opinion>) -> Result<Opinions, String> {
        Self::from_items_on(Key::DEFAULT, tokens, items)
    }

    /// Builds an `Opinions` layer indexing `tokens_key`, bounds-checking
    /// every token range against `tokens.len()` and checking all layer
    /// invariants.
    pub fn from_items_on(
        tokens_key: Key<Tokens>,
        tokens: &Tokens,
        items: impl IntoIterator<Item = Opinion>,
    ) -> Result<Opinions, String> {
        let mut out = Self::on(tokens_key);
        for item in items {
            out.try_push(tokens, item)?;
        }
        Ok(out)
    }

    /// Bounds-checks `opinion` against `tokens` and appends it in token
    /// order.
    pub fn try_push(&mut self, tokens: &Tokens, opinion: Opinion) -> Result<Ix<Opinions>, String> {
        let i = self.tokens.len();
        check_token_range("opinion", i, opinion.tokens, tokens.len())?;
        if let Some(&prev) = self.tokens.last()
            && (opinion.tokens.start, opinion.tokens.end) < (prev.start, prev.end)
        {
            return Err(format!(
                "opinion {i} token range {}..{} is not in token order after {}..{}",
                opinion.tokens.start.get(),
                opinion.tokens.end.get(),
                prev.start.get(),
                prev.end.get()
            ));
        }
        if !opinion.polarity.is_finite() {
            return Err(format!("opinion {i} has a non-finite polarity score {}", opinion.polarity));
        }
        if !opinion.intensity.is_finite() || opinion.intensity < 0.0 {
            return Err(format!(
                "opinion {i} has an invalid intensity {} (expected finite and >= 0)",
                opinion.intensity
            ));
        }
        Ok(self.push(opinion))
    }

    /// Appends `opinion` without checking bounds against a `Tokens` layer;
    /// `Layer::check` and `Document::add` still check invariants and bounds
    /// when the layer is added to a document.
    pub fn push(&mut self, opinion: Opinion) -> Ix<Opinions> {
        let ix = Ix::new(self.tokens.len() as u32);
        self.tokens.push(opinion.tokens);
        self.polarity.push(opinion.polarity);
        self.intensity.push(opinion.intensity);
        self.negated.push(opinion.negated);
        ix
    }

    /// Opinion `i`, column by column.
    pub fn get(&self, i: Ix<Opinions>) -> Opinion {
        let k = i.get();
        Opinion {
            tokens: self.tokens[k],
            polarity: self.polarity[k],
            intensity: self.intensity[k],
            negated: self.negated[k],
        }
    }

    /// The byte span of opinion `i` in `tokens`'s view, derived from the
    /// first and last token of its range without copying offsets.
    pub fn span(&self, i: Ix<Opinions>, tokens: &Tokens) -> Span {
        token_range_span(self.tokens[i.get()], tokens)
    }

    /// All opinions in order, with their index.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Ix<Opinions>, Opinion)> + '_ {
        (0..self.tokens.len() as u32).map(|i| {
            let ix = Ix::new(i);
            (ix, self.get(ix))
        })
    }
}

impl Layer for Opinions {
    const ID: &'static str = "turbo:opinions";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let max_tok = self.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0);
        vec![(self.tokens_key.id(), max_tok)]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.tokens.len();
        if self.polarity.len() != n || self.intensity.len() != n || self.negated.len() != n {
            return Err("tokens, a polarity, an intensity and a negated flag for every opinion".into());
        }
        for (i, r) in self.tokens.iter().enumerate() {
            if r.is_empty() {
                return Err(format!("opinion {i} has an empty token range {}..{}", r.start.get(), r.end.get()));
            }
            if i > 0 {
                let prev = self.tokens[i - 1];
                if (r.start, r.end) < (prev.start, prev.end) {
                    return Err(format!(
                        "opinion {i} token range {}..{} is not in token order after {}..{}",
                        r.start.get(),
                        r.end.get(),
                        prev.start.get(),
                        prev.end.get()
                    ));
                }
            }
            if !self.polarity[i].is_finite() {
                return Err(format!("opinion {i} has a non-finite polarity score"));
            }
            if !self.intensity[i].is_finite() || self.intensity[i] < 0.0 {
                return Err(format!("opinion {i} has an invalid intensity (expected finite and >= 0)"));
            }
        }
        Ok(())
    }
}

pub(super) fn check_token_range(
    what: &str,
    index: usize,
    range: IxRange<Tokens>,
    token_count: usize,
) -> Result<(), String> {
    if range.is_empty() {
        return Err(format!("{what} {index} has an empty token range {}..{}", range.start.get(), range.end.get()));
    }
    if range.end.get() > token_count {
        return Err(format!(
            "{what} {index} token range {}..{} is out of bounds for {token_count} tokens",
            range.start.get(),
            range.end.get()
        ));
    }
    Ok(())
}

pub(super) fn token_range_span(range: IxRange<Tokens>, tokens: &Tokens) -> Span {
    let first = tokens.spans.get(range.start);
    let last = tokens.spans.get(Ix::new(range.end.get() as u32 - 1));
    Span::new(first.start, last.end)
}

/// The six multi-label dimensions of [`Incivility`] (`turbo:incivility`),
/// following the Oasis (ICDMW 2019) and HKBU (ICA 2026) label set.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum IncivilityLabel {
    Insult = 0,
    Profanity = 1,
    Threat = 2,
    IdentityAttack = 3,
    Intolerance = 4,
    SevereToxicity = 5,
}

impl IncivilityLabel {
    pub const ALL: [IncivilityLabel; 6] = [
        IncivilityLabel::Insult,
        IncivilityLabel::Profanity,
        IncivilityLabel::Threat,
        IncivilityLabel::IdentityAttack,
        IncivilityLabel::Intolerance,
        IncivilityLabel::SevereToxicity,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            IncivilityLabel::Insult => "insult",
            IncivilityLabel::Profanity => "profanity",
            IncivilityLabel::Threat => "threat",
            IncivilityLabel::IdentityAttack => "identity_attack",
            IncivilityLabel::Intolerance => "intolerance",
            IncivilityLabel::SevereToxicity => "severe_toxicity",
        }
    }

    pub fn from_name(name: &str) -> Option<IncivilityLabel> {
        match name {
            "insult" => Some(IncivilityLabel::Insult),
            "profanity" => Some(IncivilityLabel::Profanity),
            "threat" => Some(IncivilityLabel::Threat),
            "identity_attack" => Some(IncivilityLabel::IdentityAttack),
            "intolerance" => Some(IncivilityLabel::Intolerance),
            "severe_toxicity" => Some(IncivilityLabel::SevereToxicity),
            _ => None,
        }
    }
}

impl fmt::Display for IncivilityLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The six multi-label scores in `f64` (`0.0..=1.0`) for one item of
/// [`Incivility`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct IncivilityScores {
    pub insult: f64,
    pub profanity: f64,
    pub threat: f64,
    pub identity_attack: f64,
    pub intolerance: f64,
    pub severe_toxicity: f64,
}

impl IncivilityScores {
    pub fn get(self, label: IncivilityLabel) -> f64 {
        match label {
            IncivilityLabel::Insult => self.insult,
            IncivilityLabel::Profanity => self.profanity,
            IncivilityLabel::Threat => self.threat,
            IncivilityLabel::IdentityAttack => self.identity_attack,
            IncivilityLabel::Intolerance => self.intolerance,
            IncivilityLabel::SevereToxicity => self.severe_toxicity,
        }
    }

    pub fn set(&mut self, label: IncivilityLabel, score: f64) {
        match label {
            IncivilityLabel::Insult => self.insult = score,
            IncivilityLabel::Profanity => self.profanity = score,
            IncivilityLabel::Threat => self.threat = score,
            IncivilityLabel::IdentityAttack => self.identity_attack = score,
            IncivilityLabel::Intolerance => self.intolerance = score,
            IncivilityLabel::SevereToxicity => self.severe_toxicity = score,
        }
    }

    pub fn iter(self) -> [(IncivilityLabel, f64); 6] {
        IncivilityLabel::ALL.map(|l| (l, self.get(l)))
    }

    pub fn is_valid(self) -> bool {
        self.iter().into_iter().all(|(_, s)| s.is_finite() && (0.0..=1.0).contains(&s))
    }
}
