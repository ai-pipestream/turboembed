//! Sentiment layers for aspect-based sentiment, opinion expressions,
//! multi-label incivility, and sentence pragmatics (sarcasm and irony).
//!
//! Every span is an [`Ix`] or [`IxRange`] into an existing layer
//! ([`Tokens`], [`Sentences`], [`Chunks`], or [`Structure`]), so byte
//! offsets follow the document's view alignment without copying spans.
//! All scores are `f64` so `1e-9` parity checks hold.

use std::fmt;
use std::sync::Arc;

use turbo_analysis::{Anchor, Ix, IxRange, Key, KeyId, Layer, Span, Tag, TagSet};

use crate::{Chunks, Sentences, Structure, Tokens};

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

/// Three-way polarity scores in `f64` for an aspect in [`Aspects`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PolarityScores {
    pub positive: f64,
    pub negative: f64,
    pub neutral: f64,
}

impl PolarityScores {
    pub const fn new(positive: f64, negative: f64, neutral: f64) -> PolarityScores {
        PolarityScores { positive, negative, neutral }
    }

    /// Whether all three scores are finite and non-negative.
    pub fn is_valid(self) -> bool {
        self.positive.is_finite()
            && self.negative.is_finite()
            && self.neutral.is_finite()
            && self.positive >= 0.0
            && self.negative >= 0.0
            && self.neutral >= 0.0
    }

    /// The dominant polarity class (ties broken Positive > Negative > Neutral).
    pub fn label(self) -> Polarity {
        if self.positive >= self.negative && self.positive >= self.neutral {
            Polarity::Positive
        } else if self.negative >= self.neutral {
            Polarity::Negative
        } else {
            Polarity::Neutral
        }
    }

    /// Net signed score (`positive - negative`).
    pub fn compound(self) -> f64 {
        self.positive - self.negative
    }
}

/// What an aspect in [`Aspects`] anchors to: an explicit aspect term over
/// a non-empty token range (with its optional enclosing sentence), or an
/// implicit aspect (no surface term in the text) anchored to its sentence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AspectTarget {
    /// An explicit aspect term covering `tokens` (`start < end`), optionally
    /// recording the `sentence` it belongs to.
    Term { tokens: IxRange<Tokens>, sentence: Option<Ix<Sentences>> },
    /// An implicit aspect (no term in the text) anchored to `sentence`.
    Implicit { sentence: Ix<Sentences> },
}

impl AspectTarget {
    pub fn tokens(self) -> Option<IxRange<Tokens>> {
        match self {
            AspectTarget::Term { tokens, .. } => Some(tokens),
            AspectTarget::Implicit { .. } => None,
        }
    }

    pub fn sentence(self) -> Option<Ix<Sentences>> {
        match self {
            AspectTarget::Term { sentence, .. } => sentence,
            AspectTarget::Implicit { sentence } => Some(sentence),
        }
    }

    pub fn is_implicit(self) -> bool {
        matches!(self, AspectTarget::Implicit { .. })
    }
}

/// One aspect entry in [`Aspects`] (`turbo:aspects`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aspect<'a> {
    pub target: AspectTarget,
    pub category: Option<Tag>,
    pub polarity: PolarityScores,
    pub opinions: &'a [Ix<Opinions>],
}

/// Aspect-based sentiment instances (`turbo:aspects`): aspect term token
/// spans (or sentence anchors for implicit aspects), optional aspect
/// category from a [`TagSet`], polarity scores in `f64`, and [`Ix`] links
/// to the [`Opinions`] instances that back each aspect.
#[derive(Clone, Debug)]
pub struct Aspects {
    /// `Some(range)` for an explicit aspect term, `None` for an implicit
    /// aspect anchored to a sentence.
    pub tokens: Vec<Option<IxRange<Tokens>>>,
    /// The sentence the aspect belongs to; always `Some` when `tokens[i]`
    /// is `None` (an implicit aspect).
    pub sentence: Vec<Option<Ix<Sentences>>>,
    /// Optional aspect category per aspect, indexed into `categories`.
    pub category: Vec<Option<Tag>>,
    pub categories: Arc<TagSet>,
    pub positive: Vec<f64>,
    pub negative: Vec<f64>,
    pub neutral: Vec<f64>,
    /// Flat CSR buffer of backing [`Opinions`] indices, kept in strictly
    /// increasing order within each aspect item.
    pub opinions: Vec<Ix<Opinions>>,
    /// End offset in `opinions` for each aspect item `0..len`.
    pub opinion_end: Vec<u32>,
    pub tokens_key: Key<Tokens>,
    pub sentences_key: Key<Sentences>,
    pub opinions_key: Key<Opinions>,
}

impl Default for Aspects {
    fn default() -> Self {
        Self::new(Arc::new(TagSet::default()))
    }
}

impl Aspects {
    /// An empty `Aspects` layer with `categories`, indexing the default
    /// [`Tokens`], [`Sentences`], and [`Opinions`] instances.
    pub fn new(categories: Arc<TagSet>) -> Aspects {
        Self::on(Key::DEFAULT, Key::DEFAULT, Key::DEFAULT, categories)
    }

    /// An empty `Aspects` layer indexing the named `tokens_key`,
    /// `sentences_key`, and `opinions_key` instances.
    pub fn on(
        tokens_key: Key<Tokens>,
        sentences_key: Key<Sentences>,
        opinions_key: Key<Opinions>,
        categories: Arc<TagSet>,
    ) -> Aspects {
        Aspects {
            tokens: Vec::new(),
            sentence: Vec::new(),
            category: Vec::new(),
            categories,
            positive: Vec::new(),
            negative: Vec::new(),
            neutral: Vec::new(),
            opinions: Vec::new(),
            opinion_end: Vec::new(),
            tokens_key,
            sentences_key,
            opinions_key,
        }
    }

    /// Builds an `Aspects` layer from `items`, bounds-checking every
    /// `IxRange<Tokens>`, `Ix<Sentences>`, `Ix<Opinions>`, and category
    /// `Tag`.
    pub fn from_items<'a>(
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        opinions: Option<&Opinions>,
        categories: Arc<TagSet>,
        items: impl IntoIterator<Item = Aspect<'a>>,
    ) -> Result<Aspects, String> {
        Self::from_items_on(Key::DEFAULT, Key::DEFAULT, Key::DEFAULT, tokens, sentences, opinions, categories, items)
    }

    /// Builds an `Aspects` layer indexing explicit keys and bounds-checking
    /// every index.
    #[allow(clippy::too_many_arguments)]
    pub fn from_items_on<'a>(
        tokens_key: Key<Tokens>,
        sentences_key: Key<Sentences>,
        opinions_key: Key<Opinions>,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        opinions: Option<&Opinions>,
        categories: Arc<TagSet>,
        items: impl IntoIterator<Item = Aspect<'a>>,
    ) -> Result<Aspects, String> {
        let mut out = Self::on(tokens_key, sentences_key, opinions_key, categories);
        for item in items {
            out.try_push(tokens, sentences, opinions, item)?;
        }
        Ok(out)
    }

    /// Bounds-checks `aspect` against `tokens`, `sentences`, `opinions`,
    /// and `self.categories`, then appends it.
    pub fn try_push(
        &mut self,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        opinions: Option<&Opinions>,
        aspect: Aspect<'_>,
    ) -> Result<Ix<Aspects>, String> {
        let i = self.tokens.len();
        if let Some(r) = aspect.target.tokens() {
            check_token_range("aspect", i, r, tokens.len())?;
        }
        if let Some(s) = aspect.target.sentence() {
            let Some(sents) = sentences else {
                return Err(format!("aspect {i} indexes sentence {}, but no Sentences layer was given", s.get()));
            };
            if s.get() >= sents.len() {
                return Err(format!(
                    "aspect {i} sentence index {} is out of bounds for {} sentences",
                    s.get(),
                    sents.len()
                ));
            }
            if let (Some(r), Some(tok_sents)) = (aspect.target.tokens(), &tokens.sentence) {
                let first_sent = tok_sents[r.start.get()];
                let last_sent = tok_sents[r.end.get() - 1];
                if first_sent != s || last_sent != s {
                    return Err(format!(
                        "aspect {i} token range {}..{} is in sentences {}..={}, not sentence {}",
                        r.start.get(),
                        r.end.get(),
                        first_sent.get(),
                        last_sent.get(),
                        s.get()
                    ));
                }
            }
        }
        if let Some(cat) = aspect.category
            && cat.0 as usize >= self.categories.len()
        {
            return Err(format!("aspect {i} has a category outside {}", self.categories.name));
        }
        if !aspect.polarity.is_valid() {
            return Err(format!("aspect {i} has non-finite or negative polarity scores"));
        }
        if !aspect.opinions.is_empty() {
            let Some(ops) = opinions else {
                return Err(format!("aspect {i} links to opinions, but no Opinions layer was given"));
            };
            for &op in aspect.opinions {
                if op.get() >= ops.len() {
                    return Err(format!(
                        "aspect {i} opinion index {} is out of bounds for {} opinions",
                        op.get(),
                        ops.len()
                    ));
                }
            }
        }
        Ok(self.push(aspect))
    }

    /// Appends `aspect`, sorting and deduplicating its backing opinion
    /// links.
    pub fn push(&mut self, aspect: Aspect<'_>) -> Ix<Aspects> {
        let ix = Ix::new(self.tokens.len() as u32);
        self.tokens.push(aspect.target.tokens());
        self.sentence.push(aspect.target.sentence());
        self.category.push(aspect.category);
        self.positive.push(aspect.polarity.positive);
        self.negative.push(aspect.polarity.negative);
        self.neutral.push(aspect.polarity.neutral);
        let start = self.opinions.len();
        self.opinions.extend_from_slice(aspect.opinions);
        self.opinions[start..].sort_unstable();
        let dedup_len = {
            let slice = &mut self.opinions[start..];
            let mut w = 0;
            for r in 0..slice.len() {
                if w == 0 || slice[r] != slice[w - 1] {
                    slice[w] = slice[r];
                    w += 1;
                }
            }
            w
        };
        self.opinions.truncate(start + dedup_len);
        self.opinion_end.push(self.opinions.len() as u32);
        ix
    }

    /// The target of aspect `i`.
    pub fn target(&self, i: Ix<Aspects>) -> AspectTarget {
        let k = i.get();
        match (self.tokens[k], self.sentence[k]) {
            (Some(tokens), sentence) => AspectTarget::Term { tokens, sentence },
            (None, Some(sentence)) => AspectTarget::Implicit { sentence },
            (None, None) => panic!("aspect {k} has neither a token range nor a sentence"),
        }
    }

    /// The category label of aspect `i`, if it has one.
    pub fn category_label(&self, i: Ix<Aspects>) -> Option<&str> {
        self.category[i.get()].map(|t| self.categories.label(t))
    }

    /// The polarity scores of aspect `i`.
    pub fn polarity(&self, i: Ix<Aspects>) -> PolarityScores {
        let k = i.get();
        PolarityScores { positive: self.positive[k], negative: self.negative[k], neutral: self.neutral[k] }
    }

    /// The backing [`Opinions`] indices of aspect `i`, in ascending order.
    pub fn opinions_of(&self, i: Ix<Aspects>) -> &[Ix<Opinions>] {
        let k = i.get();
        let start = if k == 0 { 0 } else { self.opinion_end[k - 1] as usize };
        let end = self.opinion_end[k] as usize;
        &self.opinions[start..end]
    }

    /// Aspect `i`, column by column.
    pub fn get(&self, i: Ix<Aspects>) -> Aspect<'_> {
        Aspect {
            target: self.target(i),
            category: self.category[i.get()],
            polarity: self.polarity(i),
            opinions: self.opinions_of(i),
        }
    }

    /// The byte span of aspect `i`: its token range's span in `tokens` for
    /// an explicit aspect, or its sentence's span in `sentences` for an
    /// implicit aspect.
    pub fn span(&self, i: Ix<Aspects>, tokens: &Tokens, sentences: Option<&Sentences>) -> Option<Span> {
        match self.target(i) {
            AspectTarget::Term { tokens: r, .. } => Some(token_range_span(r, tokens)),
            AspectTarget::Implicit { sentence } => sentences.map(|s| s.spans.get(sentence)),
        }
    }

    /// All aspects in order, with their index.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Ix<Aspects>, Aspect<'_>)> + '_ {
        (0..self.tokens.len() as u32).map(|i| {
            let ix = Ix::new(i);
            (ix, self.get(ix))
        })
    }
}

impl Layer for Aspects {
    const ID: &'static str = "turbo:aspects";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let max_tok = self.tokens.iter().flatten().map(|r| r.end.get()).max().unwrap_or(0);
        let mut out = vec![(self.tokens_key.id(), max_tok)];
        let max_sent = self.sentence.iter().flatten().map(|s| s.get() + 1).max().unwrap_or(0);
        if max_sent > 0 || !self.sentences_key.name.is_empty() {
            out.push((self.sentences_key.id(), max_sent));
        }
        let max_op = self.opinions.iter().map(|o| o.get() + 1).max().unwrap_or(0);
        if max_op > 0 || !self.opinions_key.name.is_empty() {
            out.push((self.opinions_key.id(), max_op));
        }
        out
    }

    fn check(&self) -> Result<(), String> {
        let n = self.tokens.len();
        if self.sentence.len() != n
            || self.category.len() != n
            || self.positive.len() != n
            || self.negative.len() != n
            || self.neutral.len() != n
            || self.opinion_end.len() != n
        {
            return Err("every aspect column must have the same length".into());
        }
        let mut prev_end = 0usize;
        for i in 0..n {
            match (self.tokens[i], self.sentence[i]) {
                (Some(r), _) if r.is_empty() => {
                    return Err(format!("aspect {i} has an empty token range {}..{}", r.start.get(), r.end.get()));
                }
                (None, None) => {
                    return Err(format!("implicit aspect {i} must anchor to a sentence"));
                }
                _ => {}
            }
            if let Some(cat) = self.category[i]
                && cat.0 as usize >= self.categories.len()
            {
                return Err(format!("a category outside {}", self.categories.name));
            }
            let scores =
                PolarityScores { positive: self.positive[i], negative: self.negative[i], neutral: self.neutral[i] };
            if !scores.is_valid() {
                return Err(format!("aspect {i} has non-finite or negative polarity scores"));
            }
            let end = self.opinion_end[i] as usize;
            if end < prev_end || end > self.opinions.len() {
                return Err(format!("aspect {i} opinion_end {end} is out of bounds or decreases"));
            }
            if !self.opinions[prev_end..end].windows(2).all(|w| w[0] < w[1]) {
                return Err(format!("aspect {i} backing opinions must be in strictly increasing order"));
            }
            prev_end = end;
        }
        if prev_end != self.opinions.len() {
            return Err("opinion_end does not reach the end of opinions".into());
        }
        Ok(())
    }
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

/// Which unit span an [`Incivility`] item scores: a sentence, a retrieval
/// chunk, or a structure block (such as a post, quote, or paragraph).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IncivilityTarget {
    Sentence(Ix<Sentences>),
    Chunk(Ix<Chunks>),
    Block(Ix<Structure>),
}

impl IncivilityTarget {
    pub const fn kind_name(self) -> &'static str {
        match self {
            IncivilityTarget::Sentence(_) => "sentence",
            IncivilityTarget::Chunk(_) => "chunk",
            IncivilityTarget::Block(_) => "block",
        }
    }

    pub fn raw_index(self) -> u32 {
        match self {
            IncivilityTarget::Sentence(i) => i.get() as u32,
            IncivilityTarget::Chunk(i) => i.get() as u32,
            IncivilityTarget::Block(i) => i.get() as u32,
        }
    }

    pub fn from_kind_and_index(kind: &str, index: u32) -> Option<IncivilityTarget> {
        match kind {
            "sentence" => Some(IncivilityTarget::Sentence(Ix::new(index))),
            "chunk" => Some(IncivilityTarget::Chunk(Ix::new(index))),
            "block" => Some(IncivilityTarget::Block(Ix::new(index))),
            _ => None,
        }
    }
}

/// One item handed to [`Incivility::try_push`] or [`Incivility::push`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IncivilityItem<'a> {
    pub target: IncivilityTarget,
    pub scores: IncivilityScores,
    pub evidence: &'a [(IncivilityLabel, IxRange<Tokens>)],
}

/// Multi-label incivility and toxicity (`turbo:incivility`) anchored to
/// sentence, chunk, or structure (post) spans, with six `f64` label scores
/// and the evidence token spans per label.
#[derive(Clone, Debug)]
pub struct Incivility {
    pub target: Vec<IncivilityTarget>,
    pub insult: Vec<f64>,
    pub profanity: Vec<f64>,
    pub threat: Vec<f64>,
    pub identity_attack: Vec<f64>,
    pub intolerance: Vec<f64>,
    pub severe_toxicity: Vec<f64>,
    /// Flat CSR column of evidence labels, sorted by `(label, tokens.start,
    /// tokens.end)` within each item.
    pub evidence_label: Vec<IncivilityLabel>,
    /// Flat CSR column of evidence token spans, parallel to
    /// `evidence_label`.
    pub evidence_tokens: Vec<IxRange<Tokens>>,
    /// End offset in `evidence_label` / `evidence_tokens` for each item
    /// `0..len`.
    pub evidence_end: Vec<u32>,
    pub tokens_key: Key<Tokens>,
    pub sentences_key: Key<Sentences>,
    pub chunks_key: Key<Chunks>,
    pub structure_key: Key<Structure>,
}

impl Default for Incivility {
    fn default() -> Self {
        Self::new()
    }
}

impl Incivility {
    /// An empty `Incivility` layer indexing the default [`Tokens`],
    /// [`Sentences`], [`Chunks`], and [`Structure`] instances.
    pub fn new() -> Incivility {
        Self::on(Key::DEFAULT, Key::DEFAULT, Key::DEFAULT, Key::DEFAULT)
    }

    /// An empty `Incivility` layer indexing the given named instances.
    pub fn on(
        tokens_key: Key<Tokens>,
        sentences_key: Key<Sentences>,
        chunks_key: Key<Chunks>,
        structure_key: Key<Structure>,
    ) -> Incivility {
        Incivility {
            target: Vec::new(),
            insult: Vec::new(),
            profanity: Vec::new(),
            threat: Vec::new(),
            identity_attack: Vec::new(),
            intolerance: Vec::new(),
            severe_toxicity: Vec::new(),
            evidence_label: Vec::new(),
            evidence_tokens: Vec::new(),
            evidence_end: Vec::new(),
            tokens_key,
            sentences_key,
            chunks_key,
            structure_key,
        }
    }

    /// Builds an `Incivility` layer from `items`, bounds-checking every
    /// target span index and every evidence token range.
    pub fn from_items<'a>(
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        structure: Option<&Structure>,
        items: impl IntoIterator<Item = IncivilityItem<'a>>,
    ) -> Result<Incivility, String> {
        Self::from_items_on(
            Key::DEFAULT,
            Key::DEFAULT,
            Key::DEFAULT,
            Key::DEFAULT,
            tokens,
            sentences,
            chunks,
            structure,
            items,
        )
    }

    /// Builds an `Incivility` layer indexing explicit keys and
    /// bounds-checking every index.
    #[allow(clippy::too_many_arguments)]
    pub fn from_items_on<'a>(
        tokens_key: Key<Tokens>,
        sentences_key: Key<Sentences>,
        chunks_key: Key<Chunks>,
        structure_key: Key<Structure>,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        structure: Option<&Structure>,
        items: impl IntoIterator<Item = IncivilityItem<'a>>,
    ) -> Result<Incivility, String> {
        let mut out = Self::on(tokens_key, sentences_key, chunks_key, structure_key);
        for item in items {
            out.try_push(tokens, sentences, chunks, structure, item)?;
        }
        Ok(out)
    }

    /// Bounds-checks `item` against the target layers and `tokens`, then
    /// appends it.
    pub fn try_push(
        &mut self,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        structure: Option<&Structure>,
        item: IncivilityItem<'_>,
    ) -> Result<Ix<Incivility>, String> {
        let i = self.target.len();
        if let Some(&prev) = self.target.last()
            && item.target <= prev
        {
            return Err(format!("incivility item {i} target is not in strictly increasing target order"));
        }
        match item.target {
            IncivilityTarget::Sentence(s) => {
                let Some(sents) = sentences else {
                    return Err(format!(
                        "incivility item {i} targets sentence {}, but no Sentences layer was given",
                        s.get()
                    ));
                };
                if s.get() >= sents.len() {
                    return Err(format!(
                        "incivility item {i} sentence index {} is out of bounds for {} sentences",
                        s.get(),
                        sents.len()
                    ));
                }
            }
            IncivilityTarget::Chunk(c) => {
                let Some(ch) = chunks else {
                    return Err(format!(
                        "incivility item {i} targets chunk {}, but no Chunks layer was given",
                        c.get()
                    ));
                };
                if c.get() >= ch.len() {
                    return Err(format!(
                        "incivility item {i} chunk index {} is out of bounds for {} chunks",
                        c.get(),
                        ch.len()
                    ));
                }
            }
            IncivilityTarget::Block(b) => {
                let Some(st) = structure else {
                    return Err(format!(
                        "incivility item {i} targets block {}, but no Structure layer was given",
                        b.get()
                    ));
                };
                if b.get() >= st.len() {
                    return Err(format!(
                        "incivility item {i} block index {} is out of bounds for {} blocks",
                        b.get(),
                        st.len()
                    ));
                }
            }
        }
        if !item.scores.is_valid() {
            return Err(format!("incivility item {i} has a score outside 0..=1 or NaN"));
        }
        for &(_, r) in item.evidence {
            check_token_range("incivility evidence", i, r, tokens.len())?;
        }
        Ok(self.push(item))
    }

    /// Appends `item`, sorting and deduplicating its evidence pairs by
    /// `(label, tokens.start, tokens.end)`.
    pub fn push(&mut self, item: IncivilityItem<'_>) -> Ix<Incivility> {
        let ix = Ix::new(self.target.len() as u32);
        self.target.push(item.target);
        self.insult.push(item.scores.insult);
        self.profanity.push(item.scores.profanity);
        self.threat.push(item.scores.threat);
        self.identity_attack.push(item.scores.identity_attack);
        self.intolerance.push(item.scores.intolerance);
        self.severe_toxicity.push(item.scores.severe_toxicity);

        let mut sorted: Vec<(IncivilityLabel, IxRange<Tokens>)> = item.evidence.to_vec();
        sorted.sort_unstable_by_key(|&(l, r)| (l, r.start, r.end));
        sorted.dedup();
        for (l, r) in sorted {
            self.evidence_label.push(l);
            self.evidence_tokens.push(r);
        }
        self.evidence_end.push(self.evidence_label.len() as u32);
        ix
    }

    /// The six multi-label scores of item `i`.
    pub fn scores(&self, i: Ix<Incivility>) -> IncivilityScores {
        let k = i.get();
        IncivilityScores {
            insult: self.insult[k],
            profanity: self.profanity[k],
            threat: self.threat[k],
            identity_attack: self.identity_attack[k],
            intolerance: self.intolerance[k],
            severe_toxicity: self.severe_toxicity[k],
        }
    }

    fn evidence_range(&self, i: Ix<Incivility>) -> std::ops::Range<usize> {
        let k = i.get();
        let start = if k == 0 { 0 } else { self.evidence_end[k - 1] as usize };
        let end = self.evidence_end[k] as usize;
        start..end
    }

    /// All `(label, tokens)` evidence entries for item `i`, in canonical
    /// `(label, start, end)` order.
    pub fn evidence_of(
        &self,
        i: Ix<Incivility>,
    ) -> impl ExactSizeIterator<Item = (IncivilityLabel, IxRange<Tokens>)> + '_ {
        let range = self.evidence_range(i);
        self.evidence_label[range.clone()].iter().copied().zip(self.evidence_tokens[range].iter().copied())
    }

    /// The contiguous slice of evidence token spans for `label` on item
    /// `i`, found by binary search with zero allocation.
    pub fn evidence_for_label(&self, i: Ix<Incivility>, label: IncivilityLabel) -> &[IxRange<Tokens>] {
        let range = self.evidence_range(i);
        let labels = &self.evidence_label[range.clone()];
        let tokens = &self.evidence_tokens[range];
        let a = labels.partition_point(|&l| l < label);
        let b = a + labels[a..].partition_point(|&l| l == label);
        &tokens[a..b]
    }

    /// The byte span of item `i`'s target in the corresponding layer.
    pub fn target_span(
        &self,
        i: Ix<Incivility>,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        structure: Option<&Structure>,
    ) -> Option<Span> {
        match self.target[i.get()] {
            IncivilityTarget::Sentence(s) => sentences.map(|l| l.spans.get(s)),
            IncivilityTarget::Chunk(c) => chunks.map(|l| l.spans.get(c)),
            IncivilityTarget::Block(b) => structure.map(|l| l.spans.get(b)),
        }
    }
}

impl Layer for Incivility {
    const ID: &'static str = "turbo:incivility";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.target.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let max_tok = self.evidence_tokens.iter().map(|r| r.end.get()).max().unwrap_or(0);
        let mut out = vec![(self.tokens_key.id(), max_tok)];
        let mut max_sent = 0usize;
        let mut max_chunk = 0usize;
        let mut max_block = 0usize;
        for &t in &self.target {
            match t {
                IncivilityTarget::Sentence(s) => max_sent = max_sent.max(s.get() + 1),
                IncivilityTarget::Chunk(c) => max_chunk = max_chunk.max(c.get() + 1),
                IncivilityTarget::Block(b) => max_block = max_block.max(b.get() + 1),
            }
        }
        if max_sent > 0 || !self.sentences_key.name.is_empty() {
            out.push((self.sentences_key.id(), max_sent));
        }
        if max_chunk > 0 || !self.chunks_key.name.is_empty() {
            out.push((self.chunks_key.id(), max_chunk));
        }
        if max_block > 0 || !self.structure_key.name.is_empty() {
            out.push((self.structure_key.id(), max_block));
        }
        out
    }

    fn check(&self) -> Result<(), String> {
        let n = self.target.len();
        let cols = [
            self.insult.len(),
            self.profanity.len(),
            self.threat.len(),
            self.identity_attack.len(),
            self.intolerance.len(),
            self.severe_toxicity.len(),
            self.evidence_end.len(),
        ];
        if cols.iter().any(|&c| c != n) {
            return Err("every incivility column must have the same length".into());
        }
        if self.evidence_label.len() != self.evidence_tokens.len() {
            return Err("an evidence token range for every evidence label".into());
        }
        if !self.target.windows(2).all(|w| w[0] < w[1]) {
            return Err("incivility targets must be in strictly increasing order".into());
        }
        let mut prev_end = 0usize;
        for i in 0..n {
            let scores = self.scores(Ix::new(i as u32));
            if !scores.is_valid() {
                return Err(format!("incivility item {i} has a score outside 0..=1 or NaN"));
            }
            let end = self.evidence_end[i] as usize;
            if end < prev_end || end > self.evidence_label.len() {
                return Err(format!("incivility item {i} evidence_end {end} is out of bounds or decreases"));
            }
            for k in prev_end..end {
                let r = self.evidence_tokens[k];
                if r.is_empty() {
                    return Err(format!(
                        "incivility item {i} has an empty evidence token range {}..{}",
                        r.start.get(),
                        r.end.get()
                    ));
                }
                if k > prev_end {
                    let prev_key = (
                        self.evidence_label[k - 1],
                        self.evidence_tokens[k - 1].start,
                        self.evidence_tokens[k - 1].end,
                    );
                    let cur_key = (self.evidence_label[k], r.start, r.end);
                    if cur_key <= prev_key {
                        return Err(format!(
                            "incivility item {i} evidence must be in strictly increasing (label, start, end) order"
                        ));
                    }
                }
            }
            prev_end = end;
        }
        if prev_end != self.evidence_label.len() {
            return Err("evidence_end does not reach the end of evidence_label".into());
        }
        Ok(())
    }
}

/// The two contrasted token spans that serve as incongruity evidence for
/// sarcasm or irony in [`Pragmatics`]: for example a positive opinion
/// expression (`positive`) contrasted against a negative situation
/// (`negative`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Incongruity {
    pub positive: IxRange<Tokens>,
    pub negative: IxRange<Tokens>,
}

const _: () = assert!(std::mem::size_of::<Option<Incongruity>>() == 16);

/// One sentence's sarcasm/irony entry in [`Pragmatics`] (`turbo:pragmatics`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PragmaticsItem {
    /// Sarcasm/irony score in `f64` (`0.0..=1.0`).
    pub score: f64,
    /// Optional contrasted token spans when explicit incongruity evidence
    /// is present (`None` when scored by an embedding head alone).
    pub evidence: Option<Incongruity>,
}

/// Sentence-level sarcasm and irony (`turbo:pragmatics`), aligned to
/// [`Sentences`] and indexing [`Tokens`] for its contrasted incongruity
/// evidence spans.
#[derive(Clone, Debug)]
pub struct Pragmatics {
    pub score: Vec<f64>,
    pub evidence: Vec<Option<Incongruity>>,
    pub tokens_key: Key<Tokens>,
}

impl Default for Pragmatics {
    fn default() -> Self {
        Self::new()
    }
}

impl Pragmatics {
    /// An empty `Pragmatics` layer indexing the default [`Tokens`] instance.
    pub fn new() -> Pragmatics {
        Self::on(Key::DEFAULT)
    }

    /// An empty `Pragmatics` layer indexing the named `tokens_key` instance.
    pub fn on(tokens_key: Key<Tokens>) -> Pragmatics {
        Pragmatics { score: Vec::new(), evidence: Vec::new(), tokens_key }
    }

    /// Builds a `Pragmatics` layer aligned to `sentences` and indexing
    /// `tokens`, bounds-checking the sentence count and every contrasted
    /// token range.
    pub fn from_items(
        sentences: &Sentences,
        tokens: &Tokens,
        items: impl IntoIterator<Item = PragmaticsItem>,
    ) -> Result<Pragmatics, String> {
        Self::from_items_on(Key::DEFAULT, sentences, tokens, items)
    }

    /// Builds a `Pragmatics` layer indexing `tokens_key`, bounds-checking
    /// the sentence count and every contrasted token range.
    pub fn from_items_on(
        tokens_key: Key<Tokens>,
        sentences: &Sentences,
        tokens: &Tokens,
        items: impl IntoIterator<Item = PragmaticsItem>,
    ) -> Result<Pragmatics, String> {
        let mut out = Self::on(tokens_key);
        for item in items {
            out.try_push(sentences, tokens, item)?;
        }
        if out.len() != sentences.len() {
            return Err(format!("pragmatics has {} items, aligned to sentences with {}", out.len(), sentences.len()));
        }
        Ok(out)
    }

    /// Bounds-checks `item` against `sentences` and `tokens`, then appends
    /// it.
    pub fn try_push(
        &mut self,
        sentences: &Sentences,
        tokens: &Tokens,
        item: PragmaticsItem,
    ) -> Result<Ix<Pragmatics>, String> {
        let i = self.score.len();
        if i >= sentences.len() {
            return Err(format!("pragmatics item {i} exceeds the {} sentences on the document", sentences.len()));
        }
        if !item.score.is_finite() || !(0.0..=1.0).contains(&item.score) {
            return Err(format!("pragmatics item {i} has a score {} outside 0..=1", item.score));
        }
        if let Some(inc) = item.evidence {
            check_token_range("pragmatics positive evidence", i, inc.positive, tokens.len())?;
            check_token_range("pragmatics negative evidence", i, inc.negative, tokens.len())?;
            if let Some(tok_sents) = &tokens.sentence {
                let s = Ix::<Sentences>::new(i as u32);
                for (which, r) in [("positive", inc.positive), ("negative", inc.negative)] {
                    if tok_sents[r.start.get()] != s || tok_sents[r.end.get() - 1] != s {
                        return Err(format!(
                            "pragmatics item {i} {which} evidence {}..{} is outside sentence {i}",
                            r.start.get(),
                            r.end.get()
                        ));
                    }
                }
            }
        }
        Ok(self.push(item))
    }

    /// Appends `item`.
    pub fn push(&mut self, item: PragmaticsItem) -> Ix<Pragmatics> {
        let ix = Ix::new(self.score.len() as u32);
        self.score.push(item.score);
        self.evidence.push(item.evidence);
        ix
    }

    /// Item `i` (for sentence `i`), column by column.
    pub fn get(&self, i: Ix<Pragmatics>) -> PragmaticsItem {
        let k = i.get();
        PragmaticsItem { score: self.score[k], evidence: self.evidence[k] }
    }

    /// The contrasted byte spans `(positive_span, negative_span)` for
    /// sentence `i` in `tokens`'s view, when `evidence[i]` is `Some`.
    pub fn evidence_spans(&self, i: Ix<Pragmatics>, tokens: &Tokens) -> Option<(Span, Span)> {
        self.evidence[i.get()]
            .map(|inc| (token_range_span(inc.positive, tokens), token_range_span(inc.negative, tokens)))
    }

    /// All sentence pragmatics items in order, with their index.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Ix<Pragmatics>, PragmaticsItem)> + '_ {
        (0..self.score.len() as u32).map(|i| {
            let ix = Ix::new(i);
            (ix, self.get(ix))
        })
    }
}

impl Layer for Pragmatics {
    const ID: &'static str = "turbo:pragmatics";

    fn anchor() -> Anchor {
        Anchor::aligned::<Sentences>()
    }

    fn len(&self) -> usize {
        self.score.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let max_tok = self
            .evidence
            .iter()
            .flatten()
            .map(|inc| inc.positive.end.get().max(inc.negative.end.get()))
            .max()
            .unwrap_or(0);
        vec![(self.tokens_key.id(), max_tok)]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.score.len();
        if self.evidence.len() != n {
            return Err("an evidence slot for every pragmatics score".into());
        }
        for i in 0..n {
            let s = self.score[i];
            if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                return Err(format!("pragmatics item {i} has a score outside 0..=1 or NaN"));
            }
            if let Some(inc) = self.evidence[i] {
                if inc.positive.is_empty() {
                    return Err(format!(
                        "pragmatics item {i} has an empty positive token range {}..{}",
                        inc.positive.start.get(),
                        inc.positive.end.get()
                    ));
                }
                if inc.negative.is_empty() {
                    return Err(format!(
                        "pragmatics item {i} has an empty negative token range {}..{}",
                        inc.negative.start.get(),
                        inc.negative.end.get()
                    ));
                }
            }
        }
        Ok(())
    }
}

fn check_token_range(what: &str, index: usize, range: IxRange<Tokens>, token_count: usize) -> Result<(), String> {
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

fn token_range_span(range: IxRange<Tokens>, tokens: &Tokens) -> Span {
    let first = tokens.spans.get(range.start);
    let last = tokens.spans.get(Ix::new(range.end.get() as u32 - 1));
    Span::new(first.start, last.end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Block, Chunks, Cut};
    use turbo_analysis::{Analyzer, Annotator, Bound, Document, Error, Provenance, Refs, Spans, ViewId};

    fn make_sentences_and_tokens() -> (Sentences, Tokens) {
        let mut s_spans = Spans::new(ViewId::SOURCE);
        s_spans.push(Span::new(0, 49));
        s_spans.push(Span::new(50, 65));
        let sentences = Sentences { spans: s_spans };

        let word_ranges: &[(u32, u32, u32)] = &[
            (0, 3, 0),   // 0: The
            (4, 8, 0),   // 1: food
            (9, 12, 0),  // 2: was
            (13, 16, 0), // 3: not
            (17, 22, 0), // 4: great
            (22, 23, 0), // 5: ,
            (24, 27, 0), // 6: but
            (28, 31, 0), // 7: the
            (32, 39, 0), // 8: service
            (40, 43, 0), // 9: was
            (44, 48, 0), // 10: fast
            (48, 49, 0), // 11: .
            (50, 55, 1), // 12: Price
            (56, 59, 1), // 13: was
            (60, 64, 1), // 14: fair
            (64, 65, 1), // 15: .
        ];
        let mut t_spans = Spans::new(ViewId::SOURCE);
        let mut t_sent = Vec::new();
        for &(a, b, s) in word_ranges {
            t_spans.push(Span::new(a, b));
            t_sent.push(Ix::new(s));
        }
        let tokens = Tokens { spans: t_spans, sentence: Some(t_sent) };
        (sentences, tokens)
    }

    fn sample_doc() -> (Document, Sentences, Tokens) {
        let text = "The food was not great, but the service was fast. Price was fair.";
        let mut doc = Document::new(text);
        let (sentences, tokens) = make_sentences_and_tokens();
        doc.add(&Key::DEFAULT, sentences, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
        let (sentences, tokens) = make_sentences_and_tokens();
        (doc, sentences, tokens)
    }

    #[test]
    fn opinions_and_aspects_construct_and_link_on_document() {
        let (mut doc, sentences, tokens) = sample_doc();
        let opinions = Opinions::from_items(
            &tokens,
            [
                Opinion { tokens: IxRange::new(4, 5), polarity: 0.8, intensity: 1.25, negated: true },
                Opinion { tokens: IxRange::new(10, 11), polarity: 0.7, intensity: 1.0, negated: false },
                Opinion { tokens: IxRange::new(14, 15), polarity: 0.5, intensity: 1.0, negated: false },
            ],
        )
        .unwrap();
        assert_eq!(opinions.len(), 3);
        assert_eq!(opinions.get(Ix::new(0)).effective_polarity(), -1.0);
        assert_eq!(opinions.span(Ix::new(0), &tokens), Span::new(17, 22));
        doc.add(&Key::DEFAULT, opinions.clone(), Provenance::by("test")).unwrap();

        let cats = TagSet::new("absa", &["food#quality", "service#general", "ambience#general", "price"]);
        let aspects = Aspects::from_items(
            &tokens,
            Some(&sentences),
            Some(&opinions),
            cats.clone(),
            [
                Aspect {
                    target: AspectTarget::Term { tokens: IxRange::new(1, 2), sentence: Some(Ix::new(0)) },
                    category: cats.tag("food#quality"),
                    polarity: PolarityScores::new(0.05, 0.90, 0.05),
                    opinions: &[Ix::new(0)],
                },
                Aspect {
                    target: AspectTarget::Term { tokens: IxRange::new(8, 9), sentence: Some(Ix::new(0)) },
                    category: cats.tag("service#general"),
                    polarity: PolarityScores::new(0.88, 0.04, 0.08),
                    opinions: &[Ix::new(1), Ix::new(1)],
                },
                Aspect {
                    target: AspectTarget::Implicit { sentence: Ix::new(1) },
                    category: cats.tag("price"),
                    polarity: PolarityScores::new(0.60, 0.10, 0.30),
                    opinions: &[Ix::new(2)],
                },
            ],
        )
        .unwrap();
        assert_eq!(aspects.len(), 3);
        assert_eq!(aspects.polarity(Ix::new(0)).label(), Polarity::Negative);
        assert_eq!(aspects.polarity(Ix::new(1)).label(), Polarity::Positive);
        assert_eq!(aspects.opinions_of(Ix::new(1)), &[Ix::new(1)]);
        assert_eq!(aspects.category_label(Ix::new(2)), Some("price"));
        assert_eq!(aspects.span(Ix::new(0), &tokens, Some(&sentences)), Some(Span::new(4, 8)));
        assert_eq!(aspects.span(Ix::new(2), &tokens, Some(&sentences)), Some(Span::new(50, 65)));

        doc.add(&Key::DEFAULT, aspects, Provenance::by("test")).unwrap();
        assert_eq!(
            doc.made_on(Key::<Aspects>::DEFAULT.id()),
            [Key::<Tokens>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id(), Key::<Opinions>::DEFAULT.id()]
        );
    }

    #[test]
    fn incivility_and_pragmatics_construct_and_check_evidence_slices() {
        let (mut doc, sentences, tokens) = sample_doc();
        let make_chunks = || {
            let mut c_spans = Spans::new(ViewId::SOURCE);
            c_spans.push(Span::new(0, 65));
            Chunks {
                spans: c_spans,
                level: vec![0],
                parent: vec![None],
                sentences: vec![IxRange::new(0, 2)],
                cut: vec![Cut::DocumentStart],
                cost: vec![10],
                model_tokens: vec![16],
                over_budget: vec![false],
            }
        };
        let chunks = make_chunks();
        doc.add(&Key::DEFAULT, make_chunks(), Provenance::by("test")).unwrap();

        let make_structure = || {
            let mut st_spans = Spans::new(ViewId::SOURCE);
            st_spans.push(Span::new(0, 65));
            Structure { spans: st_spans, kind: vec![Block::Paragraph], parent: vec![None], depth: vec![0] }
        };
        let structure = make_structure();
        doc.add(&Key::DEFAULT, make_structure(), Provenance::by("test")).unwrap();

        let mut scores0 = IncivilityScores::default();
        scores0.set(IncivilityLabel::Insult, 0.75);
        scores0.set(IncivilityLabel::Profanity, 0.25);
        let incivility = Incivility::from_items(
            &tokens,
            Some(&sentences),
            Some(&chunks),
            Some(&structure),
            [
                IncivilityItem {
                    target: IncivilityTarget::Sentence(Ix::new(0)),
                    scores: scores0,
                    evidence: &[
                        (IncivilityLabel::Profanity, IxRange::new(3, 5)),
                        (IncivilityLabel::Insult, IxRange::new(4, 5)),
                        (IncivilityLabel::Insult, IxRange::new(1, 2)),
                    ],
                },
                IncivilityItem {
                    target: IncivilityTarget::Chunk(Ix::new(0)),
                    scores: IncivilityScores::default(),
                    evidence: &[],
                },
                IncivilityItem {
                    target: IncivilityTarget::Block(Ix::new(0)),
                    scores: IncivilityScores::default(),
                    evidence: &[],
                },
            ],
        )
        .unwrap();

        assert_eq!(
            incivility.evidence_for_label(Ix::new(0), IncivilityLabel::Insult),
            &[IxRange::new(1, 2), IxRange::new(4, 5)]
        );
        assert_eq!(incivility.evidence_for_label(Ix::new(0), IncivilityLabel::Profanity), &[IxRange::new(3, 5)]);
        assert!(incivility.evidence_for_label(Ix::new(0), IncivilityLabel::Threat).is_empty());
        assert_eq!(
            incivility.target_span(Ix::new(2), Some(&sentences), Some(&chunks), Some(&structure)),
            Some(Span::new(0, 65))
        );
        doc.add(&Key::DEFAULT, incivility, Provenance::by("test")).unwrap();

        let pragmatics = Pragmatics::from_items(
            &sentences,
            &tokens,
            [
                PragmaticsItem {
                    score: 0.82,
                    evidence: Some(Incongruity { positive: IxRange::new(4, 5), negative: IxRange::new(3, 4) }),
                },
                PragmaticsItem { score: 0.05, evidence: None },
            ],
        )
        .unwrap();
        assert_eq!(pragmatics.evidence_spans(Ix::new(0), &tokens), Some((Span::new(17, 22), Span::new(13, 16))));
        assert_eq!(pragmatics.evidence_spans(Ix::new(1), &tokens), None);
        doc.add(&Key::DEFAULT, pragmatics, Provenance::by("test")).unwrap();
    }

    #[test]
    fn bounds_checking_refuses_invalid_token_sentence_and_opinion_indices() {
        let (mut doc, sentences, tokens) = sample_doc();

        // Out-of-bounds token range on Opinions.
        let err = Opinions::from_items(
            &tokens,
            [Opinion { tokens: IxRange::new(14, 17), polarity: 0.5, intensity: 1.0, negated: false }],
        )
        .unwrap_err();
        assert!(err.contains("out of bounds for 16 tokens"), "{err}");

        // Empty token range on Opinions.
        let err = Opinions::from_items(
            &tokens,
            [Opinion { tokens: IxRange::new(3, 3), polarity: 0.5, intensity: 1.0, negated: false }],
        )
        .unwrap_err();
        assert!(err.contains("empty token range"), "{err}");

        // Out-of-order token ranges on Opinions.
        let err = Opinions::from_items(
            &tokens,
            [
                Opinion { tokens: IxRange::new(4, 5), polarity: 0.5, intensity: 1.0, negated: false },
                Opinion { tokens: IxRange::new(1, 2), polarity: 0.5, intensity: 1.0, negated: false },
            ],
        )
        .unwrap_err();
        assert!(err.contains("not in token order"), "{err}");

        // Direct unchecked push with out-of-bounds index is caught by Document::add.
        let mut bad_ops = Opinions::new();
        bad_ops.push(Opinion { tokens: IxRange::new(15, 20), polarity: 0.5, intensity: 1.0, negated: false });
        let err = doc.add(&Key::DEFAULT, bad_ops, Provenance::by("test")).unwrap_err();
        assert!(matches!(err, Error::Invalid { .. }), "{err:?}");

        // Aspect whose token range crosses a sentence boundary or disagrees with its sentence index.
        let cats = TagSet::new("absa", &["price"]);
        let err = Aspects::from_items(
            &tokens,
            Some(&sentences),
            None,
            cats.clone(),
            [Aspect {
                target: AspectTarget::Term { tokens: IxRange::new(1, 2), sentence: Some(Ix::new(1)) },
                category: None,
                polarity: PolarityScores::new(0.5, 0.2, 0.3),
                opinions: &[],
            }],
        )
        .unwrap_err();
        assert!(err.contains("not sentence 1"), "{err}");

        // Aspect with out-of-bounds opinion index.
        let ops = Opinions::from_items(
            &tokens,
            [Opinion { tokens: IxRange::new(4, 5), polarity: 0.8, intensity: 1.0, negated: false }],
        )
        .unwrap();
        let err = Aspects::from_items(
            &tokens,
            Some(&sentences),
            Some(&ops),
            cats,
            [Aspect {
                target: AspectTarget::Term { tokens: IxRange::new(1, 2), sentence: Some(Ix::new(0)) },
                category: None,
                polarity: PolarityScores::new(0.5, 0.2, 0.3),
                opinions: &[Ix::new(1)],
            }],
        )
        .unwrap_err();
        assert!(err.contains("opinion index 1 is out of bounds"), "{err}");

        // Pragmatics evidence outside its own sentence.
        let err = Pragmatics::from_items(
            &sentences,
            &tokens,
            [
                PragmaticsItem {
                    score: 0.9,
                    evidence: Some(Incongruity { positive: IxRange::new(4, 5), negative: IxRange::new(12, 13) }),
                },
                PragmaticsItem { score: 0.1, evidence: None },
            ],
        )
        .unwrap_err();
        assert!(err.contains("outside sentence 0"), "{err}");
    }

    struct DummyOpinionReader;

    impl Annotator for DummyOpinionReader {
        const ID: &'static str = "test:opinion-reader";
        type Reads = (Tokens, Opinions);
        type Writes = (Aspects,);

        fn annotate(&self, (_tokens, _opinions): Refs<'_, Self>) -> Result<(Aspects,), Error> {
            Ok((Aspects::default(),))
        }
    }

    #[test]
    fn misaligned_and_unaligned_anchors_are_refused() {
        let (mut doc, _sentences, tokens) = sample_doc();

        // Pragmatics is aligned to Sentences (len 2): adding 1 item fails with Error::Misaligned.
        let mut short_prag = Pragmatics::new();
        short_prag.push(PragmaticsItem { score: 0.5, evidence: None });
        let err = doc.add(&Key::DEFAULT, short_prag, Provenance::by("test")).unwrap_err();
        assert!(
            matches!(err, Error::Misaligned { len: 1, anchor_len: 2, .. }),
            "expected Error::Misaligned, got {err:?}"
        );

        // Opinions built on Tokens@gold cannot be read alongside the default Tokens.
        let gold_tokens_key = Key::<Tokens>::named("gold");
        let (_, gold_tokens) = make_sentences_and_tokens();
        let mut p_tok = Provenance::by("test");
        p_tok.reads.push(Key::<Sentences>::DEFAULT.id());
        doc.add(&gold_tokens_key, gold_tokens, p_tok).unwrap();
        let gold_ops = Opinions::from_items_on(
            gold_tokens_key,
            &tokens,
            [Opinion { tokens: IxRange::new(4, 5), polarity: 0.8, intensity: 1.0, negated: false }],
        )
        .unwrap();
        let mut p = Provenance::by("test");
        p.reads.push(gold_tokens_key.id());
        doc.add(&Key::DEFAULT, gold_ops, p).unwrap();

        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Opinions>::DEFAULT)
            .step(Bound::new(DummyOpinionReader))
            .build()
            .unwrap();
        let results = analyzer.run(std::slice::from_mut(&mut doc));
        assert!(
            matches!(
                results[0],
                Err(Error::Unaligned {
                    layer,
                    on,
                    read,
                }) if layer == Key::<Opinions>::DEFAULT.id()
                    && on == gold_tokens_key.id()
                    && read.is_empty()
            ),
            "expected Error::Unaligned, got {:?}",
            results[0]
        );
    }
}
