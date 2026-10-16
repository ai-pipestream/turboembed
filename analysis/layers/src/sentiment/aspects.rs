use std::sync::Arc;

use turbo_analysis::{Anchor, Ix, IxRange, Key, KeyId, Layer, Span, Tag, TagSet};

use super::opinions::{Opinions, Polarity};
use super::{check_token_range, token_range_span};
use crate::{Sentences, Tokens};

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
        // Implicit aspects index no tokens, so a layer of only implicit
        // aspects does not require a Tokens layer.
        let max_tok = self.tokens.iter().flatten().map(|r| r.end.get()).max().unwrap_or(0);
        let mut out = Vec::new();
        if max_tok > 0 || !self.tokens_key.name.is_empty() {
            out.push((self.tokens_key.id(), max_tok));
        }
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
