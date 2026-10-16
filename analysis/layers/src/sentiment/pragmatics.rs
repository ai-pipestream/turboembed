use turbo_analysis::{Anchor, Ix, IxRange, Key, KeyId, Layer, Span};

use super::{check_token_range, token_range_span};
use crate::{Sentences, Tokens};

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
        // A sentence scored by an embedding head alone has no evidence, so
        // a layer without evidence does not require a Tokens layer.
        if max_tok > 0 || !self.tokens_key.name.is_empty() { vec![(self.tokens_key.id(), max_tok)] } else { Vec::new() }
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
