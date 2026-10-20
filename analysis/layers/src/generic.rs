use std::marker::PhantomData;

use turbo_analysis::{Anchor, Ix, Layer, LayerName, ModelRef, Overlap, Spans, TextSpans};

use crate::Strs;

/// One vector per item of layer `L`: per sentence, per chunk, per token.
/// Stored as one block, `[len, dim]`, on the host.
///
/// An item the producer could make no vector for (one that covers none of
/// the model's tokens, such as an empty span or whitespace alone) keeps
/// its row, all zeros, so the layer stays aligned to its items, and is
/// listed in `empty`. Cosine similarity is undefined for such a row: a
/// reader skips the items `is_empty` names rather than comparing them.
pub struct Vectors<L> {
    pub dim: u32,
    pub data: Vec<f32>,
    /// The items with no vector, in order, each once; their rows are zeros.
    pub empty: Vec<Ix<L>>,
    pub model: ModelRef,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> Vectors<L> {
    pub fn new(dim: u32, data: Vec<f32>, empty: Vec<Ix<L>>, model: ModelRef) -> Vectors<L> {
        Vectors { dim, data, empty, model, _of: PhantomData }
    }

    pub fn row(&self, i: Ix<L>) -> &[f32] {
        let d = self.dim as usize;
        &self.data[i.get() * d..(i.get() + 1) * d]
    }

    /// Whether item `i` has no vector: its row is zeros and not to be
    /// compared. A binary search over `empty`.
    pub fn is_empty(&self, i: Ix<L>) -> bool {
        self.empty.binary_search(&i).is_ok()
    }
}

impl<L: Layer> Layer for Vectors<L> {
    const ID: &'static str = "opennlp:embeddings";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        if self.dim == 0 { 0 } else { self.data.len() / self.dim as usize }
    }
    fn check(&self) -> Result<(), String> {
        if self.dim == 0 || !self.data.len().is_multiple_of(self.dim as usize) {
            return Err("a whole number of rows".into());
        }
        if !self.empty.windows(2).all(|w| w[0] < w[1]) {
            return Err("the items with no vector in order, each once".into());
        }
        if self.empty.last().is_some_and(|i| i.get() >= self.len()) {
            return Err("the items with no vector among the rows".into());
        }
        if self.empty.iter().any(|&i| self.row(i).iter().any(|&x| x != 0.0)) {
            return Err("zeros for every item with no vector".into());
        }
        Ok(())
    }
}

/// The text spans of the encoder windows that produced a
/// subword-vector layer, in text order, not overlapping. A
/// producer of contextual `Vectors<Subwords>` publishes this
/// beside the vectors; late pooling reads it as a required
/// input, which is what keeps non-contextual per-token
/// embeddings (which have no windows to present) from being
/// pooled as if they were contextual.
#[derive(Clone, Debug)]
pub struct TokenVectorWindows {
    pub spans: Spans<TokenVectorWindows>,
}

impl Layer for TokenVectorWindows {
    const ID: &'static str = "turbo:token-vector-windows";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        if self.spans.is_empty() {
            return Err("at least one encoder window".into());
        }
        // Ordered, non-overlapping and continuous, so the window
        // holding a piece is found by binary search.
        self.spans.check(Overlap::None)
    }
}

/// A confidence per item of layer `L`, for any layer whose producer has
/// one: a tokenizer's, a name finder's, a parser's arcs.
pub struct Confidence<L> {
    pub value: Vec<f64>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> Confidence<L> {
    pub fn new(value: Vec<f64>) -> Confidence<L> {
        Confidence { value, _of: PhantomData }
    }
}

impl<L: Layer> Layer for Confidence<L> {
    const ID: &'static str = "turbo:confidence";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.value.len()
    }
    fn check(&self) -> Result<(), String> {
        if self.value.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err("a confidence between 0 and 1".into());
        }
        Ok(())
    }
}

/// The UTF-16 span of every item of layer `L`, as a Java component
/// computed it, for a layer whose producer counts UTF-16 units: OpenNLP's
/// `SimpleTokenizer` and `TokenizerME` classify UTF-16 units, so a
/// character outside the Basic Multilingual Plane can be split between two
/// items. A UTF-8 byte span cannot say where inside the character the cut
/// is; `L`'s own span widens to the whole character (start rounded down,
/// end rounded up), and this layer keeps the exact span in UTF-16 units,
/// half open, and whether the item cuts a character.
///
/// `cut` is a `bool` per item, one byte each: the layer is written only
/// for texts that have a cut character, and a bit set would save little
/// beside the two `u32` columns.
pub struct Utf16Spans<L> {
    pub start16: Vec<u32>,
    pub end16: Vec<u32>,
    /// Whether the item's span starts or ends inside a surrogate pair.
    pub cut: Vec<bool>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> Utf16Spans<L> {
    pub fn new(start16: Vec<u32>, end16: Vec<u32>, cut: Vec<bool>) -> Utf16Spans<L> {
        Utf16Spans { start16, end16, cut, _of: PhantomData }
    }
}

impl<L: Layer> Layer for Utf16Spans<L> {
    const ID: &'static str = "turbo:utf16";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.start16.len()
    }
    /// The columns are of one length and no span ends before it starts.
    /// Whether a span ends inside the view's UTF-16 length needs the text,
    /// which a layer does not see, so it is not checked here.
    fn check(&self) -> Result<(), String> {
        let n = self.start16.len();
        if self.end16.len() != n || self.cut.len() != n {
            return Err("an end and a cut flag for every UTF-16 start".into());
        }
        if let Some(i) = (0..n).find(|&i| self.start16[i] > self.end16[i]) {
            return Err(format!("UTF-16 span {i} ends before it starts: {}..{}", self.start16[i], self.end16[i]));
        }
        Ok(())
    }
}

/// The document's language: a document-scoped layer holding every
/// language the detector ranked, best first, each with its score. Item 0
/// is the best language. OpenNLP's detector scores each language with
/// its probability; a step that finds the language by span
/// ([`LanguageSpans`](crate::LanguageSpans)) scores each with its share of the text.
///
/// Schema 2: schema 1 held the best language alone.
pub struct Language {
    pub code: Strs,
    pub score: Vec<f64>,
}

impl Language {
    /// The best language and its score: item 0, or `None` when the
    /// layer is empty.
    pub fn best(&self) -> Option<(&str, f64)> {
        self.score.first().map(|&s| (self.code.get(0), s))
    }

    /// The languages with their scores, best first.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        self.code.iter().zip(self.score.iter().copied())
    }
}

impl Layer for Language {
    const ID: &'static str = "opennlp:language";
    const SCHEMA: u16 = 2;
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        self.code.len()
    }
    fn check(&self) -> Result<(), String> {
        if self.code.len() != self.score.len() {
            return Err(format!("{} codes and {} scores", self.code.len(), self.score.len()));
        }
        Ok(())
    }
}
