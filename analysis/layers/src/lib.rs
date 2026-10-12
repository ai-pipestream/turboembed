//! The shared layers: the types every add-on agrees on, so a lemmatizer
//! from one add-on reads the tags of a tagger from another. An add-on's
//! own layers live in a small schema crate beside it, so another add-on
//! reads them without depending on the code that makes them.
//!
//! A layer the OpenNLP gRPC standard defines (the layer ids in
//! opennlp_document.proto) has the standard's id, so Java, Rust and the
//! wire call it one thing. A layer the standard lacks is "turbo:" until
//! the standard adds it. `Chunks` is one of those: the standard's
//! "opennlp:chunks" are phrase chunks (noun and verb groups), not
//! retrieval chunks, so `Chunks` is "turbo:chunks".

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Anchor, Ix, IxRange, Key, KeyId, Layer, ModelRef, Spans, ViewId};

/// A closed set of labels (a POS tag set, entity kinds, dependency
/// relations), interned: a layer stores a u16 per item, never a string.
#[derive(Debug, Default)]
pub struct TagSet {
    pub name: String,
    names: Vec<Box<str>>,
    index: HashMap<Box<str>, u16>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(pub u16);

impl TagSet {
    pub fn new(name: &str, labels: &[&str]) -> Arc<TagSet> {
        let names: Vec<Box<str>> = labels.iter().map(|&l| l.into()).collect();
        let index = names.iter().enumerate().map(|(i, n)| (n.clone(), i as u16)).collect();
        Arc::new(TagSet { name: name.into(), names, index })
    }

    pub fn tag(&self, label: &str) -> Option<Tag> {
        self.index.get(label).map(|&i| Tag(i))
    }

    pub fn label(&self, t: Tag) -> &str {
        &self.names[t.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Strings by column: one buffer and an end offset per item, never a
/// `String` per item. Lemmas, normalized forms, glosses.
#[derive(Clone, Debug, Default)]
pub struct Strs {
    buf: String,
    end: Vec<u32>,
}

impl Strs {
    pub fn push(&mut self, s: &str) {
        self.buf.push_str(s);
        self.end.push(self.buf.len() as u32);
    }

    pub fn get(&self, i: usize) -> &str {
        let start = if i == 0 { 0 } else { self.end[i - 1] as usize };
        &self.buf[start..self.end[i] as usize]
    }

    pub fn len(&self) -> usize {
        self.end.len()
    }

    pub fn is_empty(&self) -> bool {
        self.end.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        (0..self.len()).map(|i| self.get(i))
    }
}

// What a token costs, column by column: its span (8 bytes), its sentence
// (4), a POS tag (2), a head (4, the root included) and a relation (2).
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<Ix<Sentences>>() == 4);
    assert!(size_of::<Tag>() == 2);
    assert!(size_of::<Option<Ix<Tokens>>>() == 4);
};

macro_rules! positional {
    ($t:ident) => {
        fn len(&self) -> usize {
            self.spans.len()
        }
        fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
            let (start, end) = self.spans.columns();
            Some((self.spans.view(), start, end))
        }
    };
}

/// Document structure from the parser upstream or a structure annotator:
/// sections, headings, lists, tables, code, as a tree.
pub struct Structure {
    pub spans: Spans<Structure>,
    pub kind: Vec<Block>,
    pub parent: Vec<Option<Ix<Structure>>>,
    /// Heading depth, list depth.
    pub depth: Vec<u8>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Block {
    Section,
    Heading,
    Paragraph,
    List,
    ListItem,
    Table,
    TableRow,
    Code,
    Quote,
    Caption,
    Footnote,
    PageHeader,
    PageFooter,
}

impl Layer for Structure {
    const ID: &'static str = "turbo:structure";
    positional!(Structure);
    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.kind.len() != n || self.parent.len() != n || self.depth.len() != n {
            return Err("a kind, a parent and a depth for every block".into());
        }
        Ok(())
    }
}

pub struct Sentences {
    pub spans: Spans<Sentences>,
}

impl Layer for Sentences {
    const ID: &'static str = "opennlp:sentences";
    positional!(Sentences);
    fn check(&self) -> Result<(), String> {
        if self.spans.is_ordered() { Ok(()) } else { Err("sentences in text order, not overlapping".into()) }
    }
}

/// Word tokens (not a model's subwords: those are `Subwords`).
pub struct Tokens {
    pub spans: Spans<Tokens>,
    /// The sentence each token is in, when tokens were made per sentence.
    pub sentence: Option<Vec<Ix<Sentences>>>,
}

impl Tokens {
    /// The tokens of one sentence: a binary search, no scan.
    pub fn of(&self, s: Ix<Sentences>) -> IxRange<Tokens> {
        let col = self.sentence.as_deref().expect("tokens made per sentence");
        let a = col.partition_point(|&x| x < s);
        let b = a + col[a..].partition_point(|&x| x == s);
        IxRange::new(a as u32, b as u32)
    }
}

impl Layer for Tokens {
    const ID: &'static str = "opennlp:tokens";
    positional!(Tokens);
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let Some(s) = &self.sentence else { return Vec::new() };
        vec![(Key::<Sentences>::DEFAULT.id(), s.last().map_or(0, |i| i.get() + 1))]
    }
    fn check(&self) -> Result<(), String> {
        match &self.sentence {
            Some(s) if s.len() != self.spans.len() => Err("a sentence for every token".into()),
            Some(s) if !s.windows(2).all(|w| w[0] <= w[1]) => Err("tokens in sentence order".into()),
            _ => Ok(()),
        }
    }
}

/// A model's subword pieces, aligned to the text: what turboembed's
/// tokenizer already produces with its spans. Only the text's own pieces:
/// the special tokens a template adds and a prompt prefix are not in the
/// text, and a chunker's budget counts them as overhead. Two pieces can
/// share a span (byte-level pieces of one character).
pub struct Subwords {
    pub spans: Spans<Subwords>,
    pub ids: Vec<i32>,
    pub model: ModelRef,
}

impl Layer for Subwords {
    const ID: &'static str = "opennlp:subwords";
    positional!(Subwords);
    fn check(&self) -> Result<(), String> {
        let (start, end) = self.spans.columns();
        if self.ids.len() != start.len() {
            return Err("an id for every subword".into());
        }
        if !start.windows(2).all(|w| w[0] <= w[1]) || !end.windows(2).all(|w| w[0] <= w[1]) {
            return Err("subwords in text order".into());
        }
        Ok(())
    }
}

/// One tag per token.
pub struct PosTags {
    pub tagset: Arc<TagSet>,
    pub tags: Vec<Tag>,
    pub prob: Option<Vec<f64>>,
}

impl Layer for PosTags {
    const ID: &'static str = "opennlp:pos";
    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }
    fn len(&self) -> usize {
        self.tags.len()
    }
    fn check(&self) -> Result<(), String> {
        if self.tags.iter().any(|t| t.0 as usize >= self.tagset.len()) {
            return Err(format!("a tag outside {}", self.tagset.name));
        }
        Ok(())
    }
}

/// A lemma per token.
pub struct Lemmas {
    pub lemma: Strs,
}

impl Layer for Lemmas {
    const ID: &'static str = "opennlp:lemmas";
    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }
    fn len(&self) -> usize {
        self.lemma.len()
    }
}

/// A head and a relation per token: the tree is the columns, no nodes.
pub struct Dependencies {
    pub head: Vec<Option<Ix<Tokens>>>,
    pub relation: Vec<Tag>,
    pub relations: Arc<TagSet>,
}

impl Layer for Dependencies {
    const ID: &'static str = "opennlp:dependencies";
    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }
    fn len(&self) -> usize {
        self.head.len()
    }
}

pub struct Entities {
    pub spans: Spans<Entities>,
    pub tokens: Vec<IxRange<Tokens>>,
    pub kind: Vec<Tag>,
    pub kinds: Arc<TagSet>,
    pub score: Vec<f64>,
}

impl Layer for Entities {
    const ID: &'static str = "opennlp:entities";
    positional!(Entities);
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Tokens>::DEFAULT.id(), self.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0))]
    }
    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.tokens.len() != n || self.kind.len() != n || self.score.len() != n {
            return Err("tokens, a kind and a score for every entity".into());
        }
        if self.tokens.iter().any(|r| r.start > r.end) {
            return Err("no token range that ends before it starts".into());
        }
        if self.kind.iter().any(|k| k.0 as usize >= self.kinds.len()) {
            return Err(format!("a kind outside {}", self.kinds.name));
        }
        Ok(())
    }
}

/// Why a chunk starts where it does: the evaluation and the debugger read
/// this, and so does a person asking why a chunk was cut.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Cut {
    DocumentStart,
    /// A sentence boundary and nothing stronger.
    Sentence,
    Section,
    Heading,
    Paragraph,
    TopicShift,
    Budget,
    Clause,
}

/// What a chunk boundary after each sentence would cost, and the strongest
/// reason at it. Each signal annotator (structure, topic shifts,
/// coreference, discourse markers) writes its own named instance, and one
/// step weighs and sums them into the default instance, since a layer has
/// one writer; the chunker reads only that.
pub struct Boundaries {
    /// f32, unlike the model probabilities in other layers: costs are
    /// computed here in Rust and never compared against OpenNLP's output.
    pub cost: Vec<f32>,
    pub reason: Vec<Cut>,
}

impl Layer for Boundaries {
    const ID: &'static str = "turbo:boundaries";
    fn anchor() -> Anchor {
        Anchor::aligned::<Sentences>()
    }
    fn len(&self) -> usize {
        self.cost.len()
    }
    fn check(&self) -> Result<(), String> {
        if self.reason.len() != self.cost.len() {
            return Err("a reason for every boundary".into());
        }
        if self.cost.iter().any(|c| !c.is_finite()) {
            return Err("a finite cost for every boundary".into());
        }
        Ok(())
    }
}

/// Chunks as a hierarchy: sections hold segments hold chunks, each level
/// a coarser retrieval unit over the same text.
pub struct Chunks {
    pub spans: Spans<Chunks>,
    pub level: Vec<u8>,
    pub parent: Vec<Option<Ix<Chunks>>>,
    pub sentences: Vec<IxRange<Sentences>>,
    pub cut: Vec<Cut>,
    /// What choosing this boundary cost the optimizer.
    pub cost: Vec<f32>,
    /// Its row's length in the embedder's tokens: counted over its whole
    /// span, with the overhead every row carries.
    pub model_tokens: Vec<u32>,
}

impl Layer for Chunks {
    const ID: &'static str = "turbo:chunks";
    positional!(Chunks);
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Sentences>::DEFAULT.id(), self.sentences.iter().map(|r| r.end.get()).max().unwrap_or(0))]
    }
    fn check(&self) -> Result<(), String> {
        let columns = [
            self.level.len(),
            self.parent.len(),
            self.sentences.len(),
            self.cut.len(),
            self.cost.len(),
            self.model_tokens.len(),
        ];
        if columns.iter().any(|&l| l != self.spans.len()) {
            return Err("every column as long as the spans".into());
        }
        if self.parent.iter().enumerate().any(|(i, p)| p.is_some_and(|p| p.get() >= i)) {
            return Err("a parent before its children".into());
        }
        Ok(())
    }
}

/// One vector per item of layer `L`: per sentence, per chunk, per token.
/// Stored as one block, `[len, dim]`, on the host.
pub struct Vectors<L> {
    pub dim: u32,
    pub data: Vec<f32>,
    pub model: ModelRef,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> Vectors<L> {
    pub fn new(dim: u32, data: Vec<f32>, model: ModelRef) -> Vectors<L> {
        Vectors { dim, data, model, _of: PhantomData }
    }

    pub fn row(&self, i: Ix<L>) -> &[f32] {
        let d = self.dim as usize;
        &self.data[i.get() * d..(i.get() + 1) * d]
    }
}

impl<L: Layer> Layer for Vectors<L> {
    const ID: &'static str = "opennlp:embeddings";
    const OF: &'static str = L::ID;
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
        Ok(())
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
    const OF: &'static str = L::ID;
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

/// The document's language: a document-scoped layer.
pub struct Language {
    pub code: String,
    pub score: f64,
}

impl Layer for Language {
    const ID: &'static str = "opennlp:language";
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        1
    }
}
