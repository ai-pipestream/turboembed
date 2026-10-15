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

use std::marker::PhantomData;
use std::sync::Arc;

pub use turbo_analysis::{
    Alternative, Alternatives, Edge, Edges, FilterRule, FragmentError, Fragments, Groups, ModeSet, NBest, Overlap,
    SplitMode, Tag, TagSet, TextSpans, filter_spans,
};
use turbo_analysis::{Anchor, Ix, IxRange, Key, KeyId, Layer, LayerName, ModelRef, Span, Spans};

pub use abbreviations::{AbbreviationForm, Abbreviations, DEFINITION, MENTION, abbreviation_links};
pub use date::{Date, write_year};
pub use language_spans::{LanguageSpan, LanguageSpans, UNDETERMINED};
pub use levels::{Level, Levels, Unit, VectorKind};
pub use matches::Matches;
pub use money::{CONVERTED_MONEY, Currency, Money, MoneyMention};
pub use numbers::{DecimalRef, Multiplier, NumberMention, NumberNotation, Numbers, ScaleMarker};
pub use ordinals::{OrdinalBase, OrdinalMention, Ordinals};
pub use quantities::{PERCENT, Quantities, QuantityMention};
pub use temporal::{DocumentDate, Granularity, Origin, TemporalMention, TemporalValue, Temporals};
pub use words::{
    MorphFeatures, MwtLanguage, RuleMwtExpander, TokenExpansion, Word, WordLemmas, WordOrTokenAnchor, WordPosTags,
    Words,
};

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
        fn text_spans(&self) -> Option<turbo_analysis::TextSpans<'_>> {
            Some(self.spans.text_spans())
        }
    };
}

mod abbreviations;
mod date;
mod language_spans;
mod levels;
mod matches;
mod money;
mod numbers;
mod ordinals;
mod quantities;
mod temporal;
mod words;

/// Document structure from the parser upstream or a structure annotator:
/// sections, headings, lists, tables, code, as a tree.
pub struct Structure {
    pub spans: Spans<Structure>,
    pub kind: Vec<Block>,
    pub parent: Vec<Option<Ix<Structure>>>,
    /// A heading's or a section's level, a list's, list item's or
    /// quote's nesting (1 for the outermost), 0 otherwise.
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
    TableCell,
    Code,
    Quote,
    Caption,
    Footnote,
    PageHeader,
    PageFooter,
}

impl Layer for Structure {
    const ID: &'static str = "turbo:structure";
    const OVERLAP: Overlap = Overlap::Nesting;
    positional!(Structure);
    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.kind.len() != n || self.parent.len() != n || self.depth.len() != n {
            return Err("a kind, a parent and a depth for every block".into());
        }
        // Blocks are a tree in text order: each block starts at or after
        // the one before it, inside its parent, which comes before it, and
        // after the end of its previous sibling.
        let (start, end) = self.spans.columns();
        let mut last_child_end = vec![0u32; n];
        let mut last_top_end = 0u32;
        for i in 0..n {
            if i > 0 && start[i] < start[i - 1] {
                return Err(format!("block {i} starts before the block before it"));
            }
            let sibling_end = match self.parent[i] {
                Some(p) if p.get() >= i => return Err(format!("block {i} has a parent at or after it")),
                Some(p) => {
                    let p = p.get();
                    if start[i] < start[p] || end[i] > end[p] {
                        return Err(format!("block {i} is outside its parent {p}"));
                    }
                    &mut last_child_end[p]
                }
                None => &mut last_top_end,
            };
            if start[i] < *sibling_end {
                return Err(format!("block {i} overlaps the block before it"));
            }
            *sibling_end = end[i];
        }
        self.spans.check(Self::OVERLAP)?;
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
        self.spans.check(Self::OVERLAP).map_err(|e| {
            if e == "spans in text order, not overlapping" {
                "sentences in text order, not overlapping".into()
            } else {
                e
            }
        })
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

    /// The tokens of `mode`'s path through `alt`
    /// (`Alternatives::path_for_mode`): each alternative's raw byte span,
    /// so an alternative's `form` is not carried over. `None` when `mode`
    /// has no complete path from position `0` to `alt.num_positions`.
    pub fn from_alternatives(alt: &Alternatives<Tokens>, mode: SplitMode) -> Option<Tokens> {
        let path = alt.path_for_mode(mode)?;
        let mut spans = Spans::new(alt.spans.view());
        for &ix in &path {
            spans.push(alt.spans.get(ix));
        }
        Some(Tokens { spans, sentence: None })
    }
}

impl Layer for Tokens {
    const ID: &'static str = "opennlp:tokens";
    /// A character outside the Basic Multilingual Plane split between two
    /// UTF-16 tokens (`Utf16Spans`) widens both UTF-8 spans to that 4-byte
    /// character.
    const OVERLAP: Overlap = Overlap::Any;
    positional!(Tokens);
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let Some(s) = &self.sentence else { return Vec::new() };
        vec![(Key::<Sentences>::DEFAULT.id(), s.last().map_or(0, |i| i.get() + 1))]
    }
    fn check(&self) -> Result<(), String> {
        match &self.sentence {
            Some(s) if s.len() != self.spans.len() => Err("a sentence for every token".into()),
            Some(s) if !s.windows(2).all(|w| w[0] <= w[1]) => Err("tokens in sentence order".into()),
            _ => self.spans.check(Self::OVERLAP),
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
    /// The word token each piece belongs to, when the pieces were made
    /// from word tokens (a WordPiece or Unigram tokenizer splitting each
    /// word), so a token's pieces are a range found by binary search.
    pub token: Option<Vec<Ix<Tokens>>>,
}

impl Subwords {
    /// The pieces of one word token: a binary search, no scan.
    pub fn of(&self, t: Ix<Tokens>) -> IxRange<Subwords> {
        let col = self.token.as_deref().expect("subwords made per token");
        let a = col.partition_point(|&x| x < t);
        let b = a + col[a..].partition_point(|&x| x == t);
        IxRange::new(a as u32, b as u32)
    }
}

impl Layer for Subwords {
    const ID: &'static str = "opennlp:subwords";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(Subwords);
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let Some(t) = &self.token else { return Vec::new() };
        vec![(Key::<Tokens>::DEFAULT.id(), t.last().map_or(0, |i| i.get() + 1))]
    }
    fn check(&self) -> Result<(), String> {
        let (start, end) = self.spans.columns();
        if self.ids.len() != start.len() {
            return Err("an id for every subword".into());
        }
        if !start.windows(2).all(|w| w[0] <= w[1]) || !end.windows(2).all(|w| w[0] <= w[1]) {
            return Err("subwords in text order".into());
        }
        if self.spans.has_discontinuous() {
            return Err("discontinuous spans are not allowed in subwords".into());
        }
        match &self.token {
            Some(t) if t.len() != start.len() => Err("a token for every subword".into()),
            Some(t) if !t.windows(2).all(|w| w[0] <= w[1]) => Err("subwords in token order".into()),
            _ => self.spans.check(Self::OVERLAP),
        }
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

/// A head and a relation per token or syntactic word (`A = Tokens` by
/// default: `opennlp:dependencies`; `A = Words`:
/// `opennlp:dependencies<turbo:words>`): the tree is the columns, no nodes.
pub struct Dependencies<A: WordOrTokenAnchor = Tokens> {
    pub head: Vec<Option<Ix<A>>>,
    pub relation: Vec<Tag>,
    pub relations: Arc<TagSet>,
}

impl<A: WordOrTokenAnchor> Clone for Dependencies<A> {
    fn clone(&self) -> Self {
        Dependencies { head: self.head.clone(), relation: self.relation.clone(), relations: self.relations.clone() }
    }
}

impl<A: WordOrTokenAnchor> std::fmt::Debug for Dependencies<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dependencies")
            .field("head", &self.head)
            .field("relation", &self.relation)
            .field("relations", &self.relations.name)
            .finish()
    }
}

/// Dependencies anchored on syntactic [`Words`] (`opennlp:dependencies<turbo:words>`).
pub type WordDependencies = Dependencies<Words>;

impl<A: WordOrTokenAnchor> Layer for Dependencies<A> {
    const ID: &'static str = "opennlp:dependencies";
    const OF: Option<&'static LayerName> = A::TARGET_OF;
    fn anchor() -> Anchor {
        Anchor::aligned::<A>()
    }
    fn len(&self) -> usize {
        self.head.len()
    }
    fn check(&self) -> Result<(), String> {
        let n = self.head.len();
        if self.relation.len() != n {
            return Err("a relation for every dependency head".into());
        }
        if self.relation.iter().any(|r| r.0 as usize >= self.relations.len()) {
            return Err(format!("a relation outside {}", self.relations.name));
        }
        for (i, &h) in self.head.iter().enumerate() {
            if let Some(h) = h {
                if h.get() >= n {
                    return Err(format!("dependency head {} at {i} is out of bounds ({n})", h.get()));
                }
                if h.get() == i {
                    return Err(format!("dependency item {i} is its own head"));
                }
            }
        }
        Ok(())
    }
}

pub struct Entities {
    pub spans: Spans<Entities>,
    /// The bounding token range of each entity (for a discontinuous entity,
    /// from the first fragment's start token to the last fragment's end
    /// token, gap tokens included).
    pub tokens: Vec<IxRange<Tokens>>,
    pub kind: Vec<Tag>,
    pub kinds: Arc<TagSet>,
    pub score: Vec<f64>,
}

impl Entities {
    /// Resolves overlapping, nested or crossing entities into an
    /// [`Entities`] layer whose bounding envelopes do not overlap, in text
    /// order, according to `rule`.
    pub fn filter(&self, rule: FilterRule) -> Entities {
        let kept = self.spans.filter(Some(&self.score), rule);
        let mut spans = Spans::with_capacity(self.spans.view(), kept.len());
        let mut tokens = Vec::with_capacity(kept.len());
        let mut kind = Vec::with_capacity(kept.len());
        let mut score = Vec::with_capacity(kept.len());
        let mut frag_buf: Vec<Span> = Vec::new();
        for ix in kept {
            let i = ix.get();
            if self.spans.is_discontinuous(ix) {
                frag_buf.clear();
                frag_buf.extend(self.spans.fragments(ix));
                spans.push_fragments(&frag_buf);
            } else {
                spans.push(self.spans.get(ix));
            }
            tokens.push(self.tokens[i]);
            kind.push(self.kind[i]);
            score.push(self.score[i]);
        }
        Entities { spans, tokens, kind, kinds: Arc::clone(&self.kinds), score }
    }
}

impl Layer for Entities {
    const ID: &'static str = "opennlp:entities";
    const OVERLAP: Overlap = Overlap::Any;
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
        if self.score.iter().any(|s| s.is_nan()) {
            return Err("a score that is NaN".into());
        }
        self.spans.check(Self::OVERLAP)?;
        Ok(())
    }
}

/// A phrase chunk tag per token (`B-NP`, `I-NP`, `O`, the outcomes of a
/// chunker model) and its probability, as `ChunkerME.chunk` and `probs`
/// give them.
pub struct ChunkTags {
    pub tagset: Arc<TagSet>,
    pub tags: Vec<Tag>,
    pub prob: Option<Vec<f64>>,
}

impl Layer for ChunkTags {
    const ID: &'static str = "opennlp:chunk-tags";
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
        if self.prob.as_ref().is_some_and(|p| p.len() != self.tags.len()) {
            return Err("a probability for every tag".into());
        }
        Ok(())
    }
}

/// The phrases a chunker's tags make (`ChunkerME.chunkAsSpans`): each a
/// run of tokens with its kind (`NP`, `VP`, ...), as a span of the text
/// and a token range. Not `turbo:chunks`, which are retrieval units.
pub struct PhraseChunks {
    pub spans: Spans<PhraseChunks>,
    pub tokens: Vec<IxRange<Tokens>>,
    pub kind: Vec<Tag>,
    pub kinds: Arc<TagSet>,
}

impl Layer for PhraseChunks {
    const ID: &'static str = "opennlp:chunks";
    /// Phrases are runs of tokens, and two UTF-16 tokens that split a
    /// character outside the Basic Multilingual Plane are both widened to
    /// it, so two phrases meeting there share its bytes.
    const OVERLAP: Overlap = Overlap::Any;
    positional!(PhraseChunks);
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Tokens>::DEFAULT.id(), self.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0))]
    }
    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.tokens.len() != n || self.kind.len() != n {
            return Err("tokens and a kind for every phrase".into());
        }
        if self.tokens.iter().any(|r| r.start >= r.end) {
            return Err("a phrase has at least one token".into());
        }
        if self.kind.iter().any(|k| k.0 as usize >= self.kinds.len()) {
            return Err(format!("a kind outside {}", self.kinds.name));
        }
        self.spans.check(Self::OVERLAP)?;
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
    Clause,
    /// Between two list items or two table rows.
    Item,
    /// Inside a unit no cut may fall in (a table row, a code block, a
    /// quote), allowed only because that unit alone is over the budget.
    Inside(Block),
    /// Inside a unit that fits the budget, allowed because units that
    /// share sentences with it chain into a run over the budget, which
    /// must be cut somewhere.
    Overlap(Block),
}

/// What a chunk boundary after each sentence would cost, and the strongest
/// reason at it. The chunker reads only the default instance. With one
/// boundary signal, that signal writes the default instance itself. With
/// several (structure, topic shifts, coreference, discourse markers),
/// each should write its own named instance and a step that weighs and
/// sums them should write the default one, since a layer has one writer.
/// A new signal is one more instance of this shape, never a new column.
///
/// Costs are integers, in `UNIT`s, so every backend sums them to the same
/// totals and picks the same cuts; a signal computed in floating point is
/// rounded once, when it is written. A plain sentence boundary costs one
/// unit; a cost can be negative, for a signal that argues for a cut. A
/// step that weighs and sums several instances should compute in i64 and
/// saturate to the i32 range when it writes, so no weight or sum wraps
/// around.
pub struct Boundaries {
    pub cost: Vec<i32>,
    pub reason: Vec<Cut>,
}

impl Boundaries {
    /// The cost of a plain sentence boundary.
    pub const UNIT: i32 = 1000;
}

impl Layer for Boundaries {
    const ID: &'static str = "turbo:boundaries";
    const SCHEMA: u16 = 2;
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
    /// What choosing this boundary cost the optimizer, in
    /// `Boundaries::UNIT`s: the cost of the cut that closes it plus its
    /// size penalty and chunk term. The optimizer works with the exact
    /// cost in i128; this is it saturated to the i64 range.
    pub cost: Vec<i64>,
    /// Its row's length in the embedder's tokens: counted over its whole
    /// span, with the overhead every row carries.
    pub model_tokens: Vec<u32>,
    /// Whether it is longer than the budget's maximum. At level 0 that is
    /// one sentence alone, for a clause splitter to cut further; a row
    /// above holds the chunks cut from it. Its `cut` still says why it
    /// starts where it does.
    pub over_budget: Vec<bool>,
}

impl Layer for Chunks {
    const ID: &'static str = "turbo:chunks";
    const SCHEMA: u16 = 2;
    /// A chunk lies inside its parent, but two chunks at different levels
    /// that are not parent and child can cross: a level 0 chunk under a
    /// section row can run across the edge of a nested section's row.
    const OVERLAP: Overlap = Overlap::Any;
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
            self.over_budget.len(),
        ];
        if columns.iter().any(|&l| l != self.spans.len()) {
            return Err("every column as long as the spans".into());
        }
        for (i, p) in self.parent.iter().enumerate() {
            let Some(p) = *p else { continue };
            if p.get() >= i {
                return Err("a parent before its children".into());
            }
            if self.level[p.get()] <= self.level[i] {
                return Err(format!("chunk {i} is not at a level below its parent's"));
            }
            if !self.spans.get(p).contains(self.spans.get(Ix::new(i as u32))) {
                return Err(format!("chunk {i} is outside its parent"));
            }
        }
        self.spans.check(Self::OVERLAP)?;
        Ok(())
    }
}

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
/// ([`LanguageSpans`]) scores each with its share of the text.
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

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::ViewId;

    #[test]
    fn utf16_spans_are_named_after_their_layer_and_aligned_to_it() {
        assert_eq!(Utf16Spans::<Tokens>::NAME.to_string(), "turbo:utf16<opennlp:tokens>");
        assert_eq!(Utf16Spans::<Sentences>::NAME.to_string(), "turbo:utf16<opennlp:sentences>");
        assert_eq!(Utf16Spans::<Tokens>::anchor(), Anchor::aligned::<Tokens>());
    }

    #[test]
    fn utf16_spans_check_their_columns() {
        // Two halves of one character outside the BMP, then a whole word.
        let ok = Utf16Spans::<Tokens>::new(vec![0, 1, 3], vec![1, 2, 6], vec![true, true, false]);
        assert_eq!(ok.len(), 3);
        assert_eq!(ok.check(), Ok(()));
        let empty = Utf16Spans::<Tokens>::new(Vec::new(), Vec::new(), Vec::new());
        assert_eq!(empty.check(), Ok(()));
        let short = Utf16Spans::<Tokens>::new(vec![0, 1], vec![1, 2], vec![true]);
        assert!(short.check().unwrap_err().contains("a cut flag for every"));
        let short = Utf16Spans::<Tokens>::new(vec![0, 1], vec![1], vec![true, true]);
        assert!(short.check().is_err());
        let backwards = Utf16Spans::<Tokens>::new(vec![0, 4], vec![1, 3], vec![false, false]);
        assert_eq!(backwards.check().unwrap_err(), "UTF-16 span 1 ends before it starts: 4..3");
    }

    /// Blocks from (start, end, parent).
    fn blocks(list: &[(u32, u32, Option<u32>)]) -> Structure {
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(a, b, _) in list {
            spans.push(turbo_analysis::Span::new(a, b));
        }
        let n = list.len();
        let parent = list.iter().map(|&(_, _, p)| p.map(Ix::new)).collect();
        Structure { spans, kind: vec![Block::Paragraph; n], parent, depth: vec![0; n] }
    }

    #[test]
    fn structure_is_a_tree_in_text_order() {
        // A section holding a heading and a list of two items.
        let ok =
            [(0, 20, None), (0, 5, Some(0)), (7, 20, Some(0)), (7, 12, Some(2)), (13, 20, Some(2)), (21, 30, None)];
        assert_eq!(blocks(&ok).check(), Ok(()));
        assert_eq!(blocks(&[]).check(), Ok(()));
        // Empty blocks may touch their neighbours.
        assert_eq!(blocks(&[(0, 4, None), (1, 1, Some(0)), (1, 3, Some(0))]).check(), Ok(()));
        let err = |list: &[(u32, u32, Option<u32>)]| blocks(list).check().unwrap_err();
        assert!(err(&[(5, 9, None), (0, 4, None)]).contains("starts before"));
        assert!(err(&[(0, 9, Some(0))]).contains("parent at or after"));
        assert!(err(&[(0, 9, None), (2, 10, Some(0))]).contains("outside its parent"));
        assert!(err(&[(0, 9, None), (2, 6, Some(0)), (4, 8, Some(0))]).contains("overlaps"));
        assert!(err(&[(0, 5, None), (3, 8, None)]).contains("overlaps"));
        let mut short = blocks(&ok);
        short.depth.pop();
        assert!(short.check().unwrap_err().contains("a depth for every block"));
    }

    #[test]
    fn language_holds_the_ranking_best_first() {
        let mut code = Strs::default();
        code.push("deu");
        code.push("nld");
        let language = Language { code, score: vec![0.75, 0.25] };
        assert_eq!(language.len(), 2);
        assert_eq!(language.best(), Some(("deu", 0.75)));
        assert_eq!(language.iter().collect::<Vec<_>>(), [("deu", 0.75), ("nld", 0.25)]);
        assert_eq!(language.check(), Ok(()));
        assert_eq!(Language::anchor(), Anchor::Document);
        assert_eq!(Language::SCHEMA, 2);

        let empty = Language { code: Strs::default(), score: Vec::new() };
        assert_eq!(empty.best(), None);
        assert!(empty.is_empty());

        let mut code = Strs::default();
        code.push("deu");
        let short = Language { code, score: vec![0.5, 0.5] };
        assert_eq!(short.check().unwrap_err(), "1 codes and 2 scores");
    }

    fn toy() -> ModelRef {
        ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
    }

    /// Three sentences of two values, the middle one with no vector.
    fn three(empty: Vec<u32>) -> Vectors<Sentences> {
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        Vectors::new(2, data, empty.into_iter().map(Ix::new).collect(), toy())
    }

    #[test]
    fn vectors_name_the_items_that_have_none() {
        let v = three(vec![1]);
        assert_eq!(v.check(), Ok(()));
        assert_eq!(v.len(), 3);
        let flags: Vec<bool> = (0..3).map(|i| v.is_empty(Ix::new(i))).collect();
        assert_eq!(flags, [false, true, false]);
        assert_eq!(three(Vec::new()).check(), Ok(()));
        assert!(!three(Vec::new()).is_empty(Ix::new(1)));
    }

    #[test]
    fn vectors_check_the_items_with_no_vector() {
        assert!(three(vec![1, 1]).check().unwrap_err().contains("each once"));
        let mut v = three(vec![1]);
        v.data.extend([0.0, 0.0]);
        v.empty = vec![Ix::new(3), Ix::new(1)];
        assert!(v.check().unwrap_err().contains("in order"));
        assert!(three(vec![1, 3]).check().unwrap_err().contains("among the rows"));
        assert!(three(vec![0]).check().unwrap_err().contains("zeros for every item"));
        let v = Vectors::<Sentences>::new(2, vec![0.0; 3], Vec::new(), toy());
        assert!(v.check().unwrap_err().contains("whole number"));
    }

    #[test]
    fn tokens_from_alternatives_materializes_split_modes() {
        let s = |a: u32, b: u32| turbo_analysis::Span::new(a, b);
        let mut alt = Alternatives::<Tokens>::new(ViewId::SOURCE);
        alt.push(s(0, 6), 0, 1, SplitMode::A | SplitMode::B);
        alt.push(s(0, 18), 0, 3, SplitMode::C | SplitMode::Best);
        alt.push(s(6, 12), 1, 1, ModeSet::A);
        alt.push(s(6, 18), 1, 2, ModeSet::B);
        alt.push(s(12, 18), 2, 1, ModeSet::A);
        assert_eq!(alt.check(), Ok(()));

        let tok_a = Tokens::from_alternatives(&alt, SplitMode::A).unwrap();
        assert_eq!(tok_a.spans.len(), 3);
        let tok_b = Tokens::from_alternatives(&alt, SplitMode::B).unwrap();
        assert_eq!(tok_b.spans.len(), 2);
        let tok_c = Tokens::from_alternatives(&alt, SplitMode::C).unwrap();
        assert_eq!(tok_c.spans.len(), 1);
    }

    #[test]
    fn entities_allow_nesting_crossing_and_discontinuous_fragments() {
        use turbo_analysis::{Document, Provenance};

        let kinds = TagSet::new("ner", &["org", "loc", "verb"]);
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 16)); // "Bank of New York"
        spans.push(Span::new(8, 16)); // "New York"
        spans.push_fragments(&[Span::new(17, 21), Span::new(27, 30)]); // "took ... off"
        let entities = Entities {
            spans,
            tokens: vec![IxRange::new(0, 4), IxRange::new(2, 4), IxRange::new(4, 7)],
            kind: vec![Tag(0), Tag(1), Tag(2)],
            kinds: Arc::clone(&kinds),
            score: vec![0.6, 0.9, 0.8],
        };
        assert_eq!(Entities::OVERLAP, Overlap::Any);
        assert_eq!(entities.check(), Ok(()));
        assert!(!entities.text_spans().unwrap().frag_item_end.is_empty());

        let longest = entities.filter(FilterRule::LongestFirst);
        assert_eq!(longest.len(), 2);
        assert_eq!(longest.spans.get(Ix::new(0)), Span::new(0, 16));
        assert_eq!(longest.spans.get(Ix::new(1)), Span::new(17, 30));
        assert!(longest.spans.is_discontinuous(Ix::new(1)));

        let highest = entities.filter(FilterRule::HighestScore);
        assert_eq!(highest.len(), 2);
        assert_eq!(highest.spans.get(Ix::new(0)), Span::new(8, 16));
        assert_eq!(highest.spans.get(Ix::new(1)), Span::new(17, 30));

        // Partially crossing entities (0..10 and 6..16) are allowed on Entities
        // (Overlap::Any) so unmerged finders or crossing patterns can coexist,
        // and filter resolves them to a disjoint subset.
        let mut crossing_spans = Spans::new(ViewId::SOURCE);
        crossing_spans.push(Span::new(0, 10));
        crossing_spans.push(Span::new(6, 16));
        let crossing = Entities {
            spans: crossing_spans,
            tokens: vec![IxRange::new(0, 2), IxRange::new(1, 4)],
            kind: vec![Tag(0), Tag(1)],
            kinds: Arc::clone(&kinds),
            score: vec![0.5, 0.9],
        };
        assert_eq!(crossing.check(), Ok(()));
        let resolved = crossing.filter(FilterRule::HighestScore);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved.spans.get(Ix::new(0)), Span::new(6, 16));

        // NaN scores are refused so HighestScore has a total order.
        let mut nan_score = crossing;
        nan_score.score[0] = f64::NAN;
        assert!(nan_score.check().unwrap_err().contains("NaN"));

        // Document::add checks fragment boundaries against the view's text:
        // in "café ok" ("é" is bytes 3..5), a fragment ending at byte 4
        // cuts the character even though the envelope 0..7 does not.
        let mut doc = Document::new("café ok");
        let mut t_spans = Spans::new(ViewId::SOURCE);
        t_spans.push(Span::new(0, 5));
        t_spans.push(Span::new(6, 8));
        doc.add(&Key::DEFAULT, Tokens { spans: t_spans, sentence: None }, Provenance::by("test")).unwrap();
        let mut bad_spans = Spans::new(ViewId::SOURCE);
        bad_spans.push_fragments(&[Span::new(0, 4), Span::new(6, 8)]);
        let bad = Entities {
            spans: bad_spans,
            tokens: vec![IxRange::new(0, 2)],
            kind: vec![Tag(0)],
            kinds,
            score: vec![1.0],
        };
        let err = doc.add(&Key::DEFAULT, bad, Provenance::by("test")).unwrap_err();
        assert!(err.to_string().contains("cuts a character"), "{err}");
    }
}
