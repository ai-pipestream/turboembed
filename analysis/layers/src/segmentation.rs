use turbo_analysis::{Alternatives, Anchor, Ix, IxRange, Key, KeyId, Layer, ModelRef, Overlap, Spans, SplitMode};

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

    /// The tokens of one sentence, with or without the sentence
    /// column: the column when populated, else a binary search
    /// over the token spans inside the sentence's span. Never
    /// panics, unlike [`Tokens::of`].
    pub fn in_sentence(&self, sentences: &Sentences, s: Ix<Sentences>) -> IxRange<Tokens> {
        match &self.sentence {
            Some(_) => self.of(s),
            None => self.spans.within(sentences.spans.get(s)),
        }
    }

    /// Contiguous token runs by sentence: one `(sentence, range)`
    /// per run of equal sentence indices, or a single
    /// `(None, all)` run when the sentence column is absent.
    pub fn sentence_ranges(&self) -> impl Iterator<Item = (Option<Ix<Sentences>>, IxRange<Tokens>)> + '_ {
        let total = self.spans.len();
        let mut at = 0usize;
        std::iter::from_fn(move || {
            if at >= total {
                return None;
            }
            let (sent, end) = match &self.sentence {
                Some(col) => {
                    let s = col[at];
                    (Some(s), at + col[at..].partition_point(|&x| x == s))
                }
                None => (None, total),
            };
            let range = IxRange::new(at as u32, end as u32);
            at = end;
            Some((sent, range))
        })
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
    /// The pieces of one word token, with or without the token
    /// column: the column when populated, else a binary search
    /// over the piece spans inside the token's span. Never
    /// panics, unlike [`Subwords::of`].
    pub fn in_token(&self, tokens: &Tokens, t: Ix<Tokens>) -> IxRange<Subwords> {
        match &self.token {
            Some(_) => self.of(t),
            None => self.spans.within(tokens.spans.get(t)),
        }
    }

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
        if self.sentences.iter().any(|r| r.start > r.end) {
            return Err("no sentence range that ends before it starts".into());
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
