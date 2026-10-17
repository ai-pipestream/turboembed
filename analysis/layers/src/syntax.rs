use std::sync::Arc;

use turbo_analysis::{
    Anchor, FilterRule, Ix, IxRange, Key, KeyId, Layer, LayerName, Overlap, Span, Spans, Tag, TagSet,
};

use crate::{Strs, Tokens, WordOrTokenAnchor, Words};

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
        if self.prob.as_ref().is_some_and(|p| p.len() != self.tags.len()) {
            return Err("a probability for every tag".into());
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

/// A stem per token: item `i` is the stem of token `i`
/// (OpenNLP's `StemmerAnnotator.STEMS`).
#[derive(Clone, Debug, Default)]
pub struct Stems {
    pub stem: Strs,
}

impl Layer for Stems {
    const ID: &'static str = "turbo:stems";
    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }
    fn len(&self) -> usize {
        self.stem.len()
    }
}

/// One token's layered term forms in [`Terms`]: surface, folded, stem and
/// lemma slots.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TermSlot<'a> {
    pub surface: &'a str,
    pub folded: &'a str,
    pub stem: Option<&'a str>,
    pub lemma: Option<&'a str>,
}

/// Layered term forms per token (`turbo:terms`), aligned to [`Tokens`]:
/// `surface` (the original token text), `folded` (the character-normalized
/// form), and optional `stem` and `lemma` columns when a stemmer or
/// lemmatizer produced them.
#[derive(Clone, Debug, Default)]
pub struct Terms {
    pub surface: Strs,
    pub folded: Strs,
    pub stem: Option<Strs>,
    pub lemma: Option<Strs>,
}

impl Terms {
    /// Returns the surface, folded, stem and lemma forms for token `i`.
    pub fn get(&self, i: Ix<Tokens>) -> TermSlot<'_> {
        let idx = i.get();
        TermSlot {
            surface: self.surface.get(idx),
            folded: self.folded.get(idx),
            stem: self.stem.as_ref().map(|s| s.get(idx)),
            lemma: self.lemma.as_ref().map(|l| l.get(idx)),
        }
    }

    /// Iterates over every token's term slots in token order.
    pub fn iter(&self) -> impl Iterator<Item = TermSlot<'_>> {
        (0..self.surface.len() as u32).map(|i| self.get(Ix::new(i)))
    }
}

impl Layer for Terms {
    const ID: &'static str = "turbo:terms";
    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }
    fn len(&self) -> usize {
        self.surface.len()
    }
    fn check(&self) -> Result<(), String> {
        let n = self.surface.len();
        if self.folded.len() != n {
            return Err(format!("{} surface forms and {} folded forms", n, self.folded.len()));
        }
        if let Some(stem) = &self.stem
            && stem.len() != n
        {
            return Err(format!("{} surface forms and {} stems", n, stem.len()));
        }
        if let Some(lemma) = &self.lemma
            && lemma.len() != n
        {
            return Err(format!("{} surface forms and {} lemmas", n, lemma.len()));
        }
        Ok(())
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

/// Constituency phrases from a parser (`ParserAnnotator.PHRASES`,
/// `"opennlp:phrases"`): every non-terminal phrase node above the
/// part-of-speech level (excluding `TOP`, `INC` and `TK`), in pre-order,
/// with its span in the document text, its token range, its syntactic
/// category (`NP`, `VP`, `S`, ...), its head token and its enclosing
/// parent phrase when nested.
#[derive(Clone, Debug)]
pub struct Phrases {
    pub spans: Spans<Phrases>,
    pub tokens: Vec<IxRange<Tokens>>,
    pub kind: Vec<Tag>,
    pub kinds: Arc<TagSet>,
    /// The head token of each phrase (`Parse.getHead` / `Parse.getHeadIndex`).
    pub head: Vec<Ix<Tokens>>,
    /// The enclosing phrase in this layer, or `None` for a top-level phrase
    /// directly under `TOP` or `INC`.
    pub parent: Vec<Option<Ix<Phrases>>>,
}

impl Phrases {
    /// Returns the document byte span of the head token of phrase `i`
    /// (`ParserAnnotator.Head.span`).
    pub fn head_span(&self, i: Ix<Phrases>, tokens: &Tokens) -> Span {
        tokens.spans.get(self.head[i.get()])
    }
}

impl Layer for Phrases {
    const ID: &'static str = "opennlp:phrases";
    /// Two UTF-16 tokens that split a character outside the Basic
    /// Multilingual Plane are both widened to that character, so two
    /// adjacent phrases meeting there share its bytes.
    const OVERLAP: Overlap = Overlap::Any;
    positional!(Phrases);
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Tokens>::DEFAULT.id(), self.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0))]
    }
    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.tokens.len() != n || self.kind.len() != n || self.head.len() != n || self.parent.len() != n {
            return Err("tokens, a kind, a head and a parent for every phrase".into());
        }
        if self.tokens.iter().any(|r| r.start >= r.end) {
            return Err("a phrase has at least one token".into());
        }
        if self.kind.iter().any(|k| k.0 as usize >= self.kinds.len()) {
            return Err(format!("a kind outside {}", self.kinds.name));
        }
        for i in 0..n {
            let range = self.tokens[i];
            let head = self.head[i];
            if head < range.start || head >= range.end {
                return Err(format!(
                    "phrase {i} has head token {} outside its tokens {}..{}",
                    head.get(),
                    range.start.get(),
                    range.end.get()
                ));
            }
            if let Some(p) = self.parent[i] {
                let p = p.get();
                if p >= i {
                    return Err(format!("phrase {i} has a parent at or after it"));
                }
                let parent_range = self.tokens[p];
                if range.start < parent_range.start || range.end > parent_range.end {
                    return Err(format!("phrase {i} is outside its parent {p}"));
                }
            }
        }
        self.spans.check(Self::OVERLAP)?;
        Ok(())
    }
}
