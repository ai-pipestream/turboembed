use turbo_analysis::{Ix, Layer, Span, Spans};
use turbo_analysis_layers::{Levels, Subwords, Vectors};

use super::{Source, TokenIndex};
use crate::rows::{self, Block};
use crate::{IndexError, Metric, Space, Which, kernel};

/// A rule that drops some subword rows of a candidate before they are
/// stored (punctuation, stopwords), with the name the index records for it.
pub struct Prune<'a> {
    pub rule: &'a str,
    /// Whether to drop the piece.
    pub drop: &'a dyn Fn(Ix<Subwords>) -> bool,
}

/// Collects candidates of token vectors for a [`TokenIndex`]. A candidate
/// the builder refuses leaves it as it was.
pub struct TokenIndexBuilder<M> {
    index: TokenIndex<M>,
}

impl<M> TokenIndexBuilder<M> {
    /// A builder for token vectors of `space`, compared by dot product.
    pub fn new(space: Space) -> Result<TokenIndexBuilder<M>, IndexError> {
        if space.dim == 0 {
            return Err(IndexError::ZeroDimension);
        }
        let bpr = rows::blocks(space.dim);
        let index = TokenIndex {
            space,
            bpr,
            rows: Vec::new(),
            first: vec![0],
            sources: Vec::new(),
            meta: Vec::new(),
            pruned: Vec::new(),
            rules: Vec::new(),
            path: kernel::detect(),
        };
        Ok(TokenIndexBuilder { index })
    }

    /// The number of candidates added.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Adds a candidate of the token vectors in `tokens`, `space.dim`
    /// values a token one after another, and returns its number. The
    /// values are copied; token `i` has the source piece `i` and no span.
    pub fn push(&mut self, tokens: &[f32], meta: M) -> Result<u32, IndexError> {
        let dim = self.index.space.dim as usize;
        if !tokens.len().is_multiple_of(dim) {
            return Err(IndexError::Dimension { index: self.index.space.dim, other: tokens.len() });
        }
        let pieces = tokens.chunks_exact(dim).enumerate().map(|(i, v)| (v, Source { piece: i as u32, span: None }));
        self.add(pieces, meta, None)
    }

    /// Adds a candidate for each unit of `units` from the subword vectors
    /// of a text: a unit's tokens are the pieces of `subwords` inside its
    /// span (inside each fragment, for a discontinuous one; an empty piece
    /// counts where it lies before the fragment's end, or at its end when no
    /// non-empty piece follows, so one on the boundary of two units is the
    /// second's and one at the end of the text the last unit's),
    /// in text order,
    /// less the items `vectors` lists as empty and the pieces `prune`
    /// drops. A unit left with no token is not added. Returns how many
    /// candidates were added; `meta` names each from its unit. The layers
    /// must be of the builder's model and dimension and of one view; if any
    /// unit is refused, none is added.
    pub fn push_levels(
        &mut self,
        vectors: &Vectors<Subwords>,
        subwords: &Subwords,
        levels: &Levels,
        units: impl IntoIterator<Item = Ix<Levels>>,
        prune: Option<&Prune<'_>>,
        mut meta: impl FnMut(Ix<Levels>) -> M,
    ) -> Result<usize, IndexError> {
        let space = &self.index.space;
        for model in [&vectors.model, &subwords.model] {
            if *model != space.model {
                return Err(IndexError::Model { index: Box::new(space.model.clone()), other: Box::new(model.clone()) });
            }
        }
        if vectors.dim != space.dim {
            return Err(IndexError::Dimension { index: space.dim, other: vectors.dim as usize });
        }
        vectors.check().map_err(IndexError::Layer)?;
        subwords.check().map_err(IndexError::Layer)?;
        levels.check().map_err(IndexError::Layer)?;
        if vectors.len() != subwords.len() {
            return Err(IndexError::Layer(format!(
                "{} subword vectors for {} subwords",
                vectors.len(),
                subwords.len()
            )));
        }
        if levels.view() != subwords.spans.view() {
            return Err(IndexError::Layer("levels and subwords of different views".into()));
        }
        let before = self.mark();
        let mut added = 0;
        for unit in units {
            if unit.get() >= levels.len() {
                self.undo(before);
                return Err(IndexError::Layer(format!("unit {} of {} levels", unit.get(), levels.len())));
            }
            let mut dropped = 0u32;
            let mut kept = Vec::new();
            for fragment in levels.spans.fragments(unit) {
                let r = subwords.spans.within(fragment);
                for i in r.start.get()..r.end.get() {
                    let ix = Ix::<Subwords>::new(i as u32);
                    let span = subwords.spans.get(ix);
                    if vectors.is_empty(ix) || (span.is_empty() && !empty_piece_in(&subwords.spans, i, fragment)) {
                        continue;
                    }
                    if prune.is_some_and(|p| (p.drop)(ix)) {
                        dropped += 1;
                        continue;
                    }
                    kept.push((vectors.row(ix), Source { piece: i as u32, span: Some(span) }));
                }
            }
            if kept.is_empty() {
                continue;
            }
            let rule = match prune {
                Some(p) => Some((self.rule(p.rule)?, dropped)),
                None => None,
            };
            if let Err(e) = self.add(kept.into_iter(), meta(unit), rule) {
                self.undo(before);
                return Err(e);
            }
            added += 1;
        }
        Ok(added)
    }

    /// The index of the candidates added, in the order they were added.
    pub fn build(self) -> TokenIndex<M> {
        self.index
    }

    fn add<'v>(
        &mut self,
        tokens: impl Iterator<Item = (&'v [f32], Source)>,
        meta: M,
        pruned: Option<(u16, u32)>,
    ) -> Result<u32, IndexError> {
        let c = self.index.len();
        let id = u32::try_from(c).map_err(|_| IndexError::Full)?;
        let before = self.mark();
        let ix = &mut self.index;
        for (i, (v, source)) in tokens.enumerate() {
            if v.len() != ix.space.dim as usize {
                self.undo(before);
                return Err(IndexError::Dimension { index: self.index.space.dim, other: v.len() });
            }
            let at = ix.rows.len();
            ix.rows.resize(at + ix.bpr, Block::default());
            if let Err(e) = rows::prepare(v, &ix.space, Metric::Dot, Which::Token(c, i), &mut ix.rows[at..]) {
                self.undo(before);
                return Err(e);
            }
            ix.sources.push(source);
        }
        let ix = &mut self.index;
        if ix.sources.len() == before.1 {
            return Err(IndexError::NoTokens(Which::Candidate(c)));
        }
        let Ok(end) = u32::try_from(ix.sources.len()) else {
            self.undo(before);
            return Err(IndexError::Full);
        };
        ix.first.push(end);
        ix.meta.push(meta);
        ix.pruned.push(pruned);
        Ok(id)
    }

    fn rule(&mut self, name: &str) -> Result<u16, IndexError> {
        let rules = &mut self.index.rules;
        if let Some(i) = rules.iter().position(|r| r == name) {
            return Ok(i as u16);
        }
        let i = u16::try_from(rules.len()).map_err(|_| IndexError::Full)?;
        rules.push(name.to_owned());
        Ok(i)
    }

    /// The candidate count, token count and stored length, to undo to.
    fn mark(&self) -> (usize, usize, usize) {
        (self.index.meta.len(), self.index.sources.len(), self.index.rows.len())
    }

    fn undo(&mut self, (candidates, tokens, blocks): (usize, usize, usize)) {
        let ix = &mut self.index;
        ix.meta.truncate(candidates);
        ix.pruned.truncate(candidates);
        ix.first.truncate(candidates + 1);
        ix.sources.truncate(tokens);
        ix.rows.truncate(blocks);
    }
}

/// Whether the empty piece `i` belongs to `fragment`, which holds it: one
/// before the fragment's end does, so one on the boundary of two units is
/// the second's; one at the end does only when no non-empty piece follows,
/// so one at the end of the text stays in the units that end there.
fn empty_piece_in(spans: &Spans<Subwords>, i: usize, fragment: Span) -> bool {
    if spans.get(Ix::new(i as u32)).start < fragment.end {
        return true;
    }
    (i + 1..spans.len()).all(|j| spans.get(Ix::new(j as u32)).is_empty())
}
