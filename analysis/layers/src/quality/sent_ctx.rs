//! Sentence-level rule context.
//!
//! Every per-token datum a rule can read (surface, span, tag, lemma, morph,
//! dependency head, relation, children) is private to this module and is
//! reached only through an accessor that counts the read. Rules carry no
//! counting code of their own, so a rule cannot do work the counters miss:
//! a slow path has to read token data, and reading it is what is counted.
//!
//! What the counters cannot see is work inside this module (an accessor that
//! itself scans). Keep every accessor O(1) per unit it counts. The allocation
//! probe (`allocation_scaling` in `tests/grammar_scaling_probe.rs`) and the
//! ignored wall-clock probe measure from outside and do not rely on these
//! counters.

use std::cell::Cell;

use turbo_analysis::{Ix, IxRange, Span};

use crate::{Dependencies, Lemmas, MorphFeatures, PosTags, Tokens};

use super::grammar::GrammarChecker;
use super::work::GrammarCheckWork;

pub(crate) struct SentCtx<'a> {
    pub(crate) checker: &'a GrammarChecker,
    pub(crate) range: IxRange<Tokens>,
    text: &'a str,
    tokens: &'a Tokens,
    pos: &'a PosTags,
    lemmas: Option<&'a Lemmas>,
    deps: Option<&'a Dependencies>,
    morph: Option<&'a MorphFeatures<Tokens>>,
    dep_children: Vec<Vec<usize>>,
    token_reads: Cell<usize>,
    dep_lookups: Cell<usize>,
}

impl<'a> SentCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        checker: &'a GrammarChecker,
        text: &'a str,
        tokens: &'a Tokens,
        pos: &'a PosTags,
        lemmas: Option<&'a Lemmas>,
        deps: Option<&'a Dependencies>,
        morph: Option<&'a MorphFeatures<Tokens>>,
        range: IxRange<Tokens>,
    ) -> Self {
        let start = range.start.get();
        let end = range.end.get();
        let mut dep_children = Vec::new();
        let mut visited = 0usize;
        if let Some(d) = deps {
            dep_children.resize_with(end - start, Vec::new);
            // One unit per head entry the index build really visits, so a
            // build that walks more than this sentence shows up in the count.
            for (offset, head) in d.head[start..end].iter().enumerate() {
                visited += 1;
                if let Some(h) = head.map(Ix::get)
                    && (start..end).contains(&h)
                {
                    dep_children[h - start].push(start + offset);
                }
            }
        }
        Self {
            checker,
            range,
            text,
            tokens,
            pos,
            lemmas,
            deps,
            morph,
            dep_children,
            token_reads: Cell::new(0),
            dep_lookups: Cell::new(visited),
        }
    }

    pub(crate) fn work(&self) -> GrammarCheckWork {
        GrammarCheckWork { token_reads: self.token_reads.get(), dep_lookups: self.dep_lookups.get() }
    }

    fn read_token(&self) {
        self.token_reads.set(self.token_reads.get().saturating_add(1));
    }

    fn read_deps(&self, n: usize) {
        self.dep_lookups.set(self.dep_lookups.get().saturating_add(n));
    }

    fn raw_span(&self, i: usize) -> Span {
        self.tokens.spans.get(Ix::new(i as u32))
    }

    pub(crate) fn span(&self, i: usize) -> Span {
        self.read_token();
        self.raw_span(i)
    }

    pub(crate) fn surface(&self, i: usize) -> &'a str {
        self.read_token();
        &self.text[self.raw_span(i).range()]
    }

    /// Source text between token `i` and token `i + 1`.
    pub(crate) fn gap_after(&self, i: usize) -> &'a str {
        self.read_token();
        &self.text[self.raw_span(i).end as usize..self.raw_span(i + 1).start as usize]
    }

    pub(crate) fn pos_str(&self, i: usize) -> &'a str {
        self.read_token();
        self.pos.tagset.label(self.pos.tags[i])
    }

    pub(crate) fn lemma_str(&self, i: usize) -> Option<&'a str> {
        self.read_token();
        self.lemmas.map(|l| l.lemma.get(i))
    }

    pub(crate) fn morph_str(&self, i: usize) -> Option<&'a str> {
        self.read_token();
        self.morph.map(|m| m.feats.get(i))
    }

    pub(crate) fn has_deps(&self) -> bool {
        self.deps.is_some()
    }

    pub(crate) fn dep_head(&self, i: usize) -> Option<usize> {
        self.read_deps(1);
        self.deps.and_then(|d| d.head[i].map(Ix::get))
    }

    pub(crate) fn dep_rel(&self, i: usize) -> Option<&'a str> {
        self.read_deps(1);
        self.deps.map(|d| d.relations.label(d.relation[i]))
    }

    /// Children of `head` inside this sentence. Counts one unit per child
    /// handed out (at least one), since the caller walks the whole list.
    pub(crate) fn dep_children(&self, head: usize) -> &[usize] {
        let start = self.range.start.get();
        let end = self.range.end.get();
        if self.dep_children.is_empty() || !(start..end).contains(&head) {
            self.read_deps(1);
            return &[];
        }
        let children = &self.dep_children[head - start];
        self.read_deps(children.len().max(1));
        children
    }
}
