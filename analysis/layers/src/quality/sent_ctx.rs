//! Sentence-level rule context with private dependency data and counting accessors.

use std::cell::Cell;

use turbo_analysis::{Ix, IxRange, Span};

use crate::{Dependencies, Lemmas, MorphFeatures, PosTags, Tokens};

use super::grammar::GrammarChecker;
use super::work::GrammarCheckWork;

pub(crate) struct SentCtx<'a> {
    pub(crate) checker: &'a GrammarChecker,
    pub(crate) text: &'a str,
    pub(crate) tokens: &'a Tokens,
    pub(crate) pos: &'a PosTags,
    pub(crate) lemmas: Option<&'a Lemmas>,
    deps: Option<&'a Dependencies>,
    pub(crate) morph: Option<&'a MorphFeatures<Tokens>>,
    dep_children: Vec<Vec<usize>>,
    pub(crate) range: IxRange<Tokens>,
    work: Cell<GrammarCheckWork>,
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
        let mut work = GrammarCheckWork::default();
        if let Some(d) = deps {
            dep_children.resize_with(end - start, Vec::new);
            work.dep_lookups = end - start;
            for (t, head) in d.head.iter().enumerate().take(end).skip(start) {
                if let Some(h) = head.map(Ix::get)
                    && (start..end).contains(&h)
                {
                    dep_children[h - start].push(t);
                }
            }
        }
        Self { checker, text, tokens, pos, lemmas, deps, morph, dep_children, range, work: Cell::new(work) }
    }

    pub(crate) fn work(&self) -> GrammarCheckWork {
        self.work.get()
    }

    pub(crate) fn record_dep_lookups(&self, n: usize) {
        let mut w = self.work.get();
        w.dep_lookups = w.dep_lookups.saturating_add(n);
        self.work.set(w);
    }

    pub(crate) fn record_window_scans(&self, n: usize) {
        let mut w = self.work.get();
        w.window_scans = w.window_scans.saturating_add(n);
        self.work.set(w);
    }

    pub(crate) fn record_tokens_visited(&self, n: usize) {
        let mut w = self.work.get();
        w.tokens_visited = w.tokens_visited.saturating_add(n);
        self.work.set(w);
    }

    pub(crate) fn span(&self, i: usize) -> Span {
        self.tokens.spans.get(Ix::new(i as u32))
    }

    pub(crate) fn surface(&self, i: usize) -> &'a str {
        &self.text[self.span(i).range()]
    }

    pub(crate) fn pos_str(&self, i: usize) -> &'a str {
        self.pos.tagset.label(self.pos.tags[i])
    }

    pub(crate) fn lemma_str(&self, i: usize) -> Option<&'a str> {
        self.lemmas.map(|l| l.lemma.get(i))
    }

    pub(crate) fn morph_str(&self, i: usize) -> Option<&'a str> {
        self.morph.map(|m| m.feats.get(i))
    }

    pub(crate) fn has_deps(&self) -> bool {
        self.deps.is_some()
    }

    pub(crate) fn dep_head(&self, i: usize) -> Option<usize> {
        self.record_dep_lookups(1);
        self.deps.and_then(|d| d.head[i].map(Ix::get))
    }

    pub(crate) fn dep_rel(&self, i: usize) -> Option<&'a str> {
        self.record_dep_lookups(1);
        self.deps.map(|d| d.relations.label(d.relation[i]))
    }

    pub(crate) fn dep_children(&self, head: usize) -> &[usize] {
        let start = self.range.start.get();
        let end = self.range.end.get();
        if self.dep_children.is_empty() || !(start..end).contains(&head) {
            self.record_dep_lookups(1);
            return &[];
        }
        let children = &self.dep_children[head - start];
        self.record_dep_lookups(children.len().max(1));
        children
    }
}
