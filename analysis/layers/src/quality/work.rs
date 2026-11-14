//! Deterministic work counters for [`super::GrammarChecker`].

use turbo_analysis::{Error, View};

use crate::{Dependencies, Lemmas, PosTags, Sentences, Tokens};

use super::grammar::{GrammarChecker, GrammarDiagnostics};

/// Deterministic work counters for one [`GrammarChecker`] run.
///
/// Wall-clock time is not part of these counters: CI runners share hosts, so
/// scaling assertions use deterministic work units instead of elapsed time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GrammarCheckWork {
    /// Tokens visited across all enabled sentence rule passes.
    pub tokens_visited: usize,
    /// Dependency head/relation lookups and child-list entries scanned.
    pub dep_lookups: usize,
    /// Local token-window comparisons performed during subject and clause resolution.
    pub window_scans: usize,
}

impl GrammarCheckWork {
    /// Total deterministic work units across token passes, dependency lookups,
    /// and local window scans.
    pub fn total_units(self) -> usize {
        self.tokens_visited.saturating_add(self.dep_lookups).saturating_add(self.window_scans)
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.tokens_visited = self.tokens_visited.saturating_add(other.tokens_visited);
        self.dep_lookups = self.dep_lookups.saturating_add(other.dep_lookups);
        self.window_scans = self.window_scans.saturating_add(other.window_scans);
    }
}

impl GrammarChecker {
    /// Runs the configured grammar rules over `view` and returns deterministic
    /// [`GrammarCheckWork`] counters alongside the diagnostics.
    pub fn check_view_with_work(
        &self,
        view: &View,
        sentences: &Sentences,
        tokens: &Tokens,
        pos: &PosTags,
        lemmas: Option<&Lemmas>,
        deps: Option<&Dependencies>,
    ) -> Result<(GrammarDiagnostics, GrammarCheckWork), Error> {
        self.check_view_inner(view, sentences, tokens, pos, lemmas, deps, None)
    }
}
