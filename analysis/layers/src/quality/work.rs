//! Deterministic work counters for [`super::GrammarChecker`].

use turbo_analysis::{Error, View};

use crate::{Dependencies, Lemmas, PosTags, Sentences, Tokens};

use super::grammar::{GrammarChecker, GrammarDiagnostics};

/// Deterministic work counters for one [`GrammarChecker`] run.
///
/// The counters are taken where the rules read their input, not where the
/// rules choose to report: each read of a token's surface, span, tag, lemma
/// or morphology is one `token_reads` unit and each dependency head,
/// relation or child read is one `dep_lookups` unit. A rule that rescans the
/// sentence for every token therefore grows quadratically here whether or
/// not it was written with the counters in mind.
///
/// Wall-clock time is not part of these counters: CI runners share hosts, so
/// scaling assertions use deterministic work units instead of elapsed time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GrammarCheckWork {
    /// Reads of per-token data (surface, span, tag, lemma, morphology).
    pub token_reads: usize,
    /// Dependency reads: head and relation lookups, child-list entries handed
    /// to a rule, and head entries visited while indexing a sentence.
    pub dep_lookups: usize,
}

/// Largest growth allowed for a counter when the token count doubles.
/// Linear work doubles; `n log n` work stays under this at probe sizes;
/// quadratic work quadruples.
const MAX_GROWTH_PER_DOUBLING_PERCENT: usize = 225;

impl GrammarCheckWork {
    /// All counted reads.
    pub fn total_units(self) -> usize {
        self.token_reads.saturating_add(self.dep_lookups)
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.token_reads = self.token_reads.saturating_add(other.token_reads);
        self.dep_lookups = self.dep_lookups.saturating_add(other.dep_lookups);
    }

    /// Checks that work grows no faster than near-linearly over `samples`,
    /// given as `(token_count, work)` with the token count doubling from one
    /// sample to the next. `token_reads` and `dep_lookups` are checked
    /// separately so a large linear counter cannot hide growth in the other.
    ///
    /// The scaling probe and the tests that prove the probe catches slow
    /// lookups both call this, so they cannot drift apart.
    pub fn check_linear_scaling(samples: &[(usize, Self)]) -> Result<(), String> {
        if samples.len() < 2 {
            return Err(format!("need at least two samples, got {}", samples.len()));
        }
        for pair in samples.windows(2) {
            let (n_prev, w_prev) = pair[0];
            let (n_curr, w_curr) = pair[1];
            if n_curr != n_prev.saturating_mul(2) || n_prev == 0 {
                return Err(format!("token counts must double between samples, got {n_prev} -> {n_curr}"));
            }
            for (name, prev, curr) in [
                ("token_reads", w_prev.token_reads, w_curr.token_reads),
                ("dep_lookups", w_prev.dep_lookups, w_curr.dep_lookups),
            ] {
                if curr.saturating_mul(100) > prev.saturating_mul(MAX_GROWTH_PER_DOUBLING_PERCENT) {
                    return Err(format!(
                        "{name} grew faster than linear when tokens doubled ({n_prev} -> {n_curr}): {prev} -> {curr}"
                    ));
                }
            }
        }
        Ok(())
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
