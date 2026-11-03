//! Deterministic work counters for one fuzzy alias lookup.

/// Deterministic work counters for one [`super::FuzzyAliasIndex::for_each_candidate`] call.
///
/// Wall-clock time is not part of these counters: CI runners share hosts, so
/// scaling assertions use inverted-index probe work instead of elapsed time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FuzzyLookupWork {
    /// Entries read from inverted posting lists (length-window slices).
    pub postings_scanned: usize,
    /// Keys visited from the posting paths: token dice, the short-key table
    /// and posting slices whose sketch prefilter passed.
    pub posting_candidates_verified: usize,
    /// Keys examined in the lossless length-window fallback (O(1) sketch filters).
    pub window_keys_considered: usize,
    /// Keys visited from the window fallback after sketch filtering.
    pub window_candidates_visited: usize,
}

impl FuzzyLookupWork {
    /// Inverted-index probe work: postings scanned plus posting-path verifies.
    ///
    /// The window fallback is a lossless escape hatch when posting budgets
    /// saturate; its cost is mostly O(1) sketch filters per key in a length
    /// band, so sublinear scaling is asserted on this posting-path total.
    pub fn index_probe_units(self) -> usize {
        self.postings_scanned.saturating_add(self.posting_candidates_verified)
    }

    /// Every key passed to the visit callback (posting path or window).
    pub fn candidates_visited(self) -> usize {
        self.posting_candidates_verified.saturating_add(self.window_candidates_visited)
    }
}
