//! Fuzzy match policy integration for [`GeoEntityLinker`]: broadened
//! candidate retrieval and policy scoring over entry names.
//!
//! The policy is a black box: its `score` decides admission and confidence,
//! so broadened retrieval only needs to surface enough candidates for the
//! policy to judge. Retrieval similarities are ranking hints and are never
//! multiplied into the policy confidence, so a zero retrieval score cannot
//! zero out an admitted candidate.

use turbo_matcher::FuzzyMatchPolicy;

use crate::store::Entry;

/// Retrieval similarity for the broadened lookup. Low enough that the
/// gazetteer's own metric filter does not drop candidates the policy would
/// admit; the policy filter that runs afterwards is authoritative.
pub(crate) const POLICY_RETRIEVAL_MIN_SIMILARITY: f64 = 0.0;

/// Extra head room for the broadened lookup, since the policy filter drops
/// candidates the retrieval step only ranked.
const POLICY_RETRIEVAL_FACTOR: usize = 4;

/// The broadened retrieval limit for one mention: the normal limit with
/// policy head room, never below it.
pub(crate) fn broadened_limit(limit: usize) -> usize {
    limit.saturating_mul(POLICY_RETRIEVAL_FACTOR).max(limit)
}

/// The best policy confidence of `surface` over the entry's canonical and
/// alternate names, or `None` when the policy admits none of them. The
/// value is finite and in `[0.0, 1.0]`.
pub(crate) fn score_entry(policy: &FuzzyMatchPolicy, surface: &str, entry: &Entry<'_>) -> Option<f64> {
    let mut best = policy.score(surface, entry.name()).filter(|s| s.is_finite());
    for alt in entry.alternate_names() {
        let score = policy.score(surface, alt).filter(|s| s.is_finite());
        if score > best {
            best = score;
        }
    }
    best.map(|s| f64::from(s).clamp(0.0, 1.0))
}
