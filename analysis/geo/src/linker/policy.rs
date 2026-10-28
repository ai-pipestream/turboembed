//! Fuzzy match policy integration for [`GeoEntityLinker`]: broadened
//! candidate retrieval and policy scoring over entry names.
//!
//! The policy is a black box: its `score` decides admission and confidence,
//! so broadened retrieval only needs to surface enough candidates for the
//! policy to judge. Retrieval similarities are ranking hints and are never
//! multiplied into the policy confidence, so a zero retrieval score cannot
//! zero out an admitted candidate.

use turbo_matcher::FuzzyMatchPolicy;

use crate::geocode::by_prior_view;
use crate::store::Entry;

/// Retrieval similarity for the broadened lookup. Low enough that the
/// gazetteer's own metric filter does not drop candidates the policy would
/// admit; the policy filter that runs afterwards is authoritative.
pub(crate) const POLICY_RETRIEVAL_MIN_SIMILARITY: f64 = 0.0;

/// Policy-enabled retrieval examines the whole bounded gazetteer result set;
/// the caller's output cap applies only after policy scoring.
pub(crate) fn broadened_limit() -> usize {
    usize::MAX
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

/// Re-scores broadened fuzzy `hits` with the policy over each entry's
/// names: admitted candidates get the policy confidence as their score
/// (finite, in `[0.0, 1.0]`), refused ones are dropped unless the raw
/// retrieval similarity was exact, which a fuzzy policy must not suppress.
/// The retained hits are ranked by that policy confidence with a
/// deterministic tie break and only then capped at `limit`, so the cap and
/// the caller's rank weights see the policy order, not the raw retrieval
/// order.
pub(crate) fn score_and_retain<'a>(
    policy: &FuzzyMatchPolicy,
    surface: &str,
    hits: Vec<(Entry<'a>, f64)>,
    limit: usize,
) -> Vec<(Entry<'a>, f64)> {
    let mut retained: Vec<(Entry<'_>, f64)> = hits
        .into_iter()
        .filter_map(|(entry, sim)| {
            let score = score_entry(policy, surface, &entry).or_else(|| (sim >= 1.0).then_some(1.0))?;
            Some((entry, score))
        })
        .collect();
    retained.sort_by(|(ea, sa), (eb, sb)| sb.total_cmp(sa).then_with(|| by_prior_view(ea, eb)));
    retained.truncate(limit);
    retained
}
