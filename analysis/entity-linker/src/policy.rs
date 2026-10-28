//! Fuzzy match policy integration for candidate generation: broadened
//! retrieval through the existing provider path, then policy scoring and
//! filtering over each candidate's `matched_alias`.
//!
//! The policy is a black box: its `score` decides admission and confidence,
//! so broadened retrieval only needs to surface enough candidates for the
//! policy to judge. Retrieval similarities are never multiplied into the
//! policy confidence, so a zero retrieval score cannot zero out an admitted
//! candidate.

use turbo_matcher::FuzzyMatchPolicy;
use turbo_vector_index::Space;

use crate::providers::{KbProvider, RawCandidate};

/// Retrieval threshold for the broadened pass. Low enough that the existing
/// fuzzy retrieval does not drop candidates the policy would admit; the
/// policy filter that runs afterwards is authoritative.
const POLICY_RETRIEVAL_THRESHOLD: f64 = 0.0;

/// Policy-aware candidate generation for a [`KbProvider`]: delegates exactly
/// to [`KbProvider::generate_candidates`] when the policy is disabled, and
/// otherwise broadens retrieval, re-scores each candidate's `matched_alias`
/// with the policy, drops the ones the policy does not admit, and keeps the
/// deterministic ranking order.
pub(crate) fn generate_with_policy<P: KbProvider + ?Sized>(
    provider: &P,
    mention_text: &str,
    query_vector: Option<(&Space, &[f32])>,
    max_candidates: usize,
    fuzzy_threshold: f64,
    fuzzy_policy: FuzzyMatchPolicy,
) -> Vec<RawCandidate> {
    if !fuzzy_policy.is_enabled() {
        return provider.generate_candidates(mention_text, query_vector, max_candidates, fuzzy_threshold);
    }
    let mut candidates =
        provider.generate_candidates(mention_text, query_vector, max_candidates, POLICY_RETRIEVAL_THRESHOLD);
    score_and_retain(&fuzzy_policy, mention_text, &mut candidates);
    candidates.truncate(max_candidates);
    candidates
}

/// Re-scores `candidates` with the policy over each candidate's
/// `matched_alias`: admitted candidates get the policy confidence as their
/// `match_similarity` (finite, in `[0.0, 1.0]`), refused ones are dropped
/// unless they are exact alias matches, which a fuzzy policy must not
/// suppress. The ranking order stays deterministic.
fn score_and_retain(policy: &FuzzyMatchPolicy, mention_text: &str, candidates: &mut Vec<RawCandidate>) {
    candidates.retain_mut(|c| match policy.score(mention_text, &c.matched_alias) {
        Some(score) if score.is_finite() => {
            c.match_similarity = f64::from(score).clamp(0.0, 1.0);
            true
        }
        _ => crate::similarity::fold_alias(mention_text) == crate::similarity::fold_alias(&c.matched_alias),
    });
    candidates.sort_by(|a, b| {
        b.match_similarity
            .total_cmp(&a.match_similarity)
            .then_with(|| b.alias_prior.total_cmp(&a.alias_prior))
            .then_with(|| b.popularity_prior.total_cmp(&a.popularity_prior))
            .then_with(|| a.kb_id.cmp(&b.kb_id))
    });
}
