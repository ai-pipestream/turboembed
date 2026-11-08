//! The native default rewrite strategy is a whole-value mask, never a
//! per-document pseudonym or an unkeyed SHA-256 (issue #1882).

use turbo_pii::rewrite::{MaskPolicy, RewritePolicy, RewriteStrategy};

#[test]
fn the_default_strategy_is_a_whole_value_mask() {
    let mask = RewriteStrategy::Mask(MaskPolicy::new('*').unwrap());
    assert_eq!(RewriteStrategy::default(), mask);
    assert_eq!(RewritePolicy::default().default_strategy(), &mask);
    assert!(!matches!(RewriteStrategy::default(), RewriteStrategy::Pseudonym(_) | RewriteStrategy::Sha256(_)));
}
