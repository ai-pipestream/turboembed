//! `MaskedView::config_hash` folds every kind's effective policy keyed by
//! its stable id: views that mask any kind differently hash differently,
//! and views that mask every kind alike hash alike however they were built.

mod support;

use std::collections::HashMap;

use support::*;
use turbo_analysis::Annotator;
use turbo_pii::PiiKind;
use turbo_pii::rewrite::{MaskPolicy, MaskedView};

fn whole(c: char) -> MaskPolicy {
    MaskPolicy::new(c).unwrap()
}

/// Ids 1..=17 and 28..=30: the kinds a view had before the EU and Brazil
/// ids (18..=27) were added.
fn legacy(kind: PiiKind) -> bool {
    !(18..=27).contains(&kind.builtin_id().unwrap())
}

/// What the view does: the default policy, then the policy of every
/// built-in kind and every custom kind of its table.
fn behaviour(v: &MaskedView) -> Vec<MaskPolicy> {
    let mut out = vec![v.policy_in(v.custom_kinds(), PiiKind::Custom(turbo_pii::CustomKindId::new(u16::MAX)))];
    out.extend(PiiKind::builtins().map(|k| v.policy(k)));
    out.extend(v.custom_kinds().kinds().map(|k| v.policy(k)));
    out
}

#[test]
fn a_by_kind_view_with_legacy_kinds_starred_is_not_the_all_star_view() {
    // The pair that collided at 12419471227701003101: by kind, with every
    // legacy kind overridden to `*`, still masks the newer ids by kind.
    let star = whole('*');
    let all = MaskedView::all(star);
    let mut starred = MaskedView::by_kind('*').unwrap();
    for k in PiiKind::builtins().filter(|&k| legacy(k)) {
        starred = starred.with(k, star);
    }
    let differs: Vec<PiiKind> = PiiKind::builtins().filter(|&k| all.policy(k) != starred.policy(k)).collect();
    assert!(!differs.is_empty() && differs.iter().all(|&k| !legacy(k)), "{differs:?}");
    assert_ne!(all.config_hash(), starred.config_hash());
    assert_ne!(all.config_hash(), 12_419_471_227_701_003_101);

    // Overriding the newer ids too gives the all-star view, hash included.
    for k in differs {
        starred = starred.with(k, star);
    }
    assert_eq!(behaviour(&all), behaviour(&starred));
    assert_eq!(all.config_hash(), starred.config_hash());
}

/// A seeded generator, so the property runs the same views every time.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

#[test]
fn views_that_mask_any_kind_differently_hash_differently() {
    let pool = [
        whole('*'),
        whole('#'),
        whole('*').keeping_format(),
        whole('*').keeping_trailing(4),
        whole('*').keeping_format().keeping_trailing(4),
        whole('*').keeping_leading(2),
        whole('#').keeping_leading(1 << 16),
        whole('*').keeping_trailing(1 << 48),
    ];
    let kinds: Vec<PiiKind> = PiiKind::builtins().chain(test_kinds().kinds().take(3)).collect();
    let mut rng = Lcg(1901);
    let mut seen: HashMap<u64, Vec<MaskPolicy>> = HashMap::new();
    for i in 0..4000 {
        let mut v = match i % 3 {
            0 => MaskedView::all(pool[rng.below(pool.len())]),
            1 => MaskedView::by_kind(['*', '#'][rng.below(2)]).unwrap(),
            _ => MaskedView::all(pool[rng.below(pool.len())]).with_custom_kinds(test_kinds()),
        };
        // Up to three overrides, mostly of one or two kinds, so many views
        // differ from another in exactly one kind's policy.
        for _ in 0..rng.below(4) {
            let k = kinds[rng.below(kinds.len())];
            if let Ok(next) = v.clone().try_with(k, pool[rng.below(pool.len())]) {
                v = next;
            }
        }
        let b = behaviour(&v);
        let prior = seen.entry(v.config_hash()).or_insert_with(|| b.clone());
        assert_eq!(*prior, b, "two views that mask differently share hash {}", v.config_hash());
    }
    assert!(seen.len() > 500, "{} distinct views", seen.len());
}
