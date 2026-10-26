//! Issue #1285: the `max_pairs` cap counts only pairs the scorer can
//! take. A higher-ranked pair over the scorer's token budget must not
//! use up a slot and hide an eligible pair ranked below it.

mod claim_support;

use claim_support::{RuleNli, add_entities, doc, rows};
use turbo_analysis::{Annotator, Document, Key, Layer, View};
use turbo_analysis_layers::{ClaimRelationKind as K, Entities, Sentences, Vectors};
use turbo_doccat::{ClaimVerificationAnnotator, ClaimVerifyConfig};

/// Nine words each, sharing two entities: rank 2, but any two of them
/// are 9 + 9 + 3 specials, over the 12 token budget below.
const LONG: [&str; 3] = [
    "Acme and Beta signed a long supply deal today.",
    "Acme and Beta later extended that supply deal again.",
    "Acme and Beta finally ended the supply deal there.",
];

/// The edges, and every directed `(premise, hypothesis)` pair scored.
type Run = (Vec<(usize, usize, K)>, Vec<(String, String)>);

fn run(max_pairs: usize, max_pair_tokens: u32, sents: &[&str]) -> Run {
    let mut d: Document = doc(sents);
    add_entities(&mut d, &["Acme", "Beta", "Gamma", "Delta", "Omega"]);
    let mut nli = RuleNli::three();
    nli.max_pair_tokens = max_pair_tokens;
    let config = ClaimVerifyConfig { min_word_overlap: None, max_pairs, ..ClaimVerifyConfig::default() };
    let v = ClaimVerificationAnnotator::<_, Sentences>::with_config(nli, config);
    let (edges,) = v
        .annotate((
            d.require(&Key::<View>::DEFAULT).unwrap(),
            d.require(&Key::<Sentences>::DEFAULT).unwrap(),
            d.get::<Entities>(),
            d.get::<Vectors<Sentences>>(),
        ))
        .unwrap();
    edges.check().unwrap();
    let seen = v.scorer().seen.lock().unwrap().clone();
    (rows(&edges), seen)
}

fn pair(p: &str, h: &str) -> (String, String) {
    (p.to_string(), h.to_string())
}

#[test]
fn an_over_budget_pair_does_not_take_the_only_slot() {
    let sents = [LONG[0], LONG[1], "Gamma grew fast.", "Gamma grew fast."];
    let (edges, seen) = run(1, 12, &sents);
    assert_eq!(seen, [pair(sents[2], sents[3]), pair(sents[3], sents[2])], "both directions of 2-3, nothing else");
    assert_eq!(edges, [(2, 3, K::Redundant)]);
}

#[test]
fn eligible_claims_after_more_than_max_pairs_ineligible_pairs_are_verified() {
    // 0-1, 0-2 and 1-2 are three over-budget rank 2 pairs, more than
    // max_pairs; the eligible rank 1 pairs come after all of them, ranked
    // by word overlap: 3-4, then 5-6, then 7-8.
    let sents = [
        LONG[0],
        LONG[1],
        LONG[2],
        "Gamma grew fast.",
        "Gamma grew fast.",
        "Delta did not fall.",
        "Delta did fall.",
        "Omega rose.",
        "Omega fell sharply.",
    ];
    let (edges, seen) = run(2, 12, &sents);
    assert_eq!(seen.len(), 4, "max_pairs 2 still bounds the scorer at 2 * max_pairs evaluations");
    assert_eq!(edges, [(3, 4, K::Redundant), (5, 6, K::Contradicts)]);
    assert!(!seen.iter().any(|(p, _)| p.starts_with("Omega")), "the third eligible pair is past the cap");
}

#[test]
fn all_over_budget_scores_nothing() {
    let sents = [LONG[0], LONG[1], LONG[2]];
    let (edges, seen) = run(2, 12, &sents);
    assert!(edges.is_empty());
    assert!(seen.is_empty());
}
