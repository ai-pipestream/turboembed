//! Issue #1285 on a real scorer: `BundleNliScorer` over the tiny test
//! bundle's WordPiece tokenizer and encoder, with a fixed NLI head. The
//! pair budget is the encoder's `max_seq`, and every count is the real
//! tokenizer's, so the budget the claim verifier filters on is the one
//! the scorer enforces.
//!
//! Zero thresholds make every scored pair an edge (both directions
//! entail and contradict at 0), so the edges name exactly the pairs
//! scored.

mod nli_support;

use std::collections::BTreeSet;

use nli_support::{TempDir, make_nli_bundle};
use turbo_analysis::{Annotator, Document, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_embed::{BundleNliScorer, Device, EncoderOptions, Precision};
use turbo_analysis_layers::Sentences;
use turbo_doccat::{ClaimVerificationAnnotator, ClaimVerifyConfig, NliScorer};

/// The same words in three orders: every pair overlaps fully, so these
/// pairs rank first, and any two are over the budgets below.
const LONG: [&str; 3] = [
    "Northwind and Southgate signed the long storage agreement in Zurich today.",
    "In Zurich today Northwind and Southgate signed the long storage agreement.",
    "Today in Zurich the long storage agreement Northwind and Southgate signed.",
];
/// Two short pairs ranked below: overlap 2/3 and 1/2, none across.
const SHORT: [&str; 4] = ["Omega fell.", "Omega fell sharply.", "Gamma grew fast.", "Gamma grew slowly."];

fn bundle() -> TempDir {
    make_nli_bundle("claim-budget", &["contradiction", "entailment", "neutral"], |_| {})
}

fn scorer(dir: &TempDir, max_seq: u32) -> BundleNliScorer {
    let options = EncoderOptions { device: Device::Cpu, precision: Precision::Model, max_seq, ..Default::default() };
    BundleNliScorer::open(&dir.0, &options).unwrap()
}

/// The real tokenizer's pair size: both texts plus the pair specials.
fn pair_tokens(s: &BundleNliScorer, a: &str, b: &str) -> u32 {
    (s.count_tokens(a) + s.count_tokens(b) + s.pair_special_tokens()) as u32
}

/// The scored pairs, as sentence index pairs, and how many encoder runs.
fn verify(dir: &TempDir, max_seq: u32, max_pairs: usize, sents: &[&str]) -> (BTreeSet<(u32, u32)>, u64) {
    let text = sents.join(" ");
    let mut spans = Spans::new(ViewId(0));
    let mut at = 0;
    for s in sents {
        spans.push(Span::new(at as u32, (at + s.len()) as u32));
        at += s.len() + 1;
    }
    let mut d = Document::new(text.as_str());
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    let config =
        ClaimVerifyConfig { max_pairs, entail_threshold: 0.0, contradict_threshold: 0.0, ..Default::default() };
    let v = ClaimVerificationAnnotator::<_, Sentences>::with_config(scorer(dir, max_seq), config);
    let (edges,) = v
        .annotate((
            d.require(&Key::<View>::DEFAULT).unwrap(),
            d.require(&Key::<Sentences>::DEFAULT).unwrap(),
            None,
            None,
            None,
            None,
        ))
        .unwrap();
    let scored = edges.iter().map(|(_, e)| (e.from.get() as u32, e.to.get() as u32)).collect();
    (scored, v.scorer().run_count())
}

/// The budget at which both short pairs exactly fit, checked to leave
/// every long pair over it.
fn exact_budget(dir: &TempDir) -> u32 {
    let probe = scorer(dir, 64);
    let budget = pair_tokens(&probe, SHORT[0], SHORT[1]).max(pair_tokens(&probe, SHORT[2], SHORT[3]));
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        let long = pair_tokens(&probe, LONG[a], LONG[b]);
        assert!(long > budget, "long pair {a},{b}: {long} tokens within {budget}");
    }
    budget
}

#[test]
fn an_over_budget_pair_does_not_take_the_only_slot() {
    let dir = bundle();
    let budget = exact_budget(&dir);
    let (scored, runs) = verify(&dir, budget, 1, &[LONG[0], LONG[1], SHORT[0], SHORT[1]]);
    assert_eq!(scored, BTreeSet::from([(2, 3)]));
    assert_eq!(runs, 1);
}

#[test]
fn eligible_pairs_behind_more_than_max_pairs_over_budget_pairs_are_scored() {
    let dir = bundle();
    let budget = exact_budget(&dir);
    let sents = [LONG[0], LONG[1], LONG[2], SHORT[0], SHORT[1], SHORT[2], SHORT[3]];
    // Three long pairs rank first; the cap of 2 goes to the two short ones.
    let (scored, runs) = verify(&dir, budget, 2, &sents);
    assert_eq!(scored, BTreeSet::from([(3, 4), (5, 6)]));
    assert_eq!(runs, 1);
}

#[test]
fn a_pair_at_exactly_the_budget_is_scored_and_one_token_less_is_not() {
    let dir = bundle();
    let probe = scorer(&dir, 64);
    let exact = pair_tokens(&probe, SHORT[2], SHORT[3]);
    let sents = [SHORT[2], SHORT[3]];
    assert_eq!(verify(&dir, exact, 1, &sents), (BTreeSet::from([(0, 1)]), 1));
    assert_eq!(verify(&dir, exact - 1, 1, &sents), (BTreeSet::new(), 0));
}

#[test]
fn nothing_is_scored_when_every_pair_is_over_budget() {
    let dir = bundle();
    // Within the model's own 64 every long pair is a candidate and scored.
    assert_eq!(verify(&dir, 64, 64, &LONG), (BTreeSet::from([(0, 1), (0, 2), (1, 2)]), 1));
    let budget = exact_budget(&dir);
    let (scored, runs) = verify(&dir, budget, 64, &LONG);
    assert!(scored.is_empty(), "{scored:?}");
    assert_eq!(runs, 0, "no encoder run without a pair");
}

#[path = "claim_verify/inputs.rs"]
mod inputs;
