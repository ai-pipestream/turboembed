//! A small pruning eval for `ClaimVerificationAnnotator`. The scorer is
//! the rule-based stand-in from `claim_support`, not a trained NLI
//! model, so this measures only what candidate pruning costs: the edges
//! each configuration keeps out of those found by scoring every pair
//! with the same scorer, and how many directed pairs it scores. Run with
//! `--nocapture` to see the table; the assertions pin today's numbers.

mod claim_support;

use std::collections::BTreeSet;

use claim_support::{RuleNli, add_entities, doc, rows};
use turbo_analysis::{Annotator, Key, View};
use turbo_analysis_layers::{Entities, Sentences, Vectors};
use turbo_doccat::{ClaimVerificationAnnotator, ClaimVerifyConfig};

const SENTENCES: &[&str] = &[
    "Acme reported record revenue, strong margins and rising orders across Europe and Asia during a busy June.",
    "Acme reported revenue.",
    "Beta closed its Oslo plant.",
    "Beta never closed its Oslo plant.",
    "Gamma hired engineers.",
    "Gamma hired engineers.",
    "Delta expanded quickly in Brazil, Chile, Peru and Mexico last spring after new funding arrived.",
    "Delta expanded.",
    "Rain fell in Oslo.",
    "Prices rose sharply.",
    "Epsilon cut prices.",
    "Epsilon did not cut prices.",
];

const NAMES: &[&str] = &["Acme", "Beta", "Gamma", "Delta", "Epsilon", "Oslo"];

type Row = (usize, usize, turbo_analysis_layers::ClaimRelationKind);

fn run(config: ClaimVerifyConfig, entities: bool) -> (BTreeSet<Row>, usize) {
    let mut d = doc(SENTENCES);
    if entities {
        add_entities(&mut d, NAMES);
    }
    let v = ClaimVerificationAnnotator::<_, Sentences>::with_config(RuleNli::three(), config);
    let (edges,) = v
        .annotate((
            d.require(&Key::<View>::DEFAULT).unwrap(),
            d.require(&Key::<Sentences>::DEFAULT).unwrap(),
            d.get::<Entities>(),
            d.get::<Vectors<Sentences>>(),
            None,
            None,
        ))
        .unwrap();
    (rows(&edges).into_iter().collect(), v.scorer().seen())
}

#[test]
fn pruning_keeps_most_edges_for_a_fraction_of_the_pairs() {
    let all = ClaimVerifyConfig { min_word_overlap: Some(0.0), max_pairs: usize::MAX, ..ClaimVerifyConfig::default() };
    let (reference, all_scored) = run(all, false);
    let systems = [
        ("word overlap (default)", ClaimVerifyConfig::default(), false),
        ("entities only", ClaimVerifyConfig { min_word_overlap: None, ..ClaimVerifyConfig::default() }, true),
        ("overlap + entities", ClaimVerifyConfig::default(), true),
    ];
    println!("all pairs: {} edges, {all_scored} directed pairs scored", reference.len());
    let mut got = Vec::new();
    for (name, config, entities) in systems {
        let (edges, scored) = run(config, entities);
        let kept = edges.intersection(&reference).count();
        assert!(edges.is_subset(&reference), "{name}: pruning only removes edges");
        println!("{name}: {kept}/{} edges kept, {scored}/{all_scored} pairs scored", reference.len());
        got.push((kept, scored));
    }
    assert_eq!((reference.len(), all_scored), (5, 132));
    assert_eq!(got, [(3, 6), (5, 14), (5, 14)]);
}
