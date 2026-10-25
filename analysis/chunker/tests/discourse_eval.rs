//! A small, hand-labelled discourse eval: EDU boundaries and exact edges
//! (both unit spans, relation and nuclearity) for the dependency mode and
//! the connective fallback over the same twelve documents. Run with
//! `--nocapture` to see the table; the floors pin today's numbers.

mod discourse_support;

use discourse_support::{doc, run};
use std::collections::BTreeSet;
use turbo_analysis_layers::{DiscourseRel as R, Nuclearity as N};

type GoldEdge = (&'static str, &'static str, R, N);
type Case = (&'static [(&'static str, &'static str)], &'static [&'static str], &'static [GoldEdge]);

const CASES: &[Case] = &[
    (
        &[("Although sales fell , profits rose .", "3:mark 3:nsubj 6:advcl 6:punct 6:nsubj 0:root 6:punct")],
        &["Although sales fell ,", "profits rose ."],
        &[("Although sales fell ,", "profits rose .", R::Concession, N::SatelliteToNucleus)],
    ),
    (
        &[("Acme raised prices because costs rose .", "2:nsubj 0:root 2:obj 6:mark 6:nsubj 2:advcl 2:punct")],
        &["Acme raised prices", "because costs rose ."],
        &[("Acme raised prices", "because costs rose .", R::CauseResult, N::NucleusToSatellite)],
    ),
    (
        &[
            ("Sales rose .", "2:nsubj 0:root 2:punct"),
            ("However , margins fell .", "4:advmod 4:punct 4:nsubj 0:root 4:punct"),
        ],
        &["Sales rose .", "However , margins fell ."],
        &[("Sales rose .", "However , margins fell .", R::Contrast, N::MultiNuclear)],
    ),
    (
        &[("Acme bought Beta and sold Gamma .", "2:nsubj 0:root 2:obj 5:cc 2:conj 5:obj 2:punct")],
        &["Acme bought Beta", "and sold Gamma ."],
        &[("Acme bought Beta", "and sold Gamma .", R::Joint, N::MultiNuclear)],
    ),
    (
        &[("The firm that Acme bought grew .", "2:det 6:nsubj 5:obj 5:nsubj 2:acl:relcl 0:root 6:punct")],
        &["The firm that Acme bought grew ."],
        &[],
    ),
    (
        &[("If demand holds , Acme will expand .", "3:mark 3:nsubj 7:advcl 7:punct 7:nsubj 7:aux 0:root 7:punct")],
        &["If demand holds ,", "Acme will expand ."],
        &[("If demand holds ,", "Acme will expand .", R::Condition, N::SatelliteToNucleus)],
    ),
    (
        &[("Prices fell , so buyers returned .", "2:nsubj 0:root 6:punct 6:cc 6:nsubj 2:conj 2:punct")],
        &["Prices fell ,", "so buyers returned ."],
        &[("Prices fell ,", "so buyers returned .", R::CauseResult, N::NucleusToSatellite)],
    ),
    (&[("Sales and profits rose .", "4:nsubj 3:cc 1:conj 0:root 4:punct")], &["Sales and profits rose ."], &[]),
    (
        &[
            ("Acme cut costs .", "2:nsubj 0:root 2:obj 2:punct"),
            ("Therefore , margins improved .", "4:advmod 4:punct 4:nsubj 0:root 4:punct"),
        ],
        &["Acme cut costs .", "Therefore , margins improved ."],
        &[("Acme cut costs .", "Therefore , margins improved .", R::CauseResult, N::NucleusToSatellite)],
    ),
    (
        // RST treats a `while` contrast as multinuclear; the annotator
        // makes every subordinated unit a satellite.
        &[("Beta stalled while Acme expanded rapidly .", "2:nsubj 0:root 5:mark 5:nsubj 2:advcl 5:advmod 2:punct")],
        &["Beta stalled", "while Acme expanded rapidly ."],
        &[("Beta stalled", "while Acme expanded rapidly .", R::Contrast, N::MultiNuclear)],
    ),
    (
        // Attribution has no relation of its own; the reported clause is
        // the nucleus, which the annotator gets backwards.
        &[("Acme said profits rose .", "2:nsubj 0:root 4:nsubj 2:ccomp 2:punct")],
        &["Acme said", "profits rose ."],
        &[("Acme said", "profits rose .", R::Elaboration, N::SatelliteToNucleus)],
    ),
    (
        &[
            ("The plan failed .", "2:det 3:nsubj 0:root 3:punct"),
            (
                "For example , sales dropped after the launch .",
                "2:case 5:obl 5:punct 5:nsubj 0:root 8:case 8:det 5:obl 5:punct",
            ),
        ],
        &["The plan failed .", "For example , sales dropped after the launch ."],
        &[("The plan failed .", "For example , sales dropped after the launch .", R::Evidence, N::NucleusToSatellite)],
    ),
];

/// Boundary (tp, predicted, gold), edge (tp, predicted, gold).
fn score(parsed: bool) -> [(usize, usize, usize); 2] {
    let (mut b, mut e) = ((0, 0, 0), (0, 0, 0));
    for (sentences, gold_units, gold_edges) in CASES {
        let d = doc(sentences, parsed);
        let (units, edges) = run(&d);
        let starts = |u: &[String]| -> BTreeSet<usize> {
            let mut at = 0;
            u.iter()
                .map(|s| {
                    let s0 = at;
                    at += s.len() + 1;
                    s0
                })
                .skip(1)
                .collect()
        };
        let gold_u: Vec<String> = gold_units.iter().map(|s| s.to_string()).collect();
        let (p, g) = (starts(&units), starts(&gold_u));
        b = (b.0 + p.intersection(&g).count(), b.1 + p.len(), b.2 + g.len());
        let pred: BTreeSet<(String, String, R, N)> =
            edges.iter().map(|&(f, t, r, n)| (units[f].clone(), units[t].clone(), r, n)).collect();
        let gold: BTreeSet<(String, String, R, N)> =
            gold_edges.iter().map(|&(f, t, r, n)| (f.to_string(), t.to_string(), r, n)).collect();
        e = (e.0 + pred.intersection(&gold).count(), e.1 + pred.len(), e.2 + gold.len());
    }
    [b, e]
}

#[test]
fn dependency_mode_beats_the_connective_fallback() {
    let deps = score(true);
    let fallback = score(false);
    for (name, [b, e]) in [("dependencies", deps), ("fallback", fallback)] {
        println!(
            "{name}: boundaries {}/{} predicted, {}/{} gold; edges {}/{} predicted, {}/{} gold",
            b.0, b.1, b.0, b.2, e.0, e.1, e.0, e.2
        );
    }
    // Dependencies miss two nuclearities (`while`, attribution); the
    // fallback also misses the verb conjunct and the complement clause
    // and cuts before the prepositional `after`.
    assert_eq!(deps, [(10, 10, 10), (8, 10, 10)]);
    assert_eq!(fallback, [(8, 9, 10), (6, 9, 10)]);
}
