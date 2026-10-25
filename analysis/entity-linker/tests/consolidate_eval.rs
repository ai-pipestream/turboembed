//! A small clustering evaluation of `EntityConsolidator`: pairwise
//! precision, recall and F1 of which names are one entity, against gold
//! clusters, for the name merges alone, a coreference chain layer alone,
//! and both (the default). One document carries a tagger error (a person
//! tagged as a place) the surface merge cannot undo.

mod common;

use common::doc_with_names;
use turbo_analysis::{Groups, Ix, Key, Provenance};
use turbo_analysis_layers::Entities;
use turbo_entity_linker::{ConsolidatorConfig, EntityConsolidator};

struct Case {
    text: &'static str,
    names: &'static [(&'static str, &'static str)],
    /// The gold entity of each name.
    gold: &'static [u32],
    /// Name pairs a coreference system chained.
    chains: &'static [(u32, u32)],
}

const CASES: &[Case] = &[
    Case {
        text: "Barack Obama spoke . Obama left . Michelle Obama smiled .",
        names: &[("Barack Obama", "PERSON"), ("Obama", "PERSON"), ("Michelle Obama", "PERSON")],
        gold: &[0, 0, 1],
        chains: &[(0, 1)],
    },
    Case {
        text: "Apple Inc. sued Samsung . Apple said Samsung copied .",
        names: &[("Apple Inc.", "ORG"), ("Samsung", "ORG"), ("Apple", "ORG"), ("Samsung", "ORG")],
        gold: &[0, 1, 0, 1],
        chains: &[(0, 2)],
    },
    Case {
        text: "Michael Jordan visited Jordan . Jordan loved Jordan .",
        names: &[("Michael Jordan", "PERSON"), ("Jordan", "GPE"), ("Jordan", "PERSON"), ("Jordan", "GPE")],
        gold: &[0, 1, 0, 1],
        chains: &[],
    },
    Case {
        text: "New York City is big . NYC never sleeps . New York is NYC .",
        names: &[("New York City", "GPE"), ("NYC", "GPE"), ("New York", "GPE"), ("NYC", "GPE")],
        gold: &[0, 0, 0, 0],
        chains: &[(0, 1), (1, 2)],
    },
    Case {
        text: "Paris Hilton visited Paris . Paris was sunny . Hilton left .",
        names: &[("Paris Hilton", "PERSON"), ("Paris", "GPE"), ("Paris", "GPE"), ("Hilton", "PERSON")],
        gold: &[0, 1, 1, 0],
        chains: &[],
    },
    Case {
        text: "Washington crossed the river . Washington is the capital .",
        names: &[("Washington", "GPE"), ("Washington", "GPE")],
        gold: &[0, 1],
        chains: &[],
    },
];

/// The predicted entity of each name row.
fn predict(case: &Case, config: ConsolidatorConfig, coref: bool) -> Vec<usize> {
    let mut doc = doc_with_names(case.text, case.names);
    if coref && !case.chains.is_empty() {
        let chains: Vec<Vec<Ix<Entities>>> = case.chains.iter().map(|&(a, b)| vec![Ix::new(a), Ix::new(b)]).collect();
        doc.add(&Key::<Groups<Entities>>::DEFAULT, Groups::new(chains), Provenance::by("test:coref")).unwrap();
    }
    let graph = EntityConsolidator::new().with_config(config).consolidate(&doc).unwrap();
    let mut of = vec![usize::MAX; case.names.len()];
    for (i, e) in graph.entities.iter().enumerate() {
        for row in e.mentions.iter().filter_map(|m| m.entity) {
            of[row.get()] = i;
        }
    }
    of
}

/// Pairwise (true positive, predicted, gold) counts.
fn pairs(case: &Case, predicted: &[usize]) -> (usize, usize, usize) {
    let (mut tp, mut p, mut g) = (0, 0, 0);
    for i in 0..predicted.len() {
        for j in i + 1..predicted.len() {
            let same_p = predicted[i] == predicted[j];
            let same_g = case.gold[i] == case.gold[j];
            tp += usize::from(same_p && same_g);
            p += usize::from(same_p);
            g += usize::from(same_g);
        }
    }
    (tp, p, g)
}

fn score(config: ConsolidatorConfig, coref: bool) -> (f64, f64, f64) {
    let (mut tp, mut p, mut g) = (0, 0, 0);
    for case in CASES {
        let (a, b, c) = pairs(case, &predict(case, config, coref));
        tp += a;
        p += b;
        g += c;
    }
    let precision = if p == 0 { 1.0 } else { tp as f64 / p as f64 };
    let recall = tp as f64 / g as f64;
    (precision, recall, 2.0 * precision * recall / (precision + recall).max(f64::MIN_POSITIVE))
}

#[test]
fn consolidation_clusters_names_better_with_coref_and_name_merges() {
    let off = ConsolidatorConfig { surface_merge: false, alias_merge: false };
    let on = ConsolidatorConfig::default();
    let runs = [
        ("no merges", score(off, false)),
        ("coref chains only", score(off, true)),
        ("name merges only", score(on, false)),
        ("default (both)", score(on, true)),
    ];
    for (name, (p, r, f)) in runs {
        println!("{name:>18}: P {p:.3} R {r:.3} F1 {f:.3}");
    }
    let best = runs[3].1;
    assert!(runs.iter().all(|(_, s)| s.2 <= best.2 + 1e-12));
    assert!(best.2 >= 0.9, "default F1 {:.3}", best.2);
    assert!(best.0 < 1.0, "the tagger error is a false merge the eval should show");
}
