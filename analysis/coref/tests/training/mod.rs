//! Training fixtures: the synthetic corpus of OpenNLP's `CorefTrainerTest`.

#![allow(dead_code)]

use std::sync::Arc;

use crate::common::{Built, Fixture};
use turbo_analysis::Span;
use turbo_coref::TokenVectors;
use turbo_coref::train::{GoldMention, TrainingDocument};
use turbo_train::TrainingParameters;

/// A fixture with its gold mentions, marked by text in order.
pub struct Gold {
    pub fixture: Fixture,
    pub gold: Vec<GoldMention>,
}

impl Gold {
    pub fn new(fixture: Fixture, marks: &[(&str, usize)]) -> Gold {
        let mut from = 0;
        let mut gold = Vec::new();
        for &(phrase, chain) in marks {
            let start = fixture.text[from..].find(phrase).unwrap() + from;
            let end = start + phrase.len();
            gold.push(GoldMention { span: Span::new(start as u32, end as u32), chain });
            from = start + 1;
        }
        Gold { fixture, gold }
    }
}

/// Built fixtures that outlive the training documents borrowing them.
pub struct Corpus {
    built: Vec<(Built, Vec<GoldMention>)>,
}

impl Corpus {
    pub fn new(docs: Vec<Gold>) -> Corpus {
        Corpus { built: docs.into_iter().map(|g| (g.fixture.build(), g.gold)).collect() }
    }

    pub fn documents(&self) -> Vec<TrainingDocument<'_>> {
        self.built.iter().map(|(b, g)| TrainingDocument { input: b.input(), gold: g.clone() }).collect()
    }
}

/// "Name verb. Pronoun grew." with the pronoun in gold chain `pronoun_chain`.
pub fn document(name: &str, ty: &str, verb: &str, pronoun: &str, pronoun_chain: usize) -> Gold {
    let f = Fixture::new()
        .sentence(&[&format!("{name}/NNP"), &format!("{verb}/VBD"), "./."])
        .sentence(&[&format!("{pronoun}/PRP"), "grew/VBD", "./."])
        .entity(name, ty);
    Gold::new(f, &[(name, 0), (pronoun, pronoun_chain)])
}

/// "It" follows an organization and "He" a person; the other class
/// stays apart.
pub fn corpus() -> Corpus {
    let orgs = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay"];
    let people = ["Kowalczyk", "Nowak", "Okafor", "Tanaka", "Ivanov", "Schmidt"];
    let mut docs = Vec::new();
    for (o, p) in orgs.iter().zip(people) {
        docs.push(document(o, "organization", "expanded", "It", 0));
        docs.push(document(p, "person", "arrived", "He", 0));
        docs.push(document(o, "organization", "expanded", "He", 1));
        docs.push(document(p, "person", "arrived", "It", 1));
    }
    Corpus::new(docs)
}

pub fn parameters() -> TrainingParameters {
    let mut p = TrainingParameters::defaults();
    p.put("Iterations", 50);
    p.put("Cutoff", 1);
    p
}

/// "Acme expanded. The firm grew." with chunks and one gold chain.
pub fn firm() -> Gold {
    let f = Fixture::new()
        .sentence(&["Acme/NNP", "expanded/VBD", "./."])
        .sentence(&["The/DT", "firm/NN", "grew/VBD", "./."])
        .entity("Acme", "organization")
        .chunk("Acme", "NP")
        .chunk("The firm", "NP");
    Gold::new(f, &[("Acme", 0), ("The firm", 0)])
}

/// A test encoder from a per-token function.
pub struct Encoder<F> {
    pub dimension: usize,
    pub f: F,
}

impl<F: Fn(&[&str]) -> Vec<Vec<f32>> + Send + Sync> TokenVectors for Encoder<F> {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn vectors(&self, sentence: &[&str]) -> Vec<Vec<f32>> {
        (self.f)(sentence)
    }
}

/// Organization names and "It" point one way, other capitalized words
/// and "He" the other, everything else between.
pub fn encoder() -> Arc<dyn TokenVectors> {
    const ORGS: [&str; 9] = ["It", "Acme", "firm", "Cyberdyne", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay"];
    Arc::new(Encoder {
        dimension: 2,
        f: |tokens: &[&str]| {
            tokens
                .iter()
                .map(|w| {
                    if ORGS.contains(w) {
                        vec![1.0, 0.0]
                    } else if *w == "He" || w.chars().next().is_some_and(char::is_uppercase) {
                        vec![0.0, 1.0]
                    } else {
                        vec![0.5, 0.5]
                    }
                })
                .collect()
        },
    })
}

pub fn names(event: &turbo_train::Event) -> Vec<String> {
    event.context().iter().map(|c| c.to_string_lossy()).collect()
}
