//! The annotator as an analyzer step: it reads the document layers and
//! writes Mentions, Groups and Edges under the default keys.

mod common;

use std::slice;

use common::Fixture;
use turbo_analysis::{Analyzer, Annotator, Edges, Groups, Key};
use turbo_analysis_layers::{Entities, PosTags, Sentences, Tokens};
use turbo_coref::{CorefAnnotator, Mentions, Sieve, sieve_tags};

fn story() -> Fixture {
    Fixture::new()
        .sentence(&["Mary/NNP", "Jones/NNP", "founded/VBD", "Acme/NNP", "./."])
        .sentence(&["She/PRP", "praised/VBD", "Mary/NNP", "Jones/NNP", "./."])
        .speaker(1, "narrator")
        .entity("Mary Jones", "person")
        .entity("Acme", "organization")
        .entity("Mary Jones", "person")
}

fn analyzer() -> Analyzer {
    Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .add(CorefAnnotator::default())
        .build()
        .unwrap()
}

#[test]
fn analyzer_writes_mentions_groups_and_edges() {
    let built = story().build();
    let text = built.text.clone();
    let mut doc = built.document();
    analyzer().run(slice::from_mut(&mut doc)).remove(0).unwrap();
    let mentions = doc.get::<Mentions>().expect("mentions");
    let groups = doc.get::<Groups<Mentions>>().expect("groups");
    let edges = doc.get::<Edges<Mentions, Mentions>>().expect("edges");
    assert_eq!(common::forms(&text, mentions), ["Mary Jones", "Acme", "She", "Mary Jones"]);
    let chains: Vec<Vec<usize>> = groups.iter().map(|(_, g)| g.iter().map(|m| m.get()).collect()).collect();
    assert_eq!(chains, [vec![0, 2, 3], vec![1]]);
    // Every link joins two chains: links = mentions - chains.
    assert_eq!(edges.from.len(), mentions.spans.len() - groups.end.len());
    let tags = sieve_tags();
    for r in 0..edges.from.len() {
        assert!(edges.from[r].get() > edges.to[r].get(), "anaphor follows antecedent");
        assert_eq!(edges.score[r], 1.0);
        let label = tags.label(edges.kind[r]);
        assert!(Sieve::ALL.iter().any(|s| s.label() == label), "{label}");
    }
}

#[test]
fn rule_annotator_has_no_model_and_the_same_output_as_resolve() {
    assert!(Annotator::model(&CorefAnnotator::default()).is_none());
    assert_eq!(<CorefAnnotator as Annotator>::ID, "turbo-coref:coref");
    let built = story().build();
    let (direct, _, _) = CorefAnnotator::default().resolve(&built.input()).unwrap();
    let mut doc = story().build().document();
    analyzer().run(slice::from_mut(&mut doc)).remove(0).unwrap();
    let mentions = doc.get::<Mentions>().unwrap();
    let spans = |m: &Mentions| m.spans.iter().map(|(_, s)| s).collect::<Vec<_>>();
    assert_eq!(spans(mentions), spans(&direct));
}
