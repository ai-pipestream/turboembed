//! `CorefAnnotator::config_hash` changes with every setting that changes
//! the output, and only with those.

use turbo_analysis::Annotator;
use turbo_coref::CorefAnnotator;

fn hash(a: &CorefAnnotator) -> u64 {
    a.config_hash()
}

fn rules(person: &[&str], neutral: &[&str]) -> CorefAnnotator {
    CorefAnnotator::rules(person, neutral).unwrap()
}

#[test]
fn entity_types_change_the_hash() {
    let a = hash(&rules(&["person"], &["organization"]));
    let b = hash(&rules(&["animal"], &["location"]));
    assert_ne!(a, 0);
    assert_ne!(a, b);
    // Swapping the classes is a different configuration.
    assert_ne!(hash(&rules(&["person"], &["organization"])), hash(&rules(&["organization"], &["person"])));
    // Order, case and repeats do not change the matching, nor the hash.
    assert_eq!(hash(&rules(&["person", "agent"], &["org"])), hash(&rules(&["AGENT", "Person", "person"], &["ORG"])));
    assert_eq!(hash(&CorefAnnotator::default()), hash(&rules(&["person"], &["organization", "location"])));
}

#[test]
fn threshold_changes_the_hash() {
    let a = CorefAnnotator::new(&["person"], &["org"], None, 0.1, None, None).unwrap();
    let b = CorefAnnotator::new(&["person"], &["org"], None, 0.5, None, None).unwrap();
    assert_ne!(hash(&a), hash(&b));
}

#[test]
fn word_vectors_change_the_hash_by_content() {
    use std::sync::Arc;
    use turbo_coref::{DEFAULT_THRESHOLD, MapWordVectors, WordVectors};
    let with = |v: MapWordVectors| {
        let v: Arc<dyn WordVectors> = Arc::new(v);
        hash(&CorefAnnotator::new(&["person"], &["org"], None, DEFAULT_THRESHOLD, Some(v), None).unwrap())
    };
    let plain = hash(&CorefAnnotator::new(&["person"], &["org"], None, DEFAULT_THRESHOLD, None, None).unwrap());
    let mut a = MapWordVectors::new(2);
    a.insert("firm", vec![1.0, 0.0]).unwrap();
    a.insert("acme", vec![0.0, 1.0]).unwrap();
    let mut b = MapWordVectors::new(2);
    b.insert("acme", vec![0.0, 1.0]).unwrap();
    b.insert("firm", vec![1.0, 0.0]).unwrap();
    let mut c = MapWordVectors::new(2);
    c.insert("firm", vec![1.0, 0.5]).unwrap();
    c.insert("acme", vec![0.0, 1.0]).unwrap();
    assert_eq!(with(a.clone()), with(b), "insertion order does not matter");
    assert_ne!(with(a.clone()), with(c));
    assert_ne!(with(a), plain);
}

#[test]
fn a_model_changes_the_hash() {
    use std::sync::Arc;
    use turbo_coref::train::{DEFAULT_EPOCHS, DEFAULT_L2, DEFAULT_LEARNING_RATE, GoldMention, TrainingDocument};
    use turbo_coref::{CorefInput, DEFAULT_THRESHOLD};
    mod_fixture::with_input(|input: CorefInput<'_>, gold: Vec<GoldMention>| {
        let docs = [TrainingDocument { input, gold }];
        let train = |epochs| {
            turbo_coref::train::train_ranking(
                "eng",
                &docs,
                epochs,
                DEFAULT_LEARNING_RATE,
                DEFAULT_L2,
                &CorefAnnotator::default(),
            )
            .unwrap()
        };
        let plain = hash(&CorefAnnotator::default());
        let one = hash(&CorefAnnotator::with_model(train(1)).unwrap());
        let many = hash(&CorefAnnotator::with_model(train(DEFAULT_EPOCHS)).unwrap());
        assert_ne!(plain, one);
        assert_ne!(one, many);
        let m = Arc::new(train(1));
        let a = CorefAnnotator::new(&["person"], &["org"], Some(m.clone()), DEFAULT_THRESHOLD, None, None).unwrap();
        let b = CorefAnnotator::new(&["person"], &["org"], Some(m), DEFAULT_THRESHOLD, None, None).unwrap();
        assert_eq!(hash(&a), hash(&b), "the same model hashes the same");
    });
}

mod mod_fixture {
    use turbo_analysis::{IxRange, Span, Spans, Tag, TagSet, ViewId};
    use turbo_analysis_layers::{Entities, PosTags, Sentences, Tokens};
    use turbo_coref::CorefInput;
    use turbo_coref::train::GoldMention;

    fn spans<L>(items: &[(u32, u32)]) -> Spans<L> {
        let mut s = Spans::new(ViewId::SOURCE);
        items.iter().for_each(|&(a, b)| {
            s.push(Span::new(a, b));
        });
        s
    }

    /// "Acme grew . It won ." with Acme and It in one gold chain.
    pub fn with_input(f: impl FnOnce(CorefInput<'_>, Vec<GoldMention>)) {
        let text = "Acme grew . It won .";
        let tokens = [(0, 4), (5, 9), (10, 11), (12, 14), (15, 18), (19, 20)];
        let sentences = Sentences { spans: spans(&[(0, 11), (12, 20)]) };
        let tokens_layer = Tokens { spans: spans(&tokens), sentence: None };
        let tagset = TagSet::new("penn", &["NNP", "VBD", ".", "PRP"]);
        let tags = PosTags { tagset, tags: [0, 1, 2, 3, 1, 2].map(Tag).to_vec(), prob: None };
        let entities = Entities {
            spans: spans(&[(0, 4)]),
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("types", &["organization"]),
            score: vec![1.0],
        };
        let input = CorefInput {
            text,
            view: ViewId::SOURCE,
            sentences: &sentences,
            tokens: &tokens_layer,
            tags: &tags,
            entities: &entities,
            chunks: None,
            phrases: None,
            speakers: None,
        };
        let gold =
            vec![GoldMention { span: Span::new(0, 4), chain: 0 }, GoldMention { span: Span::new(12, 14), chain: 0 }];
        f(input, gold);
    }
}
