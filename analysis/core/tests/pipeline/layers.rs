//! Layers in a document: gold beside predicted, instances, ids, the checks
//! on adding one, and string columns.

use super::*;

#[test]
fn gold_and_predicted_layers_live_side_by_side() {
    const GOLD: Key<Tokens> = Key::named("gold");
    let mut doc = Document::new("Dogs bark.");
    let mut gold = Spans::new(ViewId::SOURCE);
    gold.push(Span::new(0, 4));
    gold.push(Span::new(5, 9));
    gold.push(Span::new(9, 10));
    doc.add(&GOLD, Tokens { spans: gold, sentence: None }, Provenance::by("corpus:gold")).unwrap();

    // The tagger reads the gold tokens instead of predicted ones.
    let a = Analyzer::builder()
        .given(GOLD)
        .step(Bound::new(Tagger { tagset: TagSet::new("toy", &["X", "VBZ"]) }).read(1, "gold"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert!(a.run(&mut docs)[0].is_ok());
    let tags = docs[0].get::<PosTags>().unwrap();
    assert_eq!(tags.tags.len(), 3);
    assert!(docs[0].get::<Tokens>().is_none());
    let pos = Key::<PosTags>::DEFAULT.id();
    assert_eq!(docs[0].provenance(pos).unwrap().reads[1], GOLD.id());
    // The tags keep which tokens they are for.
    assert_eq!(docs[0].made_on(pos), [GOLD.id()]);
}

#[test]
fn a_layer_is_read_only_beside_the_instance_it_was_made_on() {
    const GOLD: Key<Tokens> = Key::named("gold");
    let words = |spans: &[(u32, u32)]| {
        let mut s = Spans::new(ViewId::SOURCE);
        for &(a, b) in spans {
            s.push(Span::new(a, b));
        }
        Tokens { spans: s, sentence: None }
    };
    let mut doc = Document::new("Dogs bark.");
    doc.add(&GOLD, words(&[(0, 4), (5, 9), (9, 10)]), Provenance::by("corpus:gold")).unwrap();
    doc.add(&Key::DEFAULT, words(&[(0, 4), (5, 10)]), Provenance::by("corpus:predicted")).unwrap();
    let pipeline = |lemmatizer: Bound<Lemmatizer>| {
        Analyzer::builder()
            .given(GOLD)
            .given(Key::<Tokens>::DEFAULT)
            .step(Bound::new(Tagger { tagset: TagSet::new("toy", &["X", "VBZ"]) }).read(1, "gold"))
            .step(lemmatizer)
            .build()
            .unwrap()
    };

    // Tags made on the gold tokens, read beside the predicted ones: item i
    // of the one is not about item i of the other.
    let mut docs = vec![doc.clone()];
    let e = pipeline(Bound::new(Lemmatizer)).run(&mut docs).remove(0).unwrap_err();
    assert_eq!(e.to_string(), "turbo:pos was made on turbo:tokens@gold, not on the turbo:tokens the step reads");
    assert_eq!(docs[0].status(Key::<Lemmas>::DEFAULT.id()), Some(&Status::Failed(e)));

    // Beside the gold tokens they line up.
    let mut docs = vec![doc];
    assert!(pipeline(Bound::new(Lemmatizer).read(1, "gold")).run(&mut docs)[0].is_ok());
    let lemmas = docs[0].get::<Lemmas>().unwrap();
    assert_eq!(lemmas.lemma.iter().collect::<Vec<_>>(), ["dog", "bark", "."]);
}

#[test]
fn one_id_names_one_layer_type() {
    /// Another crate's idea of sentences, under the shared id.
    struct OtherSentences;
    impl Layer for OtherSentences {
        const ID: &'static str = "turbo:sentences";
        fn anchor() -> Anchor {
            Anchor::Document
        }
        fn len(&self) -> usize {
            1
        }
    }
    struct OtherSentencer;
    impl Annotator for OtherSentencer {
        const ID: &'static str = "test:other-sentences";
        type Reads = (View,);
        type Writes = (OtherSentences,);
        fn annotate(&self, _: Refs<'_, Self>) -> Result<(OtherSentences,), Error> {
            Ok((OtherSentences,))
        }
    }

    // On the wire both would be turbo:sentences, so the second is refused,
    // under any instance name.
    let mut doc = Document::new("Dogs bark.");
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 10));
    doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("t")).unwrap();
    let e = doc.add(&Key::named("other"), OtherSentences, Provenance::by("t")).unwrap_err();
    assert_eq!(e.to_string(), "turbo:sentences@other already names another layer type");

    // And a pipeline holding both is refused when it is built.
    let both = Analyzer::builder().add(Sentencer).step(Bound::new(OtherSentencer).write(0, "other")).build();
    assert_eq!(both.err().unwrap(), Error::Pipeline("two layer types are both turbo:sentences".into()));
}

#[test]
fn layers_are_checked_when_they_are_added() {
    let mut doc = Document::new("short");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 99));
    let e = doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("t")).unwrap_err();
    assert!(matches!(e, Error::Bounds { .. }));

    // "é" is two bytes, 3..5: a span ending at 4 would cut it.
    let mut cafe = Document::new("café au lait");
    let mut cut = Spans::new(ViewId::SOURCE);
    cut.push(Span::new(0, 4));
    let e = cafe.add(&Key::DEFAULT, Sentences { spans: cut }, Provenance::by("t")).unwrap_err();
    assert!(matches!(e, Error::Bounds { ref detail, .. } if detail == "a span 0..4 cuts a character"), "{e:?}");

    let mut ok = Spans::new(ViewId::SOURCE);
    ok.push(Span::new(0, 5));
    doc.add(&Key::DEFAULT, Tokens { spans: ok, sentence: None }, Provenance::by("t")).unwrap();
    let tags = PosTags { tagset: TagSet::new("toy", &["X"]), tags: vec![Tag(0), Tag(0)], prob: None };
    let e = doc.add(&Key::DEFAULT, tags, Provenance::by("t")).unwrap_err();
    assert!(matches!(e, Error::Misaligned { len: 2, anchor_len: 1, .. }));

    // An index into another layer must be inside it: a token's sentence is
    // one the document has.
    let mut doc = Document::new("Dogs bark.");
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 10));
    doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("t")).unwrap();
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(0, 4));
    let tokens = Tokens { spans: t, sentence: Some(vec![Ix::new(1)]) };
    let e = doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap_err();
    assert_eq!(e.to_string(), "turbo:tokens: an index past the end of turbo:sentences, which has 1");
}

#[test]
fn string_columns_hold_every_item_in_one_buffer() {
    let mut doc = Document::new("Dogs were barking");
    let mut spans = Spans::new(ViewId::SOURCE);
    for s in [Span::new(0, 4), Span::new(5, 9), Span::new(10, 17)] {
        spans.push(s);
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let mut lemma = Strs::default();
    for l in ["dog", "be", "bark"] {
        lemma.push(l);
    }
    doc.add(&Key::DEFAULT, Lemmas { lemma }, Provenance::by("t")).unwrap();
    let lemmas = doc.get::<Lemmas>().unwrap();
    assert_eq!(lemmas.lemma.iter().collect::<Vec<_>>(), ["dog", "be", "bark"]);
    assert_eq!(format!("{:?}", Ix::<Tokens>::new(2)), "Tokens#2");
}
