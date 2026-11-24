//! A pipeline assembled from configuration, declined and optional layers,
//! confidence columns, and merging the branches of a document.

use super::*;

#[test]
fn a_pipeline_is_assembled_from_configuration() {
    use turbo_analysis::annotate::Step;
    use turbo_analysis::registry::{Provider, Registry, Spec};

    struct Sentences;
    impl Provider<dyn Step> for Sentences {
        fn name(&self) -> &'static str {
            "sentences"
        }
        fn create(&self, _: &Spec) -> Result<Box<dyn Step>, Error> {
            Ok(Box::new(Bound::new(Sentencer)))
        }
    }
    /// A tokenizer offered twice: a fast one that needs a device this
    /// machine lacks, and the plain one.
    struct Tokens {
        priority: i32,
        available: bool,
    }
    impl Provider<dyn Step> for Tokens {
        fn name(&self) -> &'static str {
            "tokens"
        }
        fn priority(&self) -> i32 {
            self.priority
        }
        fn is_available(&self) -> bool {
            self.available
        }
        fn create(&self, _: &Spec) -> Result<Box<dyn Step>, Error> {
            Ok(Box::new(Bound::new(Tokenizer)))
        }
    }

    let mut r: Registry<dyn Step> = Registry::default();
    r.register(Sentences)
        .register(Tokens { priority: 10, available: false })
        .register(Tokens { priority: 0, available: true });
    let config = [("tokens", Spec::default()), ("sentences", Spec::default().with("language", "en"))];
    let a = r.pipeline(&config, |b| b).unwrap();
    assert_eq!(a.plan(), vec![vec!["test:sentences"], vec!["test:tokens"]]);

    // Two providers that tie are refused, not picked between.
    r.register(Tokens { priority: 0, available: true });
    let tie = r.pipeline(&config, |b| b).err().unwrap();
    assert_eq!(tie, Error::Pipeline("two providers of tokens tie at priority 0".into()));
}

#[test]
fn a_declined_layer_is_recorded_and_what_needs_it_is_skipped_too() {
    let a = Analyzer::builder().add(Stemmer).add(Tokenizer).add(Sentencer).add(Detect).build().unwrap();
    let mut docs = vec![Document::new("Dogs bark. Cats purr."), Document::new("?? 42. 17 !!")];
    let status = a.run(&mut docs);
    // Declining is not failing: both documents come through.
    assert!(status.iter().all(|s| s.is_ok()), "{status:?}");
    let lemmas = docs[0].get::<Lemmas>().unwrap();
    assert_eq!(lemmas.lemma.iter().collect::<Vec<_>>(), ["Dog", "bark.", "Cat", "purr."]);

    let d = &docs[1];
    let language = Key::<Language>::DEFAULT.id();
    assert_eq!(d.status(language), Some(&Status::Skipped("no language evidence".into())));
    assert!(d.get::<Language>().is_none());
    // The stemmer needed it, so it was skipped too, and says why.
    let why = Status::Skipped("turbo:language was skipped: no language evidence".into());
    assert_eq!(d.status(Key::<Lemmas>::DEFAULT.id()), Some(&why));
    // Skipped is neither empty nor missing: the tokens are there, and a
    // layer nothing wrote has no status at all.
    assert_eq!(d.get::<Tokens>().unwrap().len(), 4);
    assert_eq!(d.status(Key::<PosTags>::DEFAULT.id()), None);
}

#[test]
fn an_optional_input_is_used_when_it_is_there_and_recorded_only_then() {
    let (view, sentences, tokens) =
        (Key::<View>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id());

    // Alone, it tokenizes the whole text and says it read only the text.
    let alone = Analyzer::builder().add(LooseTokenizer).build().unwrap();
    let mut docs = vec![Document::new("Dogs bark. Cats purr.")];
    assert!(alone.run(&mut docs)[0].is_ok());
    let t = docs[0].get::<Tokens>().unwrap();
    assert_eq!((t.len(), t.sentence.is_none()), (4, true));
    assert_eq!(docs[0].provenance(tokens).unwrap().reads, [view]);

    // With a sentencer it runs after it, per sentence, and says so.
    let with = Analyzer::builder().add(LooseTokenizer).add(Sentencer).build().unwrap();
    assert_eq!(with.plan(), vec![vec!["test:sentences"], vec!["test:loose-tokens"]]);
    let mut docs = vec![Document::new("Dogs bark. Cats purr.")];
    assert!(with.run(&mut docs)[0].is_ok());
    assert_eq!(docs[0].get::<Tokens>().unwrap().of(Ix::new(1)), IxRange::new(2, 4));
    assert_eq!(docs[0].provenance(tokens).unwrap().reads, [view, sentences]);
}

#[test]
fn a_confidence_column_hangs_on_any_layer() {
    let mut doc = Document::new("Dogs bark");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 4));
    spans.push(Span::new(5, 9));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();

    let key = Key::<Confidence<Tokens>>::DEFAULT;
    let e = doc.add(&key, Confidence::new(vec![0.9, 1.5]), Provenance::by("t")).unwrap_err();
    assert_eq!(e.to_string(), "turbo:confidence<turbo:tokens>: a confidence between 0 and 1");
    doc.add(&key, Confidence::new(vec![0.9, 0.6]), Provenance::by("t")).unwrap();
    assert_eq!(doc.get_key(&key).unwrap().value, [0.9, 0.6]);
    // One per layer it describes: a sentence confidence is another layer.
    assert_ne!(key.id(), Key::<Confidence<Sentences>>::DEFAULT.id());
}

#[test]
fn branches_of_a_document_merge_unless_both_wrote_the_same_thing() {
    let base = Document::new("Dogs bark.");
    let sentences = || {
        let mut s = Spans::new(ViewId::SOURCE);
        s.push(Span::new(0, 10));
        Sentences { spans: s }
    };
    let tokens = || {
        let mut t = Spans::new(ViewId::SOURCE);
        t.push(Span::new(0, 4));
        t.push(Span::new(5, 10));
        Tokens { spans: t, sentence: None }
    };
    let (mut a, mut b) = (base.clone(), base.clone());
    a.add(&Key::DEFAULT, sentences(), Provenance::by("a")).unwrap();
    b.add(&Key::DEFAULT, tokens(), Provenance::by("b")).unwrap();
    a.merge(&b).unwrap();
    assert!(a.get::<Sentences>().is_some() && a.get::<Tokens>().is_some());
    // Merging again is a no-op: those are the same layers.
    a.merge(&b).unwrap();

    // A third branch that made its own tokens is refused, and changes nothing.
    let mut c = base.clone();
    c.add(&Key::DEFAULT, tokens(), Provenance::by("c")).unwrap();
    assert_eq!(a.merge(&c), Err(Error::Duplicate(Key::<Tokens>::DEFAULT.id())));
    assert_eq!(a.provenance(Key::<Tokens>::DEFAULT.id()).unwrap().annotator, "b");

    // Two branches that each made a view would both call it view 1.
    let fold = |d: &mut Document, name: &'static str| {
        let src = d.get::<View>().unwrap().clone();
        let (text, alignment) = ViewBuilder::new(src.text()).finish();
        d.add(&Key::named(name), View::derived(&src, text, alignment), Provenance::by("fold")).unwrap();
    };
    let (mut x, mut y) = (base.clone(), base);
    fold(&mut x, "x");
    fold(&mut y, "y");
    assert_eq!(x.merge(&y), Err(Error::Pipeline("both branches made a view".into())));
}
