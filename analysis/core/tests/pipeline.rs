//! A whole pipeline of toy annotators: the point is the API's mechanics
//! (types, offsets, ordering, batching, failures), not the linguistics.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_analysis::*;
use turbo_analysis_layers::{Confidence, Language, Lemmas, PosTags, Sentences, Strs, Tag, TagSet, Tokens, Vectors};

/// Folds runs of whitespace to one space and expands the "fi" ligature,
/// writing the view "folded" with its alignment to the source.
struct Fold;

impl Annotator for Fold {
    const ID: &'static str = "test:fold";
    type Reads = (View,);
    type Writes = (View,);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View,), Error> {
        let text = src.text();
        let mut b = ViewBuilder::new(text);
        let mut chars = text.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            if c == '\u{FB01}' {
                b.keep(i);
                b.replace(i + c.len_utf8(), "fi");
            } else if c.is_whitespace() {
                let mut end = i + c.len_utf8();
                while let Some(&(j, d)) = chars.peek() {
                    if !d.is_whitespace() {
                        break;
                    }
                    end = j + d.len_utf8();
                    chars.next();
                }
                if &text[i..end] != " " {
                    b.keep(i);
                    b.replace(end, " ");
                }
            }
        }
        let (out, alignment) = b.finish();
        Ok((View::derived(src, out, alignment),))
    }
}

/// Splits after ". ".
struct Sentencer;

impl Annotator for Sentencer {
    const ID: &'static str = "test:sentences";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut start = 0;
        for (i, _) in text.match_indices(". ") {
            spans.push(Span::new(start as u32, i as u32 + 1));
            start = i + 2;
        }
        if start < text.len() {
            spans.push(Span::new(start as u32, text.len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

/// Whitespace tokens, per sentence.
struct Tokenizer;

impl Annotator for Tokenizer {
    const ID: &'static str = "test:tokens";
    type Reads = (View, Sentences);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        if view.text().contains("FAIL") {
            return Err(Error::Annotator { annotator: Self::ID, detail: "asked to fail".into() });
        }
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut sentence = Vec::new();
        for (s, span) in sentences.spans.iter() {
            let mut at = span.start as usize;
            for word in text[span.range()].split(' ') {
                if !word.is_empty() {
                    spans.push(Span::new(at as u32, (at + word.len()) as u32));
                    sentence.push(s);
                }
                at += word.len() + 1;
            }
        }
        Ok((Tokens { spans, sentence: Some(sentence) },))
    }
}

/// Tags words ending in "s" as VBZ, the rest X.
struct Tagger {
    tagset: Arc<TagSet>,
}

impl Annotator for Tagger {
    const ID: &'static str = "test:tagger";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        let (vbz, x) = (self.tagset.tag("VBZ").unwrap(), self.tagset.tag("X").unwrap());
        let text = view.text();
        let tags: Vec<Tag> =
            tokens.spans.iter().map(|(_, s)| if text[s.range()].ends_with('s') { vbz } else { x }).collect();
        Ok((PosTags { tagset: self.tagset.clone(), tags, prob: None },))
    }
}

/// An embedder that takes the whole batch's sentences in one call, as a
/// device run would, and counts its calls.
struct Embedder {
    calls: Arc<AtomicUsize>,
}

impl Annotator for Embedder {
    const ID: &'static str = "test:embedder";
    type Reads = (View, Sentences);
    type Writes = (Vectors<Sentences>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<(Vectors<Sentences>,), Error> {
        self.annotate_batch(vec![input]).pop().unwrap()
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<(Vectors<Sentences>,), Error>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        // Pack every sentence of every document into one run, then split.
        let packed: Vec<&str> =
            inputs.iter().flat_map(|(v, s)| s.spans.iter().map(|(_, sp)| &v.text()[sp.range()])).collect();
        let vectors: Vec<[f32; 2]> = packed.iter().map(|t| [t.len() as f32, t.split(' ').count() as f32]).collect();
        let mut at = 0;
        inputs
            .iter()
            .map(|(_, s)| {
                let n = s.spans.len();
                let data = vectors[at..at + n].iter().flatten().copied().collect();
                at += n;
                Ok((Vectors::new(2, data, toy_model()),))
            })
            .collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Some(toy_model())
    }

    fn execution(&self) -> Option<Execution> {
        Some(Execution { backend: "cpu".into(), device: "test".into(), precision: "f32".into() })
    }
}

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// Declines a document it finds no language evidence in, rather than
/// guessing a language.
struct Detect;

impl Annotator for Detect {
    const ID: &'static str = "test:language";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Language,), Error> {
        if !view.text().chars().any(char::is_alphabetic) {
            return Err(Error::Skipped("no language evidence".into()));
        }
        Ok((Language { code: "en".into(), score: 0.9 },))
    }
}

/// Strips a plural "s" from English words. It needs the language, so a
/// document whose language was not found gets no lemmas.
struct Stemmer;

impl Annotator for Stemmer {
    const ID: &'static str = "test:stemmer";
    type Reads = (View, Language, Tokens);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, language, tokens): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        let text = view.text();
        let mut lemma = Strs::default();
        for (_, s) in tokens.spans.iter() {
            let word = &text[s.range()];
            lemma.push(if language.code == "en" { word.strip_suffix('s').unwrap_or(word) } else { word });
        }
        Ok((Lemmas { lemma },))
    }
}

/// Whitespace tokens: per sentence when the document has sentences, over
/// the whole text when it does not.
struct LooseTokenizer;

impl Annotator for LooseTokenizer {
    const ID: &'static str = "test:loose-tokens";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let text = view.text();
        let ranges: Vec<(Option<Ix<Sentences>>, Span)> = match sentences {
            Some(s) => s.spans.iter().map(|(i, span)| (Some(i), span)).collect(),
            None => vec![(None, Span::new(0, text.len() as u32))],
        };
        let mut spans = Spans::new(view.id());
        let mut sentence = Vec::new();
        for (s, range) in ranges {
            let mut at = range.start as usize;
            for word in text[range.range()].split(' ') {
                if !word.is_empty() {
                    spans.push(Span::new(at as u32, (at + word.len()) as u32));
                    sentence.extend(s);
                }
                at += word.len() + 1;
            }
        }
        Ok((Tokens { spans, sentence: sentences.map(|_| sentence) },))
    }
}

/// Lowercases each word, and strips the "s" of one tagged VBZ: it reads
/// the tags and the tokens they are for.
struct Lemmatizer;

impl Annotator for Lemmatizer {
    const ID: &'static str = "test:lemmatizer";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        let mut lemma = Strs::default();
        for ((_, s), &t) in tokens.spans.iter().zip(&tags.tags) {
            let word = view.text()[s.range()].to_lowercase();
            lemma.push(if tags.tagset.label(t) == "VBZ" { word.strip_suffix('s').unwrap_or(&word) } else { &word });
        }
        Ok((Lemmas { lemma },))
    }
}

/// Panics on a text it cannot handle. Run over a batch, the panic takes
/// every document of the batch down with it.
struct Fragile;

impl Annotator for Fragile {
    const ID: &'static str = "test:fragile";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Language,), Error> {
        assert!(!view.text().contains("PANIC"), "cannot handle this one");
        Ok((Language { code: "en".into(), score: 1.0 },))
    }
}

const FOLDED: Key<View> = Key::named("folded");

fn pipeline(calls: Arc<AtomicUsize>) -> Analyzer {
    // Added out of order on purpose: the analyzer orders by data.
    Analyzer::builder()
        .step(Bound::new(Tagger { tagset: TagSet::new("toy", &["X", "VBZ"]) }).read(0, "folded"))
        .step(Bound::new(Embedder { calls }).read(0, "folded"))
        .step(Bound::new(Tokenizer).read(0, "folded"))
        .step(Bound::new(Sentencer).read(0, "folded"))
        .step(Bound::new(Fold).write(0, "folded"))
        .build()
        .unwrap()
}

#[test]
fn a_pipeline_orders_itself_and_runs_stages_side_by_side() {
    let a = pipeline(Arc::default());
    assert_eq!(
        a.plan(),
        vec![vec!["test:fold"], vec!["test:sentences"], vec!["test:embedder", "test:tokens"], vec!["test:tagger"]]
    );
}

#[test]
fn tokens_of_a_normalized_view_land_on_exact_source_bytes() {
    let mut docs = vec![Document::new("The \u{FB01}nal   report  ships today. It runs")];
    let status = pipeline(Arc::default()).run(&mut docs);
    assert!(status.iter().all(|s| s.is_ok()), "{status:?}");
    let doc = &docs[0];
    assert_eq!(doc.get_key(&FOLDED).unwrap().text(), "The final report ships today. It runs");

    let tokens = doc.get::<Tokens>().unwrap();
    let view = tokens.spans.view();
    let words: Vec<&str> = tokens.spans.iter().map(|(_, s)| &doc.source()[doc.to_source(view, s).range()]).collect();
    assert_eq!(words, ["The", "\u{FB01}nal", "report", "ships", "today.", "It", "runs"]);

    // Typed reads: the tags are PosTags, aligned to Tokens, and a token's
    // sentence is an Ix<Sentences>.
    let tags = doc.get::<PosTags>().unwrap();
    let labels: Vec<&str> = tags.tags.iter().map(|&t| tags.tagset.label(t)).collect();
    assert_eq!(labels, ["X", "X", "X", "VBZ", "X", "X", "VBZ"]);
    assert_eq!(tokens.of(Ix::new(1)), IxRange::new(5, 7));

    // Every layer says what made it and from what.
    let p = doc.provenance(Key::<PosTags>::DEFAULT.id()).unwrap();
    assert_eq!(p.annotator, "test:tagger");
    assert_eq!(p.reads, vec![FOLDED.id(), Key::<Tokens>::DEFAULT.id()]);
}

#[test]
fn a_batch_reaches_a_batching_annotator_in_one_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut docs: Vec<Document> = (0..64).map(|i| Document::new(format!("Doc {i} starts. It ends here."))).collect();
    let status = pipeline(calls.clone()).run(&mut docs);
    assert!(status.iter().all(|s| s.is_ok()));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let v = docs[7].get::<Vectors<Sentences>>().unwrap();
    assert_eq!(v.row(Ix::new(1)), &[13.0, 3.0]); // "It ends here."

    // The vectors name the layer they describe, the model that made them
    // and where it ran: an FP16 run is not an FP32 run.
    let key = Key::<Vectors<Sentences>>::DEFAULT.id();
    assert_eq!(key.to_string(), "opennlp:embeddings<opennlp:sentences>");
    let p = docs[7].provenance(key).unwrap();
    assert_eq!(p.model.as_ref().unwrap().id, "toy");
    assert_eq!(p.execution.as_ref().unwrap().precision, "f32");
}

#[test]
fn one_document_failing_leaves_the_rest() {
    let mut docs = vec![Document::new("Fine. Fine."), Document::new("FAIL here."), Document::new("Also fine.")];
    let status = pipeline(Arc::default()).run(&mut docs);
    assert!(status[0].is_ok() && status[2].is_ok());
    let failed = Error::Annotator { annotator: "test:tokens", detail: "asked to fail".into() };
    assert_eq!(status[1], Err(failed.clone()));
    assert!(docs[2].get::<PosTags>().is_some());

    // The failed layer says so, what needs it is skipped and says why, and
    // the step that did not need it still ran.
    let d = &docs[1];
    assert_eq!(d.status(Key::<Tokens>::DEFAULT.id()), Some(&Status::Failed(failed)));
    let why = "opennlp:tokens failed: test:tokens: asked to fail";
    assert_eq!(d.status(Key::<PosTags>::DEFAULT.id()), Some(&Status::Skipped(why.into())));
    assert!(d.get::<Vectors<Sentences>>().is_some());
}

#[test]
fn a_panic_fails_only_the_document_that_caused_it() {
    let a = Analyzer::builder().add(Fragile).add(Sentencer).build().unwrap();
    let mut docs = vec![Document::new("Fine."), Document::new("PANIC now."), Document::new("Fine too.")];
    let status = a.run(&mut docs);
    assert!(status[0].is_ok() && status[2].is_ok(), "{status:?}");
    let panicked = Error::Annotator { annotator: "test:fragile", detail: "panicked: cannot handle this one".into() };
    assert_eq!(status[1], Err(panicked.clone()));
    assert_eq!(docs[1].status(Key::<Language>::DEFAULT.id()), Some(&Status::Failed(panicked)));
    // The batch was run again one document at a time, so the others have
    // their layer, and the step beside it in its stage ran on all three.
    assert!(docs[0].get::<Language>().is_some() && docs[2].get::<Language>().is_some());
    assert!(docs.iter().all(|d| d.get::<Sentences>().is_some()));
}

#[test]
fn a_broken_pipeline_is_refused_when_it_is_built() {
    let no_sentences = Analyzer::builder().step(Bound::new(Tokenizer)).build();
    assert_eq!(
        no_sentences.err().unwrap(),
        Error::Pipeline("test:tokens reads opennlp:sentences, which nothing writes".into())
    );
    let twice = Analyzer::builder().add(Sentencer).add(Sentencer).build();
    assert!(matches!(twice, Err(Error::Pipeline(m)) if m.contains("both write opennlp:sentences")));
}

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
    assert_eq!(e.to_string(), "opennlp:pos was made on opennlp:tokens@gold, not on the opennlp:tokens the step reads");
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
        const ID: &'static str = "opennlp:sentences";
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

    // On the wire both would be opennlp:sentences, so the second is refused,
    // under any instance name.
    let mut doc = Document::new("Dogs bark.");
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 10));
    doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("t")).unwrap();
    let e = doc.add(&Key::named("other"), OtherSentences, Provenance::by("t")).unwrap_err();
    assert_eq!(e.to_string(), "opennlp:sentences@other already names another layer type");

    // And a pipeline holding both is refused when it is built.
    let both = Analyzer::builder().add(Sentencer).step(Bound::new(OtherSentencer).write(0, "other")).build();
    assert_eq!(both.err().unwrap(), Error::Pipeline("two layer types are both opennlp:sentences".into()));
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
    assert_eq!(e.to_string(), "opennlp:tokens: an index past the end of opennlp:sentences, which has 1");
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
    let why = Status::Skipped("opennlp:language was skipped: no language evidence".into());
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
    assert_eq!(e.to_string(), "turbo:confidence<opennlp:tokens>: a confidence between 0 and 1");
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
