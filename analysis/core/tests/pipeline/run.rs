//! Running the pipeline: its order, where tokens of a normalized view land
//! in the source, batches, and failures of one document or of the build.

use super::*;

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
fn a_token_inside_a_run_of_replaced_characters_lands_on_its_own_source_bytes() {
    // Arabic-Indic digits (2 bytes each) folded to ASCII one character at a
    // time, so each digit is its own edit.
    let mut doc = Document::new("Room \u{661}\u{662}\u{663} is free");
    let src = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(src.text());
    for (i, c) in src.text().char_indices() {
        if let Some(d) = ('\u{660}'..='\u{669}').contains(&c).then(|| c as u32 - 0x660) {
            b.keep(i);
            b.replace(i + c.len_utf8(), &d.to_string());
        }
    }
    let (text, alignment) = b.finish();
    assert_eq!(text, "Room 123 is free");
    let digits = Key::named("digits");
    doc.add(&digits, View::derived(&src, text, alignment), Provenance::by("test:digits")).unwrap();
    let view = doc.get_key(&digits).unwrap().id();

    // One token per digit, as a digit splitter would make them.
    let mut spans = Spans::new(view);
    for at in 5..8 {
        spans.push(Span::new(at, at + 1));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test:digit-tokens")).unwrap();

    let tokens = doc.get::<Tokens>().unwrap();
    let source: Vec<Span> = tokens.spans.iter().map(|(_, s)| doc.to_source(view, s)).collect();
    assert_eq!(source, [Span::new(5, 7), Span::new(7, 9), Span::new(9, 11)]);
    assert_eq!(&doc.source()[source[1].range()], "\u{662}");
    // And back: the middle source digit is the middle view digit alone.
    assert_eq!(doc.from_source(view, Span::new(7, 9)), Span::new(6, 7));
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
    assert_eq!(key.to_string(), "turbo:embeddings<turbo:sentences>");
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
    let why = "turbo:tokens failed: test:tokens: asked to fail";
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
        Error::Pipeline("test:tokens reads turbo:sentences, which nothing writes".into())
    );
    let twice = Analyzer::builder().add(Sentencer).add(Sentencer).build();
    assert!(matches!(twice, Err(Error::Pipeline(m)) if m.contains("both write turbo:sentences")));
}
