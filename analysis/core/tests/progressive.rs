use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Error, Key, ModelRef, Span, Spans, Status, Tag, TagSet, View, export,
};
use turbo_analysis_layers::{Language, Lemmas, PosTags, Sentences, Stems, Strs, Tokens, Vectors, export::codecs};

struct Sentencer;
impl Annotator for Sentencer {
    const ID: &'static str = "test:sentences";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): (&View,)) -> Result<(Sentences,), Error> {
        let mut spans = Spans::new(view.id());
        let text = view.text();
        if !text.is_empty() {
            spans.push(Span::new(0, text.len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

struct Tokenizer {
    fail_on_marker: bool,
}
impl Annotator for Tokenizer {
    const ID: &'static str = "test:tokens";
    type Reads = (View, Sentences);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sents): (&View, &Sentences)) -> Result<(Tokens,), Error> {
        if self.fail_on_marker && view.text().contains("FAIL_TOKENS") {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokenization failed".into() });
        }
        let mut spans = Spans::new(view.id());
        let mut sentence = Vec::new();
        for (s_ix, s_span) in sents.spans.iter() {
            let slice = &view.text()[s_span.range()];
            let mut offset = 0usize;
            for word in slice.split(' ') {
                if !word.is_empty() {
                    let start = s_span.start + offset as u32;
                    let end = start + word.len() as u32;
                    spans.push(Span::new(start, end));
                    sentence.push(s_ix);
                }
                offset += word.len() + 1;
            }
        }
        Ok((Tokens { spans, sentence: Some(sentence) },))
    }
}

struct Tagger;
impl Annotator for Tagger {
    const ID: &'static str = "test:pos";
    type Reads = (Tokens,);
    type Writes = (PosTags,);

    fn annotate(&self, (tokens,): (&Tokens,)) -> Result<(PosTags,), Error> {
        let tagset = TagSet::new("pos", &["NOUN"]);
        let tags = vec![Tag(0); tokens.spans.len()];
        Ok((PosTags { tagset, tags, prob: None },))
    }
}

struct Lemmatizer;
impl Annotator for Lemmatizer {
    const ID: &'static str = "test:lemmas";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, tokens, _pos): (&View, &Tokens, &PosTags)) -> Result<(Lemmas,), Error> {
        let mut lemma = Strs::default();
        for (_, span) in tokens.spans.iter() {
            lemma.push(&view.text()[span.range()].to_ascii_lowercase());
        }
        Ok((Lemmas { lemma },))
    }
}

struct Stemmer;
impl Annotator for Stemmer {
    const ID: &'static str = "test:stems";
    type Reads = (View, Tokens);
    type Writes = (Stems,);

    fn annotate(&self, (view, tokens): (&View, &Tokens)) -> Result<(Stems,), Error> {
        let mut stem = Strs::default();
        for (_, span) in tokens.spans.iter() {
            stem.push(&view.text()[span.range()]);
        }
        Ok((Stems { stem },))
    }
}

struct SlowSentenceEmbedder {
    sleep_ms: u64,
    ran: Arc<AtomicUsize>,
}
impl Annotator for SlowSentenceEmbedder {
    const ID: &'static str = "test:sentence-embed";
    type Reads = (Sentences,);
    type Writes = (Vectors<Sentences>,);

    fn annotate(&self, (sents,): (&Sentences,)) -> Result<(Vectors<Sentences>,), Error> {
        self.ran.fetch_add(1, Ordering::SeqCst);
        if self.sleep_ms > 0 {
            std::thread::sleep(Duration::from_millis(self.sleep_ms));
        }
        let n = sents.spans.len();
        let data = vec![1.0_f32; n * 2];
        let model = ModelRef { id: "toy".into(), revision: "1".into(), sha256: "00".into() };
        Ok((Vectors::<Sentences>::new(2, data, Vec::new(), model),))
    }
}

struct LangDetector;
impl Annotator for LangDetector {
    const ID: &'static str = "test:lang";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, (_view,): (&View,)) -> Result<(Language,), Error> {
        let mut code = Strs::default();
        code.push("eng");
        Ok((Language { code, score: vec![0.99] },))
    }
}

#[test]
fn run_progressive_matches_run_and_streams_fast_dag_branches_early() {
    let ran_slow = Arc::new(AtomicUsize::new(0));
    let analyzer = Analyzer::builder()
        .add(Sentencer)
        .add(LangDetector)
        .add(Tokenizer { fail_on_marker: false })
        .add(SlowSentenceEmbedder { sleep_ms: 40, ran: ran_slow.clone() })
        .add(Tagger)
        .add(Stemmer)
        .add(Lemmatizer)
        .build()
        .unwrap();

    let mut batch_docs = vec![Document::new("Hello world"), Document::new("Rust pipelines stream fast")];
    let batch_status = analyzer.run(&mut batch_docs);
    assert!(batch_status.iter().all(|r| r.is_ok()));

    let mut prog_docs = vec![Document::new("Hello world"), Document::new("Rust pipelines stream fast")];
    let cancel = AtomicBool::new(false);
    let mut completed_steps = Vec::new();
    let prog_status = analyzer.run_progressive(&mut prog_docs, &cancel, |ev| {
        assert!(ev.failure.is_none());
        assert!(ev.skipped_dependents.is_empty());
        // Every written key of the completed step must already be present on docs.
        for &k in &ev.timing.written {
            assert!(ev.docs[0].present(k));
        }
        completed_steps.push(ev.timing.step_id);
    });
    assert_eq!(prog_status, batch_status);
    assert_eq!(completed_steps.len(), 7);

    // Verify topological ordering:
    // - test:sentences before test:tokens and test:sentence-embed
    // - test:tokens before test:pos and test:stems
    // - test:pos before test:lemmas
    let pos_of = |id: &str| completed_steps.iter().position(|&s| s == id).unwrap();
    assert!(pos_of("test:sentences") < pos_of("test:tokens"));
    assert!(pos_of("test:sentences") < pos_of("test:sentence-embed"));
    assert!(pos_of("test:tokens") < pos_of("test:pos"));
    assert!(pos_of("test:tokens") < pos_of("test:stems"));
    assert!(pos_of("test:pos") < pos_of("test:lemmas"));

    // Fast token/POS/lemma branch (stage 1..3) unlocks and finishes before the 40ms slow sentence embedder!
    assert!(
        pos_of("test:tokens") < pos_of("test:sentence-embed"),
        "expected fast token branch to finish before slow sibling sentence-embed: {completed_steps:?}"
    );

    // Final exported DocumentGraphs must be byte-for-byte identical to Analyzer::run.
    let c = codecs();
    for (b_doc, p_doc) in batch_docs.iter().zip(prog_docs.iter()) {
        let b_bytes = export(b_doc, &c).unwrap().to_bytes().unwrap();
        let p_bytes = export(p_doc, &c).unwrap().to_bytes().unwrap();
        assert_eq!(p_bytes, b_bytes);
    }
}

#[test]
fn non_backbone_step_failure_emits_skipped_dependents_and_completes_siblings() {
    let ran_embed = Arc::new(AtomicUsize::new(0));
    let analyzer = Analyzer::builder()
        .add(Sentencer)
        .add(LangDetector)
        .add(Tokenizer { fail_on_marker: true })
        .add(SlowSentenceEmbedder { sleep_ms: 0, ran: ran_embed.clone() })
        .add(Tagger)
        .add(Stemmer)
        .add(Lemmatizer)
        .build()
        .unwrap();

    let mut batch_docs = vec![Document::new("FAIL_TOKENS now")];
    let batch_status = analyzer.run(&mut batch_docs);

    let mut prog_docs = vec![Document::new("FAIL_TOKENS now")];
    let cancel = AtomicBool::new(false);
    let mut failed_event = None;
    let mut ok_steps = Vec::new();
    let prog_status = analyzer.run_progressive(&mut prog_docs, &cancel, |ev| {
        if let Some(err) = ev.failure {
            failed_event = Some((ev.timing.step_id, err, ev.skipped_dependents));
        } else {
            ok_steps.push(ev.timing.step_id);
        }
    });

    assert_eq!(prog_status, batch_status);
    let (failed_id, _err, skipped_deps) = failed_event.expect("expected test:tokens failure event");
    assert_eq!(failed_id, "test:tokens");
    assert!(skipped_deps.contains(&"test:pos"));
    assert!(skipped_deps.contains(&"test:stems"));
    assert!(skipped_deps.contains(&"test:lemmas"));

    // Independent sibling branches (test:sentences, test:lang, test:sentence-embed) all succeeded.
    assert!(ok_steps.contains(&"test:sentences"));
    assert!(ok_steps.contains(&"test:lang"));
    assert!(ok_steps.contains(&"test:sentence-embed"));
    assert_eq!(ran_embed.load(Ordering::SeqCst), 2);

    // Dependent layers are marked Status::Skipped on prog_docs[0], identical to Analyzer::run.
    assert!(matches!(prog_docs[0].status(Key::<PosTags>::DEFAULT.id()), Some(Status::Skipped(_))));
    assert!(matches!(prog_docs[0].status(Key::<Stems>::DEFAULT.id()), Some(Status::Skipped(_))));
    assert!(matches!(prog_docs[0].status(Key::<Lemmas>::DEFAULT.id()), Some(Status::Skipped(_))));

    let c = codecs();
    let b_bytes = export(&batch_docs[0], &c).unwrap().to_bytes().unwrap();
    let p_bytes = export(&prog_docs[0], &c).unwrap().to_bytes().unwrap();
    assert_eq!(p_bytes, b_bytes);
}

#[test]
fn cancellation_flag_stops_unstarted_downstream_steps() {
    let ran_downstream = Arc::new(AtomicUsize::new(0));
    let analyzer = Analyzer::builder()
        .add(Sentencer)
        .step(Bound::new(Tokenizer { fail_on_marker: false }))
        .add(Tagger)
        .add(Lemmatizer)
        .add(SlowSentenceEmbedder { sleep_ms: 0, ran: ran_downstream.clone() })
        .build()
        .unwrap();

    let mut docs = vec![Document::new("Cancel after stage 0")];
    let cancel = AtomicBool::new(false);
    let mut seen = Vec::new();
    analyzer.run_progressive(&mut docs, &cancel, |ev| {
        seen.push(ev.timing.step_id);
        if ev.timing.step_id == "test:sentences" {
            cancel.store(true, Ordering::Relaxed);
        }
    });

    assert_eq!(seen, vec!["test:sentences"]);
    assert_eq!(ran_downstream.load(Ordering::SeqCst), 0);
    assert!(docs[0].get::<Sentences>().is_some());
    assert!(docs[0].get::<Tokens>().is_none());
    assert!(docs[0].get::<PosTags>().is_none());
}
