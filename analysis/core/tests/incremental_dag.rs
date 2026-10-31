//! Integration tests for Issue #1467: content-addressed incremental DAG
//! execution, `requires`/`produces` dependency validation (`validate_dag`),
//! chained provenance digests across `DocumentGraph` binary wire round-trip,
//! and parallel vs sequential byte-identical execution.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_analysis::export::{Codecs, DocumentGraph, export, import};
use turbo_analysis::{
    Analyzer, Annotator, Document, Error, Fnv1a, Groups, Ix, IxRange, Key, KeyId, Layer, ModelRef, Refs, Span, Spans,
    StageCacheStatus, TagSet, View,
};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{AbbreviationForm, Abbreviations, Entities, Lemmas, PosTags, Sentences, Strs, Tokens};

fn test_codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs.register::<Groups<Entities>>();
    codecs
}

struct CountingSentencer {
    calls: Arc<AtomicUsize>,
    config: u64,
}

impl Annotator for CountingSentencer {
    const ID: &'static str = "test:incremental-sentencer";
    const VERSION: &'static str = "1";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn requires(&self) -> &[KeyId] {
        const { &[Key::<View>::DEFAULT.id()] }
    }
    fn produces(&self) -> &[KeyId] {
        const { &[Key::<Sentences>::DEFAULT.id()] }
    }
    fn config_hash(&self) -> u64 {
        self.config
    }
    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut start = 0usize;
        for (idx, _) in text.match_indices(". ") {
            spans.push(Span::new(start as u32, (idx + 1) as u32));
            start = idx + 2;
        }
        if start < text.len() {
            spans.push(Span::new(start as u32, text.len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

struct CountingTokenizer {
    calls: Arc<AtomicUsize>,
    config: u64,
    model_rev: &'static str,
}

impl Annotator for CountingTokenizer {
    const ID: &'static str = "test:incremental-tokenizer";
    const VERSION: &'static str = "1";
    type Reads = (View, Sentences);
    type Writes = (Tokens,);

    fn requires(&self) -> &[KeyId] {
        const { &[Key::<View>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id()] }
    }
    fn produces(&self) -> &[KeyId] {
        const { &[Key::<Tokens>::DEFAULT.id()] }
    }
    fn config_hash(&self) -> u64 {
        self.config
    }
    fn model(&self) -> Option<ModelRef> {
        Some(ModelRef {
            id: "test:tokenizer-model".into(),
            revision: self.model_rev.into(),
            sha256: format!("sha256-{}", self.model_rev),
        })
    }
    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut sentence = Vec::new();
        for (s_ix, s_span) in sentences.spans.iter() {
            let mut cursor = s_span.start as usize;
            for part in text[s_span.range()].split(' ') {
                let trimmed = part.trim_end_matches('.');
                if !trimmed.is_empty() {
                    spans.push(Span::new(cursor as u32, (cursor + trimmed.len()) as u32));
                    sentence.push(s_ix);
                }
                cursor += part.len() + 1;
            }
        }
        Ok((Tokens { spans, sentence: Some(sentence) },))
    }
}

struct CountingPosTagger {
    calls: Arc<AtomicUsize>,
    config: u64,
    tagset: Arc<TagSet>,
}

impl Annotator for CountingPosTagger {
    const ID: &'static str = "test:incremental-postagger";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    fn requires(&self) -> &[KeyId] {
        const { &[Key::<View>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()] }
    }
    fn produces(&self) -> &[KeyId] {
        const { &[Key::<PosTags>::DEFAULT.id()] }
    }
    fn config_hash(&self) -> u64 {
        self.config
    }
    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (noun, verb) = (self.tagset.tag("NOUN").unwrap(), self.tagset.tag("VERB").unwrap());
        let text = view.text();
        let flip = self.config & 1 == 1;
        let tags = tokens
            .spans
            .iter()
            .map(|(_, sp)| if text[sp.range()].ends_with('s') ^ flip { verb } else { noun })
            .collect();
        Ok((PosTags { tagset: self.tagset.clone(), tags, prob: None },))
    }
}

struct CountingNameFinder {
    calls: Arc<AtomicUsize>,
    config: u64,
    kinds: Arc<TagSet>,
}

impl Annotator for CountingNameFinder {
    const ID: &'static str = "test:incremental-namefinder";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens);
    type Writes = (Entities,);

    fn config_hash(&self) -> u64 {
        self.config
    }
    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Entities,), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let text = view.text();
        let org = self.kinds.tag("ORG").unwrap();
        let mut spans = Spans::new(view.id());
        let mut tok_ranges = Vec::new();
        let mut kind = Vec::new();
        for (tok_ix, sp) in tokens.spans.iter() {
            if text[sp.range()].chars().next().is_some_and(char::is_uppercase) {
                spans.push(sp);
                let i = tok_ix.get() as u32;
                tok_ranges.push(IxRange::new(i, i + 1));
                kind.push(org);
            }
        }
        let score = vec![1.0; kind.len()];
        Ok((Entities { spans, tokens: tok_ranges, kind, kinds: self.kinds.clone(), score },))
    }
}

struct CountingLemmatizer {
    calls: Arc<AtomicUsize>,
    config: u64,
}

impl Annotator for CountingLemmatizer {
    const ID: &'static str = "test:incremental-lemmatizer";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens);
    type Writes = (Lemmas,);

    fn config_hash(&self) -> u64 {
        self.config
    }
    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let text = view.text();
        let mut lemma = Strs::default();
        for (_, sp) in tokens.spans.iter() {
            lemma.push(&text[sp.range()].to_ascii_lowercase());
        }
        Ok((Lemmas { lemma },))
    }
}

struct CountingAbbrevFinder {
    calls: Arc<AtomicUsize>,
    config: u64,
}

impl Annotator for CountingAbbrevFinder {
    const ID: &'static str = "test:incremental-abbrev";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens);
    type Writes = (Abbreviations,);

    fn config_hash(&self) -> u64 {
        self.config
    }
    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Abbreviations,), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let text = view.text();
        let mut abbrev = Abbreviations::new(view.id());
        for (_, sp) in tokens.spans.iter() {
            let word = &text[sp.range()];
            if word.len() >= 2 && word.chars().all(|c| c.is_ascii_uppercase()) {
                abbrev.push(sp, AbbreviationForm::Short);
            }
        }
        Ok((abbrev,))
    }
}

struct CountingEntityGrouper {
    calls: Arc<AtomicUsize>,
    config: u64,
}

impl Annotator for CountingEntityGrouper {
    const ID: &'static str = "test:incremental-entity-grouper";
    const VERSION: &'static str = "1";
    type Reads = (Entities, PosTags);
    type Writes = (Groups<Entities>,);

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.str(Self::ID).u64(self.config);
        h.finish()
    }
    fn annotate(&self, (entities, pos): Refs<'_, Self>) -> Result<(Groups<Entities>,), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let members: Vec<Ix<Entities>> = (0..entities.len())
            .filter(|&i| pos.tags.get(entities.tokens[i].start.get()).is_some())
            .map(|i| Ix::new(i as u32))
            .collect();
        let groups = if members.is_empty() {
            Groups::new(std::iter::empty::<Vec<Ix<Entities>>>())
        } else {
            Groups::new([members])
        };
        Ok((groups,))
    }
}

#[derive(Default)]
struct CallCounters {
    sent: Arc<AtomicUsize>,
    tok: Arc<AtomicUsize>,
    pos: Arc<AtomicUsize>,
    ner: Arc<AtomicUsize>,
    lemma: Arc<AtomicUsize>,
    abbrev: Arc<AtomicUsize>,
    group: Arc<AtomicUsize>,
}

impl CallCounters {
    fn snapshot(&self) -> [usize; 7] {
        [
            self.sent.load(Ordering::SeqCst),
            self.tok.load(Ordering::SeqCst),
            self.pos.load(Ordering::SeqCst),
            self.ner.load(Ordering::SeqCst),
            self.lemma.load(Ordering::SeqCst),
            self.abbrev.load(Ordering::SeqCst),
            self.group.load(Ordering::SeqCst),
        ]
    }
}

fn build_dag(counters: &CallCounters, tok_rev: &'static str, pos_config: u64, group_config: u64) -> Analyzer {
    let tagset = TagSet::new("upos", &["NOUN", "VERB"]);
    let kinds = TagSet::new("ner", &["ORG"]);
    Analyzer::builder()
        .add(CountingSentencer { calls: counters.sent.clone(), config: 10 })
        .add(CountingTokenizer { calls: counters.tok.clone(), config: 20, model_rev: tok_rev })
        .add(CountingPosTagger { calls: counters.pos.clone(), config: pos_config, tagset })
        .add(CountingNameFinder { calls: counters.ner.clone(), config: 40, kinds })
        .add(CountingLemmatizer { calls: counters.lemma.clone(), config: 50 })
        .add(CountingAbbrevFinder { calls: counters.abbrev.clone(), config: 60 })
        .add(CountingEntityGrouper { calls: counters.group.clone(), config: group_config })
        .build()
        .expect("valid DAG")
}

#[test]
fn requires_and_produces_declarations_and_validate_dag_detect_missing_and_cyclic_deps() {
    let tagset = TagSet::new("upos", &["NOUN", "VERB"]);
    let pos = CountingPosTagger { calls: Arc::new(AtomicUsize::new(0)), config: 1, tagset };
    assert_eq!(pos.requires(), &[Key::<View>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()]);
    assert_eq!(pos.produces(), &[Key::<PosTags>::DEFAULT.id()]);

    let bad_builder = Analyzer::builder().add(pos);
    let err = bad_builder.validate_dag().unwrap_err();
    assert!(matches!(err, Error::Pipeline(ref msg) if msg.contains("opennlp:tokens")), "unexpected error: {err:?}");
    let unvalidated = bad_builder.build_unvalidated();
    assert!(unvalidated.validate_dag().is_err());
    let mut doc = Document::new("Acme builds systems.");
    assert!(unvalidated.analyze_incremental(&mut doc).is_err());

    struct CustomReqStep;
    impl Annotator for CustomReqStep {
        const ID: &'static str = "test:custom-req";
        type Reads = (View,);
        type Writes = (Sentences,);
        fn requires(&self) -> &[KeyId] {
            const { &[Key::<View>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()] }
        }
        fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
            Ok((Sentences { spans: Spans::new(view.id()) },))
        }
    }
    let custom_err = Analyzer::builder().add(CustomReqStep).validate_dag().unwrap_err();
    assert!(matches!(custom_err, Error::Pipeline(ref msg) if msg.contains("opennlp:tokens")));
}

#[test]
fn exported_and_imported_document_graph_skips_all_stages_in_o1() {
    let counters = CallCounters::default();
    let analyzer = build_dag(&counters, "v1", 30, 70);
    assert_eq!(analyzer.validate_dag(), Ok(()));
    assert_eq!(analyzer.plan().len(), 4);

    let mut doc = Document::new("Acme Corp builds fast parsers. NASA launches rockets.");
    let first_report = analyzer.analyze_incremental(&mut doc).unwrap();
    assert_eq!((first_report.stages_run, first_report.stages_skipped), (7, 0));
    assert_eq!(counters.snapshot(), [1, 1, 1, 1, 1, 1, 1]);

    let codecs = test_codecs();
    let graph = export(&doc, &codecs).unwrap();
    for key in [
        Key::<Sentences>::DEFAULT.id(),
        Key::<Tokens>::DEFAULT.id(),
        Key::<PosTags>::DEFAULT.id(),
        Key::<Entities>::DEFAULT.id(),
        Key::<Lemmas>::DEFAULT.id(),
        Key::<Abbreviations>::DEFAULT.id(),
        Key::<Groups<Entities>>::DEFAULT.id(),
    ] {
        let mem_digest = doc.provenance_digest(key).expect("in-memory digest");
        let wire_digest = graph.layer_provenance_digest(&key.layer.to_string(), key.name).expect("wire digest");
        assert_eq!(mem_digest, wire_digest, "digest mismatch for {key}");
    }

    let wire_bytes = graph.to_bytes().unwrap();
    let decoded_graph = DocumentGraph::from_bytes(&wire_bytes).unwrap();
    let mut imported_doc = import(&decoded_graph, &codecs).unwrap();

    let second_report = analyzer.analyze_incremental(&mut imported_doc).unwrap();
    assert_eq!((second_report.stages_run, second_report.stages_skipped), (0, 7));
    assert!(second_report.stages.iter().all(|s| s.cache_status == StageCacheStatus::Skipped));
    assert_eq!(counters.snapshot(), [1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(wire_bytes, export(&imported_doc, &codecs).unwrap().to_bytes().unwrap());
}

#[test]
fn mutating_mid_pipeline_config_or_model_recomputes_only_affected_downstream_branch() {
    let counters = CallCounters::default();
    let analyzer_v1 = build_dag(&counters, "v1", 30, 70);
    let mut doc = Document::new("Acme Corp builds fast parsers. NASA launches rockets.");
    analyzer_v1.analyze_incremental(&mut doc).unwrap();
    assert_eq!(counters.snapshot(), [1, 1, 1, 1, 1, 1, 1]);

    let analyzer_pos_v2 = build_dag(&counters, "v1", 31, 70);
    let rep2 = analyzer_pos_v2.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep2.stages_run, rep2.stages_skipped), (2, 5));
    assert!(rep2.was_skipped(CountingSentencer::ID));
    assert!(rep2.was_skipped(CountingTokenizer::ID));
    assert!(rep2.was_executed(CountingPosTagger::ID));
    assert!(rep2.was_skipped(CountingNameFinder::ID));
    assert!(rep2.was_skipped(CountingLemmatizer::ID));
    assert!(rep2.was_skipped(CountingAbbrevFinder::ID));
    assert!(rep2.was_executed(CountingEntityGrouper::ID));
    assert_eq!(counters.snapshot(), [1, 1, 2, 1, 1, 1, 2]);

    let codecs = test_codecs();
    let mut fresh_doc = Document::new("Acme Corp builds fast parsers. NASA launches rockets.");
    analyzer_pos_v2.analyze_incremental(&mut fresh_doc).unwrap();
    assert_eq!(
        export(&doc, &codecs).unwrap().to_bytes().unwrap(),
        export(&fresh_doc, &codecs).unwrap().to_bytes().unwrap()
    );

    let latest = counters.snapshot();
    let analyzer_tok_v2 = build_dag(&counters, "v2", 31, 70);
    let rep3 = analyzer_tok_v2.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep3.stages_run, rep3.stages_skipped), (6, 1));
    assert!(rep3.was_skipped(CountingSentencer::ID));
    assert!(rep3.was_executed(CountingTokenizer::ID));
    assert_eq!(
        counters.snapshot(),
        [latest[0], latest[1] + 1, latest[2] + 1, latest[3] + 1, latest[4] + 1, latest[5] + 1, latest[6] + 1]
    );
}

#[test]
fn upstream_layer_content_mutation_invalidates_downstream_stages() {
    let counters = CallCounters::default();
    let analyzer = build_dag(&counters, "v1", 30, 70);
    let mut doc = Document::new("Acme Corp builds fast parsers.");
    analyzer.analyze_incremental(&mut doc).unwrap();

    let tok_key = Key::<Tokens>::DEFAULT;
    let old_prov = doc.provenance(tok_key.id()).unwrap().clone();
    doc.remove(tok_key.id());
    let mut custom_spans = Spans::new(doc.get::<View>().unwrap().id());
    custom_spans.push(Span::new(0, 4));
    doc.add(&tok_key, Tokens { spans: custom_spans, sentence: Some(vec![Ix::new(0)]) }, old_prov).unwrap();

    let rep = analyzer.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep.stages_skipped, rep.stages_run), (2, 5));
    assert_eq!(counters.snapshot(), [1, 1, 2, 2, 2, 2, 2]);
}

#[test]
fn parallel_dag_execution_matches_sequential_wire_bytes_bit_for_bit() {
    let counters = CallCounters::default();
    let analyzer = build_dag(&counters, "v1", 30, 70);
    let codecs = test_codecs();
    let text = "Acme Corp and Globex Inc build low-latency Rust NLP pipelines. WHO and NASA test them.";

    let mut par_docs = vec![Document::new(text)];
    for r in analyzer.run(&mut par_docs) {
        r.unwrap();
    }
    let mut seq_docs = vec![Document::new(text)];
    for r in analyzer.run_sequential(&mut seq_docs) {
        r.unwrap();
    }
    let mut inc_doc = Document::new(text);
    analyzer.analyze_incremental(&mut inc_doc).unwrap();

    let par_bytes = export(&par_docs[0], &codecs).unwrap().to_bytes().unwrap();
    let seq_bytes = export(&seq_docs[0], &codecs).unwrap().to_bytes().unwrap();
    let inc_bytes = export(&inc_doc, &codecs).unwrap().to_bytes().unwrap();
    assert_eq!(par_bytes, seq_bytes);
    assert_eq!(par_bytes, inc_bytes);
}
