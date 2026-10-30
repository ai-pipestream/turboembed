//! Regression tests for incremental DAG execution probes:
//! 1. Derived-view cascade removal and `ViewId` slot reuse.
//! 2. Optional-dependency cycle back-edge masking and wire cycle guard.
//! 3. `DocumentGraph` wire persistence of `provenance_digest` and `input_snapshot`.
//! 4. Complete payload `content_hash` (`Entities.score`, `TagSet` labels, opt-in `Option<u64>`).
//! 5. Cross-document batching and `Execution` placement in `Analyzer::run_incremental`.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use turbo_analysis::annotate::Placed;
use turbo_analysis::export::{
    Codecs, DocumentGraph, ExportError, LayerRecord, LayerRef, ProvenanceRecord, SourceRecord, StatusRecord,
    ViewRecord, export, import,
};
use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Error, Execution, Ix, IxRange, Key, Layer, Provenance, ProvenanceDigest,
    Refs, Span, Spans, StageCacheStatus, Tag, TagSet, View, ViewBuilder,
};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{Entities, PosTags, Sentences, Tokens};

fn codecs() -> Codecs {
    let mut c = Codecs::new();
    register_codecs(&mut c);
    c
}

struct NormalizerStep {
    config: u64,
    trim_prefix: &'static str,
}

impl Annotator for NormalizerStep {
    const ID: &'static str = "test:norm-view";
    const VERSION: &'static str = "1";
    type Reads = (View,);
    type Writes = (View,);

    fn config_hash(&self) -> u64 {
        self.config
    }

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View,), Error> {
        let text = src.text();
        let mut vb = ViewBuilder::new(text);
        if text.starts_with(self.trim_prefix) {
            vb.replace(self.trim_prefix.len(), "");
        }
        vb.keep(text.len());
        let (out, align) = vb.finish();
        Ok((View::derived(src, out, align),))
    }
}

struct ViewTokenizer;

impl Annotator for ViewTokenizer {
    const ID: &'static str = "test:view-tokenizer";
    const VERSION: &'static str = "1";
    type Reads = (View,);
    type Writes = (Tokens,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut cursor = 0usize;
        for part in text.split(' ') {
            if !part.is_empty() {
                spans.push(Span::new(cursor as u32, (cursor + part.len()) as u32));
            }
            cursor += part.len() + 1;
        }
        Ok((Tokens { spans, sentence: None },))
    }
}

#[test]
fn probe_1_derived_view_invalidation_cascade_removes_dependents_and_reuses_view_slot() {
    let a1 = Analyzer::builder()
        .step(Bound::new(NormalizerStep { config: 1, trim_prefix: "   " }).write(0, "normalized"))
        .step(Bound::new(ViewTokenizer).read(0, "normalized"))
        .build()
        .unwrap();

    let mut doc = Document::new("   hello   world");
    let rep1 = a1.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep1.stages_run, rep1.stages_skipped), (2, 0));
    let first_view_id = doc.get_key(&Key::<View>::named("normalized")).unwrap().id();
    assert_eq!(first_view_id.0, 1);

    let a2 = Analyzer::builder()
        .step(Bound::new(NormalizerStep { config: 2, trim_prefix: " " }).write(0, "normalized"))
        .step(Bound::new(ViewTokenizer).read(0, "normalized"))
        .build()
        .unwrap();

    let rep2 = a2.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep2.stages_run, rep2.stages_skipped), (2, 0));
    let second_view_id = doc.get_key(&Key::<View>::named("normalized")).unwrap().id();
    assert_eq!(second_view_id, first_view_id, "recomputed view must reuse the reclaimed ViewId slot");

    let c = codecs();
    let graph = export(&doc, &c).expect("export after view recomputation must succeed");
    let round_tripped = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &c).unwrap();
    assert_eq!(round_tripped.get_key(&Key::<View>::named("normalized")).unwrap().text(), "  hello   world");
    assert_eq!(round_tripped.get::<Tokens>().unwrap().len(), 2);

    // Direct removal of an intermediate derived view cascades to child views,
    // span-anchored layers on those views, and aligned layers on those layers.
    let mut chain_doc = Document::new("hello world");
    let src_view = chain_doc.get::<View>().unwrap().clone();
    let mut vb1 = ViewBuilder::new(src_view.text());
    vb1.keep(src_view.text().len());
    let (t1, al1) = vb1.finish();
    chain_doc.add(&Key::<View>::named("v1"), View::derived(&src_view, t1, al1), Provenance::by("test:v1")).unwrap();
    let v1 = chain_doc.get_key(&Key::<View>::named("v1")).unwrap().clone();
    let mut vb2 = ViewBuilder::new(v1.text());
    vb2.keep(v1.text().len());
    let (t2, al2) = vb2.finish();
    let p_v2 = Provenance { reads: vec![Key::<View>::named("v1").id()], ..Provenance::by("test:v2") };
    chain_doc.add(&Key::<View>::named("v2"), View::derived(&v1, t2, al2), p_v2).unwrap();
    let v2_id = chain_doc.get_key(&Key::<View>::named("v2")).unwrap().id();
    let mut tok_spans = Spans::new(v2_id);
    tok_spans.push(Span::new(0, 5));
    let p_tok = Provenance { reads: vec![Key::<View>::named("v2").id()], ..Provenance::by("test:tok") };
    chain_doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: tok_spans, sentence: None }, p_tok).unwrap();
    let p_pos = Provenance { reads: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:pos") };
    chain_doc
        .add(
            &Key::<PosTags>::DEFAULT,
            PosTags { tagset: TagSet::new("upos", &["NOUN"]), tags: vec![Tag(0)], prob: None },
            p_pos,
        )
        .unwrap();

    assert!(chain_doc.remove(Key::<View>::named("v1").id()));
    assert!(!chain_doc.has(Key::<View>::named("v2").id()));
    assert!(!chain_doc.has(Key::<Tokens>::DEFAULT.id()));
    assert!(!chain_doc.has(Key::<PosTags>::DEFAULT.id()));
    assert!(export(&chain_doc, &c).is_ok());
}

struct OptionalBackedgeNormalizer {
    config: u64,
    saw_tokens: Arc<AtomicUsize>,
}

impl Annotator for OptionalBackedgeNormalizer {
    const ID: &'static str = "test:opt-backedge-norm";
    const VERSION: &'static str = "1";
    type Reads = (View, Option<PosTags>, Option<Tokens>);
    type Writes = (View,);

    fn config_hash(&self) -> u64 {
        self.config
    }

    fn annotate(&self, (src, _pos, tok): Refs<'_, Self>) -> Result<(View,), Error> {
        if tok.is_some() {
            self.saw_tokens.fetch_add(1, Ordering::SeqCst);
        }
        let mut vb = ViewBuilder::new(src.text());
        vb.keep(src.text().len());
        let (out, align) = vb.finish();
        Ok((View::derived(src, out, align),))
    }
}

#[test]
fn probe_2_optional_cycle_backedge_is_masked_and_wire_provenance_cycle_is_guarded() {
    let saw_tokens = Arc::new(AtomicUsize::new(0));
    let a1 = Analyzer::builder()
        .step(
            Bound::new(OptionalBackedgeNormalizer { config: 1, saw_tokens: saw_tokens.clone() }).write(0, "normalized"),
        )
        .step(Bound::new(ViewTokenizer).read(0, "normalized"))
        .build()
        .unwrap();

    let mut doc = Document::new("Can't won't.");
    let rep1 = a1.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep1.stages_run, rep1.stages_skipped), (2, 0));
    assert_eq!(saw_tokens.load(Ordering::SeqCst), 0);

    // Warm run with unchanged config must skip both stages (no false cache miss from back-edge Tokens).
    let rep_warm = a1.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep_warm.stages_run, rep_warm.stages_skipped), (0, 2));

    // Re-running with mutated config at stage 0 must not record back-edge Tokens in View("normalized").reads.
    let a2 = Analyzer::builder()
        .step(
            Bound::new(OptionalBackedgeNormalizer { config: 2, saw_tokens: saw_tokens.clone() }).write(0, "normalized"),
        )
        .step(Bound::new(ViewTokenizer).read(0, "normalized"))
        .build()
        .unwrap();
    let rep2 = a2.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep2.stages_run, rep2.stages_skipped), (2, 0));
    assert_eq!(saw_tokens.load(Ordering::SeqCst), 0, "stage 0 must never see downstream Tokens from prior run");

    let c = codecs();
    let graph = export(&doc, &c).unwrap();
    assert_eq!(graph.layer_provenance_digest(Tokens::ID, ""), doc.provenance_digest(Key::<Tokens>::DEFAULT.id()));
    assert_eq!(
        graph.view_provenance_digest("normalized"),
        doc.provenance_digest(Key::<View>::named("normalized").id())
    );

    let prov = |ann: &str, reads: Vec<LayerRef>| ProvenanceRecord {
        annotator: ann.into(),
        version: "1".into(),
        config: 0,
        model: None,
        execution: None,
        reads,
        on: vec![],
        provenance_digest: None,
        input_snapshot: None,
    };
    let cyclic_graph = DocumentGraph {
        version: 1,
        source: SourceRecord { bytes: b"hi".to_vec(), text: "hi".into(), alignment: None },
        views: vec![ViewRecord {
            instance: String::new(),
            status: StatusRecord::Complete,
            provenance: prov("turbo:source", vec![]),
            parent: None,
            text: None,
            edits: vec![],
        }],
        layers: vec![
            LayerRecord {
                layer: "test:a".into(),
                instance: String::new(),
                schema: 1,
                len: 0,
                status: StatusRecord::Complete,
                provenance: prov("ann:a", vec![LayerRef { layer: "test:b".into(), instance: String::new() }]),
                body: Some(vec![]),
            },
            LayerRecord {
                layer: "test:b".into(),
                instance: String::new(),
                schema: 1,
                len: 0,
                status: StatusRecord::Complete,
                provenance: prov("ann:b", vec![LayerRef { layer: "test:a".into(), instance: String::new() }]),
                body: Some(vec![]),
            },
        ],
    };
    assert!(matches!(cyclic_graph.try_layer_provenance_digest("test:a", ""), Err(ExportError::InvalidLayer { .. })));
    assert_eq!(cyclic_graph.layer_provenance_digest("test:a", ""), None);
}

struct ScoreReaderStep {
    calls: Arc<AtomicUsize>,
}

impl Annotator for ScoreReaderStep {
    const ID: &'static str = "test:score-reader";
    const VERSION: &'static str = "1";
    type Reads = (Entities,);
    type Writes = (Sentences,);

    fn annotate(&self, (entities,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut spans = Spans::new(entities.spans.view());
        for (ix, sp) in entities.spans.iter() {
            if entities.score[ix.get()] >= 0.5 && entities.kinds.label(entities.kind[ix.get()]) == "ORG" {
                spans.push(sp);
            }
        }
        Ok((Sentences { spans },))
    }
}

struct StaticEntityProducer {
    score: f64,
    label: &'static str,
}

impl Annotator for StaticEntityProducer {
    const ID: &'static str = "test:static-entity-producer";
    const VERSION: &'static str = "1";
    type Reads = (Tokens,);
    type Writes = (Entities,);

    fn annotate(&self, (tokens,): Refs<'_, Self>) -> Result<(Entities,), Error> {
        let kinds = TagSet::new("ner", &[self.label]);
        let mut spans = Spans::new(tokens.spans.view());
        let mut tok_ranges = Vec::new();
        let mut kind = Vec::new();
        if tokens.len() > 0 {
            spans.push(tokens.spans.get(Ix::new(0)));
            tok_ranges.push(IxRange::new(0, 1));
            kind.push(Tag(0));
        }
        Ok((Entities { spans, tokens: tok_ranges, kind, kinds, score: vec![self.score] },))
    }
}

#[test]
fn probe_3_and_4_wire_roundtrip_input_snapshot_and_entity_score_and_tagset_mutations() {
    let calls = Arc::new(AtomicUsize::new(0));
    let analyzer = Analyzer::builder()
        .add(ViewTokenizer)
        .add(StaticEntityProducer { score: 0.95, label: "ORG" })
        .add(ScoreReaderStep { calls: calls.clone() })
        .build()
        .unwrap();

    let mut doc = Document::new("Acme builds parsers");
    analyzer.analyze_incremental(&mut doc).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(doc.get::<Sentences>().unwrap().len(), 1);

    // Probe 3: export -> to_bytes -> from_bytes -> import, then mutate `Tokens` in place with the same Provenance.
    let c = codecs();
    let bytes = export(&doc, &c).unwrap().to_bytes().unwrap();
    let mut imported = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &c).unwrap();
    let tok_key = Key::<Tokens>::DEFAULT;
    let saved_prov = imported.provenance(tok_key.id()).unwrap().clone();
    imported.remove(tok_key.id());
    let mut mutated_spans = Spans::new(imported.get::<View>().unwrap().id());
    mutated_spans.push(Span::new(0, 4));
    imported.add(&tok_key, Tokens { spans: mutated_spans, sentence: None }, saved_prov).unwrap();

    let rep_imported = analyzer.analyze_incremental(&mut imported).unwrap();
    assert!(rep_imported.was_skipped(ViewTokenizer::ID));
    assert!(
        rep_imported.was_executed(StaticEntityProducer::ID),
        "persisted input_snapshot must detect in-place mutation of Tokens after import"
    );

    // Probe 4a: Mutate only `Entities.score` (`0.95` -> `0.10`) with identical provenance, spans, and Tag(0).
    let ent_key = Key::<Entities>::DEFAULT;
    let ent_prov = doc.provenance(ent_key.id()).unwrap().clone();
    let old_ent_digest = doc.content_digest(ent_key.id()).unwrap();
    doc.remove(ent_key.id());
    let mut ent_spans = Spans::new(doc.get::<View>().unwrap().id());
    ent_spans.push(Span::new(0, 4));
    doc.add(
        &ent_key,
        Entities {
            spans: ent_spans,
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("ner", &["ORG"]),
            score: vec![0.10],
        },
        ent_prov.clone(),
    )
    .unwrap();
    assert_ne!(doc.content_digest(ent_key.id()).unwrap(), old_ent_digest, "Entities.score must affect ContentDigest");
    let rep_score = analyzer.analyze_incremental(&mut doc).unwrap();
    assert!(rep_score.was_executed(ScoreReaderStep::ID), "ScoreReaderStep must re-run when Entities.score changes");
    assert_eq!(doc.get::<Sentences>().unwrap().len(), 0);

    // Probe 4b: Mutate only `TagSet` label (`"ORG"` -> `"PER"`) at `Tag(0)` with score restored to `0.95`.
    doc.remove(ent_key.id());
    let mut ent_spans2 = Spans::new(doc.get::<View>().unwrap().id());
    ent_spans2.push(Span::new(0, 4));
    doc.add(
        &ent_key,
        Entities {
            spans: ent_spans2,
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("ner", &["PER"]),
            score: vec![0.95],
        },
        ent_prov,
    )
    .unwrap();
    let rep_label = analyzer.analyze_incremental(&mut doc).unwrap();
    assert!(rep_label.was_executed(ScoreReaderStep::ID), "ScoreReaderStep must re-run when TagSet label changes");
}

struct UnhashedCustomLayer(u64);

impl Layer for UnhashedCustomLayer {
    const ID: &'static str = "test:unhashed-custom";
    fn len(&self) -> usize {
        1
    }
    fn check(&self) -> Result<(), String> {
        Ok(())
    }
}

struct UnhashedProducer(u64);

impl Annotator for UnhashedProducer {
    const ID: &'static str = "test:unhashed-producer";
    type Reads = (View,);
    type Writes = (UnhashedCustomLayer,);
    fn annotate(&self, _: Refs<'_, Self>) -> Result<(UnhashedCustomLayer,), Error> {
        Ok((UnhashedCustomLayer(self.0),))
    }
}

struct UnhashedConsumer(Arc<AtomicUsize>);

impl Annotator for UnhashedConsumer {
    const ID: &'static str = "test:unhashed-consumer";
    type Reads = (UnhashedCustomLayer,);
    type Writes = (Sentences,);
    fn annotate(&self, (u,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let mut spans = Spans::new(turbo_analysis::ViewId::SOURCE);
        if u.0 > 0 {
            spans.push(Span::new(0, 1));
        }
        Ok((Sentences { spans },))
    }
}

#[test]
fn probe_4c_custom_layer_without_content_hash_never_falsely_skips() {
    let calls = Arc::new(AtomicUsize::new(0));
    let analyzer = Analyzer::builder().add(UnhashedProducer(1)).add(UnhashedConsumer(calls.clone())).build().unwrap();
    let mut doc = Document::new("hello");
    analyzer.analyze_incremental(&mut doc).unwrap();
    assert_eq!(doc.content_digest(Key::<UnhashedCustomLayer>::DEFAULT.id()), None);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let rep2 = analyzer.analyze_incremental(&mut doc).unwrap();
    assert_eq!((rep2.stages_run, rep2.stages_skipped), (2, 0));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

struct PlacedBatchTokenizer {
    batch_sizes: Arc<Mutex<Vec<usize>>>,
}

impl Annotator for PlacedBatchTokenizer {
    const ID: &'static str = "test:placed-batch-tokenizer";
    const VERSION: &'static str = "1";
    type Reads = (View,);
    type Writes = (Tokens,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let mut spans = Spans::new(view.id());
        if !view.text().is_empty() {
            spans.push(Span::new(0, view.text().len() as u32));
        }
        Ok((Tokens { spans, sentence: None },))
    }

    fn annotate_batch_placed<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Placed<Self::Writes>> {
        let batch_len = inputs.len();
        self.batch_sizes.lock().unwrap().push(batch_len);
        let exec = Execution {
            backend: "test-batch".into(),
            device: format!("batch-size-{batch_len}"),
            precision: "fp16".into(),
        };
        inputs.into_iter().map(|i| self.annotate(i).map(|w| (w, Some(exec.clone())))).collect()
    }
}

#[test]
fn probe_5_run_incremental_coalesces_uncached_batch_and_preserves_placed_execution() {
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let analyzer = Analyzer::builder().add(PlacedBatchTokenizer { batch_sizes: batch_sizes.clone() }).build().unwrap();

    let mut doc0 = Document::new("alpha");
    let doc1 = Document::new("beta");
    let mut doc2 = Document::new("gamma");
    let doc3 = Document::new("delta");
    analyzer.analyze_incremental(&mut doc0).unwrap();
    analyzer.analyze_incremental(&mut doc2).unwrap();
    batch_sizes.lock().unwrap().clear();

    let mut docs = vec![doc0, doc1, doc2, doc3];
    let reports = analyzer.run_incremental(&mut docs);
    for r in &reports {
        assert!(r.is_ok());
    }
    assert_eq!(reports[0].as_ref().unwrap().stages[0].cache_status, StageCacheStatus::Skipped);
    assert_eq!(reports[1].as_ref().unwrap().stages[0].cache_status, StageCacheStatus::Executed);
    assert_eq!(reports[2].as_ref().unwrap().stages[0].cache_status, StageCacheStatus::Skipped);
    assert_eq!(reports[3].as_ref().unwrap().stages[0].cache_status, StageCacheStatus::Executed);

    assert_eq!(*batch_sizes.lock().unwrap(), vec![2]);
    let tok_key = Key::<Tokens>::DEFAULT.id();
    assert_eq!(docs[1].provenance(tok_key).unwrap().execution.as_ref().unwrap().device, "batch-size-2");
    assert_eq!(docs[3].provenance(tok_key).unwrap().execution.as_ref().unwrap().device, "batch-size-2");
    assert_ne!(reports[1].as_ref().unwrap().stages[0].provenance_digest, ProvenanceDigest::ZERO);
}
