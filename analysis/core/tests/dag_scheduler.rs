//! Conformance and benchmark tests for the critical-path DAG scheduler and
//! bounded [`WorkerPool`] (#2337).

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use turbo_analysis::{
    Analyzer, Anchor, Annotator, Bound, Codecs, Document, Error, ExportCx, ExportError, ImportCx, Key, Layer,
    LayerCodec, Overlap, Reader, Refs, Span, Status, TextSpans, View, ViewBuilder, ViewId, WorkerPool, Writer, export,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct TagData {
    view: ViewId,
    start: Vec<u32>,
    end: Vec<u32>,
    tags: Vec<String>,
}

impl TagData {
    fn single(view: ViewId, len: u32, tag: impl Into<String>) -> Self {
        Self { view, start: vec![0], end: vec![len], tags: vec![tag.into()] }
    }
    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(TextSpans {
            view: self.view,
            start: &self.start,
            end: &self.end,
            frag_start: &[],
            frag_end: &[],
            frag_item_end: &[],
        })
    }
    fn export_bytes(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.str(cx.view_instance(self.view)?)?;
        w.usize(self.tags.len())?;
        for (i, t) in self.tags.iter().enumerate() {
            w.span(Span::new(self.start[i], self.end[i]));
            w.str(t)?;
        }
        Ok(w.finish())
    }
    fn import_bytes(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let (view, n) = (cx.view(&r.string()?)?, r.usize()?);
        let (mut start, mut end, mut tags) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let sp = r.span()?;
            start.push(sp.start);
            end.push(sp.end);
            tags.push(r.string()?);
        }
        Ok(Self { view, start, end, tags })
    }
}

macro_rules! define_tag_layers {
    ($(($ty:ident, $id:expr)),+ $(,)?) => {
        $(
            #[derive(Clone, Debug, PartialEq, Eq)]
            struct $ty(TagData);
            impl Layer for $ty {
                const ID: &'static str = $id;
                const OVERLAP: Overlap = Overlap::Any;
                fn anchor() -> Anchor { Anchor::Text }
                fn len(&self) -> usize { self.0.tags.len() }
                fn check(&self) -> Result<(), String> { Ok(()) }
                fn text_spans(&self) -> Option<TextSpans<'_>> { self.0.text_spans() }
            }
            impl LayerCodec for $ty {
                fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> { self.0.export_bytes(cx) }
                fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
                    Ok($ty(TagData::import_bytes(data, cx)?))
                }
            }
        )+
        fn test_codecs() -> Codecs {
            let mut c = Codecs::new();
            $(c.register::<$ty>();)+
            c
        }
    };
}

define_tag_layers!(
    (TokLayer, "test:tokens"),
    (PosLayer, "test:pos"),
    (LemmaLayer, "test:lemmas"),
    (StopwordLayer, "test:stopwords"),
    (DepLayer, "turbo:dependencies"),
    (RelationLayer, "test:relations"),
    (SentimentTargetLayer, "test:sentiment-targets"),
    (ClauseLayer, "test:clauses"),
    (ModalityLayer, "test:modality"),
    (FastALayer, "test:fast-a"),
    (FastBLayer, "test:fast-b"),
    (FastCLayer, "test:fast-c"),
    (SlowALayer, "test:slow-a"),
    (SlowBLayer, "test:slow-b"),
);

macro_rules! define_simple_step {
    ($step:ident, $id:expr, $reads:ty, $out:ident, $fetch:pat => $view:ident, $tag:expr) => {
        struct $step;
        impl Annotator for $step {
            const ID: &'static str = $id;
            type Reads = $reads;
            type Writes = ($out,);
            fn annotate(&self, $fetch: Refs<'_, Self>) -> Result<Self::Writes, Error> {
                Ok(($out(TagData::single($view.id(), $view.text().len() as u32, $tag)),))
            }
        }
    };
}

define_simple_step!(TokStep, "test:tok", (View,), TokLayer, (v,) => v, "tok");
define_simple_step!(PosStep, "test:pos", (View, TokLayer), PosLayer, (v, _) => v, "pos");
define_simple_step!(SlowChildStep, "test:slow-child", (View, SlowALayer), SlowBLayer, (v, _) => v, "slow-b");
define_simple_step!(FastRootStep, "test:fast-root", (View,), FastALayer, (v,) => v, "fast-a");
define_simple_step!(FastMidStep, "test:fast-mid", (View, FastALayer), FastBLayer, (v, _) => v, "fast-b");

macro_rules! define_logged_step {
    ($step:ident, $id:expr, $reads:ty, $out:ident, $fetch:pat => $view:ident) => {
        struct $step(Arc<Mutex<Vec<&'static str>>>);
        impl Annotator for $step {
            const ID: &'static str = $id;
            type Reads = $reads;
            type Writes = ($out,);
            fn annotate(&self, $fetch: Refs<'_, Self>) -> Result<Self::Writes, Error> {
                self.0.lock().unwrap().push(Self::ID);
                Ok(($out(TagData::single($view.id(), $view.text().len() as u32, $id)),))
            }
        }
    };
}

type Vtp = (View, TokLayer, PosLayer);
type Vtd = (View, TokLayer, DepLayer);

define_logged_step!(LemmaStep, "test:lemma-leaf", Vtp, LemmaLayer, (v, _, _) => v);
define_logged_step!(StopwordStep, "test:stopword-leaf", Vtp, StopwordLayer, (v, _, _) => v);
define_logged_step!(DepParseStep, "turbo-depparse:parser", Vtp, DepLayer, (v, _, _) => v);
define_logged_step!(RelationStep, "turbo-relation:dep", Vtd, RelationLayer, (v, _, _) => v);
define_logged_step!(SentimentTargetStep, "turbo-doccat:aspect-target", Vtd, SentimentTargetLayer, (v, _, _) => v);
define_logged_step!(ClauseStep, "turbo-chunker:clause", Vtd, ClauseLayer, (v, _, _) => v);
define_logged_step!(ModalityStep, "turbo-doccat:modality", Vtd, ModalityLayer, (v, _, _) => v);

#[test]
fn dag_plan_records_exact_edges_and_schedules_depparse_early_on_critical_path() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let analyzer = Analyzer::builder()
        .add(TokStep)
        .add(PosStep)
        .add(LemmaStep(Arc::clone(&log)))
        .add(StopwordStep(Arc::clone(&log)))
        .add(DepParseStep(Arc::clone(&log)))
        .add(RelationStep(Arc::clone(&log)))
        .add(SentimentTargetStep(Arc::clone(&log)))
        .add(ClauseStep(Arc::clone(&log)))
        .add(ModalityStep(Arc::clone(&log)))
        .build()
        .unwrap();

    assert_eq!(
        analyzer.plan(),
        vec![
            vec!["test:tok"],
            vec!["test:pos"],
            vec!["test:lemma-leaf", "test:stopword-leaf", "turbo-depparse:parser"],
            vec!["turbo-relation:dep", "turbo-doccat:aspect-target", "turbo-chunker:clause", "turbo-doccat:modality"],
        ]
    );

    let dag = analyzer.dag_plan();
    assert_eq!(dag[4].id, "turbo-depparse:parser");
    assert_eq!(dag[4].predecessors, vec![0, 1]);
    assert_eq!(dag[4].dependents, vec![5, 6, 7, 8]);
    assert_eq!(dag[4].cost_hint, 10);
    assert_eq!(dag[4].transitive_dependents, 4);
    assert!(dag[4].critical_path_cost > dag[2].critical_path_cost);
    assert_eq!(analyzer.dependents(1)[0], 4);

    let cp_order = analyzer.critical_path_order();
    let dep_pos = cp_order.iter().position(|&id| id == "turbo-depparse:parser").unwrap();
    let lemma_pos = cp_order.iter().position(|&id| id == "test:lemma-leaf").unwrap();
    assert!(dep_pos < lemma_pos, "depparse must precede leaf siblings on critical path: {cp_order:?}");

    let mut docs = vec![Document::new("Syntactic head-modifier arcs unlock downstream extractors.")];
    for res in analyzer.run_with_workers(&mut docs, 1) {
        res.unwrap();
    }
    let executed = log.lock().unwrap().clone();
    assert_eq!(executed[0], "turbo-depparse:parser");
}

struct SlowRootStep {
    fast_c_finished: Arc<AtomicBool>,
    fast_c_finished_while_slow_running: Arc<AtomicBool>,
}
impl Annotator for SlowRootStep {
    const ID: &'static str = "test:slow-root";
    type Reads = (View,);
    type Writes = (SlowALayer,);
    fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let deadline = Instant::now() + Duration::from_millis(80);
        while Instant::now() < deadline {
            if self.fast_c_finished.load(Ordering::Acquire) {
                self.fast_c_finished_while_slow_running.store(true, Ordering::Release);
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok((SlowALayer(TagData::single(v.id(), v.text().len() as u32, "slow-a")),))
    }
}

struct FastLeafStep {
    finished: Arc<AtomicBool>,
}
impl Annotator for FastLeafStep {
    const ID: &'static str = "test:fast-leaf";
    type Reads = (View, FastBLayer);
    type Writes = (FastCLayer,);
    fn annotate(&self, (v, _): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.finished.store(true, Ordering::Release);
        Ok((FastCLayer(TagData::single(v.id(), v.text().len() as u32, "fast-c")),))
    }
}

#[test]
fn zero_barrier_dag_completes_multi_stage_fast_branch_while_stage0_slow_branch_is_running() {
    let fast_c_done = Arc::new(AtomicBool::new(false));
    let overlapped = Arc::new(AtomicBool::new(false));
    let analyzer = Analyzer::builder()
        .workers(2)
        .add(SlowRootStep {
            fast_c_finished: Arc::clone(&fast_c_done),
            fast_c_finished_while_slow_running: Arc::clone(&overlapped),
        })
        .add(SlowChildStep)
        .add(FastRootStep)
        .add(FastMidStep)
        .add(FastLeafStep { finished: Arc::clone(&fast_c_done) })
        .build()
        .unwrap();

    let mut docs = vec![Document::new("Zero-barrier execution test.")];
    for res in analyzer.run(&mut docs) {
        res.unwrap();
    }
    assert!(overlapped.load(Ordering::Acquire), "stage-2 fast-leaf must finish while stage-0 slow-root is running");
    assert!(docs[0].get::<SlowBLayer>().is_some());
    assert!(docs[0].get::<FastCLayer>().is_some());
}

struct TrimPrefixViewStep;
impl Annotator for TrimPrefixViewStep {
    const ID: &'static str = "test:trim-view";
    type Reads = (View,);
    type Writes = (View,);
    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View,), Error> {
        let text = src.text();
        let mut vb = ViewBuilder::new(text);
        if text.starts_with("  ") {
            vb.keep(0);
            vb.replace(2, "");
        }
        let (out, alignment) = vb.finish();
        Ok((View::derived(src, out, alignment),))
    }
}

#[test]
fn bit_identical_outputs_view_alignments_and_wire_bytes_across_1_2_4_8_workers() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let analyzer = Analyzer::builder()
        .step(Bound::new(TrimPrefixViewStep).write(0, "norm_a"))
        .step(Bound::new(TrimPrefixViewStep).write(0, "norm_b"))
        .step(Bound::new(TokStep).read(0, "norm_a"))
        .step(Bound::new(PosStep).read(0, "norm_a"))
        .step(Bound::new(LemmaStep(Arc::clone(&log))).read(0, "norm_a"))
        .step(Bound::new(StopwordStep(Arc::clone(&log))).read(0, "norm_a"))
        .step(Bound::new(DepParseStep(Arc::clone(&log))).read(0, "norm_a"))
        .step(Bound::new(RelationStep(Arc::clone(&log))).read(0, "norm_a"))
        .step(Bound::new(SentimentTargetStep(Arc::clone(&log))).read(0, "norm_a"))
        .step(Bound::new(ClauseStep(Arc::clone(&log))).read(0, "norm_a"))
        .step(Bound::new(ModalityStep(Arc::clone(&log))).read(0, "norm_a"))
        .step(Bound::new(FastRootStep).read(0, "norm_b"))
        .step(Bound::new(FastMidStep).read(0, "norm_b"))
        .step(Bound::new(FastLeafStep { finished: Arc::new(AtomicBool::new(false)) }).read(0, "norm_b"))
        .build()
        .unwrap();

    let codecs = test_codecs();
    let inputs = ["  First document with leading spaces.", "Second document.", "  Third document with two sentences."];
    let mut staged_docs: Vec<Document> = inputs.iter().map(|&s| Document::new(s)).collect();
    for r in analyzer.run_staged(&mut staged_docs) {
        r.unwrap();
    }
    let staged_wire: Vec<Vec<u8>> =
        staged_docs.iter().map(|d| export(d, &codecs).unwrap().to_bytes().unwrap()).collect();

    for workers in [1usize, 2, 4, 8] {
        let mut docs: Vec<Document> = inputs.iter().map(|&s| Document::new(s)).collect();
        for r in analyzer.run_with_workers(&mut docs, workers) {
            r.unwrap();
        }
        for (doc_idx, doc) in docs.iter().enumerate() {
            let norm_a = doc.get_key(&Key::<View>::named("norm_a")).unwrap();
            let mapped = doc.to_source(norm_a.id(), Span::new(0, norm_a.text().len() as u32));
            let expected_start = if inputs[doc_idx].starts_with("  ") { 2 } else { 0 };
            assert_eq!(mapped, Span::new(expected_start, inputs[doc_idx].len() as u32));
            assert_eq!(doc.from_source(norm_a.id(), mapped), Span::new(0, norm_a.text().len() as u32));
            assert_eq!(
                export(doc, &codecs).unwrap().to_bytes().unwrap(),
                staged_wire[doc_idx],
                "mismatch at {workers} workers"
            );
        }
    }
}

struct FailOrPanicBranchStep;
impl Annotator for FailOrPanicBranchStep {
    const ID: &'static str = "test:failing-branch";
    type Reads = (View,);
    type Writes = (SlowALayer,);
    fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        assert!(!v.text().contains("panics"), "synthetic panic on {}", v.text());
        if v.text().contains("fails") {
            return Err(Error::Annotator { annotator: Self::ID, detail: format!("synthetic error on {}", v.text()) });
        }
        Ok((SlowALayer(TagData::single(v.id(), v.text().len() as u32, "ok")),))
    }
}

#[test]
fn partial_failure_and_panic_isolation_matches_across_1_2_4_8_workers() {
    for workers in [1usize, 2, 4, 8] {
        let analyzer = Analyzer::builder()
            .add(FailOrPanicBranchStep)
            .add(SlowChildStep)
            .add(FastRootStep)
            .add(FastMidStep)
            .add(FastLeafStep { finished: Arc::new(AtomicBool::new(false)) })
            .build()
            .unwrap();

        let mut docs = vec![Document::new("doc0 panics"), Document::new("doc1 fails"), Document::new("doc2 succeeds")];
        let results = analyzer.run_with_workers(&mut docs, workers);
        assert!(results[0].is_err() && results[1].is_err() && results[2].is_ok());
        for d in [0usize, 1] {
            assert!(matches!(docs[d].status(Key::<SlowALayer>::DEFAULT.id()), Some(Status::Failed(_))));
            assert!(matches!(docs[d].status(Key::<SlowBLayer>::DEFAULT.id()), Some(Status::Skipped(_))));
            assert_eq!(docs[d].status(Key::<FastCLayer>::DEFAULT.id()), Some(&Status::Complete));
        }
        assert_eq!(docs[2].status(Key::<SlowBLayer>::DEFAULT.id()), Some(&Status::Complete));
    }
}

struct ThreadRecorderStep<W>(Arc<Mutex<HashSet<ThreadId>>>, std::marker::PhantomData<W>);
impl Annotator for ThreadRecorderStep<FastALayer> {
    const ID: &'static str = "test:rec-a";
    type Reads = (View,);
    type Writes = (FastALayer,);
    fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.0.lock().unwrap().insert(thread::current().id());
        Ok((FastALayer(TagData::single(v.id(), v.text().len() as u32, "a")),))
    }
}
impl Annotator for ThreadRecorderStep<SlowALayer> {
    const ID: &'static str = "test:rec-b";
    type Reads = (View,);
    type Writes = (SlowALayer,);
    fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.0.lock().unwrap().insert(thread::current().id());
        Ok((SlowALayer(TagData::single(v.id(), v.text().len() as u32, "b")),))
    }
}

#[test]
fn bounded_worker_pool_reuses_fixed_threads_across_stages_and_batches() {
    let pool = WorkerPool::new(3);
    assert_eq!(pool.thread_count(), 3);
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let analyzer = Analyzer::builder()
        .worker_pool(pool)
        .add(ThreadRecorderStep::<FastALayer>(Arc::clone(&seen), std::marker::PhantomData))
        .add(ThreadRecorderStep::<SlowALayer>(Arc::clone(&seen), std::marker::PhantomData))
        .add(FastMidStep)
        .add(SlowChildStep)
        .build()
        .unwrap();

    for _ in 0..12 {
        let mut docs = vec![Document::new("bounded thread pool check")];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
    }
    let threads = seen.lock().unwrap();
    assert!(!threads.is_empty() && threads.len() <= 3, "saw {} worker threads", threads.len());
}

struct TimedStep<R, W>(u64, u32, std::marker::PhantomData<(R, W)>);
impl<R, W> TimedStep<R, W> {
    const fn new(us: u64, cost: u32) -> Self {
        Self(us, cost, std::marker::PhantomData)
    }
    fn spin(&self, v: &View, tag: &'static str) -> TagData {
        let target = Instant::now() + Duration::from_micros(self.0);
        while Instant::now() < target {
            std::hint::spin_loop();
        }
        TagData::single(v.id(), v.text().len() as u32, tag)
    }
}
macro_rules! impl_timed_annotator {
    ($id:expr, $reads:ty, $out:ident, $fetch:pat => $view:ident) => {
        impl Annotator for TimedStep<$reads, $out> {
            const ID: &'static str = $id;
            type Reads = $reads;
            type Writes = ($out,);
            fn cost_hint(&self) -> u32 {
                self.1
            }
            fn annotate(&self, $fetch: Refs<'_, Self>) -> Result<Self::Writes, Error> {
                Ok(($out(self.spin($view, $id)),))
            }
        }
    };
}

impl_timed_annotator!("bench:slow-stage0", (View,), SlowALayer, (v,) => v);
impl_timed_annotator!("bench:slow-stage1", (View, SlowALayer), SlowBLayer, (v, _) => v);
impl_timed_annotator!("bench:fast-stage0", (View,), FastALayer, (v,) => v);
impl_timed_annotator!("bench:fast-stage1", (View, FastALayer), FastBLayer, (v, _) => v);
impl_timed_annotator!("bench:fast-stage2", (View, FastBLayer), FastCLayer, (v, _) => v);

#[test]
fn staged_vs_dag_scheduler_latency_and_throughput_benchmark() {
    let pool = WorkerPool::new(2);
    let analyzer = Analyzer::builder()
        .worker_pool(pool.clone())
        .add(TimedStep::<(View,), SlowALayer>::new(3000, 3))
        .add(TimedStep::<(View, SlowALayer), SlowBLayer>::new(500, 1))
        .add(TimedStep::<(View,), FastALayer>::new(1000, 2))
        .add(TimedStep::<(View, FastALayer), FastBLayer>::new(2500, 5))
        .add(TimedStep::<(View, FastBLayer), FastCLayer>::new(1500, 3))
        .build()
        .unwrap();

    let mut staged_lat: Vec<Duration> = (0..9)
        .map(|_| {
            let mut d = vec![Document::new("single query benchmark")];
            let t = Instant::now();
            analyzer.run_staged(&mut d);
            t.elapsed()
        })
        .collect();
    let mut dag_lat: Vec<Duration> = (0..9)
        .map(|_| {
            let mut d = vec![Document::new("single query benchmark")];
            let t = Instant::now();
            analyzer.run_with_pool(&mut d, &pool);
            t.elapsed()
        })
        .collect();
    staged_lat.sort();
    dag_lat.sort();
    let batch_inputs = ["doc a", "doc b", "doc c", "doc d"];
    let t_staged_batch = Instant::now();
    for _ in 0..5 {
        let mut batch: Vec<Document> = batch_inputs.iter().map(|&s| Document::new(s)).collect();
        analyzer.run_staged(&mut batch);
    }
    let staged_batch_dur = t_staged_batch.elapsed();
    let t_dag_batch = Instant::now();
    for _ in 0..5 {
        let mut batch: Vec<Document> = batch_inputs.iter().map(|&s| Document::new(s)).collect();
        analyzer.run_with_pool(&mut batch, &pool);
    }
    let dag_batch_dur = t_dag_batch.elapsed();
    eprintln!(
        "BENCHMARK single-doc: staged p50={:?} p99={:?} | dag p50={:?} p99={:?} || 20-doc batch: staged={:?} | dag={:?}",
        staged_lat[4], staged_lat[8], dag_lat[4], dag_lat[8], staged_batch_dur, dag_batch_dur
    );
    assert!(dag_lat[4] < staged_lat[4], "expected DAG p50 ({:?}) < staged p50 ({:?})", dag_lat[4], staged_lat[4]);
    assert!(dag_batch_dur < staged_batch_dur, "expected DAG batch ({dag_batch_dur:?}) < staged ({staged_batch_dur:?})");
}
