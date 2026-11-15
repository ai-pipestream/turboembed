//! Regression tests for nested `Analyzer::run` calls from inside
//! [`WorkerPool`] worker threads (#2337).

use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use turbo_analysis::{
    Analyzer, Anchor, Annotator, Document, Error, Layer, Overlap, Refs, TextSpans, View, ViewId, WorkerPool,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Tag {
    view: ViewId,
    len: u32,
}

macro_rules! define_layers {
    ($($ty:ident => $id:expr),+ $(,)?) => {
        $(
            #[derive(Clone, Debug, PartialEq, Eq)]
            struct $ty(Tag);
            impl Layer for $ty {
                const ID: &'static str = $id;
                const OVERLAP: Overlap = Overlap::Any;
                fn anchor() -> Anchor { Anchor::Text }
                fn len(&self) -> usize { 1 }
                fn check(&self) -> Result<(), String> { Ok(()) }
                fn text_spans(&self) -> Option<TextSpans<'_>> {
                    Some(TextSpans {
                        view: self.0.view,
                        start: std::slice::from_ref(&0),
                        end: std::slice::from_ref(&self.0.len),
                        frag_start: &[],
                        frag_end: &[],
                        frag_item_end: &[],
                    })
                }
            }
        )+
    };
}

define_layers!(
    InnerA => "nested:inner-a",
    InnerB => "nested:inner-b",
    Outer0 => "nested:outer-0",
    Outer1 => "nested:outer-1",
    Outer2 => "nested:outer-2",
    Outer3 => "nested:outer-3",
    Outer4 => "nested:outer-4",
    Outer5 => "nested:outer-5",
    Outer6 => "nested:outer-6",
    Outer7 => "nested:outer-7",
    Top0 => "nested:top-0",
    Top1 => "nested:top-1",
);

struct InnerStep<W>(PhantomData<W>);
impl<W: Layer + From<Tag> + 'static> InnerStep<W> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}
impl From<Tag> for InnerA {
    fn from(t: Tag) -> Self {
        Self(t)
    }
}
impl From<Tag> for InnerB {
    fn from(t: Tag) -> Self {
        Self(t)
    }
}

impl Annotator for InnerStep<InnerA> {
    const ID: &'static str = "nested:inner-step-a";
    type Reads = (View,);
    type Writes = (InnerA,);
    fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        Ok((InnerA::from(Tag { view: v.id(), len: v.text().len() as u32 }),))
    }
}

impl Annotator for InnerStep<InnerB> {
    const ID: &'static str = "nested:inner-step-b";
    type Reads = (View,);
    type Writes = (InnerB,);
    fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        Ok((InnerB::from(Tag { view: v.id(), len: v.text().len() as u32 }),))
    }
}

struct OuterStep<W> {
    inner: Arc<Analyzer>,
    _marker: PhantomData<W>,
}

macro_rules! impl_outer_step {
    ($layer:ident, $id:expr) => {
        impl Annotator for OuterStep<$layer> {
            const ID: &'static str = $id;
            type Reads = (View,);
            type Writes = ($layer,);
            fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
                let mut inner_docs = vec![Document::new(v.text())];
                for res in self.inner.run(&mut inner_docs) {
                    res?;
                }
                assert!(inner_docs[0].get::<InnerA>().is_some());
                assert!(inner_docs[0].get::<InnerB>().is_some());
                Ok(($layer(Tag { view: v.id(), len: v.text().len() as u32 }),))
            }
        }
    };
}

impl_outer_step!(Outer0, "nested:outer-step-0");
impl_outer_step!(Outer1, "nested:outer-step-1");
impl_outer_step!(Outer2, "nested:outer-step-2");
impl_outer_step!(Outer3, "nested:outer-step-3");
impl_outer_step!(Outer4, "nested:outer-step-4");
impl_outer_step!(Outer5, "nested:outer-step-5");
impl_outer_step!(Outer6, "nested:outer-step-6");
impl_outer_step!(Outer7, "nested:outer-step-7");

/// The entry point a top step uses for the outer pipeline. Each one but
/// `Run` computes side-by-side steps on threads of its own.
#[derive(Clone, Copy, Debug)]
enum Middle {
    Run,
    Incremental,
    Progressive,
}

/// A step one level further out: it runs the eight-step outer pipeline,
/// whose own steps each run the inner pipeline.
struct TopStep<W> {
    outer: Arc<Analyzer>,
    how: Middle,
    _marker: PhantomData<W>,
}

macro_rules! impl_top_step {
    ($layer:ident, $id:expr) => {
        impl Annotator for TopStep<$layer> {
            const ID: &'static str = $id;
            type Reads = (View,);
            type Writes = ($layer,);
            fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
                let mut docs = vec![Document::new(v.text())];
                match self.how {
                    Middle::Run => self.outer.run(&mut docs).into_iter().collect::<Result<(), Error>>()?,
                    Middle::Incremental => {
                        self.outer.run_incremental(&mut docs).into_iter().collect::<Result<Vec<_>, Error>>()?;
                    }
                    Middle::Progressive => self
                        .outer
                        .run_progressive(&mut docs, &AtomicBool::new(false), |_| {})
                        .into_iter()
                        .collect::<Result<(), Error>>()?,
                }
                assert!(docs[0].get::<Outer0>().is_some());
                assert!(docs[0].get::<Outer7>().is_some());
                Ok(($layer(Tag { view: v.id(), len: v.text().len() as u32 }),))
            }
        }
    };
}

impl_top_step!(Top0, "nested:top-step-0");
impl_top_step!(Top1, "nested:top-step-1");

fn build_top(pool: Option<WorkerPool>, how: Middle) -> Analyzer {
    let outer = Arc::new(build_outer(pool.clone()));
    let mut builder = Analyzer::builder()
        .add(TopStep::<Top0> { outer: Arc::clone(&outer), how, _marker: PhantomData })
        .add(TopStep::<Top1> { outer, how, _marker: PhantomData });
    if let Some(p) = pool {
        builder = builder.worker_pool(p);
    }
    builder.build().unwrap()
}

fn run_top(top: &Analyzer) {
    let mut docs = vec![Document::new("Top step runs outer steps that run inner steps.")];
    for res in top.run(&mut docs) {
        res.unwrap();
    }
    assert!(docs[0].get::<Top0>().is_some());
    assert!(docs[0].get::<Top1>().is_some());
}

fn build_outer(pool: Option<WorkerPool>) -> Analyzer {
    let mut inner_builder = Analyzer::builder().add(InnerStep::<InnerA>::new()).add(InnerStep::<InnerB>::new());
    if let Some(p) = &pool {
        inner_builder = inner_builder.worker_pool(p.clone());
    }
    let inner = Arc::new(inner_builder.build().unwrap());

    let mut outer_builder = Analyzer::builder()
        .add(OuterStep::<Outer0> { inner: Arc::clone(&inner), _marker: PhantomData })
        .add(OuterStep::<Outer1> { inner: Arc::clone(&inner), _marker: PhantomData })
        .add(OuterStep::<Outer2> { inner: Arc::clone(&inner), _marker: PhantomData })
        .add(OuterStep::<Outer3> { inner: Arc::clone(&inner), _marker: PhantomData })
        .add(OuterStep::<Outer4> { inner: Arc::clone(&inner), _marker: PhantomData })
        .add(OuterStep::<Outer5> { inner: Arc::clone(&inner), _marker: PhantomData })
        .add(OuterStep::<Outer6> { inner: Arc::clone(&inner), _marker: PhantomData })
        .add(OuterStep::<Outer7> { inner, _marker: PhantomData });
    if let Some(p) = pool {
        outer_builder = outer_builder.worker_pool(p);
    }
    outer_builder.build().unwrap()
}

fn run_with_timeout(name: &'static str, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(10)).unwrap_or_else(|_| panic!("{name} deadlocked after 10s"));
}

#[test]
fn nested_analyzer_run_across_eight_parallel_outer_steps_does_not_deadlock() {
    run_with_timeout("shared-pool nested run", || {
        let outer = build_outer(None);
        for _ in 0..4 {
            let mut docs = vec![Document::new("Eight parallel outer steps calling inner Analyzer::run.")];
            for res in outer.run(&mut docs) {
                res.unwrap();
            }
            assert!(docs[0].get::<Outer0>().is_some());
            assert!(docs[0].get::<Outer7>().is_some());
        }
    });

    run_with_timeout("single-worker-pool nested run", || {
        let outer = build_outer(Some(WorkerPool::new(1)));
        let mut docs = vec![Document::new("Single-worker pool nested Analyzer::run.")];
        for res in outer.run(&mut docs) {
            res.unwrap();
        }
        assert!(docs[0].get::<Outer0>().is_some());
        assert!(docs[0].get::<Outer7>().is_some());
    });
}

/// Three levels deep: a pool worker runs a top step, which runs the outer
/// pipeline off the pool. Whichever entry point that is, it computes the
/// eight outer steps on threads of its own, and each of them runs the inner
/// pipeline. Those threads must count as pool work too, or the inner run
/// queues on the pool whose workers are all waiting for it.
#[test]
fn pipeline_nested_three_deep_does_not_deadlock_through_any_entry_point() {
    for how in [Middle::Run, Middle::Incremental, Middle::Progressive] {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            run_top(&build_top(Some(WorkerPool::new(1)), how));
            for _ in 0..4 {
                run_top(&build_top(None, how));
            }
            let _ = tx.send(());
        });
        rx.recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|_| panic!("three-deep run through {how:?} deadlocked after 10s"));
    }
}
