//! `Analyzer::run` without a bound [`WorkerPool`] keeps the per-run threads
//! it has on main: concurrent callers are not funnelled through one
//! process-wide pool. The pool is opt-in through the builder (#2337).

use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use turbo_analysis::{Analyzer, Anchor, Annotator, Document, Error, Layer, Overlap, Refs, TextSpans, View, ViewId};

macro_rules! define_layers {
    ($($ty:ident => $id:expr),+ $(,)?) => {
        $(
            #[derive(Clone, Debug, PartialEq, Eq)]
            struct $ty(ViewId, u32);
            impl Layer for $ty {
                const ID: &'static str = $id;
                const OVERLAP: Overlap = Overlap::Any;
                fn anchor() -> Anchor { Anchor::Text }
                fn len(&self) -> usize { 1 }
                fn check(&self) -> Result<(), String> { Ok(()) }
                fn text_spans(&self) -> Option<TextSpans<'_>> {
                    Some(TextSpans {
                        view: self.0,
                        start: std::slice::from_ref(&0),
                        end: std::slice::from_ref(&self.1),
                        frag_start: &[],
                        frag_end: &[],
                        frag_item_end: &[],
                    })
                }
            }
        )+
    };
}

define_layers!(Left => "default-run:left", Right => "default-run:right");

/// Counts the steps that are inside `annotate` at once and lets none of them
/// leave until `want` are, or the wait runs out.
struct Gate {
    inside: Mutex<usize>,
    all_inside: Condvar,
    want: usize,
    wait: Duration,
}

impl Gate {
    fn new(want: usize, wait: Duration) -> Arc<Gate> {
        Arc::new(Gate { inside: Mutex::new(0), all_inside: Condvar::new(), want, wait })
    }

    fn enter(&self) {
        let mut inside = self.inside.lock().unwrap();
        *inside += 1;
        self.all_inside.notify_all();
        let (inside, timed_out) = self.all_inside.wait_timeout_while(inside, self.wait, |n| *n < self.want).unwrap();
        assert!(!timed_out.timed_out(), "only {} of {} steps ran at once", *inside, self.want);
    }
}

/// Records the name of the thread it ran on, then waits at the gate.
struct GatedStep<W> {
    gate: Arc<Gate>,
    threads: Arc<Mutex<Vec<String>>>,
    _marker: std::marker::PhantomData<W>,
}

macro_rules! impl_gated_step {
    ($layer:ident, $id:expr) => {
        impl Annotator for GatedStep<$layer> {
            const ID: &'static str = $id;
            type Reads = (View,);
            type Writes = ($layer,);
            fn annotate(&self, (v,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
                self.threads.lock().unwrap().push(thread::current().name().unwrap_or("").to_owned());
                self.gate.enter();
                Ok(($layer(v.id(), v.text().len() as u32),))
            }
        }
    };
}

impl_gated_step!(Left, "default-run:left-step");
impl_gated_step!(Right, "default-run:right-step");

fn pipeline(gate: &Arc<Gate>, threads: &Arc<Mutex<Vec<String>>>, workers: Option<usize>) -> Analyzer {
    let mut builder = Analyzer::builder()
        .add(GatedStep::<Left> {
            gate: Arc::clone(gate),
            threads: Arc::clone(threads),
            _marker: std::marker::PhantomData,
        })
        .add(GatedStep::<Right> {
            gate: Arc::clone(gate),
            threads: Arc::clone(threads),
            _marker: std::marker::PhantomData,
        });
    if let Some(n) = workers {
        builder = builder.workers(n);
    }
    builder.build().unwrap()
}

/// Sixteen callers run a two-step pipeline at once, and every step waits
/// until all thirty-two are computing together. That needs thirty-two
/// threads: a shared pool of at most eight can never get there, and the
/// steps fail at the end of their wait.
#[test]
fn concurrent_callers_of_default_run_each_get_their_own_threads() {
    const CALLERS: usize = 16;
    let gate = Gate::new(CALLERS * 2, Duration::from_secs(10));
    let threads = Arc::new(Mutex::new(Vec::new()));
    let analyzer = Arc::new(pipeline(&gate, &threads, None));

    let callers: Vec<_> = (0..CALLERS)
        .map(|_| {
            let analyzer = Arc::clone(&analyzer);
            thread::spawn(move || {
                let mut docs = vec![Document::new("One request among many.")];
                analyzer.run(&mut docs).pop().unwrap()
            })
        })
        .collect();
    for caller in callers {
        caller.join().unwrap().expect("every step of every caller computes at the same time");
    }

    let threads = threads.lock().unwrap();
    assert_eq!(threads.len(), CALLERS * 2);
    assert!(threads.iter().all(|name| !name.starts_with("turbo-dag-worker")), "default run used a pool: {threads:?}");
}

/// The pool is still what runs the steps once a pipeline asks for one.
#[test]
fn a_pipeline_built_with_workers_runs_its_steps_on_the_pool() {
    let gate = Gate::new(2, Duration::from_secs(10));
    let threads = Arc::new(Mutex::new(Vec::new()));
    let analyzer = pipeline(&gate, &threads, Some(2));

    let mut docs = vec![Document::new("One request on a bound pool.")];
    analyzer.run(&mut docs).pop().unwrap().unwrap();

    let threads = threads.lock().unwrap();
    assert_eq!(threads.len(), 2);
    assert!(threads.iter().all(|name| name.starts_with("turbo-dag-worker")), "bound pool not used: {threads:?}");
}
