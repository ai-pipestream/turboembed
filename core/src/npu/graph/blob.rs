//! Which graph a load initializes, the process's native blobs, and how
//! long each step of the last load took.
//!
//! When the graph extension exports a compiled graph
//! (`pfnGetNativeBinary2` or `pfnGetNativeBinary`), a load initializes
//! the IR as `ZE_GRAPH_FORMAT_NATIVE`: from the blob this process kept
//! for the same key, else from the blob the NGRAPH_LITE compile just
//! produced. Otherwise it initializes the NGRAPH_LITE graph. A first
//! load costs about what an NGRAPH_LITE load does (export and create are
//! a few ms next to the compile), and a later load of the same IR in the
//! process skips the compile (docs/npu.md has the device times). The device's
//! `graphFormatsSupported` is not read for this. Intel's driver sets it
//! to NATIVE and then overwrites it with NGRAPH_LITE whenever its
//! compiler loads (intel/linux-npu-driver,
//! umd/level_zero_driver/source/ext/graph.cpp, getDeviceGraphProperties),
//! and takes NATIVE at create time either way. Any refusal on the native
//! path falls back to the NGRAPH_LITE graph, so a load the compiler
//! handles is never failed here.
//!
//! The blobs are kept in memory only. A blob carries the weights, which
//! is hundreds of megabytes for a large model, and the one cache
//! directory the library knows, TURBO_AUTOTUNE_CACHE, holds small JSON
//! entries. The key names the IR's files by the SHA-256 the manifest
//! records, which the core verified before model_load: nothing is hashed
//! here.

use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::Res;

/// What a native blob is valid for: both files of the IR by the SHA-256
/// the manifest records, the build flags, the compiler, the driver and
/// the device's arch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    pub xml: [u8; 32],
    pub bin: [u8; 32],
    pub flags: String,
    pub compiler: (u16, u16),
    pub driver: String,
    pub arch: String,
}

impl Key {
    pub fn new(xml: [u8; 32], bin: [u8; 32], flags: &str, compiler: (u16, u16), driver: &str, arch: &str) -> Key {
        Key { xml, bin, flags: flags.to_owned(), compiler, driver: driver.to_owned(), arch: arch.to_owned() }
    }
}

/// The most blob bytes kept at once. The oldest entry goes first, and a
/// blob larger than this is not kept.
pub(crate) const BUDGET: usize = 1 << 30;

/// Native blobs by key, oldest first, within a byte budget.
pub(crate) struct Cache {
    entries: Vec<(Key, Arc<Vec<u8>>)>,
    bytes: usize,
    budget: usize,
}

impl Cache {
    pub const fn new(budget: usize) -> Cache {
        Cache { entries: Vec::new(), bytes: 0, budget }
    }

    /// The blob kept for `key`, shared: a hit copies nothing.
    pub fn get(&self, key: &Key) -> Option<Arc<Vec<u8>>> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, b)| Arc::clone(b))
    }

    pub fn forget(&mut self, key: &Key) {
        if let Some(i) = self.entries.iter().position(|(k, _)| k == key) {
            let (_, b) = self.entries.remove(i);
            self.bytes -= b.len();
        }
    }

    pub fn keep(&mut self, key: Key, blob: Arc<Vec<u8>>) {
        self.forget(&key);
        if blob.is_empty() || blob.len() > self.budget {
            return;
        }
        while self.bytes + blob.len() > self.budget {
            let (_, b) = self.entries.remove(0);
            self.bytes -= b.len();
        }
        self.bytes += blob.len();
        self.entries.push((key, blob));
    }

    #[cfg(test)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// The process's blobs: devices and contexts share them, since the key
/// names the driver and the arch.
pub(crate) static CACHE: Mutex<Cache> = Mutex::new(Cache::new(BUDGET));

fn lock(cache: &Mutex<Cache>) -> MutexGuard<'_, Cache> {
    cache.lock().unwrap_or_else(|p| p.into_inner())
}

/// Where the graph a load initialized came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// NATIVE, from a kept blob: nothing was compiled.
    Cache,
    /// NATIVE, from the blob this load's compile produced.
    Compiled,
    /// NGRAPH_LITE, the compiled IR itself.
    Lite,
}

impl Source {
    /// `NATIVE` or `NGRAPH_LITE`, as TURBO_NPU_GRAPH_FORMAT names it.
    pub fn format(self) -> &'static str {
        crate::npu::format_name(self != Source::Lite)
    }

    pub fn name(self) -> &'static str {
        match self {
            Source::Cache => "cache",
            Source::Compiled => "compiled",
            Source::Lite => "lite",
        }
    }
}

/// The steps of a load on the device, driven by `select`.
pub(crate) trait Loader {
    type Graph;
    type Out;
    /// The IR compiled as NGRAPH_LITE. Its refusal fails the load.
    fn compile(&mut self) -> Res<Self::Graph>;
    /// The compiled graph's native blob, copied.
    fn export(&mut self, lite: &Self::Graph) -> Res<Vec<u8>>;
    fn create_native(&mut self, blob: &[u8]) -> Res<Self::Graph>;
    /// The graph described and initialized. On a refusal the graph has
    /// been destroyed.
    fn finish(&mut self, graph: Self::Graph, source: Source) -> Res<Self::Out>;
    fn destroy(&mut self, graph: Self::Graph);
    fn note(&mut self, message: &str);
}

/// One step of a load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// A NATIVE graph created and initialized from the kept blob.
    Kept,
    /// The IR compiled as NGRAPH_LITE.
    Compile,
    /// The compiled graph's native blob exported.
    Export,
    /// A NATIVE graph created and initialized from the exported blob.
    Fresh,
    /// The NGRAPH_LITE graph initialized.
    Lite,
}

/// What follows a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Next {
    Step(Step),
    /// The load is done: the graph that runs came from here.
    Done(Source),
    /// The load fails with the step's error.
    Fail,
}

/// The first step: the kept blob when the load wants NATIVE and one is
/// kept for its key, else the compile.
pub(crate) fn first(native: bool, kept: bool) -> Step {
    if native && kept { Step::Kept } else { Step::Compile }
}

/// The step after `step`, by whether it succeeded. A refused kept blob
/// compiles the IR; any refusal after the compile initializes the
/// NGRAPH_LITE graph; only the compile and the NGRAPH_LITE initialize
/// fail the load.
pub(crate) fn next(step: Step, native: bool, ok: bool) -> Next {
    match (step, ok) {
        (Step::Kept, true) => Next::Done(Source::Cache),
        (Step::Kept, false) => Next::Step(Step::Compile),
        (Step::Compile, true) => Next::Step(if native { Step::Export } else { Step::Lite }),
        (Step::Export, true) => Next::Step(Step::Fresh),
        (Step::Fresh, true) => Next::Done(Source::Compiled),
        (Step::Export | Step::Fresh, false) => Next::Step(Step::Lite),
        (Step::Lite, true) => Next::Done(Source::Lite),
        (Step::Compile | Step::Lite, false) => Next::Fail,
    }
}

/// The load, step by step as `first` and `next` say. A kept blob that is
/// refused is dropped. The NGRAPH_LITE graph is destroyed only once a
/// native graph has been initialized in its place, and that graph's blob
/// is kept.
pub(crate) fn select<L: Loader>(l: &mut L, native: Option<&Key>, cache: &Mutex<Cache>) -> Res<(L::Out, Source)> {
    let kept = native.and_then(|key| timed("lookup", || lock(cache).get(key)));
    let mut step = first(native.is_some(), kept.is_some());
    let (mut lite, mut blob, mut out) = (None, None, None);
    loop {
        let done = match step {
            Step::Kept => match &kept {
                Some(b) => l.create_native(b).and_then(|g| l.finish(g, Source::Cache)).map(|o| out = Some(o)),
                None => Err(super::fail(crate::status::RUNTIME, "no kept blob")),
            },
            Step::Compile => l.compile().map(|g| lite = Some(g)),
            Step::Export => match &lite {
                Some(g) => l.export(g).map(|b| blob = Some(Arc::new(b))),
                None => Err(super::fail(crate::status::RUNTIME, "nothing compiled to export")),
            },
            Step::Fresh => match &blob {
                Some(b) => l.create_native(b).and_then(|g| l.finish(g, Source::Compiled)).map(|o| out = Some(o)),
                None => Err(super::fail(crate::status::RUNTIME, "no exported blob")),
            },
            Step::Lite => match lite.take() {
                Some(g) => l.finish(g, Source::Lite).map(|o| out = Some(o)),
                None => Err(super::fail(crate::status::RUNTIME, "nothing compiled to initialize")),
            },
        };
        if let Err(e) = &done {
            match step {
                Step::Kept => {
                    if let Some(key) = native {
                        lock(cache).forget(key);
                    }
                    l.note(&format!(
                        "the kept native blob was refused and is dropped; compiling the IR: {}",
                        e.message
                    ));
                }
                Step::Export => l.note(&format!("no native blob; the NGRAPH_LITE graph runs: {}", e.message)),
                Step::Fresh => {
                    l.note(&format!("the native blob was refused; the NGRAPH_LITE graph runs: {}", e.message))
                }
                Step::Compile | Step::Lite => {}
            }
        }
        match next(step, native.is_some(), done.is_ok()) {
            Next::Step(s) => step = s,
            Next::Fail => return Err(done.err().unwrap_or_else(|| super::fail(crate::status::RUNTIME, "load failed"))),
            Next::Done(source) => {
                if source == Source::Compiled {
                    if let Some(g) = lite.take() {
                        l.destroy(g);
                    }
                    if let (Some(key), Some(b)) = (native, blob.take()) {
                        lock(cache).keep(key.clone(), b);
                    }
                }
                return out.map(|o| (o, source)).ok_or_else(|| super::fail(crate::status::RUNTIME, "no graph"));
            }
        }
    }
}

thread_local! {
    static NATIVE: Cell<Option<bool>> = const { Cell::new(None) };
    static LAST: Cell<Option<Source>> = const { Cell::new(None) };
    static PHASES: RefCell<Vec<(&'static str, Duration)>> = const { RefCell::new(Vec::new()) };
}

/// Whether this thread's loads take the native path: yes, unless the
/// tests' switch says otherwise.
pub(crate) fn native_wanted() -> bool {
    NATIVE.with(Cell::get).unwrap_or(true)
}

pub(crate) fn record(source: Source) {
    LAST.with(|c| c.set(Some(source)));
}

/// A new load: its steps' times start empty.
pub(crate) fn begin() {
    PHASES.with(|p| p.borrow_mut().clear());
}

/// `f`, its time kept under `name` for this thread's load.
pub(crate) fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let out = f();
    PHASES.with(|p| p.borrow_mut().push((name, t.elapsed())));
    out
}

/// This thread's last load's steps, as the debug log says them.
pub(crate) fn phases() -> String {
    PHASES.with(|p| {
        p.borrow().iter().map(|(n, d)| format!("{n} {:.1} ms", d.as_secs_f64() * 1e3)).collect::<Vec<_>>().join(", ")
    })
}

/// This thread's loads take the native path when `Some(true)`, never
/// when `Some(false)`, and by default (native) when `None`. Built only with
/// `internals`.
#[cfg(feature = "internals")]
pub fn set_native(on: Option<bool>) {
    NATIVE.with(|c| c.set(on));
}

/// Where this thread's last npu load came from: `cache`, `compiled` or
/// `lite`. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn last_load() -> Option<&'static str> {
    LAST.with(|c| c.get().map(Source::name))
}

/// The steps of this thread's last npu load and how long each took, in
/// order: `ir`, `key`, `lookup`, `container`, `compile`, `export`,
/// `create`, `initialize`, as the load took them. Built only with
/// `internals`.
#[cfg(feature = "internals")]
pub fn last_phases() -> Vec<(&'static str, Duration)> {
    PHASES.with(|p| p.borrow().clone())
}
