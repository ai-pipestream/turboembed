//! Which graph a load initializes, the process's native blobs, and how
//! long each step of the last load took.
//!
//! A load initializes the NGRAPH_LITE graph the driver compiled, unless
//! the library is built with `npu-native`. Then a graph extension that
//! exports a compiled graph (`pfnGetNativeBinary2` or
//! `pfnGetNativeBinary`) loads the IR as `ZE_GRAPH_FORMAT_NATIVE`: from
//! the blob this process kept for the same key, else from the blob the
//! NGRAPH_LITE compile just produced. It is opt-in because the blobs are
//! kept per process: a process's first load of an IR compiles, exports
//! and creates the graph again, which is slower than NGRAPH_LITE alone,
//! and only a later load of the same IR in that process can gain. The device's
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

/// The steps of a load, so the choice between them is tested without a
/// device.
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

/// The load: NATIVE when `native` names a key, from `cache` first, else
/// from this compile's blob; NGRAPH_LITE when there is no key or any
/// native step is refused. A kept blob that is refused is dropped. The
/// NGRAPH_LITE graph is destroyed only once a native graph has been
/// initialized in its place.
pub(crate) fn select<L: Loader>(l: &mut L, native: Option<&Key>, cache: &Mutex<Cache>) -> Res<(L::Out, Source)> {
    if let Some(key) = native {
        let kept = timed("lookup", || lock(cache).get(key));
        if let Some(blob) = kept {
            match l.create_native(&blob).and_then(|g| l.finish(g, Source::Cache)) {
                Ok(out) => return Ok((out, Source::Cache)),
                Err(e) => {
                    lock(cache).forget(key);
                    l.note(&format!(
                        "the kept native blob was refused and is dropped; compiling the IR: {}",
                        e.message
                    ));
                }
            }
        }
    }
    let lite = l.compile()?;
    if let Some(key) = native {
        match l.export(&lite) {
            Ok(blob) => {
                let blob = Arc::new(blob);
                match l.create_native(&blob).and_then(|g| l.finish(g, Source::Compiled)) {
                    Ok(out) => {
                        l.destroy(lite);
                        lock(cache).keep(key.clone(), blob);
                        return Ok((out, Source::Compiled));
                    }
                    Err(e) => {
                        l.note(&format!("the native blob was refused; the NGRAPH_LITE graph runs: {}", e.message))
                    }
                }
            }
            Err(e) => l.note(&format!("no native blob; the NGRAPH_LITE graph runs: {}", e.message)),
        }
    }
    l.finish(lite, Source::Lite).map(|out| (out, Source::Lite))
}

thread_local! {
    static NATIVE: Cell<Option<bool>> = const { Cell::new(None) };
    static LAST: Cell<Option<Source>> = const { Cell::new(None) };
    static PHASES: RefCell<Vec<(&'static str, Duration)>> = const { RefCell::new(Vec::new()) };
}

/// Whether this thread's loads take the native path: the build's choice
/// (`npu-native`), unless the tests' switch says otherwise.
pub(crate) fn native_wanted() -> bool {
    NATIVE.with(Cell::get).unwrap_or(cfg!(feature = "npu-native"))
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
/// when `Some(false)`, and as built when `None`. Built only with
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
