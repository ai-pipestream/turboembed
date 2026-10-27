//! The native path's choices and fallbacks, and the blob cache, with a
//! loader that only records what it was asked to do.

use std::sync::{Arc, Mutex};

use super::blob::*;
use super::{Res, fail};
use crate::status::RUNTIME;

/// Graphs are numbers: 1 is the compiled IR, 10 plus n the nth native
/// graph. `refuse` names the steps that fail.
#[derive(Default)]
struct Fake {
    refuse: Vec<&'static str>,
    blob: Vec<u8>,
    steps: Vec<String>,
    natives: u32,
    alive: Vec<u32>,
}

impl Fake {
    fn refusing(steps: &[&'static str]) -> Fake {
        Fake { refuse: steps.to_vec(), blob: vec![7; 16], ..Fake::default() }
    }

    fn step(&mut self, s: &str) -> Res<()> {
        self.steps.push(s.to_owned());
        if self.refuse.contains(&s) { Err(fail(RUNTIME, format!("{s} refused"))) } else { Ok(()) }
    }
}

impl Loader for Fake {
    type Graph = u32;
    type Out = u32;

    fn compile(&mut self) -> Res<u32> {
        self.step("compile")?;
        self.alive.push(1);
        Ok(1)
    }

    fn export(&mut self, _: &u32) -> Res<Vec<u8>> {
        self.step("export")?;
        Ok(self.blob.clone())
    }

    fn create_native(&mut self, blob: &[u8]) -> Res<u32> {
        let what = if blob == self.blob.as_slice() { "create" } else { "create-stale" };
        self.step(what)?;
        self.natives += 1;
        self.alive.push(10 + self.natives);
        Ok(10 + self.natives)
    }

    fn finish(&mut self, g: u32, source: Source) -> Res<u32> {
        let what = if g == 1 { "finish-lite" } else { "finish-native" };
        if let Err(e) = self.step(what) {
            self.destroy(g);
            return Err(e);
        }
        let _ = source;
        Ok(g)
    }

    fn destroy(&mut self, g: u32) {
        self.alive.retain(|&a| a != g);
    }

    fn note(&mut self, m: &str) {
        self.steps.push(format!("note: {m}"));
    }
}

/// The manifest's digests stand in for the files: nothing is hashed.
const XML: [u8; 32] = [1; 32];
const BIN: [u8; 32] = [2; 32];

fn key(flags: &str) -> Key {
    Key::new(XML, BIN, flags, (7, 28), "1.38.0", "arl-npu")
}

fn run(f: &mut Fake, native: bool, cache: &Mutex<Cache>) -> (Option<u32>, Option<Source>) {
    let k = key("f");
    match select(f, native.then_some(&k), cache) {
        Ok((g, s)) => (Some(g), Some(s)),
        Err(_) => (None, None),
    }
}

fn steps(f: &Fake) -> Vec<&str> {
    f.steps.iter().map(|s| if s.starts_with("note") { "note" } else { s.as_str() }).collect()
}

#[test]
fn with_no_export_the_compiled_ir_runs_and_nothing_is_kept() {
    let cache = Mutex::new(Cache::new(BUDGET));
    let mut f = Fake::refusing(&[]);
    assert_eq!(run(&mut f, false, &cache), (Some(1), Some(Source::Lite)));
    assert_eq!(steps(&f), ["compile", "finish-lite"]);
    assert_eq!(cache.lock().unwrap().bytes(), 0);
}

#[test]
fn an_export_loads_the_native_blob_keeps_it_and_drops_the_compiled_ir() {
    let cache = Mutex::new(Cache::new(BUDGET));
    let mut f = Fake::refusing(&[]);
    assert_eq!(run(&mut f, true, &cache), (Some(11), Some(Source::Compiled)));
    assert_eq!(steps(&f), ["compile", "export", "create", "finish-native"]);
    assert_eq!(f.alive, [11], "the NGRAPH_LITE graph is destroyed once the native one is initialized");
    assert_eq!(cache.lock().unwrap().bytes(), 16);
}

#[test]
fn a_second_load_takes_the_kept_blob_and_compiles_nothing() {
    let cache = Mutex::new(Cache::new(BUDGET));
    run(&mut Fake::refusing(&[]), true, &cache);
    let mut f = Fake::refusing(&[]);
    assert_eq!(run(&mut f, true, &cache), (Some(11), Some(Source::Cache)));
    assert_eq!(steps(&f), ["create", "finish-native"]);
}

#[test]
fn each_refusal_on_the_native_path_falls_back_to_the_compiled_ir() {
    for refused in ["export", "create", "finish-native"] {
        let cache = Mutex::new(Cache::new(BUDGET));
        let mut f = Fake::refusing(&[refused]);
        assert_eq!(run(&mut f, true, &cache), (Some(1), Some(Source::Lite)), "{refused}");
        assert_eq!(steps(&f).last(), Some(&"finish-lite"), "{refused}");
        assert!(steps(&f).contains(&"note"), "{refused}: the fallback is logged");
        assert_eq!(f.alive, [1], "{refused}: only the NGRAPH_LITE graph is left");
        assert_eq!(cache.lock().unwrap().bytes(), 0, "{refused}: a refused blob is not kept");
    }
}

#[test]
fn a_refused_kept_blob_is_dropped_and_the_ir_compiled_again() {
    let cache = Mutex::new(Cache::new(BUDGET));
    cache.lock().unwrap().keep(key("f"), Arc::new(vec![9; 8]));
    let mut f = Fake::refusing(&["create-stale"]);
    assert_eq!(run(&mut f, true, &cache), (Some(11), Some(Source::Compiled)));
    assert_eq!(steps(&f), ["create-stale", "note", "compile", "export", "create", "finish-native"]);
    let kept = cache.lock().unwrap().get(&key("f")).expect("the fresh blob is kept");
    assert_eq!(*kept, vec![7; 16]);
}

#[test]
fn a_kept_blob_that_fails_to_initialize_falls_back_too() {
    let cache = Mutex::new(Cache::new(BUDGET));
    cache.lock().unwrap().keep(key("f"), Arc::new(vec![7; 16]));
    let mut f = Fake::refusing(&["finish-native"]);
    assert_eq!(run(&mut f, true, &cache), (Some(1), Some(Source::Lite)));
    assert_eq!(f.alive, [1]);
    assert_eq!(cache.lock().unwrap().bytes(), 0);
}

#[test]
fn a_compile_refusal_still_fails_the_load() {
    let cache = Mutex::new(Cache::new(BUDGET));
    for native in [false, true] {
        let mut f = Fake::refusing(&["compile"]);
        assert_eq!(run(&mut f, native, &cache), (None, None));
        assert!(f.alive.is_empty());
    }
}

#[test]
fn every_part_of_the_key_tells_blobs_apart() {
    let base = key("f");
    assert_eq!(base, key("f"));
    let other = [
        Key::new([3; 32], BIN, "f", (7, 28), "1.38.0", "arl-npu"),
        Key::new(XML, [3; 32], "f", (7, 28), "1.38.0", "arl-npu"),
        Key::new(XML, BIN, "g", (7, 28), "1.38.0", "arl-npu"),
        Key::new(XML, BIN, "f", (7, 29), "1.38.0", "arl-npu"),
        Key::new(XML, BIN, "f", (7, 28), "1.38.1", "arl-npu"),
        Key::new(XML, BIN, "f", (7, 28), "1.38.0", "lnl-npu"),
        Key::new(BIN, XML, "f", (7, 28), "1.38.0", "arl-npu"),
    ];
    for k in other {
        assert_ne!(k, base);
    }
    assert_eq!((base.xml, base.bin), (XML, BIN), "the digests are kept as given");
}

#[test]
fn the_cache_keeps_within_its_budget_oldest_out_first() {
    let mut c = Cache::new(10);
    c.keep(key("a"), Arc::new(vec![0; 4]));
    c.keep(key("b"), Arc::new(vec![0; 4]));
    c.keep(key("c"), Arc::new(vec![0; 4]));
    assert!(c.get(&key("a")).is_none(), "the oldest goes first");
    assert!(c.get(&key("b")).is_some() && c.get(&key("c")).is_some());
    assert_eq!(c.bytes(), 8);
    c.keep(key("d"), Arc::new(vec![0; 11]));
    assert!(c.get(&key("d")).is_none(), "a blob over the budget is not kept");
    c.keep(key("b"), Arc::new(vec![1; 2]));
    assert_eq!((c.bytes(), c.get(&key("b")).unwrap().len()), (6, 2), "a key is kept once");
    c.forget(&key("c"));
    assert_eq!(c.bytes(), 2);
    c.keep(key("e"), Arc::new(Vec::new()));
    assert!(c.get(&key("e")).is_none(), "an empty blob is not kept");
}

#[test]
fn a_hit_shares_the_kept_bytes() {
    let mut c = Cache::new(BUDGET);
    let blob = Arc::new(vec![3u8; 32]);
    c.keep(key("a"), Arc::clone(&blob));
    let hit = c.get(&key("a")).unwrap();
    assert_eq!(hit.as_ptr(), blob.as_ptr(), "zero-copy: the same allocation");
}

#[test]
fn the_source_names_the_format() {
    assert_eq!(Source::Cache.format(), "NATIVE");
    assert_eq!(Source::Compiled.format(), "NATIVE");
    assert_eq!(Source::Lite.format(), "NGRAPH_LITE");
}

/// The key comes from the description the core filled; a shorter one,
/// or one with a zero digest, keys nothing and the load stays lite.
#[test]
fn the_key_takes_the_manifest_digests_when_the_description_has_them() {
    use super::model::digests;
    let full = size_of::<crate::backend::turbo_backend_model>() as u32;
    let mut d = super::tests::manifest(0, 0, 0, 0, 0);
    d.artifact_sha256 = XML;
    d.artifact2_sha256 = BIN;
    d.struct_size = full;
    assert_eq!(digests(&d), Some((XML, BIN)));
    d.struct_size = std::mem::offset_of!(crate::backend::turbo_backend_model, artifact_sha256) as u32;
    assert_eq!(digests(&d), None, "a description from before the digests");
    d.struct_size = full;
    d.artifact2_sha256 = [0; 32];
    assert_eq!(digests(&d), None, "no digest for the weights");
}
