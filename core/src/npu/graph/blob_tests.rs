//! The native path's policy and the blob cache, as plain functions and
//! data: no loader, driver or device is stood in for.
//!
//! What these do not cover: the driver refusing a step. The ignored device
//! test (core/tests/npu_native.rs) runs only the paths a working driver
//! takes: compile, export, a NATIVE create and initialize, and a later
//! load from the kept blob. A missing or failing export, a driver refusal
//! of a fresh or kept blob, a NATIVE graph that fails to initialize, and
//! the graph cleanup and cache updates `select` does on each of those are
//! not run anywhere: no safe way to make a working driver refuse them is
//! known. The step order those refusals follow is `next`, tested here.

use std::sync::Arc;

use super::blob::*;

/// The manifest's digests stand in for the files: nothing is hashed.
const XML: [u8; 32] = [1; 32];
const BIN: [u8; 32] = [2; 32];

fn key(flags: &str) -> Key {
    Key::new(XML, BIN, flags, (7, 28), "1.38.0", "arl-npu")
}

#[test]
fn a_load_starts_from_the_kept_blob_only_when_it_wants_native_and_has_one() {
    assert_eq!(first(true, true), Step::Kept);
    assert_eq!(first(true, false), Step::Compile);
    assert_eq!(first(false, true), Step::Compile, "a kept blob is not read without a key");
    assert_eq!(first(false, false), Step::Compile);
}

#[test]
fn each_step_leads_where_the_policy_says() {
    use Next::{Done, Fail};
    let s = Next::Step;
    for native in [false, true] {
        assert_eq!(next(Step::Kept, native, true), Done(Source::Cache));
        assert_eq!(next(Step::Kept, native, false), s(Step::Compile), "a refused kept blob compiles the IR");
        assert_eq!(next(Step::Compile, native, false), Fail, "a compile refusal fails the load");
        assert_eq!(next(Step::Export, native, true), s(Step::Fresh));
        assert_eq!(next(Step::Export, native, false), s(Step::Lite), "no blob: the compiled IR runs");
        assert_eq!(next(Step::Fresh, native, true), Done(Source::Compiled));
        assert_eq!(next(Step::Fresh, native, false), s(Step::Lite), "a refused blob: the compiled IR runs");
        assert_eq!(next(Step::Lite, native, true), Done(Source::Lite));
        assert_eq!(next(Step::Lite, native, false), Fail);
    }
    assert_eq!(next(Step::Compile, true, true), s(Step::Export));
    assert_eq!(next(Step::Compile, false, true), s(Step::Lite), "no key: nothing is exported");
}

/// Every load the policy allows: whether it wants NATIVE, whether a blob
/// is kept, and every pattern of steps succeeding or failing. Each step
/// has what it needs, the load ends within five steps, and it fails only
/// when the compile or the NGRAPH_LITE initialize does.
#[test]
fn every_outcome_sequence_ends_with_what_its_steps_produced() {
    for native in [false, true] {
        for kept in [false, true] {
            for outcomes in 0u32..1 << 5 {
                let ok = |i: usize| outcomes >> i & 1 == 1;
                let (mut step, mut taken) = (first(native, kept), Vec::new());
                let end = loop {
                    let i = taken.len();
                    assert!(i < 5, "{native} {kept} {outcomes:05b}: {taken:?} does not end");
                    match step {
                        Step::Kept => assert!(native && kept && i == 0),
                        Step::Compile => assert!(!taken.contains(&(Step::Compile, true))),
                        Step::Export => assert!(native && taken.last() == Some(&(Step::Compile, true))),
                        Step::Fresh => assert_eq!(taken.last(), Some(&(Step::Export, true))),
                        Step::Lite => assert!(taken.contains(&(Step::Compile, true))),
                    }
                    taken.push((step, ok(i)));
                    match next(step, native, ok(i)) {
                        Next::Step(s) => step = s,
                        end => break end,
                    }
                };
                let what = format!("{native} {kept} {outcomes:05b}: {taken:?}");
                match end {
                    Next::Fail => assert!(
                        taken.last() == Some(&(Step::Compile, false)) || taken.last() == Some(&(Step::Lite, false)),
                        "{what}"
                    ),
                    Next::Done(Source::Cache) => assert_eq!(taken, [(Step::Kept, true)], "{what}"),
                    Next::Done(Source::Compiled) => {
                        assert!(native && taken.ends_with(&[(Step::Fresh, true)]), "{what}")
                    }
                    Next::Done(Source::Lite) => {
                        let refused = taken.iter().any(|&(s, ok)| !ok && matches!(s, Step::Export | Step::Fresh));
                        assert!(!native || refused, "{what}: NGRAPH_LITE runs only without a key or after a refusal");
                    }
                    Next::Step(_) => unreachable!(),
                }
            }
        }
    }
}

#[test]
fn a_forgotten_blob_is_gone_and_its_bytes_freed() {
    let mut c = Cache::new(BUDGET);
    c.keep(key("f"), Arc::new(vec![9; 8]));
    c.keep(key("g"), Arc::new(vec![9; 4]));
    c.forget(&key("f"));
    assert!(c.get(&key("f")).is_none());
    assert_eq!(c.bytes(), 4);
    c.forget(&key("f"));
    assert_eq!(c.bytes(), 4, "forgetting a key not kept changes nothing");
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
fn a_load_wants_the_native_blob_by_default() {
    assert!(std::thread::spawn(native_wanted).join().unwrap());
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
