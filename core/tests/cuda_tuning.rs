//! Tuned CUDA sessions: the cache, forcing, skipped variants, builds
//! without cuBLAS, the budget, and a RoBERTa model's position offset.
//!
//! Built with the `cuda` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none; nothing is run
//! on anything else in its place. With TURBO_TEST_REQUIRE_CUDA=1 it fails
//! instead, so a run on a GPU machine cannot pass by finding no GPU.
//! docs/cuda.md says how to run them; tests/cuda_support/ holds what the
//! files share.

#![cfg(feature = "cuda")]

mod common;
#[path = "cuda_support/mod.rs"]
mod cuda_support;

use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

use common::*;
use cuda_support::*;
use serde_json::json;
use turbo::status::UNSUPPORTED;
use turbo::*;

// ---- Tuned sessions ---------------------------------------------------------------

/// A session's desc with tuning in `mode` and a budget of `budget_ms`
/// (0 for the default's).
fn tuned_desc(max_batch: u32, max_seq: u32, precision: u32, mode: u32, budget_ms: u32) -> turbo_session_desc {
    let mut d = session_desc(max_batch, max_seq, precision);
    d.tuning = mode;
    d.tuning_budget_ms = budget_ms;
    d
}

/// f with the tuning cache in `dir` (memory only for None), in place of
/// TURBO_AUTOTUNE_CACHE, for the sessions f makes.
fn caching<T>(dir: Option<&std::path::Path>, f: impl FnOnce() -> T) -> T {
    turbo::tuning::use_cache_dir(Some(dir));
    let r = f();
    turbo::tuning::use_cache_dir(None);
    r
}

/// An empty directory of the test's own for a tuning cache.
fn cache_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The files in a cache directory.
fn cached_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect()
}

/// The log lines a runtime wrote, by level.
type Lines = Mutex<Vec<(u32, String)>>;

unsafe extern "C" fn keep_line(user: *mut c_void, level: u32, message: turbo_text) {
    let lines = unsafe { &*(user as *const Lines) };
    let bytes = unsafe { std::slice::from_raw_parts(message.ptr as *const u8, message.len as usize) };
    lines.lock().unwrap().push((level, String::from_utf8_lossy(bytes).into_owned()));
}

/// The bundle at dir loaded on the CUDA device by a runtime whose log
/// goes to the lines returned.
fn load_logged(dir: &std::path::Path) -> (Loaded, &'static Lines) {
    let lines: &'static Lines = Box::leak(Box::new(Mutex::new(Vec::new())));
    let desc = turbo_runtime_desc {
        struct_size: size_of::<turbo_runtime_desc>() as u32,
        reserved: 0,
        log: Some(keep_line),
        log_user_data: lines as *const Lines as *mut c_void,
    };
    let mut err = new_error();
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(&desc, &mut rt, &mut err) }, 0);
    let mut ctx = ptr::null_mut();
    let rc = unsafe { turbo_context_create(rt, cuda(rt), &mut ctx, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    let mut m = ptr::null_mut();
    let rc = unsafe { turbo_model_load(ctx, text(dir.to_str().unwrap()), &mut m, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    (Loaded { rt, ctx, m }, lines)
}

/// A tuned session of desc on m that measured. The tuner declines a
/// device whose times stay far apart, which a card at its power cap can
/// be for a moment: a session that says so in the log is made again, up
/// to three times; any other session that did not measure fails, with
/// the log.
fn measured_session(m: *mut turbo_model, lines: &Lines, desc: &turbo_session_desc) -> Session {
    for attempt in 1..=3 {
        let from = lines.lock().unwrap().len();
        let s = caching(None, || Session::create(m, Some(desc)).unwrap());
        let info = s.info();
        if info.tuned == TURBO_TUNED_MEASURED {
            return s;
        }
        let log: Vec<(u32, String)> = lines.lock().unwrap()[from..].to_vec();
        let busy = log.iter().any(|(level, l)| *level == 1 && l.contains("not tuned:") && l.contains("apart"));
        println!("attempt {attempt}: tuned {} ({}), log: {log:#?}", info.tuned, field(&info.choices));
        assert!(busy, "tuned {} without the device found busy: {log:#?}", info.tuned);
    }
    panic!("the device was busy for three tuned sessions in a row");
}

/// A tuned session reports MEASURED, the time it took and its choices;
/// that line forced back with tuning off runs the same kernels and gives
/// the same bits, at FASTEST and EXACT.
#[test]
fn a_measured_session_forced_back_gives_its_bits() {
    let _t = turn();
    let Some(_) = cuda_device("a_measured_session_forced_back_gives_its_bits") else { return };
    let (f, _) = small_model("cuda-measured");
    let (g, lines) = load_logged(&f.dir);
    let t = ragged_rows(&f.dir, 40, 160);
    for precision in [TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let s = measured_session(g.m, lines, &tuned_desc(40, 160, precision, TURBO_AUTOTUNE_ON, 0));
        let info = s.info();
        let choices = field(&info.choices);
        println!("precision {precision}: measured in {} ms: {choices}", info.tune_ms);
        assert!(info.tune_ms > 0 && choices.ends_with(";forced="), "{choices}");
        s.write_tokens(&t.batch(), None).unwrap();
        let want = s.run().unwrap().rows();
        let back = forcing(&choices, || Session::create(g.m, Some(&session_desc(40, 160, precision)))).unwrap();
        assert_eq!(kernels_of(&field(&back.info().choices)), kernels_of(&choices));
        back.write_tokens(&t.batch(), None).unwrap();
        assert_eq!(back.run().unwrap().rows(), want, "precision {precision}: {choices} forced back");
    }
}

/// The cache decides: a second tuned session of the same key takes the
/// first's choices unmeasured, from memory and, in another runtime, from
/// the one file the first left, whose key and choices are the session's;
/// a file that does not parse is no entry, and the session measures.
#[test]
fn the_cache_decides() {
    let _t = turn();
    let Some(_) = cuda_device("the_cache_decides") else { return };
    let (f, g) = small_model("cuda-cache");
    let dir = cache_dir("cuda-cache");
    let desc = tuned_desc(40, 160, TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_ON, 0);
    caching(Some(&dir), || {
        let first = Session::create(g.m, Some(&desc)).unwrap().info();
        assert_eq!(first.tuned, TURBO_TUNED_MEASURED);
        let choices = field(&first.choices);
        let second = Session::create(g.m, Some(&desc)).unwrap().info();
        assert_eq!((second.tuned, second.tune_ms), (TURBO_TUNED_CACHE, 0));
        assert_eq!(field(&second.choices), choices);
        let files = cached_files(&dir);
        assert_eq!(files.len(), 1, "{files:?}");
        let entry: serde_json::Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
        assert_eq!(entry["choices"], json!(choices));
        assert_eq!(entry["key"]["backend"], json!("cuda"));
        assert_eq!(entry["key"]["precision"], json!(TURBO_PRECISION_FASTEST));
        assert_eq!(entry["key"]["numerics_allowed"], json!(turbo::tuning::numerics_allowed(TURBO_PRECISION_FASTEST)));
        // Another runtime reads the file.
        let other = f.load_on(cuda).unwrap();
        let third = Session::create(other.m, Some(&desc)).unwrap().info();
        assert_eq!((third.tuned, field(&third.choices)), (TURBO_TUNED_CACHE, choices.clone()));
        // RETUNE measures over the entry.
        let again =
            Session::create(other.m, Some(&tuned_desc(40, 160, TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_RETUNE, 0)));
        assert_eq!(again.unwrap().info().tuned, TURBO_TUNED_MEASURED);
        // A file that does not parse is ignored, in a runtime of its own.
        std::fs::write(&files[0], b"{ not an entry").unwrap();
        let fresh = f.load_on(cuda).unwrap();
        assert_eq!(Session::create(fresh.m, Some(&desc)).unwrap().info().tuned, TURBO_TUNED_MEASURED);
    });
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Forcing beats tuning: a forced tile is every GEMM's, forced= names it,
/// and the session leaves no entry, so an unforced session made next
/// measures. TURBO_CUDA_TF32 at MODEL lets the tuner time TF32 kernels,
/// and a session that runs one is not cached; at EXACT it widens nothing.
#[test]
fn forcing_beats_tuning() {
    let _t = turn();
    let Some(_) = cuda_device("forcing_beats_tuning") else { return };
    use turbo::cuda::Tile;
    let (_f, g) = small_model("cuda-forcing-tuning");
    let dir = cache_dir("cuda-forcing-tuning");
    let tuned = |precision| tuned_desc(40, 160, precision, TURBO_AUTOTUNE_ON, 0);
    caching(Some(&dir), || {
        turbo::cuda::use_tile(Some(Tile::T64x64));
        let s = Session::create(g.m, Some(&tuned(TURBO_PRECISION_FASTEST)));
        turbo::cuda::use_tile(None);
        let info = s.unwrap().info();
        let choices = field(&info.choices);
        assert!(!choices.contains("w/") && choices.matches("=64x64/").count() >= 4, "{choices}");
        assert!(turbo::tuning::forced_knobs(&choices).contains(&"tile"), "{choices}");
        assert_ne!(info.tuned, TURBO_TUNED_CACHE);
        assert!(cached_files(&dir).is_empty());
        let next = Session::create(g.m, Some(&tuned(TURBO_PRECISION_FASTEST))).unwrap().info();
        assert_eq!(next.tuned, TURBO_TUNED_MEASURED);
        assert_eq!(cached_files(&dir).len(), 1);

        turbo::cuda::use_tf32(Some(true));
        let s = Session::create(g.m, Some(&tuned(TURBO_PRECISION_MODEL)));
        turbo::cuda::use_tf32(None);
        let choices = field(&s.unwrap().info().choices);
        println!("MODEL with TF32 to time: {choices}");
        let files = cached_files(&dir);
        if choices.contains("/tf32") {
            assert_eq!(files.len(), 1, "a session that runs TF32 is not cached: {files:?}");
        } else {
            assert_eq!(files.len(), 2, "{files:?}");
        }
        for f in &files {
            let entry: serde_json::Value = serde_json::from_slice(&std::fs::read(f).unwrap()).unwrap();
            assert!(!entry["choices"].as_str().unwrap().contains("/tf32"), "{entry}");
        }
        turbo::cuda::use_tf32(Some(true));
        let exact = Session::create(g.m, Some(&tuned(TURBO_PRECISION_EXACT)));
        turbo::cuda::use_tf32(None);
        let exact = field(&exact.unwrap().info().choices);
        assert!(!exact.contains("/tf32"), "{exact}");
    });
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A tuned session whose every GEMM tile is forced has nothing to time:
/// it reports FORCED, and says so at INFO.
#[test]
fn forced_tiles_leave_nothing_to_tune() {
    let _t = turn();
    let Some(dev) = cuda_device("forced_tiles_leave_nothing_to_tune") else { return };
    use turbo::cuda::Tile;
    let ordinal = Rt::new().info(dev).ordinal;
    let (f, _) = small_model("cuda-forced-tiles");
    let (g, lines) = load_logged(&f.dir);
    let desc = tuned_desc(40, 160, TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_ON, 0);
    let s = caching(None, || {
        turbo::cuda::use_tile(Some(Tile::T64x64));
        let s = Session::create(g.m, Some(&desc));
        turbo::cuda::use_tile(None);
        s
    });
    let info = s.unwrap().info();
    assert!(turbo::tuning::forced_knobs(&field(&info.choices)).contains(&"tile"));
    assert_eq!((info.tuned, info.tune_ms), (TURBO_TUNED_FORCED, 0));
    let want = format!("cuda device {ordinal}: not tuned: every GEMM tile is forced");
    let lines = lines.lock().unwrap();
    assert!(lines.iter().any(|(level, l)| *level == 2 && *l == want), "{lines:?}");
}

/// A variant that cannot launch is skipped: the tuner never chooses it
/// and says so once, at INFO; forcing it fails the session, naming it.
#[test]
fn a_variant_that_cannot_launch_is_skipped() {
    let _t = turn();
    let Some(dev) = cuda_device("a_variant_that_cannot_launch_is_skipped") else { return };
    let ordinal = Rt::new().info(dev).ordinal;
    let (f, _) = small_model("cuda-cannot-launch");
    let (g, lines) = load_logged(&f.dir);
    let precision = TURBO_PRECISION_FASTEST;
    let own = field(&Session::create(g.m, Some(&session_desc(40, 160, precision))).unwrap().info().choices);
    let allowed = turbo::tuning::numerics_allowed(precision);
    let vs = turbo::cuda::variants(ordinal, precision).unwrap();
    let v = vs
        .iter()
        .find(|v| v.candidate && v.numeric & allowed != 0 && !own.contains(&format!("={}/", v.name)))
        .expect("a candidate other than the default");
    turbo::cuda::fail_variant(Some(&v.name));
    lines.lock().unwrap().clear();
    let s = caching(None, || Session::create(g.m, Some(&tuned_desc(40, 160, precision, TURBO_AUTOTUNE_ON, 0))));
    let forced =
        forcing(&format!("all:qkv={}", v.name), || Session::create(g.m, Some(&session_desc(40, 160, precision))));
    turbo::cuda::fail_variant(None);
    let choices = field(&s.unwrap().info().choices);
    assert!(!choices.contains(&format!("={}/", v.name)), "{}: {choices}", v.name);
    let said: Vec<_> = lines
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, l)| l.contains("cannot launch") && l.contains(&v.name))
        .cloned()
        .collect();
    assert_eq!(said.len(), 1, "{said:?}");
    assert_eq!(said[0].0, 2, "at INFO: {said:?}");
    let e = forced.err().unwrap();
    assert_eq!(e.code, UNSUPPORTED, "{}", e.message);
    assert!(e.message.contains(&v.name), "{}", e.message);
}

/// A build without cuBLAS (no `cuda-cublas` feature) refuses a session
/// naming a GEMM for cuBLAS and the GEMM check against it, each as
/// UNSUPPORTED naming the feature; a build with it takes both.
#[test]
fn a_build_without_cublas_refuses_what_needs_it() {
    let _t = turn();
    let Some(dev) = cuda_device("a_build_without_cublas_refuses_what_needs_it") else { return };
    let (_f, g) = small_model("cuda-cublas-build");
    turbo::cuda::use_cublas(Some(15));
    let s = Session::create(g.m, Some(&session_desc(4, 64, TURBO_PRECISION_MODEL)));
    turbo::cuda::use_cublas(None);
    let ordinal = Rt::new().info(dev).ordinal;
    let check = turbo::cuda::gemm_check(
        ordinal,
        64,
        64,
        64,
        turbo::cuda::Epilogue::Plain,
        false,
        false,
        turbo::cuda::Tile::Default,
        0,
        1,
    );
    match (turbo::cuda::has_cublas(), s) {
        (true, s) => {
            assert!(s.is_ok(), "a build with cuBLAS takes the session");
            check.unwrap();
        }
        (false, Ok(_)) => panic!("a build without cuBLAS took a session naming GEMMs for it"),
        (false, Err(e)) => {
            assert!(e.is(UNSUPPORTED, "cuda-cublas"), "{e:?}");
            assert_eq!(check.unwrap_err(), UNSUPPORTED);
        }
    }
}

/// A session whose GEMMs cuBLAS computes is not tuned and takes no cached
/// choice, whose kernels its GEMMs would not run; the log says why.
#[test]
fn cublas_refuses_tuning() {
    let _t = turn();
    let Some(_) = cuda_device("cublas_refuses_tuning") else { return };
    if !cublas_or_skip("cublas_refuses_tuning", "the test") {
        return;
    }
    let (f, _) = small_model("cuda-cublas-tuning");
    let (g, lines) = load_logged(&f.dir);
    let desc = tuned_desc(40, 160, TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_ON, 0);
    let s = caching(None, || {
        // An entry for the key first, which the cuBLAS session must not take.
        assert_eq!(Session::create(g.m, Some(&desc)).unwrap().info().tuned, TURBO_TUNED_MEASURED);
        turbo::cuda::use_cublas(Some(15));
        let s = Session::create(g.m, Some(&desc));
        turbo::cuda::use_cublas(None);
        s
    });
    assert_eq!(s.unwrap().info().tuned, TURBO_TUNED_DEFAULT);
    let lines = lines.lock().unwrap();
    assert!(
        lines.iter().any(|(level, l)| *level == 2 && l.contains("not tuned") && l.contains("TURBO_CUDA_CUBLAS")),
        "{lines:?}"
    );
}

/// The budget holds: a session given 50 ms measures in well under five
/// times that, and names at DEBUG what it did not time.
#[test]
fn the_budget_holds() {
    let _t = turn();
    let Some(_) = cuda_device("the_budget_holds") else { return };
    let (f, _) = small_model("cuda-budget");
    let (g, lines) = load_logged(&f.dir);
    for precision in [TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        lines.lock().unwrap().clear();
        let s = caching(None, || {
            Session::create(g.m, Some(&tuned_desc(40, 160, precision, TURBO_AUTOTUNE_ON, 50))).unwrap()
        });
        let info = s.info();
        println!("precision {precision}: tuned {} in {} ms", info.tuned, info.tune_ms);
        assert!(info.tune_ms < 250, "precision {precision}: {} ms", info.tune_ms);
        let lines = lines.lock().unwrap();
        let chosen = lines.iter().find(|(level, l)| *level == 2 && l.contains("kernels chosen in"));
        if let Some((_, l)) = chosen
            && !l.ends_with("; 0 not timed)")
        {
            assert!(lines.iter().any(|(level, l)| *level == 3 && l.contains("not timed: the budget")), "{l}");
        }
    }
}

/// What a process's first tuned session of a real bundle costs, at the
/// benchmark's shape of 32 rows of 256 tokens, FASTEST and EXACT, with
/// the default budget and no disk cache: the log's line and tune_ms,
/// printed for docs/cuda.md.
#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_first_tuned_session_s_cost() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = cuda_device("the_first_tuned_session_s_cost") else { return };
    for precision in [TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT] {
        // A runtime of its own each time, so the cache is empty; written
        // to a directory of its own, whose entry prints the timings each
        // choice was made on.
        let (g, lines) = load_logged(&dir);
        let keep = Fixture::new(&format!("cuda-tuned-cost-{precision}"), model_manifest());
        let cache = keep.dir.clone();
        std::fs::create_dir_all(&cache).unwrap();
        let s = caching(Some(&cache), || {
            Session::create(g.m, Some(&tuned_desc(32, 256, precision, TURBO_AUTOTUNE_ON, 0))).unwrap()
        });
        let info = s.info();
        println!("precision {precision}: tuned {} in {} ms", info.tuned, info.tune_ms);
        for (level, l) in lines.lock().unwrap().iter().filter(|(level, _)| *level <= 2) {
            println!("  [{level}] {l}");
        }
        for entry in std::fs::read_dir(&cache).unwrap().map(|e| e.unwrap().path()) {
            if entry.extension().is_some_and(|x| x == "json") {
                let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&entry).unwrap()).unwrap();
                if let Some(timings) = v["timings"].as_array() {
                    for t in timings {
                        println!("  timing {} = {} ms", t[0].as_str().unwrap_or("?"), t[1]);
                    }
                }
            }
        }
    }
}

/// The small BERT as FAMILY_ROBERTA on the device: the positions read
/// from row 2 of the table, as the f64 encoder reads them with the
/// offset, at every precision.
#[test]
fn a_roberta_model_reads_positions_from_its_offset() {
    let _t = turn();
    let Some(_) = cuda_device("a_roberta_model_reads_positions_from_its_offset") else { return };
    let mut f = Fixture::new("cuda-roberta", {
        let mut m = model_manifest();
        m["architecture"]["family"] = json!("FAMILY_ROBERTA");
        m["architecture"]["position_offset"] = json!(2);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    f.write();
    let g = on_cuda(&f.dir);
    let tok = Tok::create(&f.dir).unwrap();
    let plain = PlainBert::new(&f.dir);
    for precision in [TURBO_PRECISION_EXACT, TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let s = Session::create(g.m, Some(&session_desc(0, 0, precision))).unwrap();
        let got = s.embed(&TEXTS, None).unwrap();
        for (i, t) in TEXTS.iter().enumerate() {
            let ids = tok.row(t, None).unwrap();
            let ones = vec![1; ids.len()];
            let want = plain.embed(&ids, &ones, &vec![0; ids.len()], TURBO_POOLING_MEAN, 8, true);
            let bound = if precision == TURBO_PRECISION_FASTEST { 1e-2 } else { 1e-5 };
            for (a, w) in got[i].iter().zip(&want) {
                assert!((*a as f64 - w).abs() < bound, "precision {precision} {t:?}: {a} vs {w}");
            }
        }
    }
    Session::create(g.m, Some(&session_desc(1, MAX_SEQ as u32, 0))).unwrap();
}
