//! The npu backend through the C interface: its table, the devices it
//! lists (Intel NPUs whose driver's graph extension answered the probe),
//! its capability, contexts and host buffers, and, on a bundle with an
//! OpenVINO IR for it, a session's embed run through the driver's
//! compiler. tests/conformance.rs with TURBO_TEST_DEVICE=npu holds the
//! vectors to the bundle's reference.
//!
//! Built with the `npu` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none; nothing is run
//! on anything else in its place. With TURBO_TEST_REQUIRE_NPU=1 it fails
//! instead, so a run on an NPU machine cannot pass by finding no device.
//! The tests that need a bundle with an IR for the npu backend are
//! ignored unless asked for, and read it from TURBO_TEST_BUNDLE.
//! docs/npu.md says how to run them.

#![cfg(feature = "npu")]

mod common;

use std::ptr;

use common::*;
use turbo::status::{BUNDLE_NO_ARTIFACT, UNSUPPORTED, UNSUPPORTED_OPTION};
use turbo::*;

struct Rt(*mut turbo_runtime);

// turbo.h: a runtime may be used from any thread.
unsafe impl Send for Rt {}
unsafe impl Sync for Rt {}

impl Rt {
    fn new() -> Rt {
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, ptr::null_mut()) }, 0);
        Rt(rt)
    }

    fn count(&self) -> u32 {
        let mut n = 0;
        assert_eq!(unsafe { turbo_runtime_device_count(self.0, &mut n, ptr::null_mut()) }, 0);
        n
    }

    fn info(&self, i: u32) -> turbo_device_info {
        let mut out: turbo_device_info = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_device_info(self.0, i, &mut out, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        out
    }

    /// The runtime's indices of the devices the npu backend listed.
    fn npu(&self) -> Vec<u32> {
        (0..self.count()).filter(|&i| field(&self.info(i).backend) == "npu").collect()
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

/// TURBO_TEST_REQUIRE_NPU=1: a test that finds no device fails.
fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_NPU").is_ok_and(|v| v == "1")
}

/// The runtime's first npu device, or None after saying the test is
/// skipped.
fn npu_device(rt: &Rt, test: &str) -> Option<u32> {
    let d = rt.npu().first().copied();
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_NPU=1 and the npu backend lists no device");
        println!("{test}: skipped: the npu feature is on and the npu backend lists no device");
    }
    d
}

// ---- Without a device ------------------------------------------------------------

#[test]
fn the_table_is_whole_and_loads_openvino_ir_alone() {
    let b = turbo::npu::backend();
    assert_eq!(b.name(), "npu");
    assert_eq!(b.struct_size as usize, size_of::<turbo::backend::turbo_backend>());
    turbo::backend::check_table(b).unwrap();
    assert_eq!(b.formats, 1 << (backend::TURBO_FORMAT_OPENVINO_IR - 1), "FORMAT_OPENVINO_IR and nothing else");
    assert!(b.model_load.is_some() && b.session_run.is_some() && b.buffer_alloc.is_some());
    assert!(b.buffer_read.is_none(), "every buffer it gives has a host address");
    assert!(b.buffer_import.is_none(), "no import path is built");
    assert!(
        b.session_create_tuned.is_some(),
        "the graph format the driver selected is reported through session_create_tuned"
    );
    assert!(turbo::backend::linked().iter().any(|l| std::ptr::eq(*l, b)));
    let v = unsafe { std::ffi::CStr::from_ptr(turbo_version()) }.to_str().unwrap();
    assert!(v.split(' ').any(|w| w == "npu"), "turbo_version() = {v}");
}

// ---- Devices ---------------------------------------------------------------------

/// Every device listed is an NPU over host memory, named, labeled and
/// versioned; with none, the runtime is made and lists the rest.
#[test]
fn the_devices_listed_are_npus_that_answered_the_graph_probe() {
    let rt = Rt::new();
    let listed = rt.npu();
    if listed.is_empty() {
        assert!(!required(), "TURBO_TEST_REQUIRE_NPU=1 and the npu backend lists no device");
        println!("the npu backend lists no device: nothing to check but that the runtime was made");
        return;
    }
    for (o, &i) in listed.iter().enumerate() {
        let d = rt.info(i);
        assert_eq!(d.ordinal, o as u32);
        assert_eq!(d.kind, TURBO_DEVICE_NPU);
        assert_eq!(d.unified_memory, 1, "an NPU computes over the host's memory");
        assert_eq!(d.memory_free, 0, "nothing says what is free");
        let arch = field(&d.arch);
        assert!(!arch.is_empty());
        let name = field(&d.name);
        assert!(!name.is_empty());
        assert_eq!(name, name.trim(), "no padding around the name");
        let vendor = field(&d.vendor);
        assert!(!vendor.is_empty());
        let driver = field(&d.driver_version);
        assert!(!driver.is_empty());
        println!(
            "npu device {o}: {name}, arch {arch}, vendor {vendor}, loader {}, driver {driver}",
            field(&d.runtime_version)
        );
    }
}

// ---- Capability ------------------------------------------------------------------

#[test]
fn embed_is_offered_and_exact_is_refused() {
    let rt = Rt::new();
    let Some(d) = npu_device(&rt, "embed_is_offered_and_exact_is_refused") else { return };
    let cap = |p| {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(rt.0, d, TURBO_TASK_EMBED, p, &mut cap, ptr::null_mut()) }, 0);
        cap
    };
    let arch = field(&rt.info(d).arch);
    // The first listed npu device is ordinal 0 in the backend's own list.
    let format = turbo::npu::load_format(0).expect("a listed device has a graph format");
    // The committed ROWS_MIXED files match windows, arl-npu, NGRAPH_LITE
    // and list cases 0 through 7, each its own [1, 128] frame.
    // falls_short is none, so decide_embedded promotes those cells.
    let backed = std::env::consts::OS == "windows" && arch == "arl-npu" && format == "NGRAPH_LITE";
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let c = cap(p);
        assert_eq!(c.dtype, TURBO_DTYPE_F16, "the recipe's declared compute dtype, without a load");
        assert!(!field(&c.reason).contains("dtype 0"), "{}", field(&c.reason));
        // The backend claims normalize, pooling and output_dim
        // (0b111000); the core sets the bits of truncate, max_tokens and
        // prompt_role itself, which it applies before any backend sees
        // the rows.
        assert_eq!(c.options_honored, 0b111111, "every field of turbo_embed_options");
        let name = match p {
            TURBO_PRECISION_MODEL => {
                "arl-npu.npu.ngraph-lite.embed.model.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json"
            }
            TURBO_PRECISION_FASTEST => {
                "arl-npu.npu.ngraph-lite.embed.fastest.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json"
            }
            _ => unreachable!(),
        };
        if backed {
            assert_eq!(c.status, backend::TURBO_CAP_SUPPORTED, "{}", field(&c.reason));
            assert_eq!(field(&c.benchmark), name);
            assert_eq!(field(&c.reason), "");
            let cell = record::Cell {
                arch: &arch,
                name: "Intel(R) AI Boost",
                cpu: false,
                backend: "npu",
                task: TURBO_TASK_EMBED,
                precision: p,
                dtype: TURBO_DTYPE_F16,
                version: record::library_version(),
                os: "windows",
                graph_format: Some("NGRAPH_LITE"),
            };
            match record::decide_embedded(&cell) {
                record::Verdict::Supported { benchmark, cosine_floor, speed_ratio } => {
                    assert_eq!(benchmark, name);
                    assert_eq!((c.cosine_floor, c.speed_ratio), (cosine_floor as f32, speed_ratio as f32));
                }
                record::Verdict::Not(why) => panic!("{name} not supported: {why}"),
            }
        } else {
            assert_eq!(c.status, backend::TURBO_CAP_EXPERIMENTAL, "{}", field(&c.reason));
            assert!(field(&c.benchmark).is_empty(), "{}", field(&c.benchmark));
            assert_eq!(field(&c.reason), "no benchmark record for this cell");
        }
    }
    let c = cap(TURBO_PRECISION_EXACT);
    assert_eq!((c.status, c.dtype, c.options_honored), (backend::TURBO_CAP_UNSUPPORTED, 0, 0));
    assert!(field(&c.reason).contains("EXACT"), "{}", field(&c.reason));
}

// ---- Contexts and buffers --------------------------------------------------------

struct Ctx(*mut turbo_context);

impl Ctx {
    fn create(rt: &Rt, d: u32) -> Ctx {
        let mut c = ptr::null_mut();
        let mut err = new_error();
        let rc = unsafe { turbo_context_create(rt.0, d, &mut c, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        Ctx(c)
    }

    fn alloc(&self, placement: u32, bytes: u64) -> Result<*mut turbo_buffer, Failure> {
        let desc = turbo_buffer_desc {
            struct_size: size_of::<turbo_buffer_desc>() as u32,
            placement,
            dtype: TURBO_DTYPE_F32,
            ndim: 1,
            shape: [bytes / 4, 0],
            bytes: 0,
        };
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_alloc(self.0, &desc, &mut b, &mut err) } {
            0 => Ok(b),
            rc => Err(failure(rc, &err)),
        }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { turbo_context_release(self.0) };
    }
}

/// A buffer is driver-allocated host memory and exports as its host
/// address; any other placement is refused.
#[test]
fn buffers_are_host_memory_the_device_reads() {
    let rt = Rt::new();
    let Some(d) = npu_device(&rt, "buffers_are_host_memory_the_device_reads") else { return };
    let ctx = Ctx::create(&rt, d);
    let buf = ctx.alloc(TURBO_PLACE_HOST, 4096).unwrap();
    let mut host = ptr::null_mut();
    assert_eq!(unsafe { turbo_buffer_host_ptr(buf, &mut host, ptr::null_mut()) }, 0);
    assert_eq!(host as usize % 64, 0, "aligned to a cache line");
    unsafe { std::ptr::write_bytes(host as *mut u8, 0xab, 4096) };
    let mut h: turbo_native_handle = unsafe { std::mem::zeroed() };
    h.struct_size = size_of::<turbo_native_handle>() as u32;
    assert_eq!(unsafe { turbo_buffer_export(buf, TURBO_HANDLE_HOST_PTR, &mut h, ptr::null_mut()) }, 0);
    assert_eq!((h.kind, h.handle as usize), (TURBO_HANDLE_HOST_PTR, host as usize));
    let mut err = new_error();
    let rc = unsafe { turbo_buffer_export(buf, TURBO_HANDLE_DMABUF_FD, &mut h, &mut err) };
    assert_eq!(failure(rc, &err).code, UNSUPPORTED);
    unsafe { turbo_buffer_release(buf) };
    for p in [TURBO_PLACE_PINNED, TURBO_PLACE_DEVICE, TURBO_PLACE_SHARED] {
        let f = ctx.alloc(p, 4096).unwrap_err();
        assert!(f.is(UNSUPPORTED, "TURBO_PLACE_HOST only"), "placement {p}: {f:?}");
    }
}

/// Contexts made and released from several threads at once.
#[test]
fn contexts_come_and_go_from_many_threads() {
    let rt = Rt::new();
    let Some(d) = npu_device(&rt, "contexts_come_and_go_from_many_threads") else { return };
    let rt = std::sync::Arc::new(rt);
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let rt = rt.clone();
            std::thread::spawn(move || {
                for _ in 0..10 {
                    let c = Ctx::create(&rt, d);
                    drop(c);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

/// A bundle of raw weights has nothing the npu backend loads: it takes
/// FORMAT_OPENVINO_IR alone, and never falls back to another device.
#[test]
fn raw_weights_are_not_an_artifact_for_npu() {
    let rt = Rt::new();
    let Some(_) = npu_device(&rt, "raw_weights_are_not_an_artifact_for_npu") else { return };
    drop(rt);
    let f = Loaded::load_on(&tiny_bundle(), |rt| first_of(rt, "npu").unwrap()).err().expect("no artifact");
    assert!(f.is(BUNDLE_NO_ARTIFACT, "npu"), "{f:?}");
}

// ---- A bundle with an OpenVINO IR ------------------------------------------------

/// TURBO_TEST_BUNDLE: a bundle with an OpenVINO IR whose artifacts[]
/// backends lists "npu".
fn ir_bundle() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set"))
}

fn on_npu() -> Loaded {
    Loaded::load_on(&ir_bundle(), |rt| first_of(rt, "npu").expect("an npu device")).unwrap_or_else(|e| panic!("{e:?}"))
}

/// LOOKUP for the artifact this load reported. The hash is the one
/// `turbo_model_info.artifact_sha256` carries, so the expectation
/// follows the file the device compiled: host for INPUT_EMBEDDINGS,
/// device for INPUT_TOKEN_IDS.
fn lookup_stage_for(bundle: &std::path::Path, artifact_sha: &str) -> u32 {
    let opened = turbo::bundle::Bundle::open(bundle).unwrap_or_else(|e| panic!("{}: {e}", bundle.display()));
    let art = opened
        .manifest
        .artifacts
        .iter()
        .find(|a| turbo::model::artifact_sha256(&opened.manifest, a) == artifact_sha)
        .unwrap_or_else(|| panic!("no artifact hashes to {artifact_sha}"));
    match art.graph_input {
        turbo::manifest::GraphInput::Embeddings => TURBO_STAGE_HOST,
        turbo::manifest::GraphInput::TokenIds => TURBO_STAGE_DEVICE,
    }
}

#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn a_session_computes_in_the_compiled_dtype_and_exact_is_refused() {
    let l = on_npu();
    let want = l.info().dtype;
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let s = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap();
        assert_eq!(s.info().compute_dtype, want, "precision {p}");
    }
    let f = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).err().expect("refused");
    assert!(f.is(UNSUPPORTED_OPTION, "EXACT") && f.field == 3, "{f:?}");
}

/// The rows go to the device as the graph's own input precisions, the
/// encoder runs on the NPU, and pooling and normalize run on the host:
/// the vectors stay in host memory, unit length by default. Lookup is
/// on the host when the loaded artifact is INPUT_EMBEDDINGS, because
/// the host gathers the word rows, and on the device for a token-id IR.
#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn a_run_reports_its_frames_and_where_each_stage_ran() {
    let l = on_npu();
    let lookup = lookup_stage_for(&ir_bundle(), &field(&l.info().artifact_sha256));
    let s = Session::create(l.m, Some(&session_desc(4, 0, TURBO_PRECISION_MODEL))).unwrap();
    let seq = s.info().max_seq as u64;
    let dim = l.info().dim as u64;
    let texts = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];
    let n = texts.len() as u64;
    s.write_text(&texts, None).unwrap();
    let r = s.run().unwrap();
    let info = r.info();
    assert_eq!((info.batch as u64, info.dim as u64), (n, dim));
    assert_eq!(info.placement, TURBO_PLACE_HOST);
    assert_eq!(field(&info.backend), "npu");
    // Per frame: the token rows over, at least a byte an element, and
    // the hidden states back.
    assert!(info.h2d_bytes >= n * seq, "{}", info.h2d_bytes);
    assert!(info.d2h_bytes >= seq * dim, "{}", info.d2h_bytes);
    assert_eq!((info.host_allocs, info.device_allocs), (0, 0));
    let st = &info.stage;
    assert_eq!(st[TURBO_EMBED_STAGE_TOKENIZE], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_UPLOAD], TURBO_STAGE_DEVICE);
    assert_eq!(st[TURBO_EMBED_STAGE_LOOKUP], lookup);
    assert_eq!(st[TURBO_EMBED_STAGE_ENCODE], TURBO_STAGE_DEVICE);
    assert_eq!(st[TURBO_EMBED_STAGE_POOL], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_NORMALIZE], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_DOWNLOAD], TURBO_STAGE_DEVICE);
    for v in r.rows() {
        let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "unit length: {norm}");
    }
}

/// Two models on one context, their sessions run at once from two
/// threads: an execute and the synchronization that waits for it are
/// one critical section on the context's immediate command list, and
/// runs on one model take turns on its graph. Interleaved runs must
/// give each thread its own rows' vectors, the same from both models.
#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn two_models_on_one_context_run_at_once() {
    struct SendModel(*mut turbo_model);
    unsafe impl Send for SendModel {}

    let rt = Rt::new();
    let d = rt.npu().first().copied().expect("an npu device");
    let ctx = Ctx::create(&rt, d);
    let path = ir_bundle();
    let load = || {
        let mut m = ptr::null_mut();
        let mut err = new_error();
        let rc = unsafe { turbo_model_load(ctx.0, text(path.to_str().unwrap()), &mut m, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        m
    };
    let models = [load(), load()];
    let texts = [
        ["threads share one context", "and two compiled graphs"],
        ["each run keeps its own rows", "whatever the other is doing"],
    ];
    let vectors: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = models
            .iter()
            .map(|&m| {
                let m = SendModel(m);
                scope.spawn(move || {
                    // The whole wrapper moves in, not its raw field.
                    let m = m;
                    let s = Session::create(m.0, Some(&session_desc(2, 0, TURBO_PRECISION_MODEL))).unwrap();
                    let mut last = Vec::new();
                    for row in texts.iter().cycle().take(8) {
                        last = s.embed(row, None).unwrap();
                    }
                    last
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // Both threads ended on texts[1]: one model's vectors match the
    // other's, so no run read another run's hidden states.
    for (a, b) in vectors[0].iter().zip(&vectors[1]) {
        let cos: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        assert!(cos > 0.9999, "the two models agree on the same rows: cosine {cos}");
    }
    for m in models {
        unsafe { turbo_model_release(m) };
    }
}

/// Pooling and normalize are the backend's, on the host: CLS differs
/// from mean, and NONE leaves the vector at its own length.
#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn pooling_and_normalize_follow_the_options() {
    let l = on_npu();
    let s = Session::create(l.m, Some(&session_desc(1, 0, TURBO_PRECISION_MODEL))).unwrap();
    let text = ["Embedding models turn text into vectors."];
    let mut o = embed_options();
    o.pooling = TURBO_POOLING_MEAN;
    let mean = s.embed(&text, Some(&o)).unwrap().remove(0);
    o.pooling = TURBO_POOLING_CLS;
    let cls = s.embed(&text, Some(&o)).unwrap().remove(0);
    let cos: f64 = mean.iter().zip(&cls).map(|(a, b)| *a as f64 * *b as f64).sum();
    assert!(cos < 0.999, "CLS and mean pooling give different vectors: cosine {cos}");
    o.pooling = TURBO_POOLING_MEAN;
    o.normalize = TURBO_NORMALIZE_NONE;
    let raw = s.embed(&text, Some(&o)).unwrap().remove(0);
    let norm = raw.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() > 1e-3, "NONE leaves the length as pooled: {norm}");
    let back: f64 = raw.iter().zip(&mean).map(|(a, b)| *a as f64 / norm * *b as f64).sum();
    assert!(back > 0.99999, "the same direction as the normalized vector: {back}");
}
