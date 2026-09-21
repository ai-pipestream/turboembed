//! Provider-agnostic conformance suite for the Turbo contract.
//!
//! The suite is one set of cases parameterized by provider and device so the
//! same tests run against the mock today and against real hardware later. The
//! harness here resolves a [`Target`] (runtime, device, bundle root) from the
//! environment; the cases live under `tests/`, one file per group, in two
//! layers: through the safe Rust API (`turbo`) and through the C ABI
//! (`turbo_capi`'s `extern "C"` functions called directly).
//!
//! Environment:
//! - `TURBO_CONFORMANCE_PROVIDER`: provider id to select explicitly
//!   (default: unset, meaning AUTO over the built-in providers).
//! - `TURBO_CONFORMANCE_ORDINAL`: device ordinal within that provider
//!   (default 0, only used with an explicit provider).
//! - `TURBO_CONFORMANCE_BUNDLES`: directory holding one subdirectory per
//!   [`BundleKind`] (default: the committed `testdata/bundles/mock`).

#![deny(missing_docs)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo::handles::{Context, Model, Session};
use turbo::provider::{ContextDesc, DeviceInfo, ModelDesc, SessionDesc};
use turbo::runtime::{DeviceSelector, Runtime, RuntimeDesc};
use turbo::types::SelectPolicy;

/// Environment variable naming the provider to test.
pub const ENV_PROVIDER: &str = "TURBO_CONFORMANCE_PROVIDER";
/// Environment variable naming the device ordinal within that provider.
pub const ENV_ORDINAL: &str = "TURBO_CONFORMANCE_ORDINAL";
/// Environment variable naming the directory holding the test bundles.
pub const ENV_BUNDLES: &str = "TURBO_CONFORMANCE_BUNDLES";
/// Environment variable listing provider libraries to load, `:`-separated.
pub const ENV_PROVIDER_PATHS: &str = "TURBO_CONFORMANCE_PROVIDER_PATHS";

/// Runtime description from the environment: any provider libraries named
/// in `TURBO_CONFORMANCE_PROVIDER_PATHS` are loaded in addition to the
/// built-in providers.
pub fn runtime_desc_from_env() -> RuntimeDesc {
    let provider_paths = std::env::var(ENV_PROVIDER_PATHS)
        .ok()
        .map(|v| v.split(':').filter(|p| !p.is_empty()).map(str::to_string).collect())
        .unwrap_or_default();
    RuntimeDesc { provider_paths, ..Default::default() }
}

/// The bundles the suite expects to find under the bundle root, one per
/// subdirectory. A provider that cannot serve one of these kinds is reported
/// by its capability matrix, and the suite asserts the refusal instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleKind {
    /// Dense embedder.
    Embedding,
    /// Cross-encoder reranker.
    Reranker,
    /// Sequence classifier.
    Classifier,
    /// Token classifier with span aggregation.
    TokenClassifier,
    /// Causal generative model.
    Generative,
    /// Generic named-tensor RUN model.
    Generic,
}

impl BundleKind {
    /// Every kind, in a stable order.
    pub const ALL: &'static [BundleKind] = &[
        BundleKind::Embedding,
        BundleKind::Reranker,
        BundleKind::Classifier,
        BundleKind::TokenClassifier,
        BundleKind::Generative,
        BundleKind::Generic,
    ];

    /// Subdirectory name under the bundle root.
    pub fn dir_name(self) -> &'static str {
        match self {
            BundleKind::Embedding => "embedding",
            BundleKind::Reranker => "reranker",
            BundleKind::Classifier => "classifier",
            BundleKind::TokenClassifier => "token-classifier",
            BundleKind::Generative => "generative",
            BundleKind::Generic => "generic",
        }
    }

    /// The equivalent kind for [`turbo::mock::write_mock_bundle`], used by
    /// tests that need a scratch copy of a bundle to tamper with.
    pub fn mock_kind(self) -> turbo::mock::MockBundleKind {
        use turbo::mock::MockBundleKind as M;
        match self {
            BundleKind::Embedding => M::Embedding,
            BundleKind::Reranker => M::Reranker,
            BundleKind::Classifier => M::Classifier,
            BundleKind::TokenClassifier => M::TokenClassifier,
            BundleKind::Generative => M::Generative,
            BundleKind::Generic => M::Generic,
        }
    }
}

/// Repository root, derived from this crate's manifest directory.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Default bundle root: the committed mock bundles.
pub fn default_bundle_root() -> PathBuf {
    repo_root().join("testdata").join("bundles").join("mock")
}

/// The system under test: one runtime, one device on it, and the directory
/// the bundles come from. Everything the suite touches goes through here so a
/// hardware run only changes the environment, never the cases.
pub struct Target {
    /// Runtime holding the provider registry.
    pub runtime: Arc<Runtime>,
    /// Index of the device under test in `runtime.devices()`.
    pub device_index: u32,
    /// Directory holding one subdirectory per [`BundleKind`].
    pub bundle_root: PathBuf,
    /// Cached static info of the device under test (immutable after creation).
    device: DeviceInfo,
}

impl Target {
    /// The default target: the built-in providers, AUTO device selection
    /// (never a CPU), and the committed mock bundles.
    pub fn mock() -> Self {
        let runtime = turbo::create_runtime(runtime_desc_from_env()).expect("create runtime");
        // The mock target is always the mock accelerator (ordinal 1), even when
        // the environment loads other providers.
        let selector = DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: "mock".into(),
            ordinal: 1,
            ..Default::default()
        };
        let device_index = runtime.select(&selector).expect("mock accelerator selection");
        let device = runtime.device(device_index).expect("device under test").info.clone();
        Self { runtime, device_index, bundle_root: default_bundle_root(), device }
    }

    /// The target described by the environment, falling back to [`Target::mock`].
    pub fn from_env() -> Self {
        let runtime = turbo::create_runtime(runtime_desc_from_env()).expect("create runtime");
        let provider = std::env::var(ENV_PROVIDER).unwrap_or_default();
        let selector = if provider.is_empty() {
            DeviceSelector::default()
        } else {
            let ordinal = std::env::var(ENV_ORDINAL)
                .ok()
                .map(|s| s.parse::<u32>().unwrap_or_else(|e| panic!("{ENV_ORDINAL}: {e}")))
                .unwrap_or(0);
            DeviceSelector { policy: SelectPolicy::Explicit, provider_id: provider, ordinal, ..Default::default() }
        };
        let device_index =
            runtime.select(&selector).unwrap_or_else(|e| panic!("selecting the device under test ({selector:?}): {e}"));
        let bundle_root = std::env::var_os(ENV_BUNDLES).map(PathBuf::from).unwrap_or_else(default_bundle_root);
        let device = runtime.device(device_index).expect("device under test").info.clone();
        Self { runtime, device_index, bundle_root, device }
    }

    /// Static info of the device under test.
    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    /// Provider id of the device under test.
    pub fn provider_id(&self) -> &str {
        &self.device().provider_id
    }

    /// Ordinal of the device under test within its provider.
    pub fn ordinal(&self) -> u32 {
        self.device().ordinal
    }

    /// Capability bitset of the device under test.
    pub fn caps(&self) -> u64 {
        self.device().caps
    }

    /// True when the device advertises every bit in `bits`.
    pub fn has(&self, bits: u64) -> bool {
        self.caps() & bits == bits
    }

    /// True when the target is the built-in mock provider, for the handful of
    /// cases that assert mock-specific numbers.
    pub fn is_mock(&self) -> bool {
        self.provider_id() == turbo::mock::MOCK_PROVIDER_ID
    }

    /// Directory of one test bundle.
    pub fn bundle(&self, kind: BundleKind) -> PathBuf {
        self.bundle_root.join(kind.dir_name())
    }

    /// An independent runtime with the same device selected. Tests that must
    /// own the last reference to a runtime use this instead of `self.runtime`,
    /// which the `Target` itself keeps alive.
    pub fn detached(&self) -> (Arc<Runtime>, u32) {
        let runtime = turbo::create_runtime(runtime_desc_from_env()).expect("create runtime");
        let selector = DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: self.provider_id().to_string(),
            ordinal: self.ordinal(),
            ..Default::default()
        };
        let index = runtime.select(&selector).expect("re-selecting the device under test");
        (runtime, index)
    }

    /// A fresh context on the device under test.
    pub fn context(&self) -> Arc<Context> {
        Context::create(self.runtime.clone(), self.device_index, &ContextDesc::default())
            .unwrap_or_else(|e| panic!("context on device {}: {e}", self.device_index))
    }

    /// A model loaded on a fresh context.
    pub fn model(&self, kind: BundleKind) -> Arc<Model> {
        self.model_on(&self.context(), kind)
    }

    /// A model loaded on an existing context.
    pub fn model_on(&self, ctx: &Arc<Context>, kind: BundleKind) -> Arc<Model> {
        ctx.load_model(&self.bundle(kind), &ModelDesc::default())
            .unwrap_or_else(|e| panic!("loading the {} bundle: {e}", kind.dir_name()))
    }

    /// A model plus one session with the model's default maxima.
    pub fn session(&self, kind: BundleKind) -> (Arc<Model>, Arc<Session>) {
        let model = self.model(kind);
        let session = model.create_session(&SessionDesc::default()).expect("create session");
        (model, session)
    }
}

/// Read one f32 output of a result into a vector of rows.
pub fn read_rows(result: &turbo::handles::ResultHandle, index: u32, cols: usize) -> Vec<Vec<f32>> {
    let flat = read_f32(result, index);
    assert_eq!(flat.len() % cols, 0, "output {index} length {} is not a multiple of {cols}", flat.len());
    flat.chunks(cols).map(|c| c.to_vec()).collect()
}

/// Read one f32 output of a result, honoring its logical shape.
pub fn read_f32(result: &turbo::handles::ResultHandle, index: u32) -> Vec<f32> {
    let out = result.output(index).expect("output");
    let bytes = out.logical_bytes().expect("logical bytes") as usize;
    let mut buf = vec![0u8; bytes];
    let n = result.read(index, &mut buf).expect("result read");
    assert_eq!(n, bytes);
    buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Read one i32 output of a result.
pub fn read_i32(result: &turbo::handles::ResultHandle, index: u32) -> Vec<i32> {
    let out = result.output(index).expect("output");
    let bytes = out.logical_bytes().expect("logical bytes") as usize;
    let mut buf = vec![0u8; bytes];
    let n = result.read(index, &mut buf).expect("result read");
    assert_eq!(n, bytes);
    buf.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Scratch copies of bundles, for the cases that must corrupt one.
pub mod fixtures {
    use std::fs;
    use std::path::{Path, PathBuf};

    use tempfile::TempDir;

    /// A scratch bundle directory that is removed when dropped.
    pub struct Scratch {
        dir: TempDir,
    }

    impl Scratch {
        /// Path of the bundle directory.
        pub fn path(&self) -> &Path {
            self.dir.path()
        }

        /// The manifest as JSON.
        pub fn manifest(&self) -> serde_json::Value {
            let text = fs::read_to_string(self.path().join("bundle.json")).expect("read bundle.json");
            serde_json::from_str(&text).expect("parse bundle.json")
        }

        /// Rewrite the manifest after `f` has edited it.
        pub fn patch_manifest(&self, f: impl FnOnce(&mut serde_json::Value)) {
            let mut v = self.manifest();
            f(&mut v);
            fs::write(self.path().join("bundle.json"), serde_json::to_string_pretty(&v).expect("serialize"))
                .expect("write bundle.json");
        }

        /// Every file the manifest lists as an artifact, by format name.
        pub fn artifact_paths(&self) -> Vec<(String, PathBuf)> {
            let m = self.manifest();
            m["artifacts"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .map(|(k, v)| (k.clone(), self.path().join(v["path"].as_str().expect("artifact path"))))
                        .collect()
                })
                .unwrap_or_default()
        }
    }

    /// Copy a bundle directory into a temporary directory.
    pub fn copy_of(src: &Path) -> Scratch {
        let dir = tempfile::tempdir().expect("temp dir");
        copy_tree(src, dir.path());
        Scratch { dir }
    }

    /// An empty temporary directory (for "not a bundle" cases).
    pub fn empty() -> Scratch {
        Scratch { dir: tempfile::tempdir().expect("temp dir") }
    }

    fn copy_tree(src: &Path, dst: &Path) {
        for entry in fs::read_dir(src).unwrap_or_else(|e| panic!("reading {}: {e}", src.display())) {
            let entry = entry.expect("dir entry");
            let target = dst.join(entry.file_name());
            if entry.file_type().expect("file type").is_dir() {
                fs::create_dir_all(&target).expect("create dir");
                copy_tree(&entry.path(), &target);
            } else {
                fs::copy(entry.path(), &target).expect("copy file");
            }
        }
    }
}

/// Every permutation of `0..n`, in a stable order.
pub fn permutations(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(n);
    let mut used = vec![false; n];
    fn walk(n: usize, used: &mut Vec<bool>, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == n {
            out.push(current.clone());
            return;
        }
        for i in 0..n {
            if used[i] {
                continue;
            }
            used[i] = true;
            current.push(i);
            walk(n, used, current, out);
            current.pop();
            used[i] = false;
        }
    }
    walk(n, &mut used, &mut current, &mut out);
    out
}

/// L2 norm of a row.
pub fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Assert an error's status code and field index exactly.
#[macro_export]
macro_rules! assert_err {
    ($expr:expr, $code:expr) => {{
        let e = match $expr {
            Ok(_) => panic!("expected {} ({}), got Ok", stringify!($code), $code),
            Err(e) => e,
        };
        assert_eq!(
            e.code(),
            $code,
            "expected {} but got {} ({}): {}",
            stringify!($code),
            turbo::error::status_name(e.code()),
            e.code(),
            e.message()
        );
        e
    }};
    ($expr:expr, $code:expr, field = $field:expr) => {{
        let e = $crate::assert_err!($expr, $code);
        assert_eq!(e.field(), $field, "expected field {}, got {} ({})", $field, e.field(), e.message());
        e
    }};
}

/// Helpers for the C ABI layer: caller-owned errors, text views, and a target
/// resolved to raw handles. These call the `extern "C"` functions in
/// `turbo_capi` directly, which is exactly what a C caller does.
pub mod c {
    // These helpers take handles this library itself produced and pass them
    // straight back to the C ABI. Marking each one `unsafe fn` would only move
    // the same obligation to every call site in the suite, so the invariant is
    // documented here instead: a handle passed to any helper below must be a
    // live handle from `turbo_capi`, and each is released exactly once.
    #![allow(clippy::not_unsafe_ptr_arg_deref)]

    use std::ffi::CStr;
    use std::mem::size_of;
    use std::os::raw::c_char;

    use turbo_abi::*;

    use super::{BundleKind, Target};

    /// `struct_size` for a descriptor type.
    pub fn ssz<T>() -> u32 {
        size_of::<T>() as u32
    }

    /// A zeroed, correctly sized `turbo_error`.
    pub fn err() -> turbo_error {
        turbo_error { struct_size: ssz::<turbo_error>(), code: 0, field: 0, message: [0; TURBO_ERROR_MESSAGE_LEN] }
    }

    /// The message of an error record as a Rust string.
    pub fn message(e: &turbo_error) -> String {
        // SAFETY: the library always NUL-terminates the message.
        unsafe { CStr::from_ptr(e.message.as_ptr()) }.to_string_lossy().into_owned()
    }

    /// A `turbo_text` view over a Rust string. The string must outlive the call.
    pub fn text(s: &str) -> turbo_text {
        turbo_text { ptr: s.as_ptr().cast::<c_char>(), len: s.len() as u64 }
    }

    /// An empty `turbo_text` with a NULL pointer.
    pub fn null_text() -> turbo_text {
        turbo_text { ptr: std::ptr::null(), len: 0 }
    }

    /// A NUL-terminated string from a fixed `c_char` array in an output struct.
    pub fn fixed(field: &[c_char]) -> String {
        let bytes: Vec<u8> = field.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// True when a fixed `c_char` array contains a NUL terminator.
    pub fn is_nul_terminated(field: &[c_char]) -> bool {
        field.contains(&0)
    }

    /// Assert a C call returned exactly `code`, reporting the message.
    #[macro_export]
    macro_rules! assert_rc {
        ($call:expr, $code:expr, $err:expr) => {{
            let rc = $call;
            assert_eq!(
                rc,
                $code,
                "{} returned {} ({}) instead of {} ({}): {}",
                stringify!($call),
                $crate::c::status_name(rc),
                rc,
                stringify!($code),
                $code,
                $crate::c::message(&$err)
            );
        }};
        ($call:expr, $code:expr, $err:expr, field = $field:expr) => {{
            $crate::assert_rc!($call, $code, $err);
            assert_eq!(
                $err.field,
                $field,
                "expected field {}, got {}: {}",
                $field,
                $err.field,
                $crate::c::message(&$err)
            );
        }};
    }

    /// Symbolic name of a status code, through the C entry point.
    pub fn status_name(code: i32) -> String {
        // SAFETY: the library returns a static NUL-terminated string.
        unsafe { CStr::from_ptr(turbo_capi::turbo_status_name(code)) }.to_string_lossy().into_owned()
    }

    /// A runtime and device resolved through the C ABI, matching a [`Target`].
    /// Releases its runtime on drop.
    pub struct CTarget {
        /// Raw runtime handle.
        pub rt: *mut turbo_runtime,
        /// Device index within that runtime.
        pub device: u32,
        /// Provider id of that device.
        pub provider_id: String,
        /// Bundle paths, kept as NUL-free Rust strings for `turbo_text`.
        root: String,
    }

    impl CTarget {
        /// Resolve the same device the Rust-layer [`Target`] uses.
        pub fn new(t: &Target) -> Self {
            let mut e = err();
            let mut rt: *mut turbo_runtime = std::ptr::null_mut();
            // SAFETY: out pointers are valid; desc is NULL for defaults.
            let rc = unsafe { turbo_capi::turbo_runtime_create(std::ptr::null(), &mut rt, &mut e) };
            assert_eq!(rc, TURBO_OK, "turbo_runtime_create: {}", message(&e));
            let provider_id = t.provider_id().to_string();
            let mut device = 0u32;
            let sel = turbo_device_selector {
                struct_size: ssz::<turbo_device_selector>(),
                policy: TURBO_SELECT_EXPLICIT,
                kind_mask: 0,
                ordinal: t.ordinal(),
                provider_id: text(&provider_id),
                vendor: null_text(),
            };
            // SAFETY: the selector borrows `provider_id`, which outlives the call.
            let rc = unsafe { turbo_capi::turbo_runtime_select_device(rt, &sel, &mut device, &mut e) };
            assert_eq!(rc, TURBO_OK, "turbo_runtime_select_device: {}", message(&e));
            Self { rt, device, provider_id, root: t.bundle_root.to_string_lossy().into_owned() }
        }

        /// Path of one bundle as an owned string (for `turbo_text`).
        pub fn bundle(&self, kind: BundleKind) -> String {
            format!("{}/{}", self.root, kind.dir_name())
        }

        /// Create a context on the device under test.
        pub fn context(&self) -> *mut turbo_context {
            let mut e = err();
            let mut ctx: *mut turbo_context = std::ptr::null_mut();
            // SAFETY: valid runtime handle and out pointer.
            let rc =
                unsafe { turbo_capi::turbo_context_create(self.rt, self.device, std::ptr::null(), &mut ctx, &mut e) };
            assert_eq!(rc, TURBO_OK, "turbo_context_create: {}", message(&e));
            ctx
        }

        /// Load a bundle on a context.
        pub fn model(&self, ctx: *mut turbo_context, kind: BundleKind) -> *mut turbo_model {
            let path = self.bundle(kind);
            let mut e = err();
            let mut m: *mut turbo_model = std::ptr::null_mut();
            // SAFETY: valid context handle; `path` outlives the call.
            let rc = unsafe { turbo_capi::turbo_model_load(ctx, text(&path), std::ptr::null(), &mut m, &mut e) };
            assert_eq!(rc, TURBO_OK, "turbo_model_load({}): {}", path, message(&e));
            m
        }

        /// Create a session with the model's default maxima.
        pub fn session(&self, m: *mut turbo_model) -> *mut turbo_session {
            let mut e = err();
            let mut s: *mut turbo_session = std::ptr::null_mut();
            // SAFETY: valid model handle and out pointer.
            let rc = unsafe { turbo_capi::turbo_session_create(m, std::ptr::null(), &mut s, &mut e) };
            assert_eq!(rc, TURBO_OK, "turbo_session_create: {}", message(&e));
            s
        }
    }

    impl Drop for CTarget {
        fn drop(&mut self) {
            // SAFETY: the handle came from turbo_runtime_create and is released once.
            unsafe { turbo_capi::turbo_runtime_release(self.rt) };
        }
    }

    /// Read `n` f32 values from result output `index`.
    pub fn read_f32(r: *mut turbo_result, index: u32, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        let mut e = err();
        let mut written = 0u64;
        // SAFETY: `out` holds `n * 4` writable bytes.
        let rc = unsafe {
            turbo_capi::turbo_result_read(r, index, out.as_mut_ptr().cast(), (n * 4) as u64, &mut written, &mut e)
        };
        assert_eq!(rc, TURBO_OK, "turbo_result_read: {}", message(&e));
        assert_eq!(written, (n * 4) as u64);
        out
    }

    /// Result info for a result handle.
    pub fn result_info(r: *mut turbo_result) -> turbo_result_info {
        let mut ri = turbo_result_info { struct_size: ssz::<turbo_result_info>(), ..Default::default() };
        let mut e = err();
        // SAFETY: valid result handle and out pointer.
        let rc = unsafe { turbo_capi::turbo_result_get_info(r, &mut ri, &mut e) };
        assert_eq!(rc, TURBO_OK, "turbo_result_get_info: {}", message(&e));
        ri
    }

    /// Default-valued embed options with the right `struct_size`.
    pub fn embed_options() -> turbo_embed_options {
        turbo_embed_options { struct_size: ssz::<turbo_embed_options>(), ..Default::default() }
    }

    /// Default-valued rerank options with the right `struct_size`.
    pub fn rerank_options() -> turbo_rerank_options {
        turbo_rerank_options { struct_size: ssz::<turbo_rerank_options>(), ..Default::default() }
    }

    /// Default-valued classify options with the right `struct_size`.
    pub fn classify_options() -> turbo_classify_options {
        turbo_classify_options { struct_size: ssz::<turbo_classify_options>(), ..Default::default() }
    }

    /// A zeroed generate descriptor with the right `struct_size`.
    pub fn generate_desc() -> turbo_generate_desc {
        turbo_generate_desc {
            struct_size: ssz::<turbo_generate_desc>(),
            max_new_tokens: 0,
            min_new_tokens: 0,
            n_sequences: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.0,
            repeat_penalty: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            has_seed: 0,
            seed: 0,
            n_stop: 0,
            n_stop_tokens: 0,
            stop: std::ptr::null(),
            stop_tokens: std::ptr::null(),
            n_logit_bias: 0,
            logprobs: 0,
            logit_bias: std::ptr::null(),
            structured_kind: TURBO_STRUCTURED_NONE,
            echo: 0,
            structured: null_text(),
            n_tools: 0,
            n_options: 0,
            tools: std::ptr::null(),
            options: std::ptr::null(),
        }
    }

    /// A zeroed generation chunk with the right `struct_size`.
    pub fn chunk() -> turbo_generation_chunk {
        turbo_generation_chunk {
            struct_size: ssz::<turbo_generation_chunk>(),
            sequence: 0,
            n_tokens: 0,
            n_logprobs: 0,
            tokens: std::ptr::null(),
            text: null_text(),
            logprobs: std::ptr::null(),
            done: 0,
            finish_reason: TURBO_FINISH_NONE,
            prompt_tokens: 0,
            generated_tokens: 0,
        }
    }

    /// The decoded text of a chunk.
    pub fn chunk_text(c: &turbo_generation_chunk) -> String {
        if c.text.len == 0 {
            return String::new();
        }
        // SAFETY: the library reports a valid UTF-8 view owned by the generation.
        let bytes = unsafe { std::slice::from_raw_parts(c.text.ptr.cast::<u8>(), c.text.len as usize) };
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// The token ids of a chunk.
    pub fn chunk_tokens(c: &turbo_generation_chunk) -> Vec<i32> {
        if c.n_tokens == 0 {
            return Vec::new();
        }
        // SAFETY: `n_tokens` entries are valid until the next call on the generation.
        unsafe { std::slice::from_raw_parts(c.tokens, c.n_tokens as usize) }.to_vec()
    }
}
