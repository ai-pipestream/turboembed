//! Group `bundle`, C ABI layer: every bundle failure mode reaches the caller
//! as its own status code through `turbo_model_load` and `turbo_can_run`, and
//! `turbo_model_info` reports the frozen contract as NUL-terminated strings.

use std::fs;
use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, fixtures, needs, BundleKind, Target};

struct Ctx {
    _ct: c::CTarget,
    rt: *mut turbo_runtime,
    device: u32,
    ctx: *mut turbo_context,
}

impl Ctx {
    fn new(t: &Target) -> Self {
        let ct = c::CTarget::new(t);
        let ctx = ct.context();
        let (rt, device) = (ct.rt, ct.device);
        Self { _ct: ct, rt, device, ctx }
    }

    /// Try to load `dir`; returns the status code (releasing any model).
    fn load(&self, dir: &std::path::Path) -> (i32, turbo_error) {
        let path = dir.to_string_lossy().into_owned();
        let mut e = c::err();
        let mut m: *mut turbo_model = ptr::null_mut();
        // SAFETY: valid context handle; `path` outlives the call.
        let rc = unsafe { turbo_model_load(self.ctx, c::text(&path), ptr::null(), &mut m, &mut e) };
        if rc == TURBO_OK {
            // SAFETY: released once.
            unsafe { turbo_model_release(m) };
        } else {
            assert!(m.is_null(), "a failed load must not hand back a model");
        }
        (rc, e)
    }

    /// Ask `turbo_can_run` about `dir` with the embedding task.
    fn can_run(&self, dir: &std::path::Path) -> i32 {
        let path = dir.to_string_lossy().into_owned();
        let mut e = c::err();
        // SAFETY: valid runtime handle; `path` outlives the call.
        unsafe { turbo_can_run(self.rt, self.device, c::text(&path), TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, &mut e) }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        // SAFETY: released once.
        unsafe { turbo_context_release(self.ctx) };
    }
}

#[test]
fn bundle_missing_directory_and_manifest_are_not_found() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let c = Ctx::new(&t);
    let scratch = fixtures::empty();
    let missing = scratch.path().join("nope");
    assert_eq!(c.load(&missing).0, TURBO_E_BUNDLE_NOT_FOUND);
    assert_eq!(c.can_run(&missing), TURBO_E_BUNDLE_NOT_FOUND);

    let copy = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    fs::remove_file(copy.path().join("bundle.json")).expect("remove the manifest");
    assert_eq!(c.load(copy.path()).0, TURBO_E_BUNDLE_NOT_FOUND);
    assert_eq!(c.can_run(copy.path()), TURBO_E_BUNDLE_NOT_FOUND);
}

#[test]
fn bundle_wrong_version_is_invalid() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let c = Ctx::new(&t);
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    scratch.patch_manifest(|m| m["bundle_version"] = serde_json::json!(1));
    let (rc, e) = c.load(scratch.path());
    assert_eq!(rc, TURBO_E_BUNDLE_INVALID, "{}", c::message(&e));
    assert!(c::message(&e).contains('1'), "{}", c::message(&e));
    assert_eq!(c.can_run(scratch.path()), TURBO_E_BUNDLE_INVALID);
}

#[test]
fn bundle_tampered_artifact_is_integrity() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let c = Ctx::new(&t);
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    let (_, path) = scratch.artifact_paths().into_iter().next().expect("an artifact");
    let mut body = fs::read(&path).expect("read");
    body.push(b' ');
    fs::write(&path, &body).expect("tamper");
    let (rc, e) = c.load(scratch.path());
    assert_eq!(rc, TURBO_E_BUNDLE_INTEGRITY, "{}", c::message(&e));
    assert_eq!(c.can_run(scratch.path()), TURBO_E_BUNDLE_INTEGRITY);
}

#[test]
fn bundle_path_escape_is_invalid() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let c = Ctx::new(&t);
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    let (format, _) = scratch.artifact_paths().into_iter().next().expect("an artifact");
    scratch.patch_manifest(|m| m["artifacts"][&format]["path"] = serde_json::json!("../../etc/passwd"));
    assert_eq!(c.load(scratch.path()).0, TURBO_E_BUNDLE_INVALID);
    assert_eq!(c.can_run(scratch.path()), TURBO_E_BUNDLE_INVALID);
}

#[test]
fn bundle_missing_contract_fields_are_invalid() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let c = Ctx::new(&t);
    for field in ["dim", "pooling", "normalize", "max_seq"] {
        let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
        scratch.patch_manifest(|m| {
            m["contract"].as_object_mut().expect("contract").remove(field);
        });
        let (rc, e) = c.load(scratch.path());
        assert_eq!(rc, TURBO_E_BUNDLE_INVALID, "contract.{field}: {}", c::message(&e));
        assert!(c::message(&e).contains(field), "{}", c::message(&e));
    }
}

#[test]
fn bundle_without_a_mock_artifact_is_no_artifact() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let c = Ctx::new(&t);
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    let body = b"a format no provider in this build knows";
    fs::write(scratch.path().join("foreign.bin"), body).expect("write");
    scratch.patch_manifest(|m| {
        m["artifacts"] = serde_json::json!({
            "not_a_real_format": { "path": "foreign.bin", "sha256": turbo::bundle::sha256_bytes(body) }
        });
    });
    let (rc, e) = c.load(scratch.path());
    assert_eq!(rc, TURBO_E_BUNDLE_NO_ARTIFACT, "{}", c::message(&e));
    assert_eq!(c.can_run(scratch.path()), TURBO_E_BUNDLE_NO_ARTIFACT);
}

#[test]
fn bundle_an_empty_path_is_invalid_argument() {
    let t = Target::from_env();
    let c = Ctx::new(&t);
    let mut e = c::err();
    let mut m: *mut turbo_model = ptr::null_mut();
    // SAFETY: valid context handle; the empty view is deliberate.
    unsafe {
        assert_rc!(turbo_model_load(c.ctx, c::text(""), ptr::null(), &mut m, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        // A NULL text view with a length is caught before the file system.
        let bad = turbo_text { ptr: ptr::null(), len: 12 };
        assert_rc!(turbo_model_load(c.ctx, bad, ptr::null(), &mut m, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        // Invalid UTF-8 in a path is refused as such.
        let bytes = [0x2fu8, 0xff, 0xfe];
        let broken = turbo_text { ptr: bytes.as_ptr().cast(), len: 3 };
        assert_rc!(turbo_model_load(c.ctx, broken, ptr::null(), &mut m, &mut e), TURBO_E_INVALID_UTF8, e);
    }
}

#[test]
fn bundle_model_descriptor_is_validated() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let c = Ctx::new(&t);
    let path = t.bundle(BundleKind::Embedding).to_string_lossy().into_owned();
    let mut e = c::err();
    let mut m: *mut turbo_model = ptr::null_mut();
    let base = turbo_model_desc {
        struct_size: ssz::<turbo_model_desc>(),
        n_options: 0,
        options: ptr::null(),
        next: ptr::null(),
    };
    // SAFETY: valid context handle; `path` and the descriptors outlive the calls.
    unsafe {
        let big = turbo_model_desc { struct_size: ssz::<turbo_model_desc>() + 4, ..base };
        assert_rc!(turbo_model_load(c.ctx, c::text(&path), &big, &mut m, &mut e), TURBO_E_INVALID_STRUCT_SIZE, e);
        let chained = turbo_model_desc { next: &base as *const _ as *const std::ffi::c_void, ..base };
        assert_rc!(turbo_model_load(c.ctx, c::text(&path), &chained, &mut m, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        let key = "definitely_not_an_option";
        let kvs = [turbo_kv { key: c::text(key), value: c::text("1") }];
        let with_options = turbo_model_desc { n_options: 1, options: kvs.as_ptr(), ..base };
        assert_rc!(turbo_model_load(c.ctx, c::text(&path), &with_options, &mut m, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        // A count without an array is caught before any read.
        let no_array = turbo_model_desc { n_options: 2, options: ptr::null(), ..base };
        assert_rc!(turbo_model_load(c.ctx, c::text(&path), &no_array, &mut m, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        // The well-formed descriptor loads.
        assert_rc!(turbo_model_load(c.ctx, c::text(&path), &base, &mut m, &mut e), TURBO_OK, e);
        turbo_model_release(m);
    }
}

#[test]
fn bundle_model_info_strings_are_nul_terminated() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let mut e = c::err();
    for kind in BundleKind::ALL {
        let path = ct.bundle(*kind);
        let mut m: *mut turbo_model = ptr::null_mut();
        // SAFETY: valid context handle; `path` outlives the call.
        let rc = unsafe { turbo_model_load(ctx, c::text(&path), ptr::null(), &mut m, &mut e) };
        if rc != TURBO_OK {
            continue;
        }
        let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
        // SAFETY: valid model handle and out pointer.
        assert_rc!(unsafe { turbo_model_get_info(m, &mut info, &mut e) }, TURBO_OK, e);
        for (what, field) in [
            ("model_id", &info.model_id[..]),
            ("revision", &info.revision[..]),
            ("tokenizer_sha256", &info.tokenizer_sha256[..]),
            ("provider_id", &info.provider_id[..]),
            ("prefix_query", &info.prefix_query[..]),
            ("prefix_document", &info.prefix_document[..]),
        ] {
            assert!(c::is_nul_terminated(field), "{}: {what} is not NUL-terminated", kind.dir_name());
        }
        assert!(!c::fixed(&info.model_id).is_empty(), "{}: model_id is empty", kind.dir_name());
        assert_eq!(c::fixed(&info.provider_id), t.provider_id());
        assert!(info.max_seq > 0 && info.max_batch > 0, "{}: unusable limits", kind.dir_name());
        assert!(info.task >= TURBO_TASK_EMBED && info.task <= TURBO_TASK_CHUNK, "{}: task", kind.dir_name());
        assert!(info.kind >= TURBO_MODEL_EMBEDDING && info.kind <= TURBO_MODEL_GENERIC, "{}: kind", kind.dir_name());
        assert_eq!(info.modality, TURBO_MODALITY_TEXT, "{}: v1 is text only", kind.dir_name());
        for stage in info.stage_placement {
            assert!(stage <= TURBO_STAGE_FUSED, "{}: stage placement {stage}", kind.dir_name());
        }
        assert!(info.fully_accelerated <= 1, "fully_accelerated is a boolean");
        // SAFETY: released once.
        unsafe { turbo_model_release(m) };
    }
    // SAFETY: released once.
    unsafe { turbo_context_release(ctx) };
    drop(ct);
}
