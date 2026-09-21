//! Group `contract`, C ABI layer, part two: `struct_size` versioning, the
//! caller-owned error record, the status-name vocabulary, and the ABI
//! version. Split from `contract_c.rs` to keep each file readable.

use std::mem::size_of;
use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, BundleKind, Target};

struct Fixture {
    _target: Target,
    _ct: c::CTarget,
    ctx: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
}

impl Fixture {
    fn new() -> Self {
        let target = Target::from_env();
        let ct = c::CTarget::new(&target);
        let ctx = ct.context();
        let model = ct.model(ctx, BundleKind::Embedding);
        let session = ct.session(model);
        Self { _target: target, _ct: ct, ctx, model, session }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // SAFETY: each handle was produced by this fixture and is released once.
        unsafe {
            turbo_session_release(self.session);
            turbo_model_release(self.model);
            turbo_context_release(self.ctx);
        }
    }
}

/// A `turbo_error` followed by a canary, to prove the message never overruns.
#[repr(C)]
struct GuardedError {
    err: turbo_error,
    canary: [u8; 64],
}

impl GuardedError {
    fn new() -> Self {
        Self {
            err: turbo_error {
                struct_size: ssz::<turbo_error>(),
                code: 0,
                field: 0,
                message: [0x41; TURBO_ERROR_MESSAGE_LEN],
            },
            canary: [0xAA; 64],
        }
    }
}

#[test]
fn contract_struct_size_larger_than_known_is_rejected() {
    let f = Fixture::new();
    let mut e = c::err();
    let s = "hello";
    let texts = [c::text(s)];
    let mut o = c::embed_options();
    o.struct_size = ssz::<turbo_embed_options>() + 4;
    // SAFETY: the library must refuse rather than read fields it does not know.
    assert_rc!(
        unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 1, &o, &mut e) },
        TURBO_E_INVALID_STRUCT_SIZE,
        e
    );
    o.struct_size = 4096;
    assert_rc!(
        unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 1, &o, &mut e) },
        TURBO_E_INVALID_STRUCT_SIZE,
        e
    );
    // Output descriptors are checked the same way.
    let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>() + 8, ..unsafe { std::mem::zeroed() } };
    assert_rc!(unsafe { turbo_model_get_info(f.model, &mut info, &mut e) }, TURBO_E_INVALID_STRUCT_SIZE, e);
    let mut sd = turbo_session_desc {
        struct_size: ssz::<turbo_session_desc>() + 1,
        max_batch: 0,
        max_seq: 0,
        n_options: 0,
        options: ptr::null(),
        next: ptr::null(),
    };
    let mut out = ptr::null_mut();
    assert_rc!(unsafe { turbo_session_create(f.model, &sd, &mut out, &mut e) }, TURBO_E_INVALID_STRUCT_SIZE, e);
    sd.struct_size = 3;
    assert_rc!(unsafe { turbo_session_create(f.model, &sd, &mut out, &mut e) }, TURBO_E_INVALID_STRUCT_SIZE, e);
}

#[test]
fn contract_struct_size_of_an_older_caller_is_accepted() {
    let f = Fixture::new();
    let mut e = c::err();
    let s = "hello world";
    let texts = [c::text(s)];
    // An older caller declares a shorter struct; the library must accept it
    // and behave as that older version did.
    let mut o = c::embed_options();
    o.struct_size = 8;
    // SAFETY: the struct is fully initialized; only `struct_size` is short.
    assert_rc!(unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 1, &o, &mut e) }, TURBO_OK, e);

    // The same rule on an output struct: only the declared prefix is written.
    let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
    let short = 16u32;
    info.struct_size = short;
    let sentinel = 0xDEAD_BEEFu32;
    info.dim = sentinel;
    assert_rc!(unsafe { turbo_model_get_info(f.model, &mut info, &mut e) }, TURBO_OK, e);
    assert_eq!(info.struct_size, short);
    assert_ne!(info.task, 0, "the declared prefix is filled in");
    assert_eq!(info.dim, sentinel, "turbo_model_info.dim lies past the declared struct_size and must not be written");
}

#[test]
fn contract_fields_beyond_the_declared_struct_size_are_ignored() {
    // PLAN.md section 4.3 and the turbo-abi docs: "The library reads only
    // fields below the size the caller declared. Appended fields must have a
    // zero value meaning 'old behavior'." A caller compiled against an older
    // header declares `struct_size = 8` (struct_size + truncate) and owns
    // only those 8 bytes; everything after is not its memory. The library
    // accepts the short size but still reads `pooling`, `normalize`,
    // `output_dim` and the rest, so an old caller gets an error about a field
    // it never set -- and the read itself is out of bounds for that caller.
    let f = Fixture::new();
    let mut e = c::err();
    let s = "hello world";
    let texts = [c::text(s)];
    let mut o = c::embed_options();
    o.struct_size = 8;
    // Everything past `truncate` is outside the declared struct and must be
    // treated as zero, whatever the bytes happen to hold.
    o.pooling = TURBO_POOLING_CLS;
    o.output_dim = 4;
    o.output_dtype = TURBO_OUTPUT_I8;
    // SAFETY: the whole struct is initialized; only `struct_size` is short.
    let rc = unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 1, &o, &mut e) };
    assert_eq!(
        rc,
        TURBO_OK,
        "fields past struct_size = 8 must be ignored, but the library read them and returned {} ({})",
        c::status_name(rc),
        c::message(&e)
    );
}

#[test]
fn contract_error_message_is_nul_terminated_and_never_overruns() {
    let f = Fixture::new();
    let mut g = GuardedError::new();
    // Force a long message: the key is echoed back in the diagnostic.
    let key = "k".repeat(2048);
    let kvs = [turbo_kv { key: c::text(&key), value: c::text("1") }];
    let sd = turbo_session_desc {
        struct_size: ssz::<turbo_session_desc>(),
        max_batch: 0,
        max_seq: 0,
        n_options: 1,
        options: kvs.as_ptr(),
        next: ptr::null(),
    };
    let mut out = ptr::null_mut();
    // SAFETY: valid model and options array.
    let rc = unsafe { turbo_session_create(f.model, &sd, &mut out, &mut g.err) };
    assert_eq!(rc, TURBO_E_INVALID_ARGUMENT, "{}", c::message(&g.err));
    assert!(out.is_null(), "a failed create must not hand back a handle");
    let terminator = g.err.message.iter().position(|&ch| ch == 0).expect("the message is NUL-terminated");
    assert!(terminator < TURBO_ERROR_MESSAGE_LEN, "the terminator is inside the array");
    assert!(terminator > 0, "the message is not empty");
    assert_eq!(g.canary, [0xAA; 64], "the message overran turbo_error");
    // The message is valid UTF-8 up to the terminator.
    let bytes: Vec<u8> = g.err.message[..terminator].iter().map(|&ch| ch as u8).collect();
    std::str::from_utf8(&bytes).expect("the message is valid UTF-8");
}

#[test]
fn contract_error_may_be_null() {
    let f = Fixture::new();
    let s = "x";
    let texts = [c::text(s)];
    // SAFETY: a NULL error record means "return the code only".
    unsafe {
        assert_eq!(
            turbo_session_write_text(ptr::null_mut(), texts.as_ptr(), 1, ptr::null(), ptr::null_mut()),
            TURBO_E_INVALID_HANDLE
        );
        assert_eq!(turbo_session_write_text(f.session, texts.as_ptr(), 1, ptr::null(), ptr::null_mut()), TURBO_OK);
        let mut r: *mut turbo_result = ptr::null_mut();
        assert_eq!(turbo_session_run(f.session, ptr::null(), &mut r, ptr::null_mut()), TURBO_OK);
        turbo_result_release(r);
    }
}

#[test]
fn contract_error_struct_size_below_the_code_field_is_rejected() {
    let f = Fixture::new();
    let s = "x";
    let texts = [c::text(s)];
    let mut tiny = c::err();
    tiny.struct_size = 4;
    // SAFETY: a record too small to hold a code cannot be written at all.
    let rc = unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 1, ptr::null(), &mut tiny) };
    assert_eq!(rc, TURBO_E_INVALID_STRUCT_SIZE);
    // A record that holds only the code still gets it.
    let mut code_only = c::err();
    code_only.struct_size = 8;
    let rc = unsafe { turbo_session_write_text(ptr::null_mut(), texts.as_ptr(), 1, ptr::null(), &mut code_only) };
    assert_eq!(rc, TURBO_E_INVALID_HANDLE);
    assert_eq!(code_only.code, TURBO_E_INVALID_HANDLE);
}

#[test]
fn contract_status_names_are_static_c_strings() {
    for code in [
        TURBO_OK,
        TURBO_E_INVALID_ARGUMENT,
        TURBO_E_INVALID_STRUCT_SIZE,
        TURBO_E_INVALID_UTF8,
        TURBO_E_INVALID_HANDLE,
        TURBO_E_INVALID_SHAPE,
        TURBO_E_INVALID_STATE,
        TURBO_E_INVALID_ENUM,
        TURBO_E_UNSUPPORTED,
        TURBO_E_UNSUPPORTED_OPTION,
        TURBO_E_UNSUPPORTED_TASK,
        TURBO_E_UNSUPPORTED_DTYPE,
        TURBO_E_UNSUPPORTED_PLACEMENT,
        TURBO_E_NOT_IMPLEMENTED,
        TURBO_E_UNSUPPORTED_MODALITY,
        TURBO_E_OUT_OF_MEMORY,
        TURBO_E_BUSY,
        TURBO_E_OVERLOADED,
        TURBO_E_CAPACITY,
        TURBO_E_DEVICE_NOT_FOUND,
        TURBO_E_DEVICE_UNAVAILABLE,
        TURBO_E_RUNTIME,
        TURBO_E_PROVIDER_LOAD,
        TURBO_E_ABI_MISMATCH,
        TURBO_E_CANCELLED,
        TURBO_E_BUNDLE_NOT_FOUND,
        TURBO_E_BUNDLE_INVALID,
        TURBO_E_BUNDLE_INTEGRITY,
        TURBO_E_BUNDLE_NO_ARTIFACT,
        TURBO_E_INTERNAL,
        TURBO_E_PANIC,
    ] {
        let name = c::status_name(code);
        assert_eq!(name, turbo::error::status_name(code), "C and Rust disagree on {code:#x}");
        assert!(name.starts_with("TURBO_"), "{name}");
        assert_ne!(name, "TURBO_E_UNKNOWN", "{code:#x} has no symbolic name");
    }
    assert_eq!(c::status_name(-7), "TURBO_E_UNKNOWN");
    assert_eq!(c::status_name(0x999), "TURBO_E_UNKNOWN");
}

#[test]
fn contract_abi_version_equals_the_header_constant() {
    let header = turbo_conformance::repo_root().join("include").join("turbo").join("turbo_types.h");
    let text = std::fs::read_to_string(&header).unwrap_or_else(|e| panic!("reading {}: {e}", header.display()));
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("#define TURBO_ABI_VERSION"))
        .unwrap_or_else(|| panic!("{} does not define TURBO_ABI_VERSION", header.display()));
    let value: u32 = line.split_whitespace().nth(2).expect("a value").parse().expect("a number");
    assert_eq!(value, TURBO_ABI_VERSION, "the committed header and turbo-abi disagree");
    assert_eq!(turbo_abi_version(), TURBO_ABI_VERSION, "the library reports a different ABI version");
}

#[test]
fn contract_error_record_layout_matches_the_abi() {
    assert_eq!(size_of::<turbo_error>(), 12 + TURBO_ERROR_MESSAGE_LEN);
    assert_eq!(TURBO_ERROR_MESSAGE_LEN, 496);
}
