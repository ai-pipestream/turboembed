//! Group `device`, C ABI layer: `turbo_runtime_select_device` policy and the
//! fixed-size, NUL-terminated strings in `turbo_device_info`.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, needs, BundleKind, Target};

fn device_info(rt: *mut turbo_runtime, index: u32) -> turbo_device_info {
    let mut e = c::err();
    let mut di = turbo_device_info { struct_size: ssz::<turbo_device_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(unsafe { turbo_runtime_device_info(rt, index, &mut di, &mut e) }, TURBO_OK, e);
    di
}

fn selector() -> turbo_device_selector {
    turbo_device_selector {
        struct_size: ssz::<turbo_device_selector>(),
        policy: TURBO_SELECT_AUTO,
        kind_mask: 0,
        ordinal: 0,
        provider_id: c::null_text(),
        vendor: c::null_text(),
    }
}

#[test]
fn device_auto_never_selects_a_cpu() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut index = u32::MAX;
    // A NULL selector means AUTO.
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(unsafe { turbo_runtime_select_device(ct.rt, ptr::null(), &mut index, &mut e) }, TURBO_OK, e);
    let di = device_info(ct.rt, index);
    assert_ne!(di.kind, TURBO_DEVICE_CPU, "AUTO selected a CPU device: {}", c::fixed(&di.name));
    // An explicit AUTO selector behaves the same.
    let sel = selector();
    let mut again = u32::MAX;
    // SAFETY: as above.
    assert_rc!(unsafe { turbo_runtime_select_device(ct.rt, &sel, &mut again, &mut e) }, TURBO_OK, e);
    assert_eq!(index, again);
}

#[test]
fn device_auto_with_a_cpu_only_mask_is_device_not_found() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut index = u32::MAX;
    let sel = turbo_device_selector { kind_mask: 1 << TURBO_DEVICE_CPU, ..selector() };
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(unsafe { turbo_runtime_select_device(ct.rt, &sel, &mut index, &mut e) }, TURBO_E_DEVICE_NOT_FOUND, e);
    let message = c::message(&e);
    assert!(message.contains("never selected automatically"), "the message must say why CPU is excluded: {message}");
}

#[test]
fn device_selector_with_unknown_bits_is_rejected() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut index = u32::MAX;
    // A kind mask with bits outside the known kinds names its field.
    let sel = turbo_device_selector { kind_mask: 1 << 31, ..selector() };
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(
        unsafe { turbo_runtime_select_device(ct.rt, &sel, &mut index, &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e,
        field = 3
    );
    // An unknown policy is an enum error naming its field.
    let sel = turbo_device_selector { policy: 42, ..selector() };
    // SAFETY: as above.
    assert_rc!(
        unsafe { turbo_runtime_select_device(ct.rt, &sel, &mut index, &mut e) },
        TURBO_E_INVALID_ENUM,
        e,
        field = 2
    );
    // A struct_size the library does not know is refused.
    let sel = turbo_device_selector { struct_size: ssz::<turbo_device_selector>() + 8, ..selector() };
    // SAFETY: as above.
    assert_rc!(unsafe { turbo_runtime_select_device(ct.rt, &sel, &mut index, &mut e) }, TURBO_E_INVALID_STRUCT_SIZE, e);
}

#[test]
fn device_explicit_without_a_provider_is_invalid_argument() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut index = u32::MAX;
    let sel = turbo_device_selector { policy: TURBO_SELECT_EXPLICIT, ..selector() };
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(
        unsafe { turbo_runtime_select_device(ct.rt, &sel, &mut index, &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e,
        field = 5
    );
}

#[test]
fn device_explicit_with_a_wrong_ordinal_is_device_not_found() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut index = u32::MAX;
    let provider = t.provider_id().to_string();
    let sel = turbo_device_selector {
        policy: TURBO_SELECT_EXPLICIT,
        provider_id: c::text(&provider),
        ordinal: 4242,
        ..selector()
    };
    // SAFETY: valid runtime handle; `provider` outlives the call.
    assert_rc!(unsafe { turbo_runtime_select_device(ct.rt, &sel, &mut index, &mut e) }, TURBO_E_DEVICE_NOT_FOUND, e);
    assert!(c::message(&e).contains("4242"), "{}", c::message(&e));
}

#[test]
fn device_explicit_cpu_loads_and_runs() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut count = 0u32;
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(unsafe { turbo_runtime_device_count(ct.rt, &mut count, &mut e) }, TURBO_OK, e);
    assert!(count > 0, "a runtime with no devices cannot be conformance-tested");

    // The provider under test's own CPU device, not the built-in mock's.
    let cpu = (0..count)
        .map(|i| (i, device_info(ct.rt, i)))
        .find(|(_, d)| d.kind == TURBO_DEVICE_CPU && c::fixed(&d.provider_id) == t.provider_id());
    let Some((_, cpu_info)) = cpu else {
        println!("device_explicit_cpu_loads_and_runs: no CPU device on this provider");
        return;
    };
    let provider = c::fixed(&cpu_info.provider_id);
    let sel = turbo_device_selector {
        policy: TURBO_SELECT_EXPLICIT,
        provider_id: c::text(&provider),
        ordinal: cpu_info.ordinal,
        ..selector()
    };
    let mut index = u32::MAX;
    // SAFETY: valid runtime handle; `provider` outlives the call.
    assert_rc!(unsafe { turbo_runtime_select_device(ct.rt, &sel, &mut index, &mut e) }, TURBO_OK, e);
    assert_eq!(device_info(ct.rt, index).kind, TURBO_DEVICE_CPU);

    let path = ct.bundle(BundleKind::Embedding);
    let mut ctx: *mut turbo_context = ptr::null_mut();
    let mut model: *mut turbo_model = ptr::null_mut();
    let mut session: *mut turbo_session = ptr::null_mut();
    let mut result: *mut turbo_result = ptr::null_mut();
    let texts = [c::text("hello world")];
    // SAFETY: every handle below is checked before use and released once.
    unsafe {
        assert_rc!(turbo_context_create(ct.rt, index, ptr::null(), &mut ctx, &mut e), TURBO_OK, e);
        assert_rc!(turbo_model_load(ctx, c::text(&path), ptr::null(), &mut model, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_create(model, ptr::null(), &mut session, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_write_text(session, texts.as_ptr(), 1, ptr::null(), &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_run(session, ptr::null(), &mut result, &mut e), TURBO_OK, e);
        let ri = c::result_info(result);
        assert!(ri.dim > 0 && ri.batch == 1);
        let v = c::read_f32(result, 0, ri.dim as usize);
        assert!(v.iter().all(|x| x.is_finite()));
        turbo_result_release(result);
        turbo_session_release(session);
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
}

#[test]
fn device_info_strings_are_nul_terminated() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut count = 0u32;
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(unsafe { turbo_runtime_device_count(ct.rt, &mut count, &mut e) }, TURBO_OK, e);
    for i in 0..count {
        let di = device_info(ct.rt, i);
        for (what, field) in [
            ("name", &di.name[..]),
            ("vendor", &di.vendor[..]),
            ("provider_id", &di.provider_id[..]),
            ("provider_version", &di.provider_version[..]),
            ("runtime_version", &di.runtime_version[..]),
            ("driver_version", &di.driver_version[..]),
        ] {
            assert!(c::is_nul_terminated(field), "device {i} {what} is not NUL-terminated");
            let s = c::fixed(field);
            assert!(s.is_ascii() || std::str::from_utf8(s.as_bytes()).is_ok(), "device {i} {what} is not UTF-8");
        }
        assert!(!c::fixed(&di.name).is_empty(), "device {i} has an empty name");
        assert!(!c::fixed(&di.provider_id).is_empty(), "device {i} has an empty provider id");
        assert!(di.kind >= TURBO_DEVICE_CPU && di.kind <= TURBO_DEVICE_ACCEL, "device {i} kind {}", di.kind);
    }
    // An index past the end is a device error, not a read of uninitialized memory.
    let mut di = turbo_device_info { struct_size: ssz::<turbo_device_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(unsafe { turbo_runtime_device_info(ct.rt, count, &mut di, &mut e) }, TURBO_E_DEVICE_NOT_FOUND, e);
}

#[test]
fn device_context_on_a_missing_device_is_device_not_found() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut count = 0u32;
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(unsafe { turbo_runtime_device_count(ct.rt, &mut count, &mut e) }, TURBO_OK, e);
    let mut ctx: *mut turbo_context = ptr::null_mut();
    // SAFETY: as above; the index is deliberately out of range.
    assert_rc!(
        unsafe { turbo_context_create(ct.rt, count, ptr::null(), &mut ctx, &mut e) },
        TURBO_E_DEVICE_NOT_FOUND,
        e
    );
    assert!(ctx.is_null());
    // A context reports the device it was created on.
    let good = ct.context();
    let mut index = u32::MAX;
    // SAFETY: valid context handle and out pointer.
    assert_rc!(unsafe { turbo_context_device(good, &mut index, &mut e) }, TURBO_OK, e);
    assert_eq!(index, ct.device);
    // SAFETY: released once.
    unsafe { turbo_context_release(good) };
}

#[test]
fn device_context_descriptor_is_validated() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut ctx: *mut turbo_context = ptr::null_mut();
    let base = turbo_context_desc {
        struct_size: ssz::<turbo_context_desc>(),
        flags: 0,
        n_options: 0,
        reserved: 0,
        options: ptr::null(),
        next: ptr::null(),
    };
    // SAFETY: valid runtime handle; each descriptor is fully initialized.
    unsafe {
        let flags = turbo_context_desc { flags: 1, ..base };
        assert_rc!(
            turbo_context_create(ct.rt, ct.device, &flags, &mut ctx, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e,
            field = 2
        );
        let big = turbo_context_desc { struct_size: ssz::<turbo_context_desc>() + 4, ..base };
        assert_rc!(turbo_context_create(ct.rt, ct.device, &big, &mut ctx, &mut e), TURBO_E_INVALID_STRUCT_SIZE, e);
        // The extension chain is declared but not implemented, and says so.
        let chained = turbo_context_desc { next: &base as *const _ as *const std::ffi::c_void, ..base };
        assert_rc!(turbo_context_create(ct.rt, ct.device, &chained, &mut ctx, &mut e), TURBO_E_NOT_IMPLEMENTED, e);
        // A well-formed descriptor works.
        assert_rc!(turbo_context_create(ct.rt, ct.device, &base, &mut ctx, &mut e), TURBO_OK, e);
        turbo_context_release(ctx);
    }
}

#[test]
fn device_runtime_descriptor_is_validated() {
    let mut e = c::err();
    let mut rt: *mut turbo_runtime = ptr::null_mut();
    let base = turbo_runtime_desc {
        struct_size: ssz::<turbo_runtime_desc>(),
        flags: 0,
        n_provider_paths: 0,
        reserved: 0,
        provider_paths: ptr::null(),
        log: None,
        log_user_data: ptr::null_mut(),
    };
    // SAFETY: each descriptor is fully initialized; out pointers are valid.
    unsafe {
        let unknown_flag = turbo_runtime_desc { flags: 1 << 20, ..base };
        assert_rc!(turbo_runtime_create(&unknown_flag, &mut rt, &mut e), TURBO_E_INVALID_ARGUMENT, e, field = 2);
        let reserved = turbo_runtime_desc { reserved: 1, ..base };
        assert_rc!(turbo_runtime_create(&reserved, &mut rt, &mut e), TURBO_E_INVALID_ARGUMENT, e, field = 4);
        let big = turbo_runtime_desc { struct_size: ssz::<turbo_runtime_desc>() + 8, ..base };
        assert_rc!(turbo_runtime_create(&big, &mut rt, &mut e), TURBO_E_INVALID_STRUCT_SIZE, e);
        assert_rc!(turbo_runtime_create(&base, &mut rt, &mut e), TURBO_OK, e);
        turbo_runtime_release(rt);
    }
}
