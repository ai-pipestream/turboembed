use super::*;

/// The CPU backend's table as a backend built before `field` was
/// appended would have it.
fn cpu_table(struct_size: usize) -> turbo_backend {
    let mut t = unsafe { std::ptr::read(&crate::cpu::BACKEND) };
    t.struct_size = struct_size as u32;
    t
}

#[test]
fn a_table_from_before_the_model_functions_is_known_and_offers_none() {
    let t = cpu_table(std::mem::offset_of!(turbo_backend, model_load));
    check_table(&t).unwrap();
    assert!(offered!(&t, buffer_alloc).is_ok());
    let e = offered!(&t, model_load).err().unwrap();
    assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer model_load"));
    let t = cpu_table(std::mem::offset_of!(turbo_backend, context_create));
    check_table(&t).unwrap();
    assert_eq!(offered!(&t, context_create).err().unwrap().code, UNSUPPORTED);
}

#[test]
fn a_table_from_before_the_session_functions_is_known_and_offers_none() {
    let t = cpu_table(std::mem::offset_of!(turbo_backend, session_create));
    check_table(&t).unwrap();
    assert!(offered!(&t, model_load).is_ok());
    let e = offered!(&t, session_create).err().unwrap();
    assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer session_create"));
    assert_eq!(offered!(&t, session_run).err().unwrap().code, UNSUPPORTED);
}

#[test]
fn a_table_from_before_buffer_read_is_known_and_offers_none() {
    let t = cpu_table(std::mem::offset_of!(turbo_backend, buffer_read));
    check_table(&t).unwrap();
    assert!(offered!(&t, session_run).is_ok());
    let e = offered!(&t, buffer_read).err().unwrap();
    assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer buffer_read"));
}

#[test]
fn a_table_from_before_formats_is_known_and_loads_raw_weights_alone() {
    let t = cpu_table(std::mem::offset_of!(turbo_backend, formats));
    check_table(&t).unwrap();
    assert!(offered!(&t, session_run).is_ok());
    assert_eq!(t.formats(), format_bit(TURBO_FORMAT_SAFETENSORS));
    // Whatever lies past the table's end is not read.
    let mut t = cpu_table(std::mem::offset_of!(turbo_backend, formats));
    t.formats = format_bit(TURBO_FORMAT_HEF);
    assert_eq!(t.formats(), format_bit(TURBO_FORMAT_SAFETENSORS));
    // A whole table that leaves formats 0 means the same.
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.formats = 0;
    assert_eq!(t.formats(), format_bit(TURBO_FORMAT_SAFETENSORS));
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.formats = format_bit(TURBO_FORMAT_SAFETENSORS) | format_bit(TURBO_FORMAT_HEF);
    assert_eq!(t.formats(), 0b101);
    let t = cpu_table(std::mem::offset_of!(turbo_backend, reserved2));
    assert_eq!(check_table(&t).unwrap_err().code, INTERNAL, "a size inside the formats group is unknown");
}

/// The CPU table with a session_create_tuned, as a backend with
/// tuned sessions has it: the CPU's own session_create under the
/// other signature, never called.
unsafe extern "C" fn no_tuned(
    _: *mut c_void,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: *mut turbo_backend_tuning,
    _: *mut u32,
    _: *mut *mut c_void,
    _: *mut turbo_error,
) -> i32 {
    crate::status::INTERNAL
}

/// A table that ends with embed_write_ranges is known and applies no
/// token head, whatever lies past its end; the whole table applies one
/// when it sets token_head.
#[test]
fn a_table_from_before_token_heads_is_known_and_applies_none() {
    let mut t = cpu_table(std::mem::offset_of!(turbo_backend, token_head));
    t.token_head = 1;
    check_table(&t).unwrap();
    assert!(!t.token_head());
    let mut t = cpu_table(size_of::<turbo_backend>());
    assert!(!t.token_head());
    t.token_head = 1;
    assert!(t.token_head());
    check_table(&t).unwrap();
}

#[test]
fn a_table_from_before_tuned_sessions_is_known_and_offers_none() {
    let t = cpu_table(std::mem::offset_of!(turbo_backend, session_create_tuned));
    assert_eq!(std::mem::offset_of!(turbo_backend, session_create_tuned), 160);
    assert_eq!(std::mem::offset_of!(turbo_backend, session_open), 168);
    assert_eq!(std::mem::offset_of!(turbo_backend, token_head), 184);
    check_table(&t).unwrap();
    assert!(offered!(&t, session_run).is_ok());
    // Whatever lies past the table's end is not read.
    let mut t = cpu_table(std::mem::offset_of!(turbo_backend, session_create_tuned));
    t.session_create_tuned = Some(no_tuned);
    let e = offered!(&t, session_create_tuned).err().unwrap();
    assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer session_create_tuned"));
    // The whole table offers it when it is there.
    let mut t = cpu_table(size_of::<turbo_backend>());
    assert_eq!(offered!(&t, session_create_tuned).err().unwrap().code, UNSUPPORTED);
    t.session_create_tuned = Some(no_tuned);
    assert!(offered!(&t, session_create_tuned).is_ok());
    check_table(&t).unwrap();
}

/// A session_open and an embed_write_ranges for tables that must
/// have them, never called.
unsafe extern "C" fn no_open(
    _: *mut c_void,
    _: *const turbo_backend_session_desc,
    _: *mut turbo_backend_tuning,
    _: *mut u32,
    _: *mut *mut c_void,
    _: *mut turbo_error,
) -> i32 {
    crate::status::INTERNAL
}

unsafe extern "C" fn no_ranges(
    _: *mut c_void,
    _: *const turbo_backend_embed_rows,
    _: *const turbo_token_range,
    _: u32,
    _: *mut turbo_error,
) -> i32 {
    crate::status::INTERNAL
}

#[test]
fn a_table_from_before_session_open_is_known_and_offers_neither() {
    let mut t = cpu_table(std::mem::offset_of!(turbo_backend, session_open));
    assert_eq!(std::mem::offset_of!(turbo_backend, session_open), 168);
    t.session_create_tuned = Some(no_tuned);
    check_table(&t).unwrap();
    assert!(offered!(&t, session_create_tuned).is_ok());
    // Whatever lies past the table's end is not read.
    t.session_open = Some(no_open);
    t.embed_write_ranges = Some(no_ranges);
    check_table(&t).unwrap();
    let e = offered!(&t, session_open).err().unwrap();
    assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer session_open"));
    let e = offered!(&t, embed_write_ranges).err().unwrap();
    assert_eq!(e, Error::new(UNSUPPORTED, "the cpu backend does not offer embed_write_ranges"));
    // The whole table offers each when it is there.
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.session_open = None;
    t.embed_write_ranges = None;
    check_table(&t).unwrap();
    assert_eq!(offered!(&t, session_open).err().unwrap().code, UNSUPPORTED);
    assert_eq!(offered!(&t, embed_write_ranges).err().unwrap().code, UNSUPPORTED);
    t.session_open = Some(no_open);
    assert!(offered!(&t, session_open).is_ok());
    check_table(&t).unwrap();
    t.embed_write_ranges = Some(no_ranges);
    assert!(offered!(&t, embed_write_ranges).is_ok());
    check_table(&t).unwrap();
    let t = cpu_table(std::mem::offset_of!(turbo_backend, embed_write_ranges));
    assert_eq!(check_table(&t).unwrap_err().code, INTERNAL, "a size inside the session_open group is unknown");
}

#[test]
fn ranges_without_session_open_are_refused() {
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.session_open = None;
    t.embed_write_ranges = Some(no_ranges);
    let e = check_table(&t).unwrap_err();
    assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has embed_write_ranges and no session_open"));
}

#[test]
fn session_open_without_its_release_or_its_run_is_refused() {
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.session_create = None;
    t.session_open = Some(no_open);
    t.session_release = None;
    let e = check_table(&t).unwrap_err();
    assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has session_open and no session_release"));
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.session_create = None;
    t.session_open = Some(no_open);
    t.session_run = None;
    let e = check_table(&t).unwrap_err();
    assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has session_open and no session_run"));
}

#[test]
fn a_tuned_session_without_its_release_or_its_run_is_refused() {
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.session_create = None;
    t.session_create_tuned = Some(no_tuned);
    t.session_release = None;
    let e = check_table(&t).unwrap_err();
    assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has session_create_tuned and no session_release"));
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.session_create = None;
    t.session_create_tuned = Some(no_tuned);
    t.session_run = None;
    let e = check_table(&t).unwrap_err();
    assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has session_create_tuned and no session_run"));
}

#[test]
fn a_session_without_its_release_or_its_run_is_refused() {
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.session_release = None;
    let e = check_table(&t).unwrap_err();
    assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has session_create and no session_release"));
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.session_run = None;
    let e = check_table(&t).unwrap_err();
    assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has session_create and no session_run"));
    let t = cpu_table(std::mem::offset_of!(turbo_backend, session_release));
    assert_eq!(check_table(&t).unwrap_err().code, INTERNAL, "a size inside the session group is unknown");
}

#[test]
fn a_table_of_an_unknown_size_or_a_load_without_its_release_is_refused() {
    let t = cpu_table(std::mem::offset_of!(turbo_backend, model_release));
    assert_eq!(check_table(&t).unwrap_err().code, INTERNAL);
    let mut t = cpu_table(size_of::<turbo_backend>());
    t.model_release = None;
    let e = check_table(&t).unwrap_err();
    assert_eq!(e, Error::new(INTERNAL, "cpu backend: its table has model_load and no model_release"));
}

#[test]
fn rerank_runs_on_the_cpu_cuda_and_levelzero_and_on_no_backend_without_a_record() {
    let mut t = cpu_table(size_of::<turbo_backend>());
    assert!(t.runs_rerank());
    for (name, runs) in [(c"cuda", true), (c"metal", false), (c"levelzero", true), (c"hailo", false), (c"npu", false)]
    {
        t.name = name.as_ptr();
        assert_eq!(t.runs_rerank(), runs, "{name:?}");
    }
}
