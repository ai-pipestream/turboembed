//! Sessions, writes, runs and results through the C interface, on the CPU
//! backend this build links: what each call resolves and reports, every
//! refusal, the session's state, and the encoder's options checked against
//! the same arithmetic written plainly in f64. tests/conformance.rs holds
//! the vectors to the upstream reference; tests/allocations.rs holds the
//! run path to its allocation count.

mod common;

use std::ffi::c_void;
use std::ptr;

use common::*;
use serde_json::json;
use turbo::status::*;
use turbo::*;

fn tiny() -> Loaded {
    Loaded::load(&tiny_bundle()).unwrap_or_else(|e| panic!("{e:?}"))
}

fn session(l: &Loaded) -> Session {
    Session::create(l.m, None).unwrap_or_else(|e| panic!("{e:?}"))
}

fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

fn null_err() -> *mut turbo_error {
    ptr::null_mut()
}

const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];

// ---- Sessions --------------------------------------------------------------------

#[test]
fn a_session_reports_what_it_resolved() {
    let l = tiny();
    let info = session(&l).info();
    assert_eq!(info.struct_size as usize, size_of::<turbo_session_info>());
    assert_eq!((info.max_batch, info.max_seq), (64, 64), "0 is the model's");
    assert_eq!((info.precision, info.compute_dtype), (TURBO_PRECISION_MODEL, TURBO_DTYPE_F32));
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let info = Session::create(l.m, Some(&session_desc(8, 16, p))).unwrap().info();
        assert_eq!((info.max_batch, info.max_seq, info.precision), (8, 16, p));
        assert_eq!(info.compute_dtype, TURBO_DTYPE_F32, "the cpu computes in F32 at every precision");
    }
}

#[test]
fn a_session_larger_than_the_model_is_refused_by_field() {
    let l = tiny();
    let e = Session::create(l.m, Some(&session_desc(65, 0, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 1), "{e:?}");
    assert!(e.message.contains("max_batch 65 is over the model's 64"), "{e:?}");
    let e = Session::create(l.m, Some(&session_desc(0, 65, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 2), "{e:?}");
    let e = Session::create(l.m, Some(&session_desc(0, 0, 3))).err().unwrap();
    assert_eq!(e.code, INVALID_ENUM, "{e:?}");
    let mut d = session_desc(0, 0, 0);
    d.struct_size -= 4;
    assert_eq!(Session::create(l.m, Some(&d)).err().unwrap().code, INVALID_STRUCT_SIZE);
}

/// A backend with one path reports no kernel choices, whatever the
/// tuning asked; the structs' sizes from before tuning are still taken,
/// the fields past them read as 0 and never written; and a tuning mode
/// that is not one is refused.
#[test]
fn a_session_of_one_path_reports_no_choices_and_old_sizes_are_known() {
    let l = tiny();
    for tuning in [TURBO_AUTOTUNE_RUNTIME, TURBO_AUTOTUNE_OFF, TURBO_AUTOTUNE_ON, TURBO_AUTOTUNE_RETUNE] {
        if tuning == TURBO_AUTOTUNE_RUNTIME && std::env::var_os("TURBO_AUTOTUNE").is_some() {
            continue;
        }
        let mut d = session_desc(8, 16, TURBO_PRECISION_EXACT);
        d.tuning = tuning;
        d.tuning_budget_ms = 20;
        let info = Session::create(l.m, Some(&d)).unwrap().info();
        assert_eq!((info.tuned, info.tune_ms, info.choices[0]), (TURBO_TUNED_DEFAULT, 0, 0), "tuning {tuning}");
    }
    let mut d = session_desc(8, 16, 0);
    d.tuning = 4;
    let e = Session::create(l.m, Some(&d)).err().unwrap();
    assert_eq!(e.code, INVALID_ENUM, "{e:?}");

    // turbo_session_desc of 16 bytes: tuning is RUNTIME.
    let d = session_desc(8, 16, 0);
    let old: [u32; 4] = [TURBO_SESSION_DESC_SIZE_V1 as u32, d.max_batch, d.max_seq, d.precision];
    assert_eq!(TURBO_SESSION_DESC_SIZE_V1, 16);
    let mut out = ptr::null_mut();
    let rc = unsafe { turbo_session_create(l.m, old.as_ptr().cast(), &mut out, null_err()) };
    assert_eq!(rc, 0);
    let s = Session(out);
    // turbo_session_info of 24 bytes: nothing past them is written.
    assert_eq!(TURBO_SESSION_INFO_SIZE_V1, 24);
    let mut buf = [0xa5u8; 64];
    buf[..4].copy_from_slice(&(TURBO_SESSION_INFO_SIZE_V1 as u32).to_ne_bytes());
    assert_eq!(unsafe { turbo_session_get_info(s.0, buf.as_mut_ptr().cast(), null_err()) }, 0);
    assert_eq!(u32::from_ne_bytes(buf[4..8].try_into().unwrap()), 8);
    assert_eq!(u32::from_ne_bytes(buf[8..12].try_into().unwrap()), 16);
    assert!(buf[24..].iter().all(|&b| b == 0xa5), "past struct_size is the caller's");
}

#[test]
fn session_calls_refuse_null_and_wrong_handles() {
    let l = tiny();
    let s = session(&l);
    let mut out = ptr::null_mut();
    let d = session_desc(0, 0, 0);
    let mut info: turbo_session_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_session_info>() as u32;
    unsafe {
        assert_eq!(turbo_session_create(ptr::null_mut(), &d, &mut out, null_err()), INVALID_HANDLE);
        assert_eq!(turbo_session_create(l.rt as *mut turbo_model, &d, &mut out, null_err()), INVALID_HANDLE);
        assert_eq!(turbo_session_create(l.m, &d, ptr::null_mut(), null_err()), INVALID_ARGUMENT);
        assert_eq!(turbo_session_get_info(ptr::null_mut(), &mut info, null_err()), INVALID_HANDLE);
        assert_eq!(turbo_session_get_info(s.0, ptr::null_mut(), null_err()), INVALID_ARGUMENT);
        info.struct_size = 8;
        assert_eq!(turbo_session_get_info(s.0, &mut info, null_err()), INVALID_STRUCT_SIZE);
        let t = text("x");
        assert_eq!(turbo_embed_write_text(ptr::null_mut(), &t, 1, ptr::null(), null_err()), INVALID_HANDLE);
        let b = Tokens::new(&[vec![101, 102]], 0);
        assert_eq!(turbo_embed_write_tokens(ptr::null_mut(), &b.batch(), ptr::null(), null_err()), INVALID_HANDLE);
        let mut r = ptr::null_mut();
        assert_eq!(turbo_session_run(ptr::null_mut(), &mut r, null_err()), INVALID_HANDLE);
        assert_eq!(turbo_session_run(l.m as *mut turbo_session, &mut r, null_err()), INVALID_HANDLE);
        s.write_text(&["x"], None).unwrap();
        assert_eq!(turbo_session_run(s.0, ptr::null_mut(), null_err()), INVALID_ARGUMENT);
        turbo_session_release(ptr::null_mut());
    }
}

#[test]
fn a_session_and_its_result_outlive_the_handles_they_were_made_from() {
    let (rt, ctx, m) = tiny().into_raw();
    let s = Session::create(m, None).unwrap();
    unsafe {
        turbo_model_release(m);
        turbo_context_release(ctx);
        turbo_runtime_release(rt);
    }
    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    drop(s);
    let rows = r.rows();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|v| v.iter().all(|x| x.is_finite())));
}

#[test]
fn sessions_on_one_model_run_from_many_threads() {
    let l = tiny();
    let want = session(&l).embed(&TEXTS, None).unwrap();
    let m = l.m as usize;
    std::thread::scope(|sc| {
        for _ in 0..4 {
            let want = &want;
            sc.spawn(move || {
                let s = Session::create(m as *mut turbo_model, None).unwrap();
                for _ in 0..3 {
                    assert_eq!(&s.embed(&TEXTS, None).unwrap(), want);
                }
            });
        }
    });
}

/// Two threads on one session at once, which turbo.h gives one owner at a
/// time: a call that finds another inside the session is TURBO_E_BUSY,
/// and every run that succeeds is one write's rows, whole.
#[test]
fn one_session_used_from_two_threads_is_busy_not_corrupt() {
    let l = tiny();
    let s = session(&l);
    let a: Vec<String> = (0..64).map(|i| format!("{} {i}", PARAGRAPH)).collect();
    let b: Vec<String> = (0..64).map(|i| format!("reset a password {i}")).collect();
    let views = |t: &[String]| t.iter().map(|x| text(x)).collect::<Vec<_>>();
    let want = [s.embed(&a.iter().map(String::as_str).collect::<Vec<_>>(), None).unwrap(), {
        s.embed(&b.iter().map(String::as_str).collect::<Vec<_>>(), None).unwrap()
    }];
    let sp = s.0 as usize;
    let inside = std::sync::atomic::AtomicU32::new(0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    std::thread::scope(|sc| {
        for t in [&a, &b] {
            let (want, inside) = (&want, &inside);
            sc.spawn(move || {
                let s = sp as *mut turbo_session;
                let v = views(t);
                while inside.load(std::sync::atomic::Ordering::Relaxed) < 3 && std::time::Instant::now() < deadline {
                    let mut err = new_error();
                    let rc = unsafe { turbo_embed_write_text(s, v.as_ptr(), 64, ptr::null(), &mut err) };
                    if rc == BUSY && failure(rc, &err).message.contains("another call is using the session") {
                        inside.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    assert!(rc == 0 || rc == BUSY, "{:?}", failure(rc, &err));
                    let mut r = ptr::null_mut();
                    let rc = unsafe { turbo_session_run(s, &mut r, &mut err) };
                    match rc {
                        0 => {
                            let rows = Outcome(r).rows();
                            assert!(rows == want[0] || rows == want[1], "a run gave rows of neither write");
                        }
                        BUSY => {
                            if failure(rc, &err).message.contains("another call is using the session") {
                                inside.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                        // The other thread's run took the write.
                        INVALID_STATE => {}
                        _ => panic!("{:?}", failure(rc, &err)),
                    }
                }
            });
        }
    });
    assert!(inside.into_inner() >= 3, "no call found another inside the session in a minute");
    // The session is whole after it.
    assert_eq!(s.embed(&b.iter().map(String::as_str).collect::<Vec<_>>(), None).unwrap(), want[1]);
}

// ---- State -----------------------------------------------------------------------

#[test]
fn a_run_takes_one_write() {
    let l = tiny();
    let s = session(&l);
    let e = s.run().err().unwrap();
    assert!(e.is(INVALID_STATE, "nothing is written"), "{e:?}");
    s.write_text(&TEXTS, None).unwrap();
    drop(s.run().unwrap());
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "the write was taken by the run");
}

#[test]
fn a_write_replaces_the_one_before() {
    let l = tiny();
    let s = session(&l);
    let alone = s.embed(&TEXTS[1..2], None).unwrap();
    s.write_text(&TEXTS, None).unwrap();
    s.write_text(&TEXTS[1..2], None).unwrap();
    let r = s.run().unwrap();
    assert_eq!(r.info().batch, 1);
    assert_eq!(r.rows(), alone);
}

#[test]
fn a_failed_write_leaves_nothing_written() {
    let l = tiny();
    let s = session(&l);
    s.write_text(&TEXTS, None).unwrap();
    let bad = opts(|o| o.pooling = 9);
    assert_eq!(s.write_text(&TEXTS, Some(&bad)).err().unwrap().code, INVALID_ENUM);
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE);
    s.write_text(&TEXTS, None).unwrap();
    let b = Tokens::new(&[vec![101, 99999, 102]], 0);
    assert_eq!(s.write_tokens(&b.batch(), None).err().unwrap().code, INVALID_ARGUMENT);
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE);
}

#[test]
fn a_held_result_or_its_buffer_makes_the_session_busy() {
    let l = tiny();
    let s = session(&l);
    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let e = s.write_text(&TEXTS, None).err().unwrap();
    assert!(e.is(BUSY, "result is held"), "{e:?}");
    assert_eq!(s.run().err().unwrap().code, BUSY);
    let b = Tokens::new(&[vec![101, 102]], 0);
    assert_eq!(s.write_tokens(&b.batch(), None).err().unwrap().code, BUSY);
    assert_eq!(s.info().max_batch, 64, "reading the session's info is not a write");

    let mut buf = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, null_err()) }, 0);
    let want = r.rows();
    drop(r);
    assert_eq!(s.write_text(&TEXTS, None).err().unwrap().code, BUSY, "the buffer holds the result");
    let mut host = ptr::null_mut();
    assert_eq!(unsafe { turbo_buffer_host_ptr(buf, &mut host, null_err()) }, 0);
    let held = unsafe { std::slice::from_raw_parts(host as *const f32, 3 * want[0].len()) };
    assert_eq!(held, want.concat().as_slice(), "the vectors are still there");
    unsafe { turbo_buffer_release(buf) };
    s.write_text(&TEXTS, None).unwrap();
    drop(s.run().unwrap());
}

// ---- Results ---------------------------------------------------------------------

#[test]
fn a_result_reports_its_run() {
    let l = tiny();
    let mi = l.info();
    let s = session(&l);
    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let i = r.info();
    assert_eq!(i.struct_size as usize, size_of::<turbo_result_info>());
    assert_eq!((i.task, i.batch, i.dim), (TURBO_TASK_EMBED, 3, 32));
    assert_eq!((i.dtype, i.compute_dtype, i.placement), (TURBO_DTYPE_F32, TURBO_DTYPE_F32, TURBO_PLACE_HOST));
    assert_eq!(i.device, cpu(l.rt));
    assert_eq!(i.bytes, 3 * 32 * 4);
    assert_eq!((i.h2d_bytes, i.d2h_bytes), (0, 0), "nothing crosses a bus on the cpu");
    assert_eq!((i.host_allocs, i.device_allocs), (0, 0));
    assert_eq!(i.stage_count as usize, TURBO_EMBED_STAGE_COUNT);
    let (h, u) = (TURBO_STAGE_HOST, TURBO_STAGE_UNUSED);
    assert_eq!(i.stage[..7], [h, u, h, h, h, h, u], "tokenize, upload, lookup, encode, pool, normalize, download");
    assert!(i.stage[7..].iter().all(|&s| s == u));
    assert_eq!(field(&i.backend), "cpu");
    assert_eq!(field(&i.arch), std::env::consts::ARCH);
    assert_eq!(field(&i.runtime_version), "");
    assert_eq!(field(&i.manifest_sha256), field(&mi.manifest_sha256));
    assert_eq!(field(&i.artifact_sha256), field(&mi.artifact_sha256));
    assert_eq!(field(&i.tokenizer_sha256), field(&mi.tokenizer_sha256));
    // A read is counted in d2h_bytes, as turbo.h says, each time.
    r.rows();
    assert_eq!(r.info().d2h_bytes, i.bytes);
    r.rows();
    assert_eq!(r.info().d2h_bytes, 2 * i.bytes);
    drop(r);

    // Rows the caller tokenized were not tokenized here; a vector left
    // unnormalized was not normalized.
    let b = Tokens::new(&[vec![101, 7592, 102]], 0);
    s.write_tokens(&b.batch(), Some(&opts(|o| o.normalize = TURBO_NORMALIZE_NONE))).unwrap();
    let i = s.run().unwrap().info();
    assert_eq!(i.stage[..7], [u, u, h, h, h, u, u]);
    assert_eq!((i.batch, i.d2h_bytes), (1, 0), "a run's count starts again");
}

#[test]
fn the_result_buffer_is_the_vectors_where_they_are() {
    let l = tiny();
    let s = session(&l);
    s.write_text(&TEXTS, Some(&opts(|o| o.output_dim = 32))).unwrap();
    let r = s.run().unwrap();
    let want = r.rows().concat();
    let mut buf = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, null_err()) }, 0);
    let mut d: turbo_buffer_desc = unsafe { std::mem::zeroed() };
    d.struct_size = size_of::<turbo_buffer_desc>() as u32;
    let mut host = ptr::null_mut();
    let mut h: turbo_native_handle = unsafe { std::mem::zeroed() };
    h.struct_size = size_of::<turbo_native_handle>() as u32;
    unsafe {
        assert_eq!(turbo_buffer_get_desc(buf, &mut d, null_err()), 0);
        assert_eq!(turbo_buffer_host_ptr(buf, &mut host, null_err()), 0);
        assert_eq!(turbo_buffer_export(buf, TURBO_HANDLE_HOST_PTR, &mut h, null_err()), 0);
    }
    assert_eq!((d.placement, d.dtype, d.ndim, d.shape, d.bytes), (TURBO_PLACE_HOST, TURBO_DTYPE_F32, 2, [3, 32], 384));
    assert_eq!((h.handle, h.offset), (host as u64, 0), "exported where it is");
    assert_eq!(host as usize % 64, 0);
    // A second buffer is the same memory; neither is a copy.
    let mut buf2 = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf2, null_err()) }, 0);
    let mut host2 = ptr::null_mut();
    assert_eq!(unsafe { turbo_buffer_host_ptr(buf2, &mut host2, null_err()) }, 0);
    assert_eq!(host, host2);
    drop(r);
    drop(s);
    drop(l);
    // The buffers hold the result, its session and its model.
    let v = unsafe { std::slice::from_raw_parts(host as *const f32, want.len()) };
    assert_eq!(v, want.as_slice());
    unsafe {
        turbo_buffer_release(buf);
        turbo_buffer_release(buf2);
    }
}

#[test]
fn result_calls_refuse_what_they_cannot_take() {
    let l = tiny();
    let s = session(&l);
    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let mut info: turbo_result_info = unsafe { std::mem::zeroed() };
    let mut dst = vec![0f32; 3 * 32];
    let mut written = 7u64;
    let mut buf = ptr::null_mut();
    unsafe {
        info.struct_size = 8;
        assert_eq!(turbo_result_get_info(r.0, &mut info, null_err()), INVALID_STRUCT_SIZE);
        assert_eq!(turbo_result_get_info(r.0, ptr::null_mut(), null_err()), INVALID_ARGUMENT);
        info.struct_size = size_of::<turbo_result_info>() as u32;
        assert_eq!(turbo_result_get_info(ptr::null_mut(), &mut info, null_err()), INVALID_HANDLE);
        assert_eq!(turbo_result_get_info(s.0 as *mut turbo_result, &mut info, null_err()), INVALID_HANDLE);
        let p = dst.as_mut_ptr() as *mut c_void;
        assert_eq!(turbo_result_read(r.0, ptr::null_mut(), 384, &mut written, null_err()), INVALID_ARGUMENT);
        let mut err = new_error();
        assert_eq!(turbo_result_read(r.0, p, 383, &mut written, &mut err), INVALID_ARGUMENT);
        assert!(failure(INVALID_ARGUMENT, &err).message.contains("capacity 383 is under the result's 384 bytes"));
        assert_eq!(written, 7, "a failed read writes nothing");
        assert_eq!(turbo_result_read(r.0, p, 1000, ptr::null_mut(), null_err()), 0, "written may be NULL");
        assert_eq!(turbo_result_buffer(r.0, ptr::null_mut(), null_err()), INVALID_ARGUMENT);
        assert_eq!(turbo_result_buffer(ptr::null_mut(), &mut buf, null_err()), INVALID_HANDLE);
        turbo_result_release(ptr::null_mut());
    }
    // A released result is not a result.
    let raw = r.0;
    drop(r);
    unsafe {
        assert_eq!(turbo_result_get_info(raw, &mut info, null_err()), INVALID_HANDLE);
        let p = dst.as_mut_ptr() as *mut c_void;
        assert_eq!(turbo_result_read(raw, p, 384, &mut written, null_err()), INVALID_HANDLE);
        assert_eq!(turbo_result_buffer(raw, &mut buf, null_err()), INVALID_HANDLE);
        turbo_result_release(raw);
    }
}

// ---- write_text ------------------------------------------------------------------

#[test]
fn write_text_refuses_what_it_cannot_take() {
    let l = tiny();
    let s = Session::create(l.m, Some(&session_desc(2, 16, 0))).unwrap();
    let e = s.write_text(&[], None).err().unwrap();
    assert!(e.is(INVALID_ARGUMENT, "count is 0"), "{e:?}");
    assert_eq!(unsafe { turbo_embed_write_text(s.0, ptr::null(), 1, ptr::null(), null_err()) }, INVALID_ARGUMENT);
    let e = s.write_text(&TEXTS, None).err().unwrap();
    assert!(e.is(CAPACITY, "3 texts is over the session's max_batch 2"), "{e:?}");
    let bad = [0xffu8, 0xfe];
    let t = turbo_text { ptr: bad.as_ptr() as *const _, len: 2 };
    let mut err = new_error();
    assert_eq!(unsafe { turbo_embed_write_text(s.0, &t, 1, ptr::null(), &mut err) }, INVALID_UTF8);
    assert!(failure(INVALID_UTF8, &err).message.contains("texts[0]"));
    // Cut at the bundle's max_seq, 64, as on every device; then longer
    // than this session's 16.
    let long = PARAGRAPH;
    let e = s.write_text(&["short", long], None).err().unwrap();
    assert!(e.is(CAPACITY, "texts[1]: 64 tokens is over the session's max_seq 16"), "{e:?}");
    // Asked to fit, it fits.
    s.write_text(&[long], Some(&opts(|o| o.max_tokens = 16))).unwrap();
    let e = s.write_text(&[long], Some(&opts(|o| o.max_tokens = 17))).err().unwrap();
    assert!(e.is(CAPACITY, "max_tokens 17 is over the session's max_seq 16"), "{e:?}");
    let e = s.write_text(&[long], Some(&opts(|o| o.truncate = TURBO_TRUNCATE_NONE))).err().unwrap();
    assert!(e.is(CAPACITY, "TRUNCATE_NONE"), "{e:?}");
    let e = s.write_text(&["a"], Some(&opts(|o| o.max_tokens = 1))).err().unwrap();
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 2), "{e:?}");
}

#[test]
fn every_option_is_checked_by_name() {
    let l = tiny();
    let s = session(&l);
    let refused = |o: turbo_embed_options| s.write_text(&["a"], Some(&o)).err().unwrap();
    for (o, name) in [
        (opts(|o| o.truncate = 4), "truncate: 4"),
        (opts(|o| o.prompt_role = 3), "prompt_role: 3"),
        (opts(|o| o.normalize = 3), "normalize: 3"),
        (opts(|o| o.pooling = 4), "pooling: 4"),
    ] {
        let e = refused(o);
        assert!(e.is(INVALID_ENUM, name), "{e:?}");
    }
    let e = refused(opts(|o| o.output_dim = 33));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 6), "{e:?}");
    assert!(e.message.contains("output_dim 33 is over the model's dim 32"), "{e:?}");
    let e = refused(opts(|o| o.output_dim = 16));
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
    let mut o = embed_options();
    o.struct_size += 4;
    assert_eq!(refused(o).code, INVALID_STRUCT_SIZE);
    // The model's own width is always there.
    assert_eq!(s.embed(&["a"], Some(&opts(|o| o.output_dim = 32))).unwrap()[0].len(), 32);
}

#[test]
fn prompt_roles_prefix_the_text_as_the_bundle_says() {
    let l = tiny();
    assert_eq!(field(&l.info().prefix_query), "query: ");
    let s = session(&l);
    let q = s.embed(&["reset"], Some(&opts(|o| o.prompt_role = TURBO_PROMPT_QUERY))).unwrap();
    assert_eq!(q, s.embed(&["query: reset"], None).unwrap());
    assert_ne!(q, s.embed(&["reset"], None).unwrap());
    // The bundle has no document prefix.
    let d = s.embed(&["reset"], Some(&opts(|o| o.prompt_role = TURBO_PROMPT_DOCUMENT))).unwrap();
    assert_eq!(d, s.embed(&["reset"], None).unwrap());
}

// ---- write_tokens ----------------------------------------------------------------

#[test]
fn write_tokens_refuses_what_it_cannot_take() {
    let l = tiny();
    let s = Session::create(l.m, Some(&session_desc(2, 8, 0))).unwrap();
    let ok = Tokens::new(&[vec![101, 7592, 102], vec![101, 102]], 0);
    s.write_tokens(&ok.batch(), None).unwrap();
    let fails = |b: &turbo_token_batch, o: Option<&turbo_embed_options>| s.write_tokens(b, o).err().unwrap();

    assert_eq!(unsafe { turbo_embed_write_tokens(s.0, ptr::null(), ptr::null(), null_err()) }, INVALID_ARGUMENT);
    let mut b = ok.batch();
    b.struct_size = 8;
    assert_eq!(fails(&b, None).code, INVALID_STRUCT_SIZE);
    for (edit, code, what) in [
        ((|b: &mut turbo_token_batch| b.batch = 0) as fn(&mut turbo_token_batch), INVALID_SHAPE, "neither may be 0"),
        (|b| b.seq = 0, INVALID_SHAPE, "neither may be 0"),
        (|b| b.batch = 3, CAPACITY, "batch 3 is over the session's max_batch 2"),
        (|b| b.seq = 9, CAPACITY, "seq 9 is over the session's max_seq 8"),
        (|b| b.row_stride = 2, INVALID_SHAPE, "row_stride 2 is under seq 3"),
        (|b| b.ids = ptr::null(), INVALID_ARGUMENT, "may not be NULL"),
        (|b| b.mask = ptr::null(), INVALID_ARGUMENT, "may not be NULL"),
    ] {
        let mut b = ok.batch();
        edit(&mut b);
        let e = fails(&b, None);
        assert!(e.is(code, what), "{what}: {e:?}");
    }

    let with = |edit: fn(&mut Tokens)| {
        let mut t = Tokens::new(&[vec![101, 7592, 102], vec![101, 102]], 0);
        edit(&mut t);
        t
    };
    for (t, what) in [
        (with(|t| t.ids[1] = -1), "ids[0][1] is -1"),
        (with(|t| t.ids[4] = 30522), "ids[1][1] is 30522, outside the model's vocabulary of 30522"),
        (with(|t| t.mask[2] = 2), "mask[0][2] is 2"),
        (with(|t| t.mask[3..].fill(0)), "mask row 1 has no 1"),
        (with(|t| t.types = Some(vec![0, 2, 0, 0, 0, 0])), "types[0][1] is 2, and the model has 2 token types"),
    ] {
        let e = fails(&t.batch(), None);
        assert!(e.is(INVALID_ARGUMENT, what), "{what}: {e:?}");
    }
    let e = fails(&ok.batch(), Some(&opts(|o| o.truncate = TURBO_TRUNCATE_RIGHT)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 1), "{e:?}");
    let e = fails(&ok.batch(), Some(&opts(|o| o.prompt_role = TURBO_PROMPT_QUERY)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 3), "{e:?}");
    let e = fails(&ok.batch(), Some(&opts(|o| o.max_tokens = 2)));
    assert!(e.is(CAPACITY, "row 0: 3 tokens through its last live one is over max_tokens 2"), "{e:?}");
    let e = fails(&ok.batch(), Some(&opts(|o| o.max_tokens = 9)));
    assert!(e.is(CAPACITY, "max_tokens 9 is over the session's max_seq 8"), "{e:?}");
    s.write_tokens(&ok.batch(), Some(&opts(|o| o.max_tokens = 3))).unwrap();
}

#[test]
fn tokens_and_text_give_the_same_vectors() {
    let l = tiny();
    let s = session(&l);
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let from_text = s.embed(&TEXTS, None).unwrap();
    let t = Tokens::new(&rows, 0);
    s.write_tokens(&t.batch(), None).unwrap();
    assert_eq!(s.run().unwrap().rows(), from_text);

    // The same rows at a wider stride, and with explicit zero types.
    let seq = t.seq as usize;
    let mut wide = Tokens::new(&rows, 0);
    wide.stride = seq as u32 + 5;
    wide.ids = vec![-9; rows.len() * wide.stride as usize];
    wide.mask = vec![-9; rows.len() * wide.stride as usize];
    for r in 0..rows.len() {
        let at = r * wide.stride as usize;
        wide.ids[at..at + seq].copy_from_slice(&t.ids[r * seq..(r + 1) * seq]);
        wide.mask[at..at + seq].copy_from_slice(&t.mask[r * seq..(r + 1) * seq]);
    }
    wide.types = Some(vec![0; wide.ids.len()]);
    s.write_tokens(&wide.batch(), None).unwrap();
    assert_eq!(s.run().unwrap().rows(), from_text, "the stride's gap is never read");
}

// ---- The encoder against plain arithmetic -----------------------------------------

/// Every pooling, with and without normalization, cut or not, with token
/// types and with padding on either side, against PlainBert in f64.
#[test]
fn every_option_matches_the_arithmetic_written_plainly() {
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let l = tiny();
    let s = session(&l);
    let tok = Tok::create(&dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let seq = rows.iter().map(Vec::len).max().unwrap() + 2;
    // Row 0 right-padded, row 1 left-padded, row 2 with a masked token in
    // the middle and types of 1 after its first token.
    let mut t = Tokens::new(&[vec![0; seq], vec![0; seq], vec![0; seq]], 0);
    t.mask.fill(0);
    let mut types = vec![0; 3 * seq];
    for (r, row) in rows.iter().enumerate() {
        let start = if r == 1 { seq - row.len() } else { 0 };
        for (p, &id) in row.iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
            if r == 2 && p > 0 {
                types[r * seq + start + p] = 1;
            }
        }
    }
    t.mask[2 * seq + 1] = 0;
    t.types = Some(types.clone());

    let mut worst = 0f64;
    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| {
                o.pooling = pooling;
                o.normalize = normalize;
            });
            s.write_tokens(&t.batch(), Some(&o)).unwrap();
            let got = s.run().unwrap().rows();
            for (r, row) in got.iter().enumerate() {
                let at = r * seq..(r + 1) * seq;
                let (ids, mask, ty) = (&t.ids[at.clone()], &t.mask[at.clone()], &types[at]);
                let want = plain.embed(ids, mask, ty, pooling, 32, normalize == TURBO_NORMALIZE_L2);
                for (g, w) in row.iter().zip(&want) {
                    let d = (*g as f64 - w).abs();
                    worst = worst.max(d);
                    assert!(d < 1e-5 * (1.0 + w.abs()), "pooling {pooling} normalize {normalize} row {r}: {g} vs {w}");
                }
            }
        }
    }
    println!("largest difference from the f64 encoder: {worst:.3e}");
}

#[test]
fn the_bundles_pooling_and_normalization_are_the_default() {
    let l = tiny();
    let mi = l.info();
    assert_eq!((mi.pooling, mi.normalize), (TURBO_POOLING_MEAN, TURBO_NORMALIZE_L2));
    let s = session(&l);
    let explicit = opts(|o| {
        o.pooling = TURBO_POOLING_MEAN;
        o.normalize = TURBO_NORMALIZE_L2;
    });
    assert_eq!(s.embed(&TEXTS, None).unwrap(), s.embed(&TEXTS, Some(&explicit)).unwrap());
    let raw = s.embed(&TEXTS, Some(&opts(|o| o.normalize = TURBO_NORMALIZE_NONE))).unwrap();
    for v in s.embed(&TEXTS, None).unwrap() {
        let n: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "{n}");
    }
    assert!(raw.iter().any(|v| (v.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() > 1e-3));
}

// ---- Output widths and weight dtypes, on the small BERT of tests/common --------------

#[test]
fn an_output_dim_the_bundle_lists_is_cut_then_normalized() {
    let mut f = Fixture::new("output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load().unwrap();
    let s = session(&l);
    let e = s.write_text(&["a"], Some(&opts(|o| o.output_dim = 3))).err().unwrap();
    assert!(e.is(UNSUPPORTED_OPTION, "cut to [4]"), "{e:?}");
    let o = opts(|o| o.output_dim = 4);
    let got = s.embed(&TEXTS, Some(&o)).unwrap();
    let plain = PlainBert::new(&f.dir);
    let tok = Tok::create(&f.dir).unwrap();
    for (t, g) in TEXTS.iter().zip(&got) {
        assert_eq!(g.len(), 4);
        let ids = tok.row(t, None).unwrap();
        let ones = vec![1; ids.len()];
        let want = plain.embed(&ids, &ones, &vec![0; ids.len()], TURBO_POOLING_MEAN, 4, true);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
    let r = {
        s.write_text(&TEXTS, Some(&o)).unwrap();
        s.run().unwrap()
    };
    assert_eq!((r.info().dim, r.info().bytes), (4, 3 * 4 * 4));
}

/// f32 to the nearest half, ties to even; for the small BERT's values,
/// which are normal halves or 0.
fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if x == 0.0 {
        return sign;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exp), "{x} is not a normal half");
    let man = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn from_f16(h: u16) -> f32 {
    let (sign, exp, man) = ((h as u32 & 0x8000) << 16, (h >> 10) & 0x1f, h as u32 & 0x3ff);
    if exp == 0 && man == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13))
}

fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The small BERT stored as `dtype`, and the same values widened to F32.
fn narrowed(dtype: &'static str) -> (Vec<Tensor>, Vec<Tensor>) {
    type Narrow = fn(f32) -> u16;
    type Widen = fn(u16) -> f32;
    let (narrow, wide): (Narrow, Widen) = match dtype {
        "F16" => (to_f16, from_f16),
        _ => (to_bf16, |h| f32::from_bits((h as u32) << 16)),
    };
    tiny_weights(0)
        .into_iter()
        .map(|t| {
            let halves: Vec<u16> =
                t.data.chunks(4).map(|c| narrow(f32::from_le_bytes(c.try_into().unwrap()))).collect();
            let n = Tensor {
                name: t.name.clone(),
                dtype,
                shape: t.shape.clone(),
                data: halves.iter().flat_map(|h| h.to_le_bytes()).collect(),
            };
            let w = Tensor { data: halves.iter().flat_map(|&h| wide(h).to_le_bytes()).collect(), ..t };
            (n, w)
        })
        .unzip()
}

#[test]
fn half_weights_compute_in_f32_from_one_shared_copy() {
    for dtype in ["F16", "BF16"] {
        let (narrow, wide) = narrowed(dtype);
        let mut f = Fixture::model(&format!("half-{dtype}"));
        f.weights("weights/model.safetensors", &narrow);
        let l = f.load().unwrap();
        assert_ne!(l.info().dtype, TURBO_DTYPE_F32);
        let e = Session::create(l.m, None).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{dtype}: {e:?}");
        assert!(e.message.contains("computes in F32 only"), "{e:?}");
        assert!(unsafe { model_converted_weights(l.m) }.is_none(), "a refused session makes no copy");

        let a = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        assert_eq!(a.info().compute_dtype, TURBO_DTYPE_F32);
        let copy = unsafe { model_converted_weights(l.m) }.expect("the first F32 session made the copy");
        let b = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
        assert_eq!(b.info().compute_dtype, TURBO_DTYPE_F32);
        assert_eq!(unsafe { model_converted_weights(l.m) }.unwrap(), copy, "one copy, shared");

        // The same values stored as F32 give the same vectors, bit for bit.
        let mut g = Fixture::model(&format!("half-{dtype}-wide"));
        g.weights("weights/model.safetensors", &wide);
        let lw = g.load().unwrap();
        let want = session(&lw).embed(&TEXTS, None).unwrap();
        assert_eq!(a.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert_eq!(b.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert!(unsafe { model_converted_weights(lw.m) }.is_none(), "F32 weights are read in place");
    }
}

// ---- The capability matrix ------------------------------------------------------------

#[test]
fn the_cpu_offers_embed_as_its_sessions_run_it() {
    let l = tiny();
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(l.rt, cpu(l.rt), TURBO_TASK_EMBED, p, &mut cap, null_err()) }, 0);
        let info = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap().info();
        assert_eq!(cap.dtype, info.compute_dtype, "precision {p}");
        assert_eq!(cap.options_honored, 0b111111, "every field of turbo_embed_options");
    }
}
