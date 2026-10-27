//! Sessions: what each call resolves and reports, and from how many
//! threads they can run.

use std::ptr;

use crate::common::*;
use crate::support::*;
use turbo::status::*;
use turbo::*;

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

    // turbo_session_desc of 24 bytes: cpu_threads and max_ranges are 0.
    // The words after it hold values either field would refuse, so a read
    // past struct_size fails the create.
    assert_eq!(TURBO_SESSION_DESC_SIZE_V2, 24);
    let mid: [u32; 8] = [TURBO_SESSION_DESC_SIZE_V2 as u32, 8, 16, 0, TURBO_AUTOTUNE_OFF, 0, u32::MAX, u32::MAX];
    let mut out = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_session_create(l.m, mid.as_ptr().cast(), &mut out, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    let s = Session(out);
    assert_eq!((s.info().max_batch, s.info().max_seq), (8, 16));
    // max_ranges of 0 is max_batch: nine ranges are one too many.
    let t = Tokens::new(&[vec![101, 7592, 102]], 0);
    let e = s.write_ranges(&t.batch(), &[range(0, 0, 3); 9], None).err().unwrap();
    assert!(e.is(CAPACITY, "9 ranges is over the session's max_ranges 8"), "{e:?}");
}

/// max_ranges is 1 to max_batch * max_seq, counted after 0 in either is
/// resolved to the model's; 0 is max_batch.
#[test]
fn max_ranges_over_the_sessions_tokens_is_refused_by_field() {
    let l = tiny();
    let ranges = |max_batch, max_seq, max_ranges| {
        let mut d = session_desc(max_batch, max_seq, 0);
        d.max_ranges = max_ranges;
        Session::create(l.m, Some(&d))
    };
    let e = ranges(2, 8, 17).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 7), "{e:?}");
    assert!(e.message.contains("max_ranges 17 is over max_batch 2 times max_seq 8, 16"), "{e:?}");
    ranges(2, 8, 16).unwrap();
    // The model's 64 by 64.
    let e = ranges(0, 0, 64 * 64 + 1).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 7), "{e:?}");
    ranges(0, 0, 64 * 64).unwrap();
    ranges(0, 0, u32::MAX).err().unwrap();
}

/// cpu_threads is 1 to 1024 on the CPU, and 0 leaves the count to the
/// environment.
#[test]
fn cpu_threads_out_of_range_is_refused_by_field() {
    let l = tiny();
    let threads = |n| {
        let mut d = session_desc(2, 32, 0);
        d.cpu_threads = n;
        Session::create(l.m, Some(&d))
    };
    let e = threads(1025).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
    assert!(e.message.contains("cpu_threads 1025"), "{e:?}");
    let one = threads(1).unwrap().embed(&TEXTS[..2], None).unwrap();
    let three = threads(3).unwrap().embed(&TEXTS[..2], None).unwrap();
    assert_eq!(one, three, "the thread count does not change the bits");
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
