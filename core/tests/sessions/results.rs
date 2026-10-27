//! Results: what a run reports, the buffer as the vectors where they are,
//! and the refusals of the result calls.

use std::ffi::c_void;
use std::ptr;

use crate::common::*;
use crate::support::*;
use turbo::status::*;
use turbo::*;

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
