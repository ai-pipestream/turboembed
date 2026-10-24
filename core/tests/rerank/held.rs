//! A rerank run's scores as a held result and a buffer of it, through the
//! C API: the stages say where each part ran, the buffer's host pointer is
//! what it exports, and the scores stay while the buffer holds them.

use std::ptr;

use crate::common::{Loaded, Session, failure, new_error, session_desc, text};
use turbo::*;

const PAIRS: [(&str, &str); 2] = [("what is it", "a short passage"), ("who", "another passage about someone")];

fn write(s: &Session) -> i32 {
    let pairs: Vec<turbo_text_pair> =
        PAIRS.iter().map(|&(q, p)| turbo_text_pair { query: text(q), passage: text(p) }).collect();
    let mut err = new_error();
    unsafe { turbo_rerank_write_pairs(s.0, pairs.as_ptr(), pairs.len() as u32, ptr::null(), &mut err) }
}

/// Runs the pairs on `loaded` and checks the result: the encoder's stage
/// is `encode`, the head and the window reduction the host's; the buffer
/// exports its host pointer as TURBO_HANDLE_HOST_PTR and refuses
/// `refused`; the scores read the same through the result, the pointer and
/// after the result is released; the session is busy until the buffer is.
pub fn held_scores(loaded: &Loaded, encode: u32, refused: u32) {
    let s = Session::create(loaded.m, Some(&session_desc(2, 32, TURBO_PRECISION_EXACT))).unwrap();
    assert_eq!(write(&s), 0);
    let mut err = new_error();
    let mut r = ptr::null_mut();
    assert_eq!(unsafe { turbo_session_run(s.0, &mut r, &mut err) }, 0, "{:?}", failure(-1, &err));
    let mut info: turbo_result_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_result_info>() as u32;
    assert_eq!(unsafe { turbo_result_get_info(r, &mut info, &mut err) }, 0);
    let mut want = [TURBO_STAGE_UNUSED; TURBO_STAGE_MAX];
    want[..TURBO_RERANK_STAGE_COUNT].fill(TURBO_STAGE_HOST);
    want[TURBO_RERANK_STAGE_ENCODE] = encode;
    assert_eq!(info.stage, want);
    assert_eq!(info.placement, TURBO_PLACE_HOST);

    let mut read = [f32::NAN; PAIRS.len()];
    let (bytes, mut written) = (size_of_val(&read) as u64, 0);
    let rc = unsafe { turbo_result_read(r, read.as_mut_ptr().cast(), bytes, &mut written, &mut err) };
    assert_eq!((rc, written), (0, bytes));
    assert!(read.iter().all(|v| v.is_finite()), "{read:?}");

    let mut buf = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r, &mut buf, &mut err) }, 0, "{:?}", failure(-1, &err));
    let mut host = ptr::null_mut();
    assert_eq!(unsafe { turbo_buffer_host_ptr(buf, &mut host, &mut err) }, 0);
    let mut h: turbo_native_handle = unsafe { std::mem::zeroed() };
    h.struct_size = size_of::<turbo_native_handle>() as u32;
    assert_eq!(unsafe { turbo_buffer_export(buf, TURBO_HANDLE_HOST_PTR, &mut h, &mut err) }, 0);
    assert_eq!((h.kind, h.handle, h.offset), (TURBO_HANDLE_HOST_PTR, host as usize as u64, 0));
    let rc = unsafe { turbo_buffer_export(buf, refused, &mut h, &mut err) };
    assert_eq!(rc, turbo::status::UNSUPPORTED, "kind {refused}");
    let at = |p: *mut std::ffi::c_void| unsafe { std::slice::from_raw_parts(p.cast::<f32>(), PAIRS.len()).to_vec() };
    assert_eq!(at(host), read);

    unsafe { turbo_result_release(r) };
    assert_eq!(at(host), read, "the buffer holds the scores after the result is released");
    assert_eq!(write(&s), turbo::status::BUSY, "the buffer still holds the session");
    unsafe { turbo_buffer_release(buf) };
    assert_eq!(write(&s), 0);
}
