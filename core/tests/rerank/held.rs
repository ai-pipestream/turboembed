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
// metal_rerank.rs, whose scores are not host memory, calls held_scores_in alone.
#[allow(dead_code)]
pub fn held_scores(loaded: &Loaded, encode: u32, refused: u32) {
    held_scores_in(loaded, encode, refused, TURBO_PLACE_HOST);
}

/// As held_scores, with the scores in `placement`: the session's own
/// host memory, or a backend's output buffer the core writes them into,
/// as Metal's shared one.
pub fn held_scores_in(loaded: &Loaded, encode: u32, refused: u32, placement: u32) {
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
    assert_eq!(info.placement, placement);

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

fn run_info(s: &Session, pairs: &[(&str, &str)], max_tokens: u32) -> turbo_result_info {
    let views: Vec<turbo_text_pair> =
        pairs.iter().map(|&(q, p)| turbo_text_pair { query: text(q), passage: text(p) }).collect();
    let o = turbo_rerank_options {
        struct_size: size_of::<turbo_rerank_options>() as u32,
        max_tokens,
        ..Default::default()
    };
    let mut err = new_error();
    let rc = unsafe { turbo_rerank_write_pairs(s.0, views.as_ptr(), views.len() as u32, &o, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    let mut r = ptr::null_mut();
    assert_eq!(unsafe { turbo_session_run(s.0, &mut r, &mut err) }, 0, "{:?}", failure(-1, &err));
    let mut info: turbo_result_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_result_info>() as u32;
    assert_eq!(unsafe { turbo_result_get_info(r, &mut info, &mut err) }, 0);
    unsafe { turbo_result_release(r) };
    info
}

/// Windows past max_batch take more than one encoder run, and the result
/// counts the uploads and read-backs of every one: two full batches of
/// full windows move twice the bytes of one. On a `device`, one run moves
/// some bytes each way.
pub fn transfers_add_up_over_window_batches(loaded: &Loaded, dir: &std::path::Path, device: bool) {
    const MAX: u32 = 16;
    let bundle = turbo::bundle::Bundle::open(dir).unwrap();
    let tok = turbo::tokenizer::Tokenizer::load(&bundle).unwrap();
    let windows = |p: &str| tok.encode_pair("q", p, 0, MAX, 0).unwrap();
    // The longest run of one word that still makes one full window.
    let passage = |n: usize| vec!["word"; n].join(" ");
    let n = (1..64).find(|&n| windows(&passage(n + 1)).len() > 1).unwrap();
    let (one, two) = (passage(n), passage(2 * n));
    assert!(windows(&one).iter().all(|w| w.ids.len() == MAX as usize) && windows(&one).len() == 1);
    assert!(windows(&two).iter().all(|w| w.ids.len() == MAX as usize) && windows(&two).len() == 2);
    let s = Session::create(loaded.m, Some(&session_desc(2, MAX, TURBO_PRECISION_EXACT))).unwrap();
    // Two windows: one run. Four windows: two runs of two.
    let a = run_info(&s, &[("q", &one), ("q", &one)], MAX);
    let b = run_info(&s, &[("q", &two), ("q", &two)], MAX);
    if device {
        assert!(a.h2d_bytes > 0 && a.d2h_bytes > 0, "one run's uploads {} and read-backs {}", a.h2d_bytes, a.d2h_bytes);
    }
    assert_eq!(b.h2d_bytes, 2 * a.h2d_bytes, "uploads of both runs");
    assert_eq!(b.d2h_bytes, 2 * a.d2h_bytes, "read-backs of both runs");
}
