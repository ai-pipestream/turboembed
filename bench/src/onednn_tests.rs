use super::*;
use std::path::Path;
use std::ptr;

use turbo::{TURBO_PRECISION_EXACT, turbo_text};

const SAMPLE_WARNING: &str = "levelzero device 0: oneDNN not used, the backend's own kernels run the linear \
                              layers: levelzero: oneDNN: cannot open libturbo_onednn.so on the library path \
                              (libdnnl.so.3: cannot open shared object file: No such file or directory)";

#[test]
fn prints_one_line_with_logged_reason_when_onednn_cannot_open() {
    let mut out = Vec::new();
    let warnings = vec![
        "levelzero: building the encoder's kernels failed; building without the flag".to_owned(),
        SAMPLE_WARNING.to_owned(),
        SAMPLE_WARNING.to_owned(),
    ];
    write_fallback(&mut out, "levelzero", "", &warnings);
    let text = String::from_utf8(out).unwrap();
    assert_eq!(text.lines().count(), 1);
    assert_eq!(text, format!("turbo-bench: {SAMPLE_WARNING}\n"));
}

#[test]
fn collapses_multiline_logged_reason_onto_one_stderr_line() {
    let mut out = Vec::new();
    let multiline = "levelzero device 1: oneDNN not used, the backend's own kernels run the linear layers: \
                     sycl open failed\nmissing oneAPI runtime\n";
    write_fallback(&mut out, "levelzero", "", &[multiline.to_owned()]);
    let text = String::from_utf8(out).unwrap();
    assert_eq!(text.lines().count(), 1);
    assert_eq!(
        text,
        "turbo-bench: levelzero device 1: oneDNN not used, the backend's own kernels run the linear layers: sycl \
         open failed; missing oneAPI runtime\n"
    );
}

#[test]
fn stays_silent_when_onednn_runs_or_own_was_forced_or_backend_differs() {
    let warnings = vec![SAMPLE_WARNING.to_owned()];
    for (backend, choices, ws) in [
        ("levelzero", "linear=onednn;forced=", warnings.as_slice()),
        ("levelzero", "linear=onednn;forced=linear", warnings.as_slice()),
        ("levelzero", "linear=own;forced=linear", warnings.as_slice()),
        ("levelzero", "linear=own", warnings.as_slice()),
        ("levelzero", "", &[][..]),
        ("levelzero", "", &["levelzero: building without the flag".to_owned()][..]),
        ("cpu", "", warnings.as_slice()),
        ("cuda", "", warnings.as_slice()),
        ("metal", "", warnings.as_slice()),
        ("npu", "", warnings.as_slice()),
    ] {
        let mut out = Vec::new();
        write_fallback(&mut out, backend, choices, ws);
        assert!(out.is_empty(), "expected silence for backend={backend}, choices={choices}: {out:?}");
    }
}

#[test]
fn runtime_log_callback_and_out_of_order_handle_drop_keep_warnings_intact() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let session = {
        let rt = crate::api::Runtime::create().unwrap();
        assert!(rt.warnings().is_empty());
        let cpu = rt.find("cpu").unwrap();
        let ctx = rt.context(cpu).unwrap();
        let tok = rt.tokenizer(&bundle).unwrap();
        assert!(tok.info().unwrap().vocab_size > 0);
        drop(rt);
        let model = ctx.load(&bundle).unwrap();
        drop(ctx);
        let s = model.session(1, 8, TURBO_PRECISION_EXACT).unwrap();
        drop(model);
        s
    };
    assert_eq!(session.info().unwrap().max_batch, 1);
    drop(session);

    let sink: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let user_data = std::sync::Arc::as_ptr(&sink).cast_mut().cast();
    for (level, msg) in [(0u32, "err"), (1, SAMPLE_WARNING), (2, "info"), (3, "debug")] {
        let t = turbo_text { ptr: msg.as_ptr().cast(), len: msg.len() as u64 };
        unsafe { crate::api::collect_warning(user_data, level, t) };
    }
    unsafe { crate::api::collect_warning(user_data, 1, turbo_text { ptr: ptr::null(), len: 0 }) };
    let recorded = sink.lock().unwrap().clone();
    assert_eq!(recorded, vec!["err".to_owned(), SAMPLE_WARNING.to_owned(), String::new()]);
}
