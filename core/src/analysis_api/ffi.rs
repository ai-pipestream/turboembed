//! C ABI entrypoints for `turbo_analyzer` and `turbo_analysis_result`.

use std::ffi::c_char;
use std::sync::Arc;

use super::{
    ANALYSIS_RESULT_MAGIC, ANALYZER_MAGIC, AnalysisResultInner, AnalyzerEngine, analysis_result_handle,
    analyzer_handle, turbo_analysis_result, turbo_analyzer, turbo_analyzer_desc, turbo_span,
};
use crate::ffi::{array_len, call, out_ptr, sized, text};
use crate::status::{Error, INVALID_ARGUMENT, Result};
use crate::{turbo_error, turbo_text};

unsafe fn text_slice<'a>(items: *const turbo_text, count: u32, what: &str) -> Result<Vec<&'a str>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if items.is_null() {
        return Err(Error::new(INVALID_ARGUMENT, format!("{what} is NULL with count {count}")));
    }
    let len = array_len::<turbo_text>(count as u64, what)?;
    let raw = unsafe { std::slice::from_raw_parts(items, len) };
    let mut out = Vec::with_capacity(len);
    for (idx, &t) in raw.iter().enumerate() {
        out.push(unsafe { text(t, &format!("{what}[{idx}]"))? });
    }
    Ok(out)
}

fn box_result(inner: AnalysisResultInner) -> *mut turbo_analysis_result {
    Box::into_raw(Box::new(turbo_analysis_result { magic: ANALYSIS_RESULT_MAGIC, inner }))
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_create(
    desc: *const turbo_analyzer_desc,
    out: *mut *mut turbo_analyzer,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let out = out_ptr(out, "out")?;
            let desc_ref = if desc.is_null() {
                None
            } else {
                let d = &*desc;
                sized(d.struct_size, size_of::<turbo_analyzer_desc>(), "turbo_analyzer_desc")?;
                if d.flags != 0 {
                    return Err(Error::new(INVALID_ARGUMENT, "turbo_analyzer_desc.flags must be 0"));
                }
                Some(d)
            };
            let engine = AnalyzerEngine::from_desc(desc_ref)?;
            *out = Box::into_raw(Box::new(turbo_analyzer { magic: ANALYZER_MAGIC, inner: Arc::new(engine) }));
            Ok(())
        })
    }
}

/// # Safety
/// `a` is NULL or a live handle returned by `turbo_analyzer_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_release(a: *mut turbo_analyzer) {
    if unsafe { analyzer_handle(a) }.is_ok() {
        let mut b = unsafe { Box::from_raw(a) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_detect_sentences(
    a: *mut turbo_analyzer,
    input: turbo_text,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let a = analyzer_handle(a)?;
            let out = out_ptr(out, "out")?;
            let s = text(input, "text")?;
            *out = box_result(a.inner.detect_sentences(s)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_tokenize(
    a: *mut turbo_analyzer,
    input: turbo_text,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let a = analyzer_handle(a)?;
            let out = out_ptr(out, "out")?;
            let s = text(input, "text")?;
            *out = box_result(a.inner.tokenize(s)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_pos_tag(
    a: *mut turbo_analyzer,
    tokens: *const turbo_text,
    count: u32,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let a = analyzer_handle(a)?;
            let out = out_ptr(out, "out")?;
            let toks = text_slice(tokens, count, "tokens")?;
            *out = box_result(a.inner.pos_tag(&toks)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_lemmatize(
    a: *mut turbo_analyzer,
    tokens: *const turbo_text,
    tags: *const turbo_text,
    count: u32,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let a = analyzer_handle(a)?;
            let out = out_ptr(out, "out")?;
            let toks = text_slice(tokens, count, "tokens")?;
            let pos = text_slice(tags, count, "tags")?;
            *out = box_result(a.inner.lemmatize(&toks, &pos)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_find_tokens(
    a: *mut turbo_analyzer,
    tokens: *const turbo_text,
    count: u32,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let a = analyzer_handle(a)?;
            let out = out_ptr(out, "out")?;
            let toks = text_slice(tokens, count, "tokens")?;
            *out = box_result(a.inner.find_tokens(&toks)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_find_text(
    a: *mut turbo_analyzer,
    input: turbo_text,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let a = analyzer_handle(a)?;
            let out = out_ptr(out, "out")?;
            let s = text(input, "text")?;
            *out = box_result(a.inner.find_text(s)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_analyze_graph(
    a: *mut turbo_analyzer,
    input: turbo_text,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let a = analyzer_handle(a)?;
            let out = out_ptr(out, "out")?;
            let s = text(input, "text")?;
            *out = box_result(a.inner.analyze_graph(s)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_inspect_graph(
    a: *mut turbo_analyzer,
    input: turbo_text,
    graph_bytes: *const u8,
    graph_len: u64,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let a = analyzer_handle(a)?;
            let out = out_ptr(out, "out")?;
            let s = text(input, "text")?;
            if graph_len == 0 || graph_bytes.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "graph_bytes is NULL or empty"));
            }
            let n = array_len::<u8>(graph_len, "graph_bytes")?;
            let bytes = std::slice::from_raw_parts(graph_bytes, n);
            *out = box_result(a.inner.inspect_graph(s, bytes)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_spans(
    r: *const turbo_analysis_result,
    spans: *mut *const turbo_span,
    count: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = analysis_result_handle(r)?;
            let spans = out_ptr(spans, "spans")?;
            let count = out_ptr(count, "count")?;
            *count = r.inner.spans.len() as u32;
            *spans = if r.inner.spans.is_empty() { std::ptr::null() } else { r.inner.spans.as_ptr() };
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_strings(
    r: *const turbo_analysis_result,
    pool: *mut *const c_char,
    bytes: *mut u64,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = analysis_result_handle(r)?;
            let pool = out_ptr(pool, "pool")?;
            let bytes = out_ptr(bytes, "bytes")?;
            *bytes = r.inner.pool.len() as u64;
            *pool = if r.inner.pool.is_empty() { std::ptr::null() } else { r.inner.pool.as_ptr().cast::<c_char>() };
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_graph(
    r: *const turbo_analysis_result,
    data: *mut *const u8,
    bytes: *mut u64,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = analysis_result_handle(r)?;
            let data = out_ptr(data, "data")?;
            let bytes = out_ptr(bytes, "bytes")?;
            *bytes = r.inner.graph.len() as u64;
            *data = if r.inner.graph.is_empty() { std::ptr::null() } else { r.inner.graph.as_ptr() };
            Ok(())
        })
    }
}

/// # Safety
/// `r` is NULL or a live handle returned by a `turbo_analyzer_*` call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_release(r: *mut turbo_analysis_result) {
    if unsafe { analysis_result_handle(r) }.is_ok() {
        let mut b = unsafe { Box::from_raw(r) };
        b.magic = 0;
    }
}
