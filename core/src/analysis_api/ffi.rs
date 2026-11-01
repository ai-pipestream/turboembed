//! C ABI entrypoints for `turbo_analyzer` and `turbo_analysis_result`.

use std::ffi::c_char;
use std::sync::Arc;

use super::{
    ANALYSIS_RESULT_MAGIC, ANALYZER_MAGIC, AnalysisResultInner, AnalyzerEngine, TURBO_ANALYZER_DESC_SIZE_V1,
    analysis_result_handle, analyzer_handle, turbo_analysis_result, turbo_analyzer, turbo_analyzer_desc, turbo_span,
};
use crate::ffi::{array_len, call, out_ptr, text};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_STRUCT_SIZE, Result};
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
    // Do not reserve from count alone: growth follows elements actually read.
    let mut out = Vec::new();
    for (idx, &t) in raw.iter().enumerate() {
        out.push(unsafe { text(t, &format!("{what}[{idx}]"))? });
    }
    Ok(out)
}

fn box_result(inner: AnalysisResultInner) -> *mut turbo_analysis_result {
    Arc::into_raw(Arc::new(turbo_analysis_result { magic: ANALYSIS_RESULT_MAGIC, inner })).cast_mut()
}

unsafe fn result_out<'a>(out: *mut *mut turbo_analysis_result) -> Result<&'a mut *mut turbo_analysis_result> {
    let out = unsafe { out_ptr(out, "out") }?;
    *out = std::ptr::null_mut();
    Ok(out)
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
            *out = std::ptr::null_mut();
            let desc_ref = if desc.is_null() {
                None
            } else {
                let d = &*desc;
                // A program built before bundle_path was appended passes
                // the earlier size; it then names no bundle.
                let known = [TURBO_ANALYZER_DESC_SIZE_V1, size_of::<turbo_analyzer_desc>()];
                if !known.contains(&(d.struct_size as usize)) {
                    return Err(Error::new(
                        INVALID_STRUCT_SIZE,
                        format!(
                            "turbo_analyzer_desc.struct_size is {}, this library knows {} and {}",
                            d.struct_size, known[0], known[1]
                        ),
                    ));
                }
                if d.flags != 0 {
                    return Err(Error::new(INVALID_ARGUMENT, "turbo_analyzer_desc.flags must be 0"));
                }
                Some((d, d.struct_size as usize > TURBO_ANALYZER_DESC_SIZE_V1))
            };
            let engine = AnalyzerEngine::from_desc(desc_ref)?;
            *out =
                Arc::into_raw(Arc::new(turbo_analyzer { magic: ANALYZER_MAGIC, inner: Arc::new(engine) })).cast_mut();
            Ok(())
        })
    }
}

/// # Safety
/// `a` is NULL or a live reference returned by `turbo_analyzer_create` or
/// this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_retain(a: *mut turbo_analyzer) -> *mut turbo_analyzer {
    if a.is_null() || unsafe { analyzer_handle(a) }.is_err() {
        return std::ptr::null_mut();
    }
    unsafe { Arc::increment_strong_count(a.cast_const()) };
    a
}

/// # Safety
/// `a` is NULL or one live reference owned by the caller. The caller must
/// release each reference exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_release(a: *mut turbo_analyzer) {
    if !a.is_null() && unsafe { analyzer_handle(a) }.is_ok() {
        drop(unsafe { Arc::from_raw(a.cast_const()) });
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
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
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
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
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
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
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
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
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
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
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
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
            let s = text(input, "text")?;
            *out = box_result(a.inner.find_text(s)?);
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_detect_pii(
    a: *mut turbo_analyzer,
    input: turbo_text,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
            let s = text(input, "text")?;
            *out = box_result(a.inner.detect_pii(s)?);
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
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
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
    graph_bytes: *const u8,
    graph_len: u64,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let out = result_out(out)?;
            let a = analyzer_handle(a)?;
            if graph_len == 0 || graph_bytes.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "graph_bytes is NULL or empty"));
            }
            let n = array_len::<u8>(graph_len, "graph_bytes")?;
            let bytes = std::slice::from_raw_parts(graph_bytes, n);
            *out = box_result(a.inner.inspect_graph(bytes)?);
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
            let spans = out_ptr(spans, "spans")?;
            *spans = std::ptr::null();
            let count = out_ptr(count, "count")?;
            *count = 0;
            let r = analysis_result_handle(r)?;
            *count = u32::try_from(r.inner.spans.len())
                .map_err(|_| Error::new(crate::status::CAPACITY, "analysis span count exceeds u32"))?;
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
            let pool = out_ptr(pool, "pool")?;
            *pool = std::ptr::null();
            let bytes = out_ptr(bytes, "bytes")?;
            *bytes = 0;
            let r = analysis_result_handle(r)?;
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
            let data = out_ptr(data, "data")?;
            *data = std::ptr::null();
            let bytes = out_ptr(bytes, "bytes")?;
            *bytes = 0;
            let r = analysis_result_handle(r)?;
            *bytes = r.inner.graph.len() as u64;
            *data = if r.inner.graph.is_empty() { std::ptr::null() } else { r.inner.graph.as_ptr() };
            Ok(())
        })
    }
}

/// # Safety
/// `r` is NULL or a live reference returned by a `turbo_analyzer_*` call or
/// this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_retain(r: *mut turbo_analysis_result) -> *mut turbo_analysis_result {
    if r.is_null() || unsafe { analysis_result_handle(r) }.is_err() {
        return std::ptr::null_mut();
    }
    unsafe { Arc::increment_strong_count(r.cast_const()) };
    r
}

/// # Safety
/// `r` is NULL or one live reference owned by the caller. The caller must
/// release each reference exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_release(r: *mut turbo_analysis_result) {
    if !r.is_null() && unsafe { analysis_result_handle(r) }.is_ok() {
        drop(unsafe { Arc::from_raw(r.cast_const()) });
    }
}
