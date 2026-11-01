//! The analyzer and analysis result entry points: `turbo_analyzer_*` and
//! `turbo_analysis_result_*` for the Java class `TurboJni`.

use std::ffi::c_char;

use super::sys::{JNIEnv, jbyteArray, jclass, jdoubleArray, jint, jintArray, jlong};
use super::{check, count, error, handle, java_array, run, split, text};
use crate::analysis_api::{
    TURBO_ANALYZER_DESC_SIZE_V1, turbo_analysis_result, turbo_analysis_result_graph, turbo_analysis_result_release,
    turbo_analysis_result_retain, turbo_analysis_result_spans, turbo_analysis_result_strings, turbo_analyzer,
    turbo_analyzer_analyze_graph, turbo_analyzer_create, turbo_analyzer_desc, turbo_analyzer_detect_pii,
    turbo_analyzer_detect_sentences, turbo_analyzer_find_text, turbo_analyzer_find_tokens,
    turbo_analyzer_inspect_graph, turbo_analyzer_lemmatize, turbo_analyzer_pos_tag, turbo_analyzer_retain,
    turbo_analyzer_tokenize, turbo_span,
};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_HANDLE, Result};
use crate::turbo_text;

/// The `uint32_t` fields of a `turbo_span` before `prob`, in the header's
/// order: a row crosses to Java as this many `int`s and one `double`.
pub const SPAN_INTS: usize = 28;
const _: () = assert!(std::mem::offset_of!(turbo_span, prob) == SPAN_INTS * 4);
const _: () = assert!(size_of::<turbo_span>() == SPAN_INTS * 4 + size_of::<f64>());
// A struct_size of 120: every field, bundle_path included.
const _: () = assert!(size_of::<turbo_analyzer_desc>() == TURBO_ANALYZER_DESC_SIZE_V1 + size_of::<turbo_text>());

fn analyzer(value: jlong) -> Result<*mut turbo_analyzer> {
    handle(value, "analyzer")
}

fn result(value: jlong) -> Result<*mut turbo_analysis_result> {
    handle(value, "analysis result")
}

/// Run one analyzer task writing a result, and hand its handle to Java.
fn task(f: impl FnOnce(*mut *mut turbo_analysis_result, *mut crate::turbo_error) -> i32) -> Result<jlong> {
    let mut out: *mut turbo_analysis_result = std::ptr::null_mut();
    let mut e = error();
    check(f(&mut out, &mut e), &e)?;
    if out.is_null() {
        return Err(Error::new(INVALID_HANDLE, "the analyzer returned no result"));
    }
    Ok(out as usize as jlong)
}

/// `static native long analyzerCreate(byte[] language, byte[] bundlePath)`:
/// `turbo_analyzer_create` with a full-size descriptor naming the language
/// and the bundle, the retired fields empty.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_analyzerCreate(
    env: *mut JNIEnv,
    _class: jclass,
    language: jbyteArray,
    bundle_path: jbyteArray,
) -> jlong {
    run(env, 0, |env| {
        let language = env.byte_array(language);
        let bundle_path = env.byte_array(bundle_path);
        let empty = text(&[]);
        let desc = turbo_analyzer_desc {
            struct_size: size_of::<turbo_analyzer_desc>() as u32,
            flags: 0,
            language: text(&language),
            sent_model_path: empty,
            tok_model_path: empty,
            pos_model_path: empty,
            lemma_model_path: empty,
            ner_model_path: empty,
            bundle_path: text(&bundle_path),
        };
        let mut out: *mut turbo_analyzer = std::ptr::null_mut();
        let mut e = error();
        check(unsafe { turbo_analyzer_create(&desc, &mut out, &mut e) }, &e)?;
        if out.is_null() {
            return Err(Error::new(INVALID_HANDLE, "turbo_analyzer_create returned no analyzer"));
        }
        Ok(out as usize as jlong)
    })
}

/// `static native long analyzerRetain(long analyzer)`: another reference.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_analyzerRetain(
    env: *mut JNIEnv,
    _class: jclass,
    a: jlong,
) -> jlong {
    run(env, 0, |_| {
        let retained = unsafe { turbo_analyzer_retain(analyzer(a)?) };
        if retained.is_null() {
            return Err(Error::new(INVALID_HANDLE, "turbo_analyzer_retain refused the handle"));
        }
        Ok(retained as usize as jlong)
    })
}

/// `static native void analyzerRelease(long analyzer)`: drops one reference.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_analyzerRelease(
    env: *mut JNIEnv,
    _class: jclass,
    a: jlong,
) {
    run(env, (), |_| {
        unsafe { crate::analysis_api::turbo_analyzer_release(analyzer(a)?) };
        Ok(())
    })
}

/// The entry points taking one UTF-8 text: `turbo_analyzer_<name>(a, text, out, err)`.
macro_rules! text_task {
    ($java:ident, $c:ident) => {
        #[unsafe(no_mangle)]
        pub extern "system" fn $java(env: *mut JNIEnv, _class: jclass, a: jlong, text_bytes: jbyteArray) -> jlong {
            run(env, 0, |env| {
                let a = analyzer(a)?;
                let bytes = env.byte_array(text_bytes);
                task(|out, e| unsafe { $c(a, text(&bytes), out, e) })
            })
        }
    };
}

text_task!(Java_ai_pipestream_turboembed_jni_TurboJni_analyzerDetectSentences, turbo_analyzer_detect_sentences);
text_task!(Java_ai_pipestream_turboembed_jni_TurboJni_analyzerTokenize, turbo_analyzer_tokenize);
text_task!(Java_ai_pipestream_turboembed_jni_TurboJni_analyzerFindText, turbo_analyzer_find_text);
text_task!(Java_ai_pipestream_turboembed_jni_TurboJni_analyzerDetectPii, turbo_analyzer_detect_pii);
text_task!(Java_ai_pipestream_turboembed_jni_TurboJni_analyzerAnalyzeGraph, turbo_analyzer_analyze_graph);

/// The entry points taking one packed token sequence:
/// `turbo_analyzer_<name>(a, tokens, count, out, err)`.
macro_rules! token_task {
    ($java:ident, $c:ident) => {
        #[unsafe(no_mangle)]
        pub extern "system" fn $java(
            env: *mut JNIEnv,
            _class: jclass,
            a: jlong,
            packed: jbyteArray,
            lengths: jintArray,
        ) -> jlong {
            run(env, 0, |env| {
                let a = analyzer(a)?;
                let packed = env.byte_array(packed);
                let lengths = env.int_array(lengths);
                let tokens = split(&packed, &lengths, "tokens")?;
                let n = count(tokens.len(), "tokens")?;
                task(|out, e| unsafe { $c(a, tokens.as_ptr(), n, out, e) })
            })
        }
    };
}

token_task!(Java_ai_pipestream_turboembed_jni_TurboJni_analyzerPosTag, turbo_analyzer_pos_tag);
token_task!(Java_ai_pipestream_turboembed_jni_TurboJni_analyzerFindTokens, turbo_analyzer_find_tokens);

/// `static native long analyzerLemmatize(long analyzer, byte[] tokens,
/// int[] tokenLengths, byte[] tags, int[] tagLengths)`: `turbo_analyzer_lemmatize`
/// over aligned token and tag sequences.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_analyzerLemmatize(
    env: *mut JNIEnv,
    _class: jclass,
    a: jlong,
    tokens: jbyteArray,
    token_lengths: jintArray,
    tags: jbyteArray,
    tag_lengths: jintArray,
) -> jlong {
    run(env, 0, |env| {
        let a = analyzer(a)?;
        let token_bytes = env.byte_array(tokens);
        let tag_bytes = env.byte_array(tags);
        let tokens = split(&token_bytes, &env.int_array(token_lengths), "tokens")?;
        let tags = split(&tag_bytes, &env.int_array(tag_lengths), "tags")?;
        if tokens.len() != tags.len() {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("{} tokens with {} tags; the sequences must align", tokens.len(), tags.len()),
            ));
        }
        let n = count(tokens.len(), "tokens")?;
        task(|out, e| unsafe { turbo_analyzer_lemmatize(a, tokens.as_ptr(), tags.as_ptr(), n, out, e) })
    })
}

/// `static native long analyzerInspectGraph(long analyzer, byte[] graph, int offset, int length)`:
/// `turbo_analyzer_inspect_graph` over one slice of the array.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_analyzerInspectGraph(
    env: *mut JNIEnv,
    _class: jclass,
    a: jlong,
    graph: jbyteArray,
    offset: jint,
    length: jint,
) -> jlong {
    run(env, 0, |env| {
        let a = analyzer(a)?;
        let bytes = env.byte_array(graph);
        let (Ok(offset), Ok(length)) = (usize::try_from(offset), usize::try_from(length)) else {
            return Err(Error::new(INVALID_ARGUMENT, "graph offset and length must not be negative"));
        };
        let slice = offset
            .checked_add(length)
            .filter(|&end| end <= bytes.len())
            .map(|end| &bytes[offset..end])
            .ok_or_else(|| Error::new(INVALID_ARGUMENT, "graph slice is outside the array"))?;
        let ptr = if slice.is_empty() { std::ptr::null() } else { slice.as_ptr() };
        task(|out, e| unsafe { turbo_analyzer_inspect_graph(a, ptr, slice.len() as u64, out, e) })
    })
}

/// `static native int[] resultRows(long result)`: every row's 28 `uint32_t`
/// fields in the header's order, as the bit patterns of `int`s.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_resultRows(
    env: *mut JNIEnv,
    _class: jclass,
    r: jlong,
) -> jintArray {
    run(env, std::ptr::null_mut(), |env| {
        let spans = spans(r)?;
        let mut rows = Vec::with_capacity(spans.len() * SPAN_INTS);
        for span in spans {
            // repr(C), u32 fields up to prob: the static asserts above hold the layout.
            let words = unsafe { std::slice::from_raw_parts((span as *const turbo_span).cast::<u32>(), SPAN_INTS) };
            rows.extend(words.iter().map(|&w| w as i32));
        }
        java_array(env.new_int_array(&rows), "rows")
    })
}

/// `static native double[] resultProbabilities(long result)`: every row's `prob`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_resultProbabilities(
    env: *mut JNIEnv,
    _class: jclass,
    r: jlong,
) -> jdoubleArray {
    run(env, std::ptr::null_mut(), |env| {
        let probabilities: Vec<f64> = spans(r)?.iter().map(|span| span.prob).collect();
        java_array(env.new_double_array(&probabilities), "probabilities")
    })
}

fn spans<'a>(r: jlong) -> Result<&'a [turbo_span]> {
    let r = result(r)?;
    let mut ptr: *const turbo_span = std::ptr::null();
    let mut n: u32 = 0;
    let mut e = error();
    check(unsafe { turbo_analysis_result_spans(r, &mut ptr, &mut n, &mut e) }, &e)?;
    if n == 0 || ptr.is_null() {
        return Ok(&[]);
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, n as usize) })
}

/// `static native byte[] resultStrings(long result)`: the UTF-8 string pool.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_resultStrings(
    env: *mut JNIEnv,
    _class: jclass,
    r: jlong,
) -> jbyteArray {
    run(env, std::ptr::null_mut(), |env| {
        let r = result(r)?;
        let mut ptr: *const c_char = std::ptr::null();
        let mut bytes: u64 = 0;
        let mut e = error();
        check(unsafe { turbo_analysis_result_strings(r, &mut ptr, &mut bytes, &mut e) }, &e)?;
        java_array(env.new_byte_array(unsafe { borrowed(ptr.cast(), bytes) }), "string pool")
    })
}

/// `static native long resultGraphLength(long result)`: the graph's byte length.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_resultGraphLength(
    env: *mut JNIEnv,
    _class: jclass,
    r: jlong,
) -> jlong {
    run(env, 0, |_| {
        let (_, bytes) = graph(r)?;
        Ok(bytes as jlong)
    })
}

/// `static native byte[] resultGraph(long result)`: the serialized graph.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_resultGraph(
    env: *mut JNIEnv,
    _class: jclass,
    r: jlong,
) -> jbyteArray {
    run(env, std::ptr::null_mut(), |env| {
        let (ptr, bytes) = graph(r)?;
        java_array(env.new_byte_array(unsafe { borrowed(ptr, bytes) }), "graph")
    })
}

fn graph(r: jlong) -> Result<(*const u8, u64)> {
    let r = result(r)?;
    let mut ptr: *const u8 = std::ptr::null();
    let mut bytes: u64 = 0;
    let mut e = error();
    check(unsafe { turbo_analysis_result_graph(r, &mut ptr, &mut bytes, &mut e) }, &e)?;
    Ok((ptr, bytes))
}

/// A borrowed buffer the C ABI lent, as a slice; NULL or 0 bytes is empty.
unsafe fn borrowed<'a>(ptr: *const u8, bytes: u64) -> &'a [u8] {
    if ptr.is_null() || bytes == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(ptr, bytes as usize) }
}

/// `static native long resultRetain(long result)`: another reference.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_resultRetain(
    env: *mut JNIEnv,
    _class: jclass,
    r: jlong,
) -> jlong {
    run(env, 0, |_| {
        let retained = unsafe { turbo_analysis_result_retain(result(r)?) };
        if retained.is_null() {
            return Err(Error::new(INVALID_HANDLE, "turbo_analysis_result_retain refused the handle"));
        }
        Ok(retained as usize as jlong)
    })
}

/// `static native void resultRelease(long result)`: drops one reference.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_resultRelease(
    env: *mut JNIEnv,
    _class: jclass,
    r: jlong,
) {
    run(env, (), |_| {
        unsafe { turbo_analysis_result_release(result(r)?) };
        Ok(())
    })
}
