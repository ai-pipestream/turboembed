//! The embedding and reranking session entry points for the Java class
//! `TurboJni`: a runtime, a context on the device the options name (an
//! index, or the one `turbo_runtime_select` picks for a task), a loaded
//! bundle and one session over it with the options' precision, batch,
//! sequence and threads, opened and released together, as the bindings'
//! `TurboSession` does over the C ABI. Options cross as `int[]` in the
//! order `TurboSessionOptions.toNative`, `TurboEmbedOptions.toNative` and
//! `TurboRerankOptions.toNative` write them.

use std::ffi::c_char;

use super::sys::{JNIEnv, jbyteArray, jclass, jfloatArray, jintArray, jlong, jobjectArray};
use super::{check, count, error, handle, java_array, run, split, text};
use crate::status::{Error, INTERNAL, INVALID_ARGUMENT, INVALID_HANDLE, Result};
use crate::{
    TURBO_DTYPE_F32, TURBO_DTYPE_I8, TURBO_DTYPE_U8, turbo_context, turbo_context_create, turbo_context_release,
    turbo_embed_options, turbo_embed_write_text, turbo_model, turbo_model_get_info, turbo_model_info, turbo_model_load,
    turbo_model_release, turbo_rerank_options, turbo_rerank_write_pairs, turbo_result, turbo_result_get_info,
    turbo_result_info, turbo_result_read, turbo_result_release, turbo_runtime, turbo_runtime_create,
    turbo_runtime_release, turbo_runtime_select, turbo_session, turbo_session_create, turbo_session_desc,
    turbo_session_release, turbo_session_run, turbo_text_pair,
};

/// The handles behind one Java `TurboSession`, released in reverse order,
/// and the device index its context is on.
struct Session {
    runtime: *mut turbo_runtime,
    context: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
    device: u32,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                turbo_session_release(self.session);
            }
            if !self.model.is_null() {
                turbo_model_release(self.model);
            }
            if !self.context.is_null() {
                turbo_context_release(self.context);
            }
            if !self.runtime.is_null() {
                turbo_runtime_release(self.runtime);
            }
        }
    }
}

fn session<'a>(value: jlong) -> Result<&'a Session> {
    Ok(unsafe { &*handle::<Session>(value, "session")? })
}

/// The `int[]` a Java options record wrote, checked to hold `n` values,
/// each a `u32`.
fn options(values: &[i32], n: usize, what: &str) -> Result<Vec<u32>> {
    if values.len() != n {
        return Err(Error::new(INVALID_ARGUMENT, format!("{what}: {} values, not {n}", values.len())));
    }
    values
        .iter()
        .map(|&v| u32::try_from(v).map_err(|_| Error::new(INVALID_ARGUMENT, format!("{what}: {v} is negative"))))
        .collect()
}

/// `static native long sessionOpen(byte[] bundlePath, int[] options)`:
/// `turbo_runtime_create`; the options' device, or the one
/// `turbo_runtime_select` picks when they name a task; `turbo_context_create`
/// on it, `turbo_model_load`, and `turbo_session_create` with the options'
/// max batch, max sequence, precision and CPU threads (0 each for the
/// model's or the runtime's own). Options: `[device, task, precision,
/// maxBatch, maxSeq, cpuThreads]`, task 0 for none. On a failure, what was
/// made is released before the error is thrown.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionOpen(
    env: *mut JNIEnv,
    _class: jclass,
    bundle_path: jbyteArray,
    opts: jintArray,
) -> jlong {
    run(env, 0, |env| {
        let path = env.byte_array(bundle_path);
        let o = options(&env.int_array(opts), 6, "session options")?;
        let mut s = Box::new(Session {
            runtime: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
            model: std::ptr::null_mut(),
            session: std::ptr::null_mut(),
            device: o[0],
        });
        let desc = turbo_session_desc {
            struct_size: size_of::<turbo_session_desc>() as u32,
            max_batch: o[3],
            max_seq: o[4],
            precision: o[2],
            tuning: 0,
            tuning_budget_ms: 0,
            cpu_threads: o[5],
            max_ranges: 0,
        };
        let mut e = error();
        unsafe {
            check(turbo_runtime_create(std::ptr::null(), &mut s.runtime, &mut e), &e)?;
            if o[1] != 0 {
                check(turbo_runtime_select(s.runtime, o[1], &mut s.device, std::ptr::null_mut(), 0, &mut e), &e)?;
            }
            check(turbo_context_create(s.runtime, s.device, &mut s.context, &mut e), &e)?;
            check(turbo_model_load(s.context, text(&path), &mut s.model, &mut e), &e)?;
            check(turbo_session_create(s.model, &desc, &mut s.session, &mut e), &e)?;
        }
        Ok(Box::into_raw(s) as usize as jlong)
    })
}

/// The embed options `[truncate, maxTokens, promptRole, normalize, pooling,
/// outputDim, quantization]` with `ranges` as the int8 ranges (none when
/// empty); the ranges must outlive the write call.
fn embed_options(values: &[i32], ranges: &[f32]) -> Result<turbo_embed_options> {
    let o = options(values, 7, "embed options")?;
    Ok(turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: o[0],
        max_tokens: o[1],
        prompt_role: o[2],
        normalize: o[3],
        pooling: o[4],
        output_dim: o[5],
        quantization: o[6],
        int8_ranges: if ranges.is_empty() { std::ptr::null() } else { ranges.as_ptr() },
        layers: 0,
        reserved: 0,
    })
}

/// `turbo_embed_write_text` of the packed texts with the options, then the
/// run: the result's info and its bytes.
fn embed_run(
    env: &super::sys::Env,
    s: jlong,
    packed: jbyteArray,
    lengths: jintArray,
    opts: jintArray,
    ranges: jfloatArray,
) -> Result<(turbo_result_info, Vec<u8>)> {
    let s = session(s)?;
    let packed = env.byte_array(packed);
    let texts = split(&packed, &env.int_array(lengths), "texts")?;
    let n = count(texts.len(), "texts")?;
    let ranges = env.float_array(ranges);
    let o = embed_options(&env.int_array(opts), &ranges)?;
    let mut e = error();
    check(unsafe { turbo_embed_write_text(s.session, texts.as_ptr(), n, &o, &mut e) }, &e)?;
    run_bytes(s)
}

/// `static native float[][] sessionEmbed(long session, byte[] texts, int[] lengths, int[] options, float[] int8Ranges)`:
/// `turbo_embed_write_text` over the packed texts with the options, one
/// run, and the F32 vectors as one `float[]` a text. A quantization that
/// makes codes is refused here: `sessionEmbedCodes` returns those.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionEmbed(
    env: *mut JNIEnv,
    _class: jclass,
    s: jlong,
    packed: jbyteArray,
    lengths: jintArray,
    opts: jintArray,
    ranges: jfloatArray,
) -> jobjectArray {
    run(env, std::ptr::null_mut(), |env| {
        let (info, bytes) = embed_run(env, s, packed, lengths, opts, ranges)?;
        if info.dtype != TURBO_DTYPE_F32 {
            return Err(Error::new(INVALID_ARGUMENT, "the options quantize the vectors: read them with embedCodes"));
        }
        let values: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
        java_array(env.new_float_matrix(info.batch as usize, &values), "embedding")
    })
}

/// `static native byte[][] sessionEmbedCodes(long session, byte[] texts, int[] lengths, int[] options, float[] int8Ranges)`:
/// as `sessionEmbed`, for options that quantize: each text's int8 codes or
/// packed sign bits (TURBO_QUANTIZE_INT8, TURBO_QUANTIZE_UBINARY) as one
/// `byte[]`. F32 vectors are refused here.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionEmbedCodes(
    env: *mut JNIEnv,
    _class: jclass,
    s: jlong,
    packed: jbyteArray,
    lengths: jintArray,
    opts: jintArray,
    ranges: jfloatArray,
) -> jobjectArray {
    run(env, std::ptr::null_mut(), |env| {
        let (info, bytes) = embed_run(env, s, packed, lengths, opts, ranges)?;
        if info.dtype != TURBO_DTYPE_I8 && info.dtype != TURBO_DTYPE_U8 {
            return Err(Error::new(INVALID_ARGUMENT, "the options do not quantize the vectors: read them with embed"));
        }
        java_array(env.new_byte_matrix(info.batch as usize, &bytes), "codes")
    })
}

/// `static native int[] sessionInfo(long session)`: `turbo_model_get_info`
/// as `[device, task, dim, pooling, normalize, maxSeq, maxBatch, dtype,
/// rerankQueryTokens, rerankPairTemplate, outputDims...]`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionInfo(
    env: *mut JNIEnv,
    _class: jclass,
    s: jlong,
) -> jintArray {
    run(env, std::ptr::null_mut(), |env| {
        let s = session(s)?;
        let i = model_info(s)?;
        let head = [
            s.device,
            i.task,
            i.dim,
            i.pooling,
            i.normalize,
            i.max_seq,
            i.max_batch,
            i.dtype,
            i.rerank_query_tokens,
            i.rerank_pair_template,
        ];
        let dims = &i.output_dims[..(i.output_dims_count as usize).min(i.output_dims.len())];
        let values: Vec<i32> = head.iter().chain(dims).map(|&v| v as i32).collect();
        java_array(env.new_int_array(&values), "model info")
    })
}

/// `static native byte[][] sessionInfoText(long session)`: the strings of
/// `turbo_model_get_info` as UTF-8, `[modelId, revision, manifestSha256,
/// artifactSha256, tokenizerSha256, prefixQuery, prefixDocument]`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionInfoText(
    env: *mut JNIEnv,
    _class: jclass,
    s: jlong,
) -> jobjectArray {
    run(env, std::ptr::null_mut(), |env| {
        let i = model_info(session(s)?)?;
        let fields: [&[c_char]; 7] = [
            &i.model_id,
            &i.revision,
            &i.manifest_sha256,
            &i.artifact_sha256,
            &i.tokenizer_sha256,
            &i.prefix_query,
            &i.prefix_document,
        ];
        let rows: Vec<Vec<u8>> =
            fields.iter().map(|f| f.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect()).collect();
        java_array(env.new_byte_rows(&rows), "model info text")
    })
}

fn model_info(s: &Session) -> Result<turbo_model_info> {
    let mut e = error();
    unsafe {
        let mut info: turbo_model_info = std::mem::zeroed();
        info.struct_size = size_of::<turbo_model_info>() as u32;
        check(turbo_model_get_info(s.model, &mut info, &mut e), &e)?;
        Ok(info)
    }
}

/// `static native void sessionRelease(long session)`: releases the four handles.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionRelease(
    env: *mut JNIEnv,
    _class: jclass,
    s: jlong,
) {
    run(env, (), |_| {
        drop(unsafe { Box::from_raw(handle::<Session>(s, "session")?) });
        Ok(())
    })
}

/// `static native float[] sessionRerank(long session, byte[] query, byte[] passages, int[] lengths, int[] options)`:
/// `turbo_rerank_write_pairs` of the query with each packed passage and
/// the options `[queryTokens, maxTokens, stride]`, one run, and the
/// `[passages, 1]` F32 scores as a `float[]`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionRerank(
    env: *mut JNIEnv,
    _class: jclass,
    s: jlong,
    query: jbyteArray,
    packed: jbyteArray,
    lengths: jintArray,
    opts: jintArray,
) -> jfloatArray {
    run(env, std::ptr::null_mut(), |env| {
        let s = session(s)?;
        let query = env.byte_array(query);
        let packed = env.byte_array(packed);
        let passages = split(&packed, &env.int_array(lengths), "passages")?;
        let pairs: Vec<turbo_text_pair> =
            passages.iter().map(|&passage| turbo_text_pair { query: text(&query), passage }).collect();
        let n = count(pairs.len(), "passages")?;
        let o = options(&env.int_array(opts), 3, "rerank options")?;
        let o = turbo_rerank_options {
            struct_size: size_of::<turbo_rerank_options>() as u32,
            query_tokens: o[0],
            max_tokens: o[1],
            stride: o[2],
        };
        let mut e = error();
        check(unsafe { turbo_rerank_write_pairs(s.session, pairs.as_ptr(), n, &o, &mut e) }, &e)?;
        let (info, bytes) = run_bytes(s)?;
        if info.dtype != TURBO_DTYPE_F32 {
            return Err(Error::new(INTERNAL, format!("the rerank result's dtype is {}, not F32", info.dtype)));
        }
        let scores: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
        if info.batch as usize != pairs.len() || info.dim != 1 {
            return Err(Error::new(
                INTERNAL,
                format!("rerank returned [{}, {}] scores for {} passages", info.batch, info.dim, pairs.len()),
            ));
        }
        java_array(env.new_float_array(&scores), "scores")
    })
}

/// A result the session holds a hold on; released when dropped.
struct OwnedResult(*mut turbo_result);

impl Drop for OwnedResult {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { turbo_result_release(self.0) };
        }
    }
}

/// `turbo_session_run`, then the result's info and its bytes read out,
/// checked against the shape the info reports: four bytes a value for
/// F32, a byte a code for I8, and whole rows for U8 sign bits.
fn run_bytes(s: &Session) -> Result<(turbo_result_info, Vec<u8>)> {
    let mut e = error();
    let mut result = OwnedResult(std::ptr::null_mut());
    unsafe {
        check(turbo_session_run(s.session, &mut result.0, &mut e), &e)?;
        if result.0.is_null() {
            return Err(Error::new(INVALID_HANDLE, "turbo_session_run returned no result"));
        }
        let mut info: turbo_result_info = std::mem::zeroed();
        info.struct_size = size_of::<turbo_result_info>() as u32;
        check(turbo_result_get_info(result.0, &mut info, &mut e), &e)?;
        let (batch, dim) = (info.batch as u64, info.dim as u64);
        let want = match info.dtype {
            TURBO_DTYPE_F32 => batch * dim * 4,
            TURBO_DTYPE_I8 => batch * dim,
            TURBO_DTYPE_U8 => batch * dim.div_ceil(8),
            other => return Err(Error::new(INTERNAL, format!("the result's dtype is {other}"))),
        };
        if info.bytes != want {
            return Err(Error::new(
                INTERNAL,
                format!(
                    "the result holds {} bytes for [{}, {}] of dtype {}",
                    info.bytes, info.batch, info.dim, info.dtype
                ),
            ));
        }
        let mut bytes = vec![0u8; info.bytes as usize];
        let mut written = 0u64;
        check(turbo_result_read(result.0, bytes.as_mut_ptr().cast(), info.bytes, &mut written, &mut e), &e)?;
        if written != info.bytes {
            return Err(Error::new(INTERNAL, format!("read {written} of {} result bytes", info.bytes)));
        }
        Ok((info, bytes))
    }
}
