//! The embedding and reranking session entry points for the Java class
//! `TurboJni`: a runtime, a context on the first device, a loaded bundle
//! and one session over it, opened and released together, as the
//! bindings' `TurboSession` does over the C ABI.

use super::sys::{JNIEnv, jbyteArray, jclass, jfloatArray, jintArray, jlong, jobjectArray};
use super::{check, count, error, handle, java_array, run, split, text};
use crate::status::{Error, INTERNAL, INVALID_HANDLE, Result};
use crate::{
    TURBO_DTYPE_F32, turbo_context, turbo_context_create, turbo_context_release, turbo_embed_write_text, turbo_model,
    turbo_model_load, turbo_model_release, turbo_rerank_write_pairs, turbo_result, turbo_result_get_info,
    turbo_result_info, turbo_result_read, turbo_result_release, turbo_runtime, turbo_runtime_create,
    turbo_runtime_release, turbo_session, turbo_session_create, turbo_session_release, turbo_session_run,
    turbo_text_pair,
};

/// The handles behind one Java `TurboSession`, released in reverse order.
struct Session {
    runtime: *mut turbo_runtime,
    context: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
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

/// `static native long sessionOpen(byte[] bundlePath)`: `turbo_runtime_create`,
/// `turbo_context_create` on device 0, `turbo_model_load` and
/// `turbo_session_create` with the model's defaults. On a failure, what
/// was made is released before the error is thrown.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionOpen(
    env: *mut JNIEnv,
    _class: jclass,
    bundle_path: jbyteArray,
) -> jlong {
    run(env, 0, |env| {
        let path = env.byte_array(bundle_path);
        let mut s = Box::new(Session {
            runtime: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
            model: std::ptr::null_mut(),
            session: std::ptr::null_mut(),
        });
        let mut e = error();
        unsafe {
            check(turbo_runtime_create(std::ptr::null(), &mut s.runtime, &mut e), &e)?;
            check(turbo_context_create(s.runtime, 0, &mut s.context, &mut e), &e)?;
            check(turbo_model_load(s.context, text(&path), &mut s.model, &mut e), &e)?;
            check(turbo_session_create(s.model, std::ptr::null(), &mut s.session, &mut e), &e)?;
        }
        Ok(Box::into_raw(s) as usize as jlong)
    })
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

/// `static native float[][] sessionEmbed(long session, byte[] texts, int[] lengths)`:
/// `turbo_embed_write_text` over the packed texts with the session's
/// default options, one run, and the F32 vectors as one `float[]` a text.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionEmbed(
    env: *mut JNIEnv,
    _class: jclass,
    s: jlong,
    packed: jbyteArray,
    lengths: jintArray,
) -> jobjectArray {
    run(env, std::ptr::null_mut(), |env| {
        let s = session(s)?;
        let packed = env.byte_array(packed);
        let texts = split(&packed, &env.int_array(lengths), "texts")?;
        let n = count(texts.len(), "texts")?;
        let mut e = error();
        check(unsafe { turbo_embed_write_text(s.session, texts.as_ptr(), n, std::ptr::null(), &mut e) }, &e)?;
        let (info, values) = run_f32(s)?;
        java_array(env.new_float_matrix(info.batch as usize, &values), "embedding")
    })
}

/// `static native float[] sessionRerank(long session, byte[] query, byte[] passages, int[] lengths)`:
/// `turbo_rerank_write_pairs` of the query with each packed passage, one
/// run, and the `[passages, 1]` F32 scores as a `float[]`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_sessionRerank(
    env: *mut JNIEnv,
    _class: jclass,
    s: jlong,
    query: jbyteArray,
    packed: jbyteArray,
    lengths: jintArray,
) -> jfloatArray {
    run(env, std::ptr::null_mut(), |env| {
        let s = session(s)?;
        let query = env.byte_array(query);
        let packed = env.byte_array(packed);
        let passages = split(&packed, &env.int_array(lengths), "passages")?;
        let pairs: Vec<turbo_text_pair> =
            passages.iter().map(|&passage| turbo_text_pair { query: text(&query), passage }).collect();
        let n = count(pairs.len(), "passages")?;
        let mut e = error();
        check(unsafe { turbo_rerank_write_pairs(s.session, pairs.as_ptr(), n, std::ptr::null(), &mut e) }, &e)?;
        let (info, scores) = run_f32(s)?;
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

/// `turbo_session_run`, then the result's info and its F32 values read
/// out, checked against the shape the info reports.
fn run_f32(s: &Session) -> Result<(turbo_result_info, Vec<f32>)> {
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
        if info.dtype != TURBO_DTYPE_F32 {
            return Err(Error::new(INTERNAL, format!("the result's dtype is {}, not F32", info.dtype)));
        }
        let count = (info.batch as usize) * (info.dim as usize);
        if info.bytes != (count * size_of::<f32>()) as u64 {
            return Err(Error::new(
                INTERNAL,
                format!("the result holds {} bytes for [{}, {}] F32 values", info.bytes, info.batch, info.dim),
            ));
        }
        let mut values = vec![0f32; count];
        let mut written = 0u64;
        check(turbo_result_read(result.0, values.as_mut_ptr().cast(), info.bytes, &mut written, &mut e), &e)?;
        if written != info.bytes {
            return Err(Error::new(INTERNAL, format!("read {written} of {} result bytes", info.bytes)));
        }
        Ok((info, values))
    }
}
