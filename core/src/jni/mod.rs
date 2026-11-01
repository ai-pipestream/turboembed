//! The Java entry points of libturbo: the `native` methods of the class
//! `ai.pipestream.turboembed.jni.TurboJni`, exported from the shared
//! library beside the C ABI so one `libturbo` serves a Java virtual machine
//! through the Java Native Interface (Android, and any Java from 8 up)
//! as well as through the Foreign Function and Memory API.
//!
//! Every entry point is a thin shim over the C ABI functions of this
//! crate: it copies its Java arrays once, calls the same `turbo_*`
//! function a C caller would, and copies the answer into new Java arrays.
//! Text crosses as UTF-8 bytes the Java side encoded, so what reaches the
//! analyzer or the session is byte for byte what the Panama binding sends.
//! A failed call becomes a thrown `TurboException` carrying the status
//! code, its name, the field and the message of the `turbo_error`. Native
//! handles cross as `long`: an analyzer, a result or a session the Java
//! side releases explicitly through the matching entry point.

#![allow(non_snake_case)]

mod analyzer;
mod session;
mod sys;

use std::ffi::CStr;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::ffi::{new_error, panic_error};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_HANDLE, Result};
use crate::{turbo_error, turbo_text};
use sys::{Env, JNIEnv, jclass, jint, jobject, jstring, jvalue};

const EXCEPTION_CLASS: &CStr = c"ai/pipestream/turboembed/jni/TurboException";
const EXCEPTION_INIT: &CStr = c"(ILjava/lang/String;ILjava/lang/String;)V";

/// Run `f` as the body of an entry point: a panic stops here, and an error
/// is thrown as a `TurboException` while `fail` is returned to Java, which
/// discards it. An exception the virtual machine raised meanwhile (an
/// allocation that failed) is left pending in place of ours.
fn run<T>(env: *mut JNIEnv, fail: T, f: impl FnOnce(&Env) -> Result<T>) -> T {
    let env = unsafe { Env::new(env) };
    let outcome = catch_unwind(AssertUnwindSafe(|| f(&env))).unwrap_or_else(|p| Err(panic_error(p)));
    match outcome {
        Ok(value) => value,
        Err(e) => {
            if !env.exception_check() {
                throw(&env, &e);
            }
            fail
        }
    }
}

fn throw(env: &Env, e: &Error) {
    let class = env.find_class(EXCEPTION_CLASS);
    if class.is_null() {
        return;
    }
    let init = env.get_method_id(class, c"<init>", EXCEPTION_INIT);
    if init.is_null() {
        return;
    }
    let name = env.new_string(crate::status::name(e.code).to_str().unwrap_or("TURBO_E_UNKNOWN"));
    let message = env.new_string(&e.message);
    if name.is_null() || message.is_null() {
        return;
    }
    let args = [jvalue { i: e.code }, jvalue { l: name }, jvalue { i: e.field as jint }, jvalue { l: message }];
    let exception = env.new_object(class, init, &args);
    if !exception.is_null() {
        env.throw(exception);
    }
}

/// A fresh `turbo_error` for one C ABI call.
fn error() -> turbo_error {
    new_error()
}

/// The status of a C ABI call as a result, with the error's field and message.
fn check(code: i32, e: &turbo_error) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(e.message.as_ptr()) }.to_string_lossy().into_owned();
    Err(Error::field(code, e.field, message))
}

/// A `turbo_text` over bytes the Java side encoded as UTF-8.
fn text(bytes: &[u8]) -> turbo_text {
    turbo_text { ptr: if bytes.is_empty() { std::ptr::null() } else { bytes.as_ptr().cast() }, len: bytes.len() as u64 }
}

/// Texts the Java side packed end to end into one `byte[]` with a length
/// each in an `int[]`, as `turbo_text` views into the packed bytes.
fn split(packed: &[u8], lengths: &[i32], what: &str) -> Result<Vec<turbo_text>> {
    let mut out = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for (index, &len) in lengths.iter().enumerate() {
        let len = usize::try_from(len)
            .map_err(|_| Error::new(INVALID_ARGUMENT, format!("{what}[{index}] has a negative length {len}")))?;
        let end = offset
            .checked_add(len)
            .filter(|&end| end <= packed.len())
            .ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("{what} lengths run past the packed bytes")))?;
        out.push(text(&packed[offset..end]));
        offset = end;
    }
    if offset != packed.len() {
        return Err(Error::new(INVALID_ARGUMENT, format!("{what} lengths do not cover the packed bytes")));
    }
    Ok(out)
}

/// A count as the C ABI takes it.
fn count(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::new(INVALID_ARGUMENT, format!("{what}: {n} items is over the C ABI's u32")))
}

/// A native handle the Java side holds as a `long`.
fn handle<T>(value: i64, what: &str) -> Result<*mut T> {
    if value == 0 {
        return Err(Error::new(INVALID_HANDLE, format!("{what} handle is released")));
    }
    Ok(value as usize as *mut T)
}

/// A new Java string for `s`, or an error when the virtual machine could
/// not make it (its own exception is then pending).
fn java_string(env: &Env, s: &str) -> Result<jstring> {
    let string = env.new_string(s);
    if string.is_null() {
        return Err(Error::new(crate::status::OUT_OF_MEMORY, "the Java virtual machine could not allocate a string"));
    }
    Ok(string)
}

/// A new Java array, or the error for a failed allocation as [`java_string`].
fn java_array(array: jobject, what: &str) -> Result<jobject> {
    if array.is_null() {
        return Err(Error::new(
            crate::status::OUT_OF_MEMORY,
            format!("the Java virtual machine could not allocate the {what} array"),
        ));
    }
    Ok(array)
}

/// `static native String version()`: `turbo_version()`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_version(env: *mut JNIEnv, _class: jclass) -> jstring {
    run(env, std::ptr::null_mut(), |env| {
        let version = unsafe { CStr::from_ptr(crate::turbo_version()) }.to_string_lossy();
        java_string(env, &version)
    })
}

/// `static native String statusName(int code)`: `turbo_status_name(code)`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_statusName(
    env: *mut JNIEnv,
    _class: jclass,
    code: jint,
) -> jstring {
    run(env, std::ptr::null_mut(), |env| java_string(env, crate::status::name(code).to_str().unwrap_or_default()))
}

/// `static native String lastErrorMessage()`: `turbo_last_error_message()`
/// on the calling thread.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_pipestream_turboembed_jni_TurboJni_lastErrorMessage(
    env: *mut JNIEnv,
    _class: jclass,
) -> jstring {
    run(env, std::ptr::null_mut(), |env| {
        let message = unsafe { CStr::from_ptr(crate::turbo_last_error_message()) }.to_string_lossy();
        java_string(env, &message)
    })
}
