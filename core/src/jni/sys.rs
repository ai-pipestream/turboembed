//! The Java Native Interface as the entry points use it: the primitive
//! types, the reference types and the function table every `JNIEnv` points
//! at, declared from the JNI specification so the library builds without
//! Java headers and without C glue. Only the table entries the entry points
//! call are typed; the table's shape is the specification's, fixed since
//! JNI 1.2, so an entry is found by its position.

use std::ffi::{CStr, c_char, c_void};

pub type jint = i32;
pub type jlong = i64;
pub type jsize = i32;
pub type jboolean = u8;
pub type jobject = *mut c_void;
pub type jclass = jobject;
pub type jstring = jobject;
pub type jthrowable = jobject;
pub type jbyteArray = jobject;
pub type jintArray = jobject;
pub type jfloatArray = jobject;
pub type jdoubleArray = jobject;
pub type jobjectArray = jobject;
pub type jmethodID = *mut c_void;

/// One argument of a Java constructor or method, as `NewObjectA` reads it.
#[repr(C)]
pub union jvalue {
    pub z: jboolean,
    pub i: jint,
    pub j: jlong,
    pub f: f32,
    pub d: f64,
    pub l: jobject,
}

/// `struct JNINativeInterface_`: the function table, by position. The
/// entries the entry points call are named below; the rest are padding.
#[repr(C)]
pub struct JniTable {
    pub fns: [*const c_void; 235],
}

/// `JNIEnv`: a thread's pointer to the table, as the first argument of
/// every entry point.
pub type JNIEnv = *const JniTable;

const FIND_CLASS: usize = 6;
const THROW: usize = 13;
const GET_METHOD_ID: usize = 33;
const NEW_OBJECT_A: usize = 30;
const NEW_STRING: usize = 163;
const GET_ARRAY_LENGTH: usize = 171;
const NEW_OBJECT_ARRAY: usize = 172;
const SET_OBJECT_ARRAY_ELEMENT: usize = 174;
const NEW_BYTE_ARRAY: usize = 176;
const NEW_INT_ARRAY: usize = 179;
const NEW_FLOAT_ARRAY: usize = 181;
const NEW_DOUBLE_ARRAY: usize = 182;
const GET_BYTE_ARRAY_REGION: usize = 200;
const GET_INT_ARRAY_REGION: usize = 203;
const SET_BYTE_ARRAY_REGION: usize = 208;
const SET_INT_ARRAY_REGION: usize = 211;
const SET_FLOAT_ARRAY_REGION: usize = 213;
const SET_DOUBLE_ARRAY_REGION: usize = 214;
const EXCEPTION_CHECK: usize = 228;

/// A table entry as the function it is. `JNICALL` is the platform's
/// system calling convention, which `extern "system"` names.
macro_rules! entry {
    ($env:expr, $index:expr, fn($($arg:ty),*) -> $ret:ty) => {{
        let raw: *const c_void = unsafe { (**$env.env).fns[$index] };
        unsafe { std::mem::transmute::<*const c_void, unsafe extern "system" fn(*mut JNIEnv, $($arg),*) -> $ret>(raw) }
    }};
}

/// The calling thread's `JNIEnv`, valid for one entry point call. A
/// reference it returns is a local reference, freed when the call returns.
#[derive(Clone, Copy)]
pub struct Env {
    env: *mut JNIEnv,
}

impl Env {
    /// # Safety
    /// `env` is the `JNIEnv *` the Java virtual machine passed to the
    /// running entry point, on the thread running it.
    pub unsafe fn new(env: *mut JNIEnv) -> Env {
        Env { env }
    }

    /// `FindClass`: NULL, with an exception pending, when the class is not found.
    pub fn find_class(&self, name: &CStr) -> jclass {
        let f = entry!(self, FIND_CLASS, fn(*const c_char) -> jclass);
        unsafe { f(self.env, name.as_ptr()) }
    }

    /// `GetMethodID`: NULL, with an exception pending, when the method is not found.
    pub fn get_method_id(&self, class: jclass, name: &CStr, signature: &CStr) -> jmethodID {
        let f = entry!(self, GET_METHOD_ID, fn(jclass, *const c_char, *const c_char) -> jmethodID);
        unsafe { f(self.env, class, name.as_ptr(), signature.as_ptr()) }
    }

    /// `NewObjectA`: the constructor `method` of `class` over `args`.
    pub fn new_object(&self, class: jclass, method: jmethodID, args: &[jvalue]) -> jobject {
        let f = entry!(self, NEW_OBJECT_A, fn(jclass, jmethodID, *const jvalue) -> jobject);
        unsafe { f(self.env, class, method, args.as_ptr()) }
    }

    /// `Throw`: makes `throwable` the pending exception.
    pub fn throw(&self, throwable: jthrowable) -> jint {
        let f = entry!(self, THROW, fn(jthrowable) -> jint);
        unsafe { f(self.env, throwable) }
    }

    /// `ExceptionCheck`: whether an exception is pending on this thread.
    pub fn exception_check(&self) -> bool {
        let f = entry!(self, EXCEPTION_CHECK, fn() -> jboolean);
        unsafe { f(self.env) != 0 }
    }

    /// `NewString` over the UTF-16 code units of `s`: every scalar value,
    /// including the ones outside the Basic Multilingual Plane, as Java
    /// spells it. NULL when the virtual machine is out of memory.
    pub fn new_string(&self, s: &str) -> jstring {
        let units: Vec<u16> = s.encode_utf16().collect();
        let Ok(len) = jsize::try_from(units.len()) else { return std::ptr::null_mut() };
        let f = entry!(self, NEW_STRING, fn(*const u16, jsize) -> jstring);
        unsafe { f(self.env, units.as_ptr(), len) }
    }

    /// `GetArrayLength`; NULL is an empty array.
    pub fn array_length(&self, array: jobject) -> usize {
        if array.is_null() {
            return 0;
        }
        let f = entry!(self, GET_ARRAY_LENGTH, fn(jobject) -> jsize);
        usize::try_from(unsafe { f(self.env, array) }).unwrap_or(0)
    }

    /// The bytes of a `byte[]`, copied; NULL is empty.
    pub fn byte_array(&self, array: jbyteArray) -> Vec<u8> {
        let len = self.array_length(array);
        let mut out = vec![0u8; len];
        if len > 0 {
            let f = entry!(self, GET_BYTE_ARRAY_REGION, fn(jbyteArray, jsize, jsize, *mut u8) -> ());
            unsafe { f(self.env, array, 0, len as jsize, out.as_mut_ptr()) };
        }
        out
    }

    /// The values of an `int[]`, copied; NULL is empty.
    pub fn int_array(&self, array: jintArray) -> Vec<i32> {
        let len = self.array_length(array);
        let mut out = vec![0i32; len];
        if len > 0 {
            let f = entry!(self, GET_INT_ARRAY_REGION, fn(jintArray, jsize, jsize, *mut i32) -> ());
            unsafe { f(self.env, array, 0, len as jsize, out.as_mut_ptr()) };
        }
        out
    }

    /// A new `byte[]` holding `bytes`; NULL, with an exception pending, when
    /// the virtual machine is out of memory or the length is over `jsize`.
    pub fn new_byte_array(&self, bytes: &[u8]) -> jbyteArray {
        let Ok(len) = jsize::try_from(bytes.len()) else { return std::ptr::null_mut() };
        let new = entry!(self, NEW_BYTE_ARRAY, fn(jsize) -> jbyteArray);
        let array = unsafe { new(self.env, len) };
        if !array.is_null() && len > 0 {
            let set = entry!(self, SET_BYTE_ARRAY_REGION, fn(jbyteArray, jsize, jsize, *const u8) -> ());
            unsafe { set(self.env, array, 0, len, bytes.as_ptr()) };
        }
        array
    }

    /// A new `int[]` holding `values`, as [`Env::new_byte_array`].
    pub fn new_int_array(&self, values: &[i32]) -> jintArray {
        let Ok(len) = jsize::try_from(values.len()) else { return std::ptr::null_mut() };
        let new = entry!(self, NEW_INT_ARRAY, fn(jsize) -> jintArray);
        let array = unsafe { new(self.env, len) };
        if !array.is_null() && len > 0 {
            let set = entry!(self, SET_INT_ARRAY_REGION, fn(jintArray, jsize, jsize, *const i32) -> ());
            unsafe { set(self.env, array, 0, len, values.as_ptr()) };
        }
        array
    }

    /// A new `float[]` holding `values`, as [`Env::new_byte_array`].
    pub fn new_float_array(&self, values: &[f32]) -> jfloatArray {
        let Ok(len) = jsize::try_from(values.len()) else { return std::ptr::null_mut() };
        let new = entry!(self, NEW_FLOAT_ARRAY, fn(jsize) -> jfloatArray);
        let array = unsafe { new(self.env, len) };
        if !array.is_null() && len > 0 {
            let set = entry!(self, SET_FLOAT_ARRAY_REGION, fn(jfloatArray, jsize, jsize, *const f32) -> ());
            unsafe { set(self.env, array, 0, len, values.as_ptr()) };
        }
        array
    }

    /// A new `double[]` holding `values`, as [`Env::new_byte_array`].
    pub fn new_double_array(&self, values: &[f64]) -> jdoubleArray {
        let Ok(len) = jsize::try_from(values.len()) else { return std::ptr::null_mut() };
        let new = entry!(self, NEW_DOUBLE_ARRAY, fn(jsize) -> jdoubleArray);
        let array = unsafe { new(self.env, len) };
        if !array.is_null() && len > 0 {
            let set = entry!(self, SET_DOUBLE_ARRAY_REGION, fn(jdoubleArray, jsize, jsize, *const f64) -> ());
            unsafe { set(self.env, array, 0, len, values.as_ptr()) };
        }
        array
    }

    /// A new `float[][]` of `rows` rows, each a copy of its slice of
    /// `values`; NULL, with an exception pending, when an allocation fails.
    pub fn new_float_matrix(&self, rows: usize, values: &[f32]) -> jobjectArray {
        let Ok(count) = jsize::try_from(rows) else { return std::ptr::null_mut() };
        let row_class = self.find_class(c"[F");
        if row_class.is_null() {
            return std::ptr::null_mut();
        }
        let new = entry!(self, NEW_OBJECT_ARRAY, fn(jsize, jclass, jobject) -> jobjectArray);
        let matrix = unsafe { new(self.env, count, row_class, std::ptr::null_mut()) };
        if matrix.is_null() || rows == 0 {
            return matrix;
        }
        let width = values.len() / rows;
        let set = entry!(self, SET_OBJECT_ARRAY_ELEMENT, fn(jobjectArray, jsize, jobject) -> ());
        for (index, row) in values.chunks(width.max(1)).take(rows).enumerate() {
            let row = self.new_float_array(if width == 0 { &[] } else { row });
            if row.is_null() {
                return std::ptr::null_mut();
            }
            unsafe { set(self.env, matrix, index as jsize, row) };
        }
        matrix
    }
}
