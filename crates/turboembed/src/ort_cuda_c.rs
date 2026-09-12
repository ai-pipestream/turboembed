//! C hooks called from `native/turboembed/src/stub.cpp` when
//! `TURBOEMBED_ORT_CUDA` is set. The C++ stub never talks to ORT itself.
//!
//! Texts are pointer+length views (not required to be NUL-terminated).

use std::ffi::{c_char, c_int, CStr};
use std::path::PathBuf;
use std::ptr;
use std::slice;

use inferstream_backend_ort::Pooling;

use crate::catalog::{Arch, Catalog};
use crate::ort_cuda::OrtCudaSession;

fn write_err(err: *mut c_char, err_len: usize, msg: &str) {
    if err.is_null() || err_len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(err_len.saturating_sub(1));
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), err.cast(), n);
        *err.add(n) = 0;
    }
}

fn view_to_str<'a>(ptr: *const c_char, len: usize) -> Result<&'a str, String> {
    if ptr.is_null() {
        if len == 0 {
            return Ok("");
        }
        return Err("null pointer with non-zero length".into());
    }
    let bytes = unsafe { slice::from_raw_parts(ptr.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|e| e.to_string())
}

fn c_string_opt(ptr: *const c_char) -> Result<Option<PathBuf>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|e| e.to_string())?;
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(PathBuf::from(s)))
    }
}

#[no_mangle]
pub unsafe extern "C" fn turboembed_ort_cuda_open(
    alias: *const c_char,
    alias_len: usize,
    config_path: *const c_char,
    workspace_root: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> *mut OrtCudaSession {
    let result = (|| -> Result<OrtCudaSession, String> {
        let alias = view_to_str(alias, alias_len)?;
        let cfg = c_string_opt(config_path)?;
        let root = match c_string_opt(workspace_root)? {
            Some(p) => p,
            None => std::env::current_dir().map_err(|e| e.to_string())?,
        };
        let catalog = match cfg {
            Some(p) => Catalog::from_file(&p).map_err(|e| e.to_string())?,
            None => Catalog::builtin(),
        };
        let spec = catalog
            .resolve_embed(alias, Arch::Nvidia)
            .map_err(|e| e.to_string())?;
        OrtCudaSession::load(spec, &root)
    })();
    match result {
        Ok(session) => Box::into_raw(Box::new(session)),
        Err(e) => {
            write_err(err, err_len, &e);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn turboembed_ort_cuda_embed(
    session: *mut OrtCudaSession,
    ptrs: *const *const c_char,
    lens: *const usize,
    n_texts: usize,
    requested_pooling: i32,
    requested_normalize: i32,
    out_values: *mut *mut f32,
    out_dim: *mut usize,
    out_count: *mut usize,
    err: *mut c_char,
    err_len: usize,
) -> c_int {
    if session.is_null()
        || ptrs.is_null()
        || lens.is_null()
        || out_values.is_null()
        || out_dim.is_null()
        || out_count.is_null()
    {
        write_err(err, err_len, "null pointer in turboembed_ort_cuda_embed");
        return -1;
    }
    let session = unsafe { &*session };

    // 0 = DEFAULT, 1 = MEAN, 2 = CLS, 3 = LAST (matches turboembed_pooling).
    match requested_pooling {
        0 | 1 => {
            if session.pooling() != Pooling::Mean {
                write_err(
                    err,
                    err_len,
                    "embed requested mean/default pooling but the loaded \
                     catalog alias is not mean",
                );
                return -1;
            }
        }
        2 => {
            if session.pooling() != Pooling::Cls {
                write_err(
                    err,
                    err_len,
                    "embed requested CLS pooling but the loaded catalog \
                     alias is mean (MiniLM goldens are mean+L2)",
                );
                return -1;
            }
        }
        3 => {
            write_err(
                err,
                err_len,
                "LAST pooling is not implemented on the ORT CUDA path",
            );
            return -1;
        }
        _ => {
            write_err(err, err_len, "unknown pooling enum");
            return -1;
        }
    }
    if requested_normalize == 0 && session.normalize() {
        write_err(
            err,
            err_len,
            "normalize=false is not the catalog MiniLM path \
             (goldens are L2-normalized)",
        );
        return -1;
    }

    let texts: Result<Vec<String>, String> = (0..n_texts)
        .map(|i| {
            let p = unsafe { *ptrs.add(i) };
            let n = unsafe { *lens.add(i) };
            view_to_str(p, n).map(str::to_string)
        })
        .collect();
    let texts = match texts {
        Ok(t) => t,
        Err(e) => {
            write_err(err, err_len, &e);
            return -1;
        }
    };
    match session.embed_batch(&texts) {
        Ok((dim, flat)) => {
            let n = texts.len();
            if dim == 0 || flat.len() != n * dim {
                write_err(err, err_len, "ragged embedding batch");
                return -1;
            }
            let ptr = Box::into_raw(flat.into_boxed_slice()) as *mut f32;
            unsafe {
                *out_values = ptr;
                *out_dim = dim;
                *out_count = n;
            }
            0
        }
        Err(e) => {
            write_err(err, err_len, &e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn turboembed_ort_cuda_close(session: *mut OrtCudaSession) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

#[no_mangle]
pub unsafe extern "C" fn turboembed_ort_cuda_free_values(values: *mut f32, n: usize) {
    if values.is_null() || n == 0 {
        return;
    }
    drop(unsafe { Box::from_raw(slice::from_raw_parts_mut(values, n)) });
}

#[no_mangle]
pub unsafe extern "C" fn turboembed_ort_cuda_dim(session: *const OrtCudaSession) -> u32 {
    if session.is_null() {
        return 0;
    }
    unsafe { &*session }.embedding_dim() as u32
}

/// Keep C hooks in the rlib so `--gc-sections` cannot drop them before
/// the C++ stub (same crate) resolves the symbols.
#[used]
static ORT_CUDA_C_ABI: [*const (); 5] = [
    turboembed_ort_cuda_open as *const (),
    turboembed_ort_cuda_embed as *const (),
    turboembed_ort_cuda_close as *const (),
    turboembed_ort_cuda_free_values as *const (),
    turboembed_ort_cuda_dim as *const (),
];
