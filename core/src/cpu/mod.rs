//! The CPU backend: the host processor, through turbo_backend.h like any
//! other backend. It lists the device and its memory (here and host.rs),
//! gives out host memory (buffers.rs), holds the models loaded on it
//! (model.rs), and runs embed sessions (session.rs, run.rs) with the
//! encoder in encoder/, or for a static model the table in table/.

mod buffers;
mod encoder;
mod host;
mod kernels;
mod model;
pub(crate) mod pool;
mod run;
mod session;
mod table;

#[cfg(test)]
mod boundary_tests;

use std::ffi::c_char;

use crate::backend::{TURBO_CAP_EXPERIMENTAL, TURBO_FORMAT_SAFETENSORS, format_bit, refuse, turbo_backend};
use crate::status::INVALID_ARGUMENT;
use crate::{TURBO_DEVICE_CPU, TURBO_DTYPE_F32, turbo_device_info, turbo_error, write_str};

use buffers::*;
use host::Host;
use model::*;
#[cfg(feature = "internals")]
pub(crate) use model::{converted_data, tensor_data};
use run::*;
use session::*;

pub use kernels::{Matrix, Mode};
pub(crate) use kernels::{gelu, softmax};

pub static BACKEND: turbo_backend = turbo_backend {
    struct_size: size_of::<turbo_backend>() as u32,
    reserved: 0,
    name: c"cpu".as_ptr(),
    runtime_version: c"".as_ptr(),
    device_count,
    device_info,
    capability,
    context_create: Some(context_create),
    context_release: Some(context_release),
    buffer_alloc: Some(buffer_alloc),
    buffer_import: Some(buffer_import),
    buffer_release: Some(buffer_release),
    buffer_export: Some(buffer_export),
    model_load: Some(model_load),
    model_release: Some(model_release),
    session_create: Some(session_create),
    session_release: Some(session_release),
    embed_write: Some(embed_write),
    session_run: Some(session_run),
    // Every buffer here has a host address, which the core reads itself.
    buffer_read: None,
    formats: format_bit(TURBO_FORMAT_SAFETENSORS),
    reserved2: 0,
    session_create_tuned: None,
    session_open: Some(session_open),
    embed_write_ranges: Some(embed_write_ranges),
    token_head: 0,
    reserved3: 0,
};

unsafe extern "C" fn device_count(out: *mut u32, _err: *mut turbo_error) -> i32 {
    unsafe { *out = 1 };
    0
}

/// The host is the one device this backend lists.
unsafe fn only(ordinal: u32, err: *mut turbo_error) -> Option<i32> {
    (ordinal != 0).then(|| unsafe { refuse(err, INVALID_ARGUMENT, &format!("cpu device {ordinal}: only 0 is listed")) })
}

unsafe extern "C" fn device_info(ordinal: u32, out: *mut turbo_device_info, err: *mut turbo_error) -> i32 {
    if let Some(rc) = unsafe { only(ordinal, err) } {
        return rc;
    }
    let out = unsafe { &mut *out };
    let host = Host::read();
    out.kind = TURBO_DEVICE_CPU;
    out.ordinal = 0;
    out.unified_memory = 1;
    out.memory_total = host.memory_total;
    out.memory_free = host.memory_free;
    // The instruction set, not the processor: arch[32] cannot hold a model
    // name. A CPU benchmark record is filed under arch and name together,
    // so a record for one x86_64 processor backs no other.
    write_str(&mut out.arch, std::env::consts::ARCH);
    write_str(&mut out.name, &host.name);
    write_str(&mut out.vendor, &host.vendor);
    write_str(&mut out.runtime_version, "");
    write_str(&mut out.driver_version, "");
    0
}

/// Bits of turbo_embed_options the run honors: normalize (4), pooling (5)
/// and output_dim (6), every value of each.
const EMBED_HONORED: u32 = 0b111000;

/// Embed runs at every precision, in F32: the one dtype the encoder
/// computes in. FASTEST is F32 too, and says so. A model whose weights are
/// F16 or BF16 computes in F32 at EXACT and FASTEST, from a converted copy
/// (see session_create); at MODEL it would compute in its storage dtype,
/// which the encoder does not, so that session is refused. A cell cannot
/// say that, since it is not for one model; turbo_session_get_info does.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn capability(
    ordinal: u32,
    _task: u32,
    _precision: u32,
    status: *mut u32,
    dtype: *mut u32,
    options_honored: *mut u32,
    reason: *mut c_char,
    reason_len: u32,
    err: *mut turbo_error,
) -> i32 {
    if let Some(rc) = unsafe { only(ordinal, err) } {
        return rc;
    }
    unsafe {
        *status = TURBO_CAP_EXPERIMENTAL;
        *dtype = TURBO_DTYPE_F32;
        *options_honored = EMBED_HONORED;
        let r = std::slice::from_raw_parts_mut(reason, reason_len as usize);
        write_str(r, "");
    }
    0
}
