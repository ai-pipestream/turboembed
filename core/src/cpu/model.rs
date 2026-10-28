//! Models: the weights read in place, and the copies made once from them.
//!
//! The weights stay where the core read them: its one verified host copy of
//! each weights file, which it keeps unchanged until model_release returns.
//! The CPU reads them in place. A model here is the architecture and a table
//! of pointers into those bytes. F16 and BF16 weights are widened once, for
//! the sessions that compute in F32. The linear layers' weight matrices are
//! also copied once into the panel layout the matrix kernel reads
//! (kernels/): for all-MiniLM-L6-v2, 42.5 MB beside its 90.9 MB file.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, TURBO_FAMILY_BERT, TURBO_FAMILY_DEBERTA_V2,
    TURBO_FAMILY_ROBERTA, TURBO_FAMILY_STATIC, refuse, turbo_backend_model, turbo_backend_tensor,
};
use crate::status::{INVALID_ARGUMENT, INVALID_STRUCT_SIZE, UNSUPPORTED};
use crate::{TURBO_DTYPE_F16, TURBO_DTYPE_F32, turbo_error};

use super::{kernels, table};

pub(super) struct Model {
    pub(super) desc: turbo_backend_model,
    /// Each tensor as the core described it, its name dropped: the name is
    /// valid only for the load.
    pub(super) tensors: Vec<turbo_backend_tensor>,
    /// For F16 or BF16 weights, each tensor converted to F32: turbo.h's
    /// one resident copy per compute dtype. The first session that
    /// computes in F32 makes it, every later one shares it, and it goes
    /// with the model. It is made outside any run, so no result counts it.
    pub(super) f32: OnceLock<Vec<Vec<f32>>>,
    /// The linear layers packed for this processor's matrix kernel, made
    /// by the first session and shared by every later one, like `f32`.
    pub(super) packed: OnceLock<kernels::Packed>,
    /// Held while `packed` is made. A failed pack leaves it unset, and the
    /// next session tries again.
    pub(super) packing: Mutex<()>,
    /// A static model's table, its weights and mapping widened, made by the
    /// first session and shared by every later one.
    pub(super) table: OnceLock<std::sync::Arc<table::Table>>,
    /// The table in I8 for FASTEST sessions, made by the first, or why
    /// it cannot be.
    pub(super) quantized: OnceLock<Result<std::sync::Arc<table::Quantized>, String>>,
    /// A DeBERTa-v2 model's encoder, made from the F32 tensors by the
    /// first session, or why it cannot be.
    pub(super) deberta: OnceLock<Result<std::sync::Arc<crate::deberta::Encoder>, String>>,
}

// The tensors point into the core's weights, which it keeps unchanged
// until model_release; the copy is written once, then only read.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// A DeBERTa-v2 model's encoder, made once from `tensors`, the
    /// model's F32 tensors.
    pub(super) fn deberta(&self, tensors: &[&[f32]]) -> Result<std::sync::Arc<crate::deberta::Encoder>, String> {
        self.deberta
            .get_or_init(|| {
                let config = super::deberta::config(&self.desc);
                crate::deberta::Encoder::from_table(config, tensors).map(std::sync::Arc::new).map_err(|e| e.message)
            })
            .clone()
    }

    /// Each tensor as F32 values, in TURBO_BERT_* order.
    pub(super) fn f32_tensors(&self) -> Vec<&[f32]> {
        if self.desc.dtype == TURBO_DTYPE_F32 {
            // The loader placed each tensor at a multiple of its element
            // size in a 64-byte aligned file.
            return self
                .tensors
                .iter()
                .map(|t| unsafe { std::slice::from_raw_parts(t.data as *const f32, t.bytes as usize / 4) })
                .collect();
        }
        let copy = self.f32.get_or_init(|| {
            self.tensors
                .iter()
                .map(|t| {
                    let h = unsafe { std::slice::from_raw_parts(t.data as *const u16, t.bytes as usize / 2) };
                    let widen = if self.desc.dtype == TURBO_DTYPE_F16 { f16_to_f32 } else { bf16_to_f32 };
                    h.iter().map(|&v| widen(v)).collect()
                })
                .collect()
        });
        copy.iter().map(Vec::as_slice).collect()
    }

    /// The packed linear layers, made on first use from `tensors`, which
    /// are f32_tensors(). Err is the bytes that could not be allocated.
    pub(super) fn packed(&self, tensors: &[&[f32]]) -> Result<&kernels::Packed, usize> {
        if let Some(p) = self.packed.get() {
            return Ok(p);
        }
        // Sessions made at once wait here, so the weights are packed once.
        let _one = self.packing.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = self.packed.get() {
            return Ok(p);
        }
        let layer = |l: usize, r: usize| {
            tensors[(TURBO_BERT_EMBEDDING_TENSORS + l as u32 * TURBO_BERT_LAYER_TENSORS) as usize + r]
        };
        let p = kernels::Packed::new(&self.desc, layer, kernels::Isa::detect())?;
        Ok(self.packed.get_or_init(|| p))
    }
}

/// An IEEE half to the single it names exactly.
pub(super) fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as u32;
    match exp {
        // Zero and subnormals: man * 2^-24, exact in F32.
        0 => sign * man as f32 * f32::from_bits(0x3380_0000),
        0x1f => f32::from_bits(((h as u32 & 0x8000) << 16) | 0x7f80_0000 | (man << 13)),
        e => f32::from_bits(((h as u32 & 0x8000) << 16) | ((e as u32 + 112) << 23) | (man << 13)),
    }
}

/// A bfloat16 is the top half of the single it names.
pub(super) fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

pub(super) unsafe extern "C" fn model_load(
    _ctx: *mut c_void,
    desc: *const turbo_backend_model,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let desc = unsafe { *desc };
    if !matches!(desc.family, TURBO_FAMILY_BERT | TURBO_FAMILY_ROBERTA | TURBO_FAMILY_STATIC | TURBO_FAMILY_DEBERTA_V2)
    {
        let msg = format!("family {}: the cpu holds BERT and DeBERTa-v2 encoders and static models", desc.family);
        return unsafe { refuse(err, UNSUPPORTED, &msg) };
    }
    if desc.family == TURBO_FAMILY_DEBERTA_V2 {
        let need = std::mem::offset_of!(turbo_backend_model, max_relative_positions) + size_of::<u32>();
        if (desc.struct_size as usize) < need {
            let msg = format!("turbo_backend_model of {} bytes: a DeBERTa-v2 model needs {need}", desc.struct_size);
            return unsafe { refuse(err, INVALID_STRUCT_SIZE, &msg) };
        }
        if desc.tensor_count as u64 != super::deberta::tensor_count(desc.layers) || desc.tensors.is_null() {
            let msg = format!(
                "tensor_count {}: a DeBERTa-v2 of {} layers has {}",
                desc.tensor_count,
                desc.layers,
                super::deberta::tensor_count(desc.layers)
            );
            return unsafe { refuse(err, INVALID_ARGUMENT, &msg) };
        }
    }
    let tensors = unsafe { std::slice::from_raw_parts(desc.tensors, desc.tensor_count as usize) }
        .iter()
        .map(|t| turbo_backend_tensor { name: std::ptr::null(), ..*t })
        .collect();
    let desc = turbo_backend_model { tensors: std::ptr::null(), ..desc };
    unsafe {
        *out = Box::into_raw(Box::new(Model {
            desc,
            tensors,
            f32: OnceLock::new(),
            packed: OnceLock::new(),
            packing: Mutex::new(()),
            table: OnceLock::new(),
            quantized: OnceLock::new(),
            deberta: OnceLock::new(),
        })) as *mut c_void
    };
    0
}

pub(super) unsafe extern "C" fn model_release(model: *mut c_void) {
    drop(unsafe { Box::from_raw(model as *mut Model) });
}

/// Where each tensor of a model this backend loaded is read from, in the
/// order it was handed them.
///
/// # Safety
/// `model` is one model_load returned and model_release has not taken.
#[cfg(feature = "internals")]
pub(crate) unsafe fn tensor_data(model: *mut c_void) -> Vec<*const c_void> {
    let m = unsafe { &*(model as *const Model) };
    debug_assert_eq!(m.desc.tensor_count as usize, m.tensors.len());
    m.tensors.iter().map(|t| t.data).collect()
}

/// Where the F32 copy of each tensor of a model this backend loaded is,
/// if one was made.
///
/// # Safety
/// As for tensor_data.
#[cfg(feature = "internals")]
pub(crate) unsafe fn converted_data(model: *mut c_void) -> Option<Vec<*const c_void>> {
    let m = unsafe { &*(model as *const Model) };
    m.f32.get().map(|c| c.iter().map(|t| t.as_ptr() as *const c_void).collect())
}
