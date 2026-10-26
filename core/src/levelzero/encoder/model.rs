//! Models: the weights on the device, and the copies sessions compute from.

use std::ffi::c_void;
use std::sync::Mutex;

use super::staging::Staging;
use super::*;
use crate::backend::{
    TURBO_BERT_LAYER_TENSORS, TURBO_FAMILY_BERT, TURBO_FAMILY_DEBERTA_V2, TURBO_FAMILY_ROBERTA, turbo_backend_model,
};
use crate::levelzero::gpu::{Arg, Context, LOG_DEBUG, Res, fail, guarded, quietly};
use crate::status::{INVALID_ARGUMENT, UNSUPPORTED};
use crate::{TURBO_DTYPE_F16, TURBO_DTYPE_F32, turbo_error};

// ---- Models --------------------------------------------------------------------

pub(crate) struct Model {
    pub(super) ctx: *const Context,
    pub(super) desc: turbo_backend_model,
    /// Every tensor as stored, in one allocation, and where each starts.
    pub(super) stored: *mut c_void,
    pub(super) offsets: Vec<usize>,
    pub(super) counts: Vec<u64>,
    /// The weights sessions compute from, made by the first one.
    pub(super) f32: Mutex<Option<Weights>>,
    /// The linear layers' weights in F16, for the matrix engines, made by
    /// the first session at FASTEST.
    pub(super) f16: Mutex<Option<Transposed>>,
    /// The same in F32, for the sessions that compute in F32.
    pub(super) f32t: Mutex<Option<Transposed>>,
}

/// Each layer's linear weights transposed to [n_in, n_out] for the 2D
/// block reads, in F16 for the matrix engines or in F32, in one
/// allocation: the fused Q, K and V, the attention output, and the
/// feed-forward input and output.
#[derive(Clone)]
pub(super) struct Transposed {
    pub(super) layers: Vec<[u64; 4]>,
    pub(super) alloc: *mut c_void,
    /// With oneDNN (levelzero-onednn): each layer's four weights packed in
    /// oneDNN's layout, in one allocation; empty without it.
    #[cfg_attr(not(feature = "levelzero-onednn"), allow(dead_code))]
    pub(super) packed: Vec<[u64; 4]>,
    #[cfg_attr(not(feature = "levelzero-onednn"), allow(dead_code))]
    pub(super) packed_alloc: *mut c_void,
}

/// A model's weights in F32 on the device: every tensor, into stored for
/// an F32 model and into a widened copy for an F16 or BF16 one; and each
/// layer's Q, K and V weights and biases side by side, [3 * hidden,
/// hidden] and [3 * hidden], for one projection.
#[derive(Clone)]
pub(super) struct Weights {
    pub(super) tensors: Vec<u64>,
    pub(super) qkv: Vec<(u64, u64)>,
    /// The widened copy; null for an F32 model.
    pub(super) widened: *mut c_void,
    pub(super) fused: *mut c_void,
    /// A DeBERTa-v2 model's relative rows through each layer's fused
    /// projection (deberta.rs), and their allocation; empty and null for
    /// any other family.
    pub(super) relative: Vec<u64>,
    pub(super) relative_alloc: *mut c_void,
}

// The context outlives the model (the core releases models first); the
// device memory is the driver's, for any thread.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    #[inline]
    pub(super) fn ctx(&self) -> &Context {
        unsafe { &*self.ctx }
    }

    /// The weights in F32, made on first need.
    pub(super) fn f32_weights(&self) -> Res<Weights> {
        let mut f = self.f32.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(w) = f.as_ref() {
            return Ok(w.clone());
        }
        let c = self.ctx();
        let base = self.stored as u64;
        let (tensors, widened) = if self.desc.dtype == TURBO_DTYPE_F32 {
            (self.offsets.iter().map(|&o| base + o as u64).collect::<Vec<_>>(), std::ptr::null_mut())
        } else {
            self.widen()?
        };
        let fused = match self.fuse(&tensors) {
            Ok(p) => p,
            Err(e) => {
                if !widened.is_null() {
                    c.free(widened, "the widened weights");
                }
                return Err(e);
            }
        };
        let (h, layers) = (self.desc.hidden as u64, self.desc.layers as u64);
        let per_layer = 3 * h * h * 4 + 3 * h * 4;
        let qkv: Vec<_> =
            (0..layers).map(|l| (fused as u64 + l * per_layer, fused as u64 + l * per_layer + 3 * h * h * 4)).collect();
        let (relative, relative_alloc) = if self.desc.family == TURBO_FAMILY_DEBERTA_V2 {
            match self.relative(&tensors, &qkv) {
                Ok(r) => r,
                Err(e) => {
                    c.free(fused, "the fused projections");
                    if !widened.is_null() {
                        c.free(widened, "the widened weights");
                    }
                    return Err(e);
                }
            }
        } else {
            (Vec::new(), std::ptr::null_mut())
        };
        let w = Weights { tensors, qkv, widened, fused, relative, relative_alloc };
        *f = Some(w.clone());
        Ok(w)
    }

    /// The linear layers' weights in F16, made on first need from the F32
    /// ones, and packed for oneDNN where it runs. The lock is held from the
    /// look to the store, so sessions made at once make them once.
    pub(super) fn f16_weights(&self, w: &Weights) -> Res<Transposed> {
        let mut slot = self.f16.lock().unwrap_or_else(|p| p.into_inner());
        let t = self.transposed(w, &mut slot, "narrow_f16_transposed", 2, "the F16 weights")?;
        #[cfg(feature = "levelzero-onednn")]
        if t.packed.is_empty() {
            return self.packed_for_onednn(t, &mut slot);
        }
        Ok(t)
    }

    /// The F16 weights packed in oneDNN's layout as well, kept in `slot`;
    /// the Transposed without them where oneDNN is not open.
    #[cfg(feature = "levelzero-onednn")]
    pub(super) fn packed_for_onednn(&self, t: Transposed, slot: &mut Option<Transposed>) -> Res<Transposed> {
        let c = self.ctx();
        let Some(dn) = c.dnnl() else { return Ok(t) };
        let d = &self.desc;
        let (h, i) = (d.hidden, d.intermediate);
        // (k, n) of each layer's four weights, as transposed() orders them.
        let shapes = [(h, 3 * h), (h, h), (h, i), (i, h)];
        let mut at = Vec::with_capacity(t.layers.len() * 4);
        let mut total = 0usize;
        for _ in &t.layers {
            for &(k, n) in &shapes {
                at.push(total as u64);
                total += round_up(dn.weights_bytes(k, n)?, DEVICE_ALIGN);
            }
        }
        let alloc = c.alloc_device(total)?;
        let packed = (|| {
            // oneDNN's handle is the context's, used under its queue lock.
            let _q = c.lock_queue()?;
            for (l, layer) in t.layers.iter().enumerate() {
                for (j, &(k, n)) in shapes.iter().enumerate() {
                    dn.pack(layer[j], k, n, alloc as u64 + at[l * 4 + j])?;
                }
            }
            Ok(())
        })();
        if let Err(e) = packed {
            c.free(alloc, "the packed F16 weights");
            return Err(e);
        }
        let packed = at.chunks(4).map(|a| [0, 1, 2, 3].map(|j| alloc as u64 + a[j])).collect();
        let done = Transposed { layers: t.layers, alloc: t.alloc, packed, packed_alloc: alloc };
        *slot = Some(done.clone());
        Ok(done)
    }

    /// The linear layers' weights in F32 transposed, made on first need,
    /// for linear_sgemm.
    pub(super) fn f32_transposed(&self, w: &Weights) -> Res<Transposed> {
        let mut slot = self.f32t.lock().unwrap_or_else(|p| p.into_inner());
        self.transposed(w, &mut slot, "transpose_f32", 4, "the transposed F32 weights")
    }

    /// The linear layers' weights transposed to [n_in, n_out] by `kernel`,
    /// `bytes` a value, kept in `slot`, whose lock the caller holds, for
    /// the sessions after the first.
    pub(super) fn transposed(
        &self,
        w: &Weights,
        slot: &mut Option<Transposed>,
        kernel: &str,
        bytes: u64,
        what: &str,
    ) -> Res<Transposed> {
        if let Some(h) = slot.as_ref() {
            return Ok(h.clone());
        }
        let c = self.ctx();
        let d = &self.desc;
        let (h, i) = (d.hidden as u64, d.intermediate as u64);
        let base = deberta::layer_base(d);
        let t = |l: u32, r: u32| w.tensors[(base + l * TURBO_BERT_LAYER_TENSORS + r) as usize];
        // Each layer's four weights: where they are in F32, and their
        // outputs and inputs.
        let mut parts = Vec::with_capacity(4 * d.layers as usize);
        for l in 0..d.layers {
            parts.push((w.qkv[l as usize].0, 3 * h, h));
            parts.push((t(l, ATTN_OUT_WEIGHT), h, h));
            parts.push((t(l, FFN_IN_WEIGHT), i, h));
            parts.push((t(l, FFN_OUT_WEIGHT), h, i));
        }
        let mut at = Vec::with_capacity(parts.len());
        let mut total = 0usize;
        for &(_, n_out, n_in) in &parts {
            at.push(total as u64);
            total += round_up((n_out * n_in * bytes) as usize, DEVICE_ALIGN);
        }
        let alloc = c.alloc_device(total)?;
        let narrowed = (|| {
            let k = c.kernel(kernel, [WIDE, 1, 1])?;
            let mut q = c.lock_queue()?;
            let appended = (|| {
                for (p, &(src, n_out, n_in)) in parts.iter().enumerate() {
                    let args =
                        [Arg::Ptr(src), Arg::I32(n_out as i32), Arg::I32(n_in as i32), Arg::Ptr(alloc as u64 + at[p])];
                    let groups = [elementwise_groups(n_out * n_in), 1, 1];
                    k.launch(c, &mut q, kernel, &args, groups)?;
                }
                Ok(())
            })();
            let synced = c.sync(&mut q);
            appended?;
            synced
        })();
        if let Err(e) = narrowed {
            c.free(alloc, what);
            return Err(e);
        }
        let layers = at.chunks(4).map(|a| [0, 1, 2, 3].map(|j| alloc as u64 + a[j])).collect();
        let half = Transposed { layers, alloc, packed: Vec::new(), packed_alloc: std::ptr::null_mut() };
        *slot = Some(half.clone());
        Ok(half)
    }

    /// An F16 or BF16 model's tensors widened to F32 in a new allocation.
    pub(super) fn widen(&self) -> Res<(Vec<u64>, *mut c_void)> {
        let c = self.ctx();
        let base = self.stored as u64;
        let mut at = Vec::with_capacity(self.counts.len());
        let mut total = 0usize;
        for &n in &self.counts {
            at.push(total);
            total += round_up(n as usize * 4, DEVICE_ALIGN);
        }
        let wide = c.alloc_device(total)?;
        let name = if self.desc.dtype == TURBO_DTYPE_F16 { "widen_f16" } else { "widen_bf16" };
        let widened = (|| {
            let k = c.kernel(name, [WIDE, 1, 1])?;
            let mut q = c.lock_queue()?;
            // The queue is left idle whether or not every launch was
            // appended, before a failure frees what they write to.
            let appended = (|| {
                for (i, &n) in self.counts.iter().enumerate() {
                    let args =
                        [Arg::Ptr(base + self.offsets[i] as u64), Arg::U64(n), Arg::Ptr(wide as u64 + at[i] as u64)];
                    k.launch(c, &mut q, name, &args, [elementwise_groups(n), 1, 1])?;
                }
                Ok(())
            })();
            let synced = c.sync(&mut q);
            appended?;
            synced
        })();
        if let Err(e) = widened {
            c.free(wide, "the widened weights");
            return Err(e);
        }
        Ok((at.iter().map(|&a| wide as u64 + a as u64).collect(), wide))
    }

    /// Each layer's Q, K and V weights, then their biases, copied side by
    /// side on the device from the F32 tensors.
    pub(super) fn fuse(&self, tensors: &[u64]) -> Res<*mut c_void> {
        let c = self.ctx();
        let (h, layers) = (self.desc.hidden as usize, self.desc.layers as usize);
        let (weight, bias) = (h * h * 4, h * 4);
        let fused = c.alloc_device(layers * 3 * (weight + bias))?;
        let base = deberta::layer_base(&self.desc);
        let copied = (|| {
            let mut q = c.lock_queue()?;
            let appended = (|| {
                for l in 0..layers {
                    let dst = fused as usize + l * 3 * (weight + bias);
                    let t = |r: u32| tensors[(base + l as u32 * TURBO_BERT_LAYER_TENSORS + r) as usize];
                    for (i, (w, b)) in
                        [(Q_WEIGHT, Q_BIAS), (K_WEIGHT, K_BIAS), (V_WEIGHT, V_BIAS)].into_iter().enumerate()
                    {
                        unsafe {
                            c.copy(&mut q, (dst + i * weight) as *mut c_void, t(w) as usize as *const c_void, weight)?;
                            c.copy(
                                &mut q,
                                (dst + 3 * weight + i * bias) as *mut c_void,
                                t(b) as usize as *const c_void,
                                bias,
                            )?;
                        }
                    }
                }
                Ok(())
            })();
            let synced = c.sync(&mut q);
            appended?;
            synced
        })();
        if let Err(e) = copied {
            c.free(fused, "the fused projections");
            return Err(e);
        }
        Ok(fused)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let c = unsafe { &*self.ctx };
        let f = self.f32.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(w) = f.take() {
            if !w.widened.is_null() {
                c.free(w.widened, "the widened weights");
            }
            c.free(w.fused, "the fused projections");
            if !w.relative_alloc.is_null() {
                c.free(w.relative_alloc, "the relative projections");
            }
        }
        let h = self.f16.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(h) = h.take() {
            if !h.packed_alloc.is_null() {
                c.free(h.packed_alloc, "the packed F16 weights");
            }
            c.free(h.alloc, "the F16 weights");
        }
        let t = self.f32t.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(t) = t.take() {
            c.free(t.alloc, "the transposed F32 weights");
        }
        if !self.stored.is_null() {
            c.free(self.stored, "the weights");
        }
    }
}

pub(crate) unsafe extern "C" fn model_load(
    ctx: *mut c_void,
    desc: *const turbo_backend_model,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let (c, d) = (&*(ctx as *const Context), &*desc);
            if d.family == TURBO_FAMILY_DEBERTA_V2 {
                deberta::check_load(d)?;
            } else if d.family != TURBO_FAMILY_BERT && d.family != TURBO_FAMILY_ROBERTA {
                return Err(fail(
                    UNSUPPORTED,
                    format!("family {}: the levelzero backend holds BERT and DeBERTa-v2 encoders", d.family),
                ));
            }
            if d.heads == 0 || d.hidden % d.heads != 0 {
                return Err(fail(UNSUPPORTED, format!("hidden {} is not a multiple of heads {}", d.hidden, d.heads)));
            }
            // Each layer's tensors are read by index when a session is made:
            // the table holds the embeddings and every layer's, no fewer.
            let expected = deberta::layer_base(d) as u64 + d.layers as u64 * TURBO_BERT_LAYER_TENSORS as u64;
            if d.tensor_count as u64 != expected || d.tensors.is_null() {
                return Err(fail(
                    INVALID_ARGUMENT,
                    format!(
                        "tensor_count {}: a BERT of {} layers is {expected} tensors, the embeddings and each layer's",
                        d.tensor_count, d.layers
                    ),
                ));
            }
            let elem = if d.dtype == TURBO_DTYPE_F32 { 4 } else { 2 };
            let tensors = std::slice::from_raw_parts(d.tensors, d.tensor_count as usize);
            let mut offsets = Vec::with_capacity(tensors.len());
            let mut counts = Vec::with_capacity(tensors.len());
            let mut total = 0usize;
            for t in tensors {
                offsets.push(total);
                counts.push(t.bytes / elem);
                total += round_up(t.bytes as usize, DEVICE_ALIGN);
            }
            let stored = c.alloc_device(total)?;
            // From here the model's Drop frees what it holds.
            let m = Box::new(Model {
                ctx: c,
                desc: turbo_backend_model { tensors: std::ptr::null(), ..*d },
                stored,
                offsets,
                counts,
                f32: Mutex::new(None),
                f16: Mutex::new(None),
                f32t: Mutex::new(None),
            });
            {
                // The weights go through staging the context allocated, not
                // straight from the caller's pages (a mapped bundle): the
                // driver would pin every page of them until the queue
                // synchronized, and a second load at once ran it out of
                // memory (#1355).
                let stage = Staging::new(c)?;
                let mut q = c.lock_queue()?;
                // The queue is left idle whether or not every copy was
                // appended, before a failure frees what they write to.
                let appended = (|| {
                    let mut s = stage.fill();
                    for (i, t) in tensors.iter().enumerate() {
                        let dst = (stored as *mut u8).add(m.offsets[i]);
                        let src = std::slice::from_raw_parts(t.data as *const u8, t.bytes as usize);
                        s.copy(c, &mut q, dst, src)?;
                    }
                    Ok(())
                })();
                let synced = c.sync(&mut q);
                drop(q);
                drop(stage);
                appended?;
                synced?;
            }
            c.say(
                LOG_DEBUG,
                &format!(
                    "levelzero device {}: a {} of {} layers, {total} bytes of {} weights",
                    c.ordinal,
                    if d.family == TURBO_FAMILY_DEBERTA_V2 { "DeBERTa-v2" } else { "BERT" },
                    d.layers,
                    dtype_name(d.dtype)
                ),
            );
            *out = Box::into_raw(m) as *mut c_void;
            Ok(())
        })
    }
}

/// The device address of the F32 copy of an F16 or BF16 model's weights,
/// once a session made it.
///
/// # Safety
/// `model` is one model_load returned, not yet released.
#[cfg(feature = "internals")]
pub(crate) unsafe fn widened(model: *mut c_void) -> Option<*const c_void> {
    let m = unsafe { &*(model as *const Model) };
    let f = m.f32.lock().unwrap_or_else(|p| p.into_inner());
    f.as_ref().map(|w| w.widened as *const c_void).filter(|p| !p.is_null())
}

pub(crate) unsafe extern "C" fn model_release(model: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(model as *mut Model) }));
}
