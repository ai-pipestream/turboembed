//! The encoder's forward and backward on a CUDA device (#1519): the
//! weights, the activations one forward keeps and the gradients live on
//! the device, the kernels of trainer/cuda/train.cu run the ops of the
//! CPU encoder one for one in the order `layer.rs` and `pass.rs` run
//! them, and only the token ids, the states and the states' gradient
//! cross to and from the host for a sequence. The gradients add up on
//! the device over as many sequences as the caller runs, and come to
//! the host at once. Dropout draws the CPU's masks from the same keys;
//! bfloat16 compute rounds at the CPU emulation's points. What comes
//! back is held to the CPU encoder within the tolerance the tests
//! state: the sums inside each op run in another order. Built for the
//! `cuda` feature (the device) or `cuda-shim` (the same kernels as C++
//! on the host).

use std::ffi::{CStr, c_void};
use std::ptr;

use turbo::status::{Result, invalid};

use super::{Compute, Dropout, Encoder, Geometry, Lora};

mod ffi;
mod pass;

pub use pass::Timing;

/// Turns a CUDA error number into the trainer's error.
fn check(error: i32) -> Result<()> {
    if error == 0 {
        return Ok(());
    }
    let what = unsafe { CStr::from_ptr(ffi::tt_error_string(error)) }.to_string_lossy();
    Err(invalid(format!("cuda: {what} ({error})")))
}

/// Device memory of `bytes` bytes, freed with it.
pub(super) struct Buf {
    ptr: *mut c_void,
    bytes: usize,
}

impl Buf {
    fn new(bytes: usize) -> Result<Buf> {
        let mut ptr = ptr::null_mut();
        check(unsafe { ffi::tt_malloc(bytes.max(1), &mut ptr) })?;
        Ok(Buf { ptr, bytes })
    }

    /// Room for `n` f32.
    fn floats(n: usize) -> Result<Buf> {
        Self::new(n * 4)
    }

    /// Room for `n` f64.
    fn doubles(n: usize) -> Result<Buf> {
        Self::new(n * 8)
    }

    fn f32(&self) -> *mut f32 {
        self.ptr.cast()
    }

    fn f64(&self) -> *mut f64 {
        self.ptr.cast()
    }

    /// The f32 the buffer holds.
    fn len(&self) -> usize {
        self.bytes / 4
    }

    fn upload<T: Copy>(&self, src: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.bytes {
            return Err(invalid("cuda: an upload past the buffer"));
        }
        check(unsafe { ffi::tt_upload(self.ptr, src.as_ptr().cast(), bytes) })
    }

    fn download(&self, dst: &mut [f32]) -> Result<()> {
        if dst.len() * 4 > self.bytes {
            return Err(invalid("cuda: a download past the buffer"));
        }
        check(unsafe { ffi::tt_download(dst.as_mut_ptr().cast(), self.ptr, dst.len() * 4) })
    }

    fn zero(&self, floats: usize) -> Result<()> {
        check(unsafe { ffi::tt_zero(self.ptr, floats * 4) })
    }

    fn copy_from(&self, src: &Buf, floats: usize) -> Result<()> {
        check(unsafe { ffi::tt_copy(self.ptr, src.ptr, floats * 4) })
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        // A failed free leaves nothing to do; the device keeps the error.
        unsafe { ffi::tt_free(self.ptr) };
    }
}

/// An encoder on the device: the master weights' copy (rounded under
/// bfloat16), the gradients, and the workspace of the last forward.
pub struct CudaEncoder {
    geo: Geometry,
    vocab: usize,
    lora: Option<Lora>,
    lora_from: usize,
    frozen: bool,
    checkpoint: bool,
    dropout: Option<Dropout>,
    compute: Compute,
    lens: Vec<usize>,
    weights: Vec<Buf>,
    /// One a slot; a frozen base slot's holds nothing.
    grads: Vec<Buf>,
    /// The rotary table, `[2 (global, local)][2 (cos, sin)][positions][pairs]`.
    table: Buf,
    positions: usize,
    pairs: usize,
    work: pass::Workspace,
}

impl CudaEncoder {
    /// The devices the runtime lists.
    pub fn device_count() -> Result<usize> {
        let mut n = 0;
        check(unsafe { ffi::tt_device_count(&mut n) })?;
        Ok(n.max(0) as usize)
    }

    /// `e` on device `device`: its weights as its forward reads them, its
    /// adapters, dropout and compute, and zero gradients.
    pub fn new(e: &Encoder<f32>, device: i32) -> Result<CudaEncoder> {
        check(unsafe { ffi::tt_set_device(device) })?;
        let weights = e
            .weights()
            .iter()
            .map(|p| {
                let b = Buf::floats(p.len())?;
                b.upload(p)?;
                Ok(b)
            })
            .collect::<Result<Vec<_>>>()?;
        let grads = e
            .grads()
            .iter()
            .map(|g| {
                let b = Buf::floats(g.len())?;
                b.zero(g.len())?;
                Ok(b)
            })
            .collect::<Result<Vec<_>>>()?;
        let table = Buf::floats(e.table.data().len())?;
        table.upload(e.table.data())?;
        let mut out = CudaEncoder {
            geo: e.geo,
            vocab: e.vocab,
            lora: e.lora,
            lora_from: e.lora_from,
            frozen: e.frozen,
            checkpoint: e.checkpoint,
            dropout: e.dropout,
            compute: e.compute,
            lens: e.lens.clone(),
            weights,
            grads,
            table,
            positions: e.table.positions(),
            pairs: e.table.pairs(),
            work: pass::Workspace::default(),
        };
        out.work.ensure(&out.geo, out.lora, out.checkpoint, 1)?;
        Ok(out)
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geo
    }

    pub fn compute(&self) -> Compute {
        self.compute
    }

    pub fn dropout(&self) -> Option<Dropout> {
        self.dropout
    }

    pub fn frozen(&self) -> bool {
        self.frozen
    }

    pub fn is_checkpointed(&self) -> bool {
        self.checkpoint
    }

    /// How many layers' inner workspaces are held on the device.
    pub fn buffered_layers(&self) -> usize {
        self.work.buffered_layers()
    }

    /// Replaces slot `s`'s weights with `p`, which must keep the length,
    /// rounded under bfloat16 as the CPU's shadow is.
    pub fn put_slot(&mut self, s: usize, p: &[f32]) -> Result<()> {
        if s >= self.lens.len() || p.len() != self.lens[s] {
            return Err(invalid("cuda: the weights do not match the slot's"));
        }
        self.weights[s].upload(p)?;
        self.store(&self.weights[s], p.len())
    }

    /// Zeroes the gradients.
    pub fn zero_grads(&self) -> Result<()> {
        for g in &self.grads {
            g.zero(g.len())?;
        }
        Ok(())
    }

    /// The gradients, into `g` shaped as [`Encoder::grads`] gives them.
    pub fn grads(&self, g: &mut [Vec<f32>]) -> Result<()> {
        if g.len() != self.grads.len() || g.iter().zip(&self.grads).any(|(a, b)| a.len() != b.len()) {
            return Err(invalid("cuda: the gradients are not shaped as the weights"));
        }
        self.sync()?;
        for (dst, src) in g.iter_mut().zip(&self.grads) {
            src.download(dst)?;
        }
        Ok(())
    }

    /// The gradients, shaped as [`Encoder::grads`] gives them.
    pub fn grads_vec(&self) -> Result<Vec<Vec<f32>>> {
        let mut g: Vec<Vec<f32>> = self.grads.iter().map(|b| vec![0.0; b.len()]).collect();
        self.grads(&mut g)?;
        Ok(g)
    }

    /// Waits for every launch so far.
    pub fn sync(&self) -> Result<()> {
        check(unsafe { ffi::tt_sync() })
    }

    /// Rounds `b`'s first `count` values to bfloat16 under that compute.
    fn store(&self, b: &Buf, count: usize) -> Result<()> {
        if self.compute == Compute::Bf16 {
            check(unsafe { ffi::tt_round_bf16(b.f32(), count) })?;
        }
        Ok(())
    }

    /// The adapters of layer `l`, when it has them.
    fn adapter(&self, l: usize) -> Option<Lora> {
        self.lora.filter(|_| l >= self.lora_from)
    }

    /// The keys each side of a query on layer `l`, -1 for all of them.
    fn window(&self, l: usize) -> i32 {
        if self.geo.global(l) { -1 } else { self.geo.window as i32 }
    }

    /// Refuses a sequence the encoder cannot run.
    fn check_ids(&self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() || ids.len() > self.geo.max_positions || ids.len() > self.positions {
            return Err(invalid(format!("encoder: {} tokens, of 1 to {}", ids.len(), self.geo.max_positions)));
        }
        if let Some(id) = ids.iter().find(|&&id| id as usize >= self.vocab) {
            return Err(invalid(format!("encoder: token id {id} is past the {} words", self.vocab)));
        }
        Ok(())
    }
}
