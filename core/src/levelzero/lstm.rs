//! The batched LSTM of turbo::device::lstm on an Intel GPU (#860): the
//! kernels lstm_project and lstm_step of core/levelzero/encoder.cl on one
//! context per device, which every LSTM on that device shares, so a table
//! one LSTM made is memory of the context another one reads it in. The
//! weights and a table stay in device memory; a run makes every buffer it
//! needs first, then stages its rows, lengths and starts in pinned memory,
//! appends the copies, the projection and one launch per step to the
//! context's in-order queue under its lock, and waits once for the states.
//!
//! Anything appended is waited for before its memory can be freed or
//! written again: an append or a wait that fails is followed by a wait
//! for what was appended before it, and when even that fails the context
//! is lost, and nothing in it is freed again (its memory, kernels and the
//! context itself are kept to the end of the process).
//!
//! The caller (core/src/device/lstm.rs) has checked shapes, lengths and
//! inputs; a non-finite gate or state comes back as the least row it was
//! found at.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::gpu::{self, Arg, Context, Fail, Kernel};
use crate::status::{DEVICE_NOT_FOUND, DEVICE_UNAVAILABLE, Error, INVALID_ARGUMENT, Result};

const TILE: u32 = 16;

fn error(f: Fail) -> Error {
    Error::new(f.code, f.message)
}

/// A device's LSTM context, released when its last user goes unless it
/// was lost.
struct Ctx {
    c: ManuallyDrop<Box<Context>>,
    ordinal: u32,
    /// Work appended to it may still be running: nothing is freed.
    lost: AtomicBool,
}

impl Drop for Ctx {
    fn drop(&mut self) {
        if !self.lost.load(Ordering::Relaxed) {
            unsafe { ManuallyDrop::drop(&mut self.c) };
        }
    }
}

/// The live LSTM context of each device.
static CONTEXTS: Mutex<Vec<Weak<Ctx>>> = Mutex::new(Vec::new());

/// The context of listed device `ordinal`, made when no LSTM holds one.
fn context(ordinal: u32) -> Result<Arc<Ctx>> {
    let mut all = CONTEXTS.lock().unwrap_or_else(|p| p.into_inner());
    all.retain(|w| w.strong_count() > 0);
    if let Some(c) = all.iter().filter_map(Weak::upgrade).find(|c| c.ordinal == ordinal) {
        return Ok(c);
    }
    let d = match super::driver() {
        Ok(Some(d)) => d,
        Ok(None) => return Err(Error::new(DEVICE_NOT_FOUND, "lstm: no levelzero device is listed")),
        Err(m) => return Err(Error::new(DEVICE_UNAVAILABLE, format!("lstm: {m}"))),
    };
    let Some(dev) = d.devices.get(ordinal as usize) else {
        let message = format!("lstm: levelzero device {ordinal}: {} are listed", d.devices.len());
        return Err(Error::new(DEVICE_NOT_FOUND, message));
    };
    let mut raw = std::ptr::null_mut();
    unsafe { gpu::create(d, dev, ordinal, None, std::ptr::null_mut(), &mut raw) }.map_err(error)?;
    let c = ManuallyDrop::new(unsafe { Box::from_raw(raw as *mut Context) });
    let ctx = Arc::new(Ctx { c, ordinal, lost: AtomicBool::new(false) });
    all.push(Arc::downgrade(&ctx));
    Ok(ctx)
}

/// Appends left before an injected failure (`internals`, for the tests):
/// 0 is none.
#[cfg(feature = "internals")]
static FAIL_IN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Fails the LSTM's `n`th append from now, once, as a refused append
/// would; None clears it. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn fail_append(n: Option<u32>) {
    FAIL_IN.store(n.unwrap_or(0), Ordering::Relaxed);
}

/// Whether this append is the one to fail.
fn injected() -> Result<()> {
    #[cfg(feature = "internals")]
    {
        let left = FAIL_IN.load(Ordering::Relaxed);
        if left > 0 {
            FAIL_IN.store(left - 1, Ordering::Relaxed);
            if left == 1 {
                return Err(Error::new(crate::status::RUNTIME, "lstm on levelzero: an injected append failure"));
            }
        }
    }
    Ok(())
}

/// `r`, after waiting for what the queue holds when it is an error; a
/// queue that cannot be waited for loses the context.
fn settle<T>(ctx: &Ctx, q: &mut gpu::Queue, r: Result<T>) -> Result<T> {
    if r.is_err() && ctx.c.sync(q).is_err() {
        ctx.lost.store(true, Ordering::Relaxed);
    }
    r
}

/// Device memory of one context, freed with it.
struct Mem {
    ctx: Arc<Ctx>,
    ptr: *mut c_void,
    bytes: usize,
    pinned: bool,
}

unsafe impl Send for Mem {}
unsafe impl Sync for Mem {}

impl Mem {
    fn new(ctx: &Arc<Ctx>, bytes: usize, pinned: bool) -> Result<Mem> {
        let c = &ctx.c;
        let ptr = if pinned { c.alloc_pinned(bytes) } else { c.alloc_device(bytes) }.map_err(error)?;
        Ok(Mem { ctx: ctx.clone(), ptr, bytes, pinned })
    }

    fn addr(&self) -> Arg {
        Arg::Ptr(self.ptr as u64)
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        if !self.ctx.lost.load(Ordering::Relaxed) {
            self.ctx.c.free(self.ptr, if self.pinned { "LSTM staging" } else { "LSTM device memory" });
        }
    }
}

/// A buffer that grows and is kept, in one placement.
#[derive(Default)]
struct Grown(Option<Mem>);

impl Grown {
    /// The buffer's address, at least `bytes` long, on `ctx`.
    fn fit(&mut self, ctx: &Arc<Ctx>, bytes: usize, pinned: bool) -> Result<*mut c_void> {
        if self.0.as_ref().is_none_or(|m| m.bytes < bytes || !Arc::ptr_eq(&m.ctx, ctx)) {
            self.0 = None;
            self.0 = Some(Mem::new(ctx, bytes, pinned)?);
        }
        Ok(self.0.as_ref().expect("fitted").ptr)
    }
}

/// An LSTM's weights and kernels on one GPU.
pub(crate) struct Lstm {
    /// Their arguments are set and their launches appended under the
    /// queue's lock, so runs on one LSTM share them. They are dropped
    /// first: the context's module goes with the context.
    project: ManuallyDrop<Kernel>,
    step: ManuallyDrop<Kernel>,
    w: Mem,
    u: Mem,
    b: Mem,
    reverse: Mem,
    input: usize,
    hidden: usize,
    dirs: usize,
    ctx: Arc<Ctx>,
}

impl Drop for Lstm {
    fn drop(&mut self) {
        if !self.ctx.lost.load(Ordering::Relaxed) {
            unsafe {
                ManuallyDrop::drop(&mut self.project);
                ManuallyDrop::drop(&mut self.step);
            }
        }
    }
}

/// A gathered run's projections on the LSTM's GPU.
pub(crate) struct Table(Mem);

/// A run's device buffers and its pinned staging, made on first use.
#[derive(Default)]
pub(crate) struct Work {
    x: Grown,
    index: Grown,
    proj: Grown,
    out: Grown,
    lens: Grown,
    starts: Grown,
    state: Grown,
    cell: Grown,
    bad: Grown,
    up: Grown,
    down: Grown,
}

/// Copies `values` into pinned memory and appends their copy to `dst`.
///
/// # Safety
/// `up` and `dst` stay valid until the queue synchronizes.
unsafe fn upload(
    c: &Context,
    q: &mut gpu::Queue,
    up: *mut u8,
    at: &mut usize,
    dst: *mut c_void,
    bytes: &[u8],
) -> Result<()> {
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), up.add(*at), bytes.len());
        c.copy(q, dst, up.add(*at) as *const c_void, bytes.len()).map_err(error)?;
    }
    *at += bytes.len().next_multiple_of(64);
    Ok(())
}

fn bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl Lstm {
    /// Checked weights as the CPU form packs them, on listed GPU `ordinal`.
    pub(crate) fn new(
        ordinal: u32,
        input: usize,
        hidden: usize,
        w: &[f32],
        u: &[f32],
        b: &[f32],
        reverse: &[i32],
    ) -> Result<Lstm> {
        if i32::try_from(4 * hidden * reverse.len()).is_err() || i32::try_from(input).is_err() {
            return Err(Error::new(INVALID_ARGUMENT, "lstm on levelzero: a width past i32"));
        }
        let ctx = context(ordinal)?;
        let c = &ctx.c;
        let (project, step) = (
            ManuallyDrop::new(c.kernel("lstm_project", [TILE, TILE, 1]).map_err(error)?),
            ManuallyDrop::new(c.kernel("lstm_step", [TILE, TILE, 1]).map_err(error)?),
        );
        let parts: [&[u8]; 4] = [bytes(w), bytes(u), bytes(b), bytes(reverse)];
        let total: usize = parts.iter().map(|p| p.len().next_multiple_of(64)).sum();
        let up = Mem::new(&ctx, total, true)?;
        let mems = parts.iter().map(|p| Mem::new(&ctx, p.len(), false)).collect::<Result<Vec<_>>>()?;
        {
            let mut q = c.lock_queue().map_err(error)?;
            let mut at = 0;
            let sent = parts
                .iter()
                .zip(&mems)
                .try_for_each(|(p, m)| unsafe { upload(c, &mut q, up.ptr as *mut u8, &mut at, m.ptr, p) });
            let sent = sent.and_then(|()| c.sync(&mut q).map_err(error));
            settle(&ctx, &mut q, sent)?;
        }
        drop(up);
        let mut it = mems.into_iter();
        let mut next = || it.next().expect("four weights");
        let (w, u, b, reverse_mem) = (next(), next(), next(), next());
        Ok(Lstm { project, step, w, u, b, reverse: reverse_mem, input, hidden, dirs: reverse.len(), ctx })
    }

    pub(crate) fn table(&self, values: &[f32]) -> Result<Table> {
        let c = &self.ctx.c;
        let (mem, up) = (Mem::new(&self.ctx, values.len() * 4, false)?, Mem::new(&self.ctx, values.len() * 4, true)?);
        let mut q = c.lock_queue().map_err(error)?;
        let sent = unsafe { upload(c, &mut q, up.ptr as *mut u8, &mut 0, mem.ptr, bytes(values)) };
        let sent = sent.and_then(|()| c.sync(&mut q).map_err(error));
        settle(&self.ctx, &mut q, sent)?;
        Ok(Table(mem))
    }

    /// A checked, nonempty run: `x` the rows, or else `table` read by
    /// `index`. Returns the least row with a non-finite gate or state.
    pub(crate) fn run(
        &self,
        x: Option<&[f32]>,
        gathered: Option<(&Table, &[i64])>,
        lengths: &[i64],
        rows: usize,
        out: &mut [f32],
        w: &mut Work,
    ) -> Result<Option<i64>> {
        let (ctx, c) = (&self.ctx, &self.ctx.c);
        let (h, batch) = (self.hidden, lengths.len());
        let (pw, width) = (4 * h * self.dirs, h * self.dirs);
        if i32::try_from(rows).is_err() || i32::try_from(rows * pw).is_err() {
            return Err(Error::new(INVALID_ARGUMENT, format!("lstm on levelzero: {rows} rows past i32")));
        }
        let mut starts = Vec::with_capacity(batch);
        let mut at = 0i64;
        for &n in lengths {
            starts.push(at);
            at += n;
        }
        let state = self.dirs * batch * h * 4;
        let src_bytes = x.map_or(rows * 8, |x| x.len() * 4);
        let up_bytes = src_bytes.next_multiple_of(64) + 2 * (batch * 8).next_multiple_of(64) + 64;
        if let Some((t, _)) = gathered
            && !Arc::ptr_eq(&t.0.ctx, ctx)
        {
            return Err(Error::new(INVALID_ARGUMENT, "lstm on levelzero: a table made in another context"));
        }
        // Every buffer first: nothing is freed or made once work is appended.
        let up = w.up.fit(ctx, up_bytes, true)? as *mut u8;
        let down = w.down.fit(ctx, rows * width * 4 + 64, true)? as *mut u8;
        let outp = w.out.fit(ctx, rows * width * 4, false)?;
        let (lens, startp, badp) =
            (w.lens.fit(ctx, batch * 8, false)?, w.starts.fit(ctx, batch * 8, false)?, w.bad.fit(ctx, 4, false)?);
        let (xp, projp, ip) = match x {
            Some(x) => {
                (w.x.fit(ctx, x.len() * 4, false)?, w.proj.fit(ctx, rows * pw * 4, false)?, std::ptr::null_mut())
            }
            None => (std::ptr::null_mut(), std::ptr::null_mut(), w.index.fit(ctx, rows * 8, false)?),
        };
        let st = w.state.fit(ctx, 2 * state, false)? as u64;
        let cellp = w.cell.fit(ctx, state, false)?;
        let ptr = |p: *mut c_void| Arg::Ptr(p as u64);
        let mut q = c.lock_queue().map_err(error)?;
        let appended = (|| {
            let q = &mut *q;
            let mut used = 0;
            unsafe {
                upload(c, q, up, &mut used, lens, bytes(lengths))?;
                upload(c, q, up, &mut used, startp, bytes(&starts))?;
                upload(c, q, up, &mut used, badp, bytes(&[i32::MAX]))?;
            }
            let (proj, index) = match (x, gathered) {
                (Some(x), _) => {
                    unsafe { upload(c, q, up, &mut used, xp, bytes(x))? };
                    let (n, k) = (Arg::I32(pw as i32), Arg::I32(self.input as i32));
                    let args = [ptr(xp), self.w.addr(), self.b.addr(), ptr(projp), Arg::U64(rows as u64), n, k];
                    let groups = [(pw as u32).div_ceil(TILE), (rows as u32).div_ceil(TILE), 1];
                    self.project.launch(c, q, "lstm_project", &args, groups).map_err(error)?;
                    (ptr(projp), Arg::Ptr(0))
                }
                (None, Some((table, idx))) => {
                    unsafe { upload(c, q, up, &mut used, ip, bytes(idx))? };
                    (table.0.addr(), ptr(ip))
                }
                (None, None) => unreachable!("a run with neither rows nor a table"),
            };
            let states = [Arg::Ptr(st), Arg::Ptr(st + state as u64)];
            let mut active = batch;
            for t in 0..lengths[0] {
                while lengths[active - 1] <= t {
                    active -= 1;
                }
                let args = [
                    proj,
                    index,
                    self.u.addr(),
                    ptr(lens),
                    ptr(startp),
                    states[(t & 1) as usize],
                    states[((t + 1) & 1) as usize],
                    ptr(cellp),
                    ptr(outp),
                    ptr(badp),
                    self.reverse.addr(),
                    Arg::U64(batch as u64),
                    Arg::I32(active as i32),
                    Arg::I32(h as i32),
                    Arg::I32(pw as i32),
                    Arg::I32(width as i32),
                    Arg::I32(t as i32),
                ];
                let groups = [(h as u32).div_ceil(TILE), (active as u32).div_ceil(TILE), self.dirs as u32];
                injected()?;
                self.step.launch(c, q, "lstm_step", &args, groups).map_err(error)?;
            }
            unsafe {
                c.copy(q, down as *mut c_void, outp, rows * width * 4).map_err(error)?;
                c.copy(q, down.add(rows * width * 4) as *mut c_void, badp, 4).map_err(error)?;
            }
            c.sync(q).map_err(error)
        })();
        settle(ctx, &mut q, appended)?;
        drop(q);
        let got = unsafe { std::slice::from_raw_parts(down as *const f32, rows * width) };
        out.copy_from_slice(got);
        let row = unsafe { (down.add(rows * width * 4) as *const i32).read_unaligned() };
        Ok((row != i32::MAX).then_some(i64::from(row)))
    }
}
