//! The Metal side of [`super::lstm`]: core/metal/backend.mm's LSTM calls
//! (kernels.metal, LSTM). The weights and a table stay in Metal buffers on
//! the device's unified memory; a [`Work`] holds its own command queue and
//! buffers, so runs on different workspaces may run at once on one LSTM.

use std::ffi::{c_char, c_int};
use std::ptr;

use crate::status::{DEVICE_NOT_FOUND, Error, INVALID_ARGUMENT, OUT_OF_MEMORY, RUNTIME, Result};

#[repr(C)]
struct RawLstm {
    _private: [u8; 0],
}

#[repr(C)]
struct RawTable {
    _private: [u8; 0],
}

#[repr(C)]
struct RawWork {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn turbo_metal_lstm_new(
        ordinal: u32,
        input: c_int,
        hidden: c_int,
        dirs: c_int,
        w: *const f32,
        u: *const f32,
        b: *const f32,
        reverse: *const c_int,
        out: *mut *mut RawLstm,
        msg: *mut c_char,
        len: usize,
    ) -> c_int;
    fn turbo_metal_lstm_free(l: *mut RawLstm);
    fn turbo_metal_lstm_table_new(
        l: *const RawLstm,
        values: *const f32,
        count: usize,
        out: *mut *mut RawTable,
        msg: *mut c_char,
        len: usize,
    ) -> c_int;
    fn turbo_metal_lstm_table_free(t: *mut RawTable);
    fn turbo_metal_lstm_work_free(w: *mut RawWork);
    fn turbo_metal_lstm_run(
        l: *const RawLstm,
        work: *mut *mut RawWork,
        x: *const f32,
        table: *const RawTable,
        index: *const i64,
        lengths: *const i64,
        batch: i64,
        rows: i64,
        out: *mut f32,
        bad_row: *mut i64,
        msg: *mut c_char,
        len: usize,
    ) -> c_int;
}

/// backend.mm's LSTM statuses.
const NOT_FINITE: c_int = 1;
const NO_DEVICE: c_int = 2;
const NO_MEMORY: c_int = 3;

fn status(code: c_int, msg: &[u8], bad_row: i64) -> Result<()> {
    let text = || {
        let end = msg.iter().position(|&b| b == 0).unwrap_or(msg.len());
        format!("lstm on Metal: {}", String::from_utf8_lossy(&msg[..end]))
    };
    match code {
        0 => Ok(()),
        NOT_FINITE => {
            Err(Error::new(INVALID_ARGUMENT, format!("lstm: a gate or state is not finite at row {bad_row}")))
        }
        NO_DEVICE => Err(Error::new(DEVICE_NOT_FOUND, text())),
        NO_MEMORY => Err(Error::new(OUT_OF_MEMORY, text())),
        _ => Err(Error::new(RUNTIME, text())),
    }
}

/// An LSTM's weights on one GPU.
pub(super) struct Lstm(*mut RawLstm);

// The weights are written once, when made, and only read after; every run
// queues on its own workspace's command queue.
unsafe impl Send for Lstm {}
unsafe impl Sync for Lstm {}

/// A gathered run's projections on the GPU of the LSTM they were made for.
pub(super) struct Table(*mut RawTable);

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

/// A run's command queue and buffers, made on first use.
pub(super) struct Work(*mut RawWork);

unsafe impl Send for Work {}

impl Default for Work {
    fn default() -> Work {
        Work(ptr::null_mut())
    }
}

fn int(v: usize, what: &str) -> Result<c_int> {
    c_int::try_from(v).map_err(|_| Error::new(INVALID_ARGUMENT, format!("lstm on Metal: {what} {v} is past i32")))
}

impl Lstm {
    /// Checked weights, as the CPU form packs them: every direction's W
    /// stacked, each U, the biases one after another.
    pub(super) fn new(
        ordinal: u32,
        input: usize,
        hidden: usize,
        w: &[f32],
        u: &[f32],
        b: &[f32],
        reverse: &[c_int],
    ) -> Result<Lstm> {
        let (input, hidden, dirs) = (int(input, "input")?, int(hidden, "hidden")?, int(reverse.len(), "directions")?);
        let (mut raw, mut msg) = (ptr::null_mut(), [0u8; 256]);
        // The shapes were checked by the CPU form's constructor.
        let code = unsafe {
            turbo_metal_lstm_new(
                ordinal,
                input,
                hidden,
                dirs,
                w.as_ptr(),
                u.as_ptr(),
                b.as_ptr(),
                reverse.as_ptr(),
                &mut raw,
                msg.as_mut_ptr().cast(),
                msg.len(),
            )
        };
        status(code, &msg, 0).map(|()| Lstm(raw))
    }

    pub(super) fn table(&self, values: &[f32]) -> Result<Table> {
        let (mut raw, mut msg) = (ptr::null_mut(), [0u8; 256]);
        let code = unsafe {
            turbo_metal_lstm_table_new(
                self.0,
                values.as_ptr(),
                values.len(),
                &mut raw,
                msg.as_mut_ptr().cast(),
                msg.len(),
            )
        };
        status(code, &msg, 0).map(|()| Table(raw))
    }

    /// A checked run: `x` the rows, or else `table` read by `index`.
    pub(super) fn run(
        &self,
        x: Option<&[f32]>,
        gathered: Option<(&Table, &[i64])>,
        lengths: &[i64],
        rows: usize,
        out: &mut [f32],
        w: &mut Work,
    ) -> Result<()> {
        let (table, index) = gathered.map_or((ptr::null(), ptr::null()), |(t, i)| (t.0.cast_const(), i.as_ptr()));
        let (mut bad, mut msg) = (0i64, [0u8; 256]);
        let code = unsafe {
            turbo_metal_lstm_run(
                self.0,
                &mut w.0,
                x.map_or(ptr::null(), <[f32]>::as_ptr),
                table,
                index,
                lengths.as_ptr(),
                lengths.len() as i64,
                rows as i64,
                out.as_mut_ptr(),
                &mut bad,
                msg.as_mut_ptr().cast(),
                msg.len(),
            )
        };
        status(code, &msg, bad)
    }
}

impl Drop for Lstm {
    fn drop(&mut self) {
        unsafe { turbo_metal_lstm_free(self.0) }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        unsafe { turbo_metal_lstm_table_free(self.0) }
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        unsafe { turbo_metal_lstm_work_free(self.0) }
    }
}
