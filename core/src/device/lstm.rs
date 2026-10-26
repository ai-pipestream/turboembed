//! [`crate::cpu::lstm::Lstm`] on a [`Device`]: the same packed weights of
//! every direction, the same rows (each sequence's one after another, the
//! longest first), the same outputs and the same refusals. On a GPU the
//! weights and a gathered run's table stay in device memory; a run copies
//! its inputs in and its states out.
//!
//! The CPU form is the reference: a device's states are held to it within
//! a bound its tests name, not to its bits.

use super::Device;
use crate::cpu::lstm as cpu;
use crate::status::{Error, Result, UNSUPPORTED};

pub use crate::cpu::lstm::Weights;

/// Checked weights of one or more directions over one input, on a device.
pub struct Lstm {
    device: Device,
    inner: Inner,
}

enum Inner {
    Cpu(cpu::Lstm),
}

/// Input projections made once per distinct row (a character table, say),
/// where the device reads them, for [`Lstm::run_gathered`].
pub struct Table {
    device: Device,
    rows: usize,
    inner: TableInner,
}

enum TableInner {
    Cpu(Vec<f32>),
}

/// What a run computes in, kept across runs so that a run of the same or
/// smaller shape allocates nothing on the host.
#[derive(Default)]
pub struct Work {
    cpu: cpu::Work,
}

/// A device this library cannot run the op on.
fn unsupported(device: Device) -> Error {
    let why = match device {
        Device::Cpu => "is not built in",
        Device::Cuda(_) => "has no LSTM in this build",
    };
    Error::new(UNSUPPORTED, format!("lstm: device {device:?} {why}"))
}

impl Lstm {
    /// The directions in `dirs` on `device`, as
    /// [`cpu::Lstm::new`](crate::cpu::lstm::Lstm::new) checks them. A
    /// device the library has no LSTM for is refused with UNSUPPORTED.
    pub fn new(device: Device, input: usize, hidden: usize, dirs: &[Weights<'_>]) -> Result<Lstm> {
        let inner = match device {
            Device::Cpu => Inner::Cpu(cpu::Lstm::new(input, hidden, dirs)?),
            Device::Cuda(_) => return Err(unsupported(device)),
        };
        Ok(Lstm { device, inner })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn cpu(&self) -> &cpu::Lstm {
        match &self.inner {
            Inner::Cpu(l) => l,
        }
    }

    pub fn input(&self) -> usize {
        self.cpu().input()
    }

    pub fn hidden(&self) -> usize {
        self.cpu().hidden()
    }

    /// The width of an output row: every direction's state, in the order
    /// the directions were given.
    pub fn width(&self) -> usize {
        self.cpu().width()
    }

    /// The width of a row of input projections, every direction's four
    /// gates.
    pub fn projected_width(&self) -> usize {
        self.cpu().projected_width()
    }

    /// `values` [distinct, projected_width] as a table on this LSTM's
    /// device. Refused with INVALID_SHAPE for a length that is not whole
    /// rows; non-finite values are refused by the run that reads them, as
    /// the CPU form refuses them.
    pub fn table(&self, values: &[f32]) -> Result<Table> {
        let pw = self.projected_width();
        if !values.len().is_multiple_of(pw) {
            let message = format!("lstm: a table of {} values is not whole rows of {pw}", values.len());
            return Err(Error::new(crate::status::INVALID_SHAPE, message));
        }
        let inner = match self.inner {
            Inner::Cpu(_) => TableInner::Cpu(values.to_vec()),
        };
        Ok(Table { device: self.device, rows: values.len() / pw, inner })
    }

    /// [`cpu::Lstm::run`](crate::cpu::lstm::Lstm::run) on the device: the
    /// sequences of `lengths` (the longest first) over rows `x`, into `out`
    /// [sum, width], refused as the CPU form refuses.
    pub fn run(&self, x: &[f32], lengths: &[usize], out: &mut [f32], w: &mut Work) -> Result<()> {
        match &self.inner {
            Inner::Cpu(l) => l.run(x, lengths, out, &mut w.cpu),
        }
    }

    /// [`cpu::Lstm::run_gathered`](crate::cpu::lstm::Lstm::run_gathered)
    /// from a table on this LSTM's device: row r reads row `index[r]` of
    /// `table`. A table made for another device is refused with
    /// INVALID_ARGUMENT, and the rest as the CPU form refuses.
    pub fn run_gathered(
        &self,
        table: &Table,
        index: &[usize],
        lengths: &[usize],
        out: &mut [f32],
        w: &mut Work,
    ) -> Result<()> {
        if table.device != self.device {
            let message = format!("lstm: a table on {:?} for an LSTM on {:?}", table.device, self.device);
            return Err(Error::new(crate::status::INVALID_ARGUMENT, message));
        }
        match (&self.inner, &table.inner) {
            (Inner::Cpu(l), TableInner::Cpu(t)) => l.run_gathered(t, index, lengths, out, &mut w.cpu),
        }
    }
}

impl Table {
    pub fn device(&self) -> Device {
        self.device
    }

    /// The table's distinct rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
}
