//! [`crate::cpu::lstm::Lstm`] on a [`Device`]: the same packed weights of
//! every direction, the same rows (each sequence's one after another, the
//! longest first), the same outputs and the same refusals. On a GPU the
//! weights and a gathered run's table stay in device memory; a run copies
//! its inputs in and its states out.
//!
//! The CPU form is the reference: a device's states are held to it within
//! a bound its tests name, not to its bits.

use super::Device;
use super::gpu::{self, Gpu};
use crate::cpu::lstm as cpu;
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result};

pub use crate::cpu::lstm::Weights;

/// Checked weights of one or more directions over one input, on a device.
pub struct Lstm {
    device: Device,
    /// The CPU form, which checked the weights and runs on the CPU.
    cpu: cpu::Lstm,
    gpu: Option<Gpu>,
}

/// Input projections made once per distinct row (a character table, say),
/// where the device reads them, for [`Lstm::run_gathered`].
pub struct Table {
    device: Device,
    /// The values held, and the rows of the LSTM that made the table: a
    /// run reads it by its own projected width, so it checks the values.
    values: usize,
    rows: usize,
    /// Whether every value is finite, which a run reading it needs.
    finite: bool,
    inner: TableInner,
}

enum TableInner {
    Cpu(Vec<f32>),
    Gpu(gpu::Table),
}

/// What a run computes in, kept across runs so that a run of the same or
/// smaller shape allocates nothing.
#[derive(Default)]
pub struct Work {
    cpu: cpu::Work,
    gpu: gpu::Work,
    lengths: Vec<i64>,
    index: Vec<i64>,
}

fn finite(v: &[f32]) -> bool {
    v.chunks(64).all(|c| c.iter().fold(true, |a, x| a & x.is_finite()))
}

impl Lstm {
    /// The directions in `dirs` on `device`, as
    /// [`cpu::Lstm::new`](crate::cpu::lstm::Lstm::new) checks them. A
    /// device the library was built without is refused with UNSUPPORTED,
    /// and one that is not there with DEVICE_NOT_FOUND.
    pub fn new(device: Device, input: usize, hidden: usize, dirs: &[Weights<'_>]) -> Result<Lstm> {
        let cpu = cpu::Lstm::new(input, hidden, dirs)?;
        if device == Device::Cpu {
            return Ok(Lstm { device, cpu, gpu: None });
        }
        let gpu = Gpu::new(device, input, hidden, dirs)?;
        Ok(Lstm { device, cpu, gpu: Some(gpu) })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn input(&self) -> usize {
        self.cpu.input()
    }

    pub fn hidden(&self) -> usize {
        self.cpu.hidden()
    }

    /// The width of an output row: every direction's state, in the order
    /// the directions were given.
    pub fn width(&self) -> usize {
        self.cpu.width()
    }

    /// The width of a row of input projections, every direction's four
    /// gates.
    pub fn projected_width(&self) -> usize {
        self.cpu.projected_width()
    }

    /// `values` [distinct, projected_width] as a table on this LSTM's
    /// device. Refused with INVALID_SHAPE for a length that is not whole
    /// rows; non-finite values are refused by the run that reads them, as
    /// the CPU form refuses them.
    pub fn table(&self, values: &[f32]) -> Result<Table> {
        let pw = self.projected_width();
        if !values.len().is_multiple_of(pw) {
            let message = format!("lstm: a table of {} values is not whole rows of {pw}", values.len());
            return Err(Error::new(INVALID_SHAPE, message));
        }
        let inner = match &self.gpu {
            None => TableInner::Cpu(values.to_vec()),
            Some(g) => TableInner::Gpu(g.table(values)?),
        };
        Ok(Table { device: self.device, values: values.len(), rows: values.len() / pw, finite: finite(values), inner })
    }

    /// [`cpu::Lstm::run`](crate::cpu::lstm::Lstm::run) on the device: the
    /// sequences of `lengths` (the longest first) over rows `x`, into `out`
    /// [sum, width], refused as the CPU form refuses.
    pub fn run(&self, x: &[f32], lengths: &[usize], out: &mut [f32], w: &mut Work) -> Result<()> {
        if self.device == Device::Cpu {
            return self.cpu.run(x, lengths, out, &mut w.cpu);
        }
        let rows = self.rows(lengths, out)?;
        if Some(x.len()) != rows.checked_mul(self.input()) {
            let message = format!("lstm: {} input values for {rows} rows of {}", x.len(), self.input());
            return Err(Error::new(INVALID_SHAPE, message));
        }
        if !finite(x) {
            return Err(Error::new(INVALID_ARGUMENT, "lstm: an input is not finite"));
        }
        self.on_device(Some(x), None, lengths, rows, out, w)
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
            return Err(Error::new(INVALID_ARGUMENT, message));
        }
        if let Some(t) = table.on_cpu() {
            return self.cpu.run_gathered(t, index, lengths, out, &mut w.cpu);
        }
        let rows = self.rows(lengths, out)?;
        // As the CPU form reads a table: whole rows of this LSTM's width,
        // whatever LSTM made it, so no index reads past its values.
        let (pw, n) = (self.projected_width(), table.values);
        if !n.is_multiple_of(pw) || index.len() != rows || index.iter().any(|&i| i >= n / pw) {
            let message = format!("lstm: a table of {n} values and {} indices for {rows} rows of {pw}", index.len());
            return Err(Error::new(INVALID_SHAPE, message));
        }
        if !table.finite {
            return Err(Error::new(INVALID_ARGUMENT, "lstm: a projection is not finite"));
        }
        let mut wide = std::mem::take(&mut w.index);
        wide.clear();
        wide.extend(index.iter().map(|&i| i as i64));
        let result = self.on_device(None, Some((table, &wide)), lengths, rows, out, w);
        w.index = wide;
        result
    }

    /// The rows of `lengths`, checked as the CPU form checks them, and
    /// `out` checked against them.
    fn rows(&self, lengths: &[usize], out: &[f32]) -> Result<usize> {
        let rows = lengths.iter().try_fold(0usize, |a, &n| a.checked_add(n));
        let rows = rows.ok_or_else(|| Error::new(INVALID_ARGUMENT, "lstm: lengths overflow"))?;
        if lengths.contains(&0) || lengths.windows(2).any(|p| p[0] < p[1]) {
            return Err(Error::new(INVALID_ARGUMENT, "lstm: lengths must be at least 1 and the longest first"));
        }
        if Some(out.len()) != rows.checked_mul(self.width()) {
            let message = format!("lstm: {} output values for {rows} rows of {}", out.len(), self.width());
            return Err(Error::new(INVALID_SHAPE, message));
        }
        Ok(rows)
    }

    /// A checked run on a GPU.
    fn on_device(
        &self,
        x: Option<&[f32]>,
        table: Option<(&Table, &[i64])>,
        lengths: &[usize],
        rows: usize,
        out: &mut [f32],
        w: &mut Work,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        w.lengths.clear();
        w.lengths.extend(lengths.iter().map(|&n| n as i64));
        let table = table.map(|(t, i)| match &t.inner {
            TableInner::Gpu(t) => (t, i),
            TableInner::Cpu(_) => unreachable!("a CPU table on a GPU run"),
        });
        let gpu = self.gpu.as_ref().expect("an LSTM on a GPU holds its weights there");
        gpu.run(x, table, &w.lengths, rows, out, &mut w.gpu)
    }
}

impl Table {
    fn on_cpu(&self) -> Option<&[f32]> {
        match &self.inner {
            TableInner::Cpu(t) => Some(t),
            TableInner::Gpu(_) => None,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The table's distinct rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
}
