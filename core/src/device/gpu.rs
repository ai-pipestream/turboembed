//! [`super::lstm`]'s GPUs, by backend: the backends built in, each behind
//! its feature. A build with none has no GPU here, and an LSTM on a GPU is
//! refused with UNSUPPORTED when it is made.

#![allow(unused_variables)]

use super::lstm::Weights;
use super::{Device, Precision};
use crate::status::{Error, Result, UNSUPPORTED};

/// An LSTM's weights on a GPU.
pub(super) enum Gpu {
    #[cfg(feature = "cuda")]
    Cuda(super::cuda_lstm::Lstm),
    #[cfg(feature = "levelzero")]
    LevelZero(crate::levelzero::lstm::Lstm),
}

/// A table on a GPU.
pub(super) enum Table {
    #[cfg(feature = "cuda")]
    Cuda(super::cuda_lstm::Table),
    #[cfg(feature = "levelzero")]
    LevelZero(crate::levelzero::lstm::Table),
}

/// A run's buffers on each backend, made on first use.
#[derive(Default)]
pub(super) struct Work {
    #[cfg(feature = "cuda")]
    cuda: super::cuda_lstm::Work,
    #[cfg(feature = "levelzero")]
    levelzero: crate::levelzero::lstm::Work,
}

#[cfg(feature = "levelzero")]
fn not_finite(row: i64) -> Error {
    Error::new(crate::status::INVALID_ARGUMENT, format!("lstm: a gate or state is not finite at row {row}"))
}

impl Gpu {
    /// Weights the CPU form has checked, on `device`.
    pub(super) fn new(
        device: Device,
        precision: Precision,
        input: usize,
        hidden: usize,
        dirs: &[Weights<'_>],
    ) -> Result<Gpu> {
        let (mut w, mut u, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for d in dirs {
            w.extend_from_slice(d.w);
            u.extend_from_slice(d.u);
            b.extend_from_slice(d.b);
        }
        let reverse: Vec<i32> = dirs.iter().map(|d| i32::from(d.reverse)).collect();
        match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(n) => {
                let f16 = precision == Precision::Fastest;
                Ok(Gpu::Cuda(super::cuda_lstm::Lstm::new(n, input, hidden, &w, &u, &b, &reverse, f16)?))
            }
            #[cfg(feature = "levelzero")]
            Device::LevelZero(_) if precision == Precision::Fastest => {
                Err(Error::new(UNSUPPORTED, "lstm: levelzero has no FASTEST (F16 weights) yet"))
            }
            #[cfg(feature = "levelzero")]
            Device::LevelZero(n) => {
                Ok(Gpu::LevelZero(crate::levelzero::lstm::Lstm::new(n, input, hidden, &w, &u, &b, &reverse)?))
            }
            _ => Err(Error::new(UNSUPPORTED, format!("lstm: device {device:?} is not built into this library"))),
        }
    }

    pub(super) fn table(&self, values: &[f32]) -> Result<Table> {
        match *self {
            #[cfg(feature = "cuda")]
            Gpu::Cuda(ref g) => Ok(Table::Cuda(g.table(values)?)),
            #[cfg(feature = "levelzero")]
            Gpu::LevelZero(ref g) => Ok(Table::LevelZero(g.table(values)?)),
        }
    }

    /// A checked, nonempty run: `x` the rows, or else `table` read by
    /// `index`, a table this GPU made.
    pub(super) fn run(
        &self,
        x: Option<&[f32]>,
        table: Option<(&Table, &[i64])>,
        lengths: &[i64],
        rows: usize,
        out: &mut [f32],
        w: &mut Work,
    ) -> Result<()> {
        match *self {
            #[cfg(feature = "cuda")]
            Gpu::Cuda(ref g) => {
                let table = table.map(|(t, i)| match *t {
                    Table::Cuda(ref t) => (t, i),
                    #[cfg(feature = "levelzero")]
                    Table::LevelZero(_) => unreachable!("a levelzero table on a CUDA run"),
                });
                g.run(x, table, lengths, rows, out, &mut w.cuda)
            }
            #[cfg(feature = "levelzero")]
            Gpu::LevelZero(ref g) => {
                let table = table.map(|(t, i)| match *t {
                    Table::LevelZero(ref t) => (t, i),
                    #[cfg(feature = "cuda")]
                    Table::Cuda(_) => unreachable!("a CUDA table on a levelzero run"),
                });
                g.run(x, table, lengths, rows, out, &mut w.levelzero)?.map_or(Ok(()), |row| Err(not_finite(row)))
            }
        }
    }
}
