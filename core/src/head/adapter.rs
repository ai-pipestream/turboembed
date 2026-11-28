//! Stacked word-level BiLSTM adapter (`HeadSpec::adapter`) on [`Device`],
//! shared by [`super::words::WordReader`] across every word head.

use std::sync::{Mutex, PoisonError};

use super::words::Take;
use crate::device::lstm::{Lstm, Weights, Work as LstmWork};
use crate::device::{Device, Precision};
use crate::manifest::HeadSpec;
use crate::status::Result;

struct Layer {
    lstm: Lstm,
    input: usize,
    hidden: usize,
    weights: [Vec<f32>; 6],
}

impl Layer {
    fn new(device: Device, precision: Precision, input: usize, hidden: usize, weights: [Vec<f32>; 6]) -> Result<Self> {
        let lstm = Self::make(device, precision, input, hidden, &weights)?;
        Ok(Self { lstm, input, hidden, weights })
    }

    fn make(device: Device, precision: Precision, input: usize, hidden: usize, w: &[Vec<f32>; 6]) -> Result<Lstm> {
        let dirs = [
            Weights { w: &w[0], u: &w[1], b: &w[2], reverse: false },
            Weights { w: &w[3], u: &w[4], b: &w[5], reverse: true },
        ];
        Lstm::new_at(device, precision, input, hidden, &dirs)
    }
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lstm")
            .field("device", &self.lstm.device())
            .field("precision", &self.lstm.precision())
            .field("input", &self.lstm.input())
            .field("hidden", &self.lstm.hidden())
            .finish()
    }
}

/// Optional stacked BiLSTM over pooled word vectors, with reusable GPU workspaces.
pub(crate) struct Adapter {
    layers: Vec<Layer>,
    spare: Mutex<Vec<LstmWork>>,
}

impl std::fmt::Debug for Adapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(&self.layers).finish()
    }
}

impl Adapter {
    pub(crate) fn new(spec: &HeadSpec, mut width: usize, t: &Take<'_>) -> Result<(Self, usize)> {
        let mut layers = Vec::new();
        if let Some(a) = &spec.adapter {
            let h = a.bilstm.hidden as usize;
            for k in 0..a.bilstm.layers {
                let part = |dir: &str, x: &str| t.take(&format!("lstm.{k}.{dir}.{x}"));
                let weights = [
                    part("fwd", "w")?,
                    part("fwd", "u")?,
                    part("fwd", "b")?,
                    part("bwd", "w")?,
                    part("bwd", "u")?,
                    part("bwd", "b")?,
                ];
                layers.push(Layer::new(Device::Cpu, Precision::Model, width, h, weights)?);
                width = 2 * h;
            }
        }
        Ok((Self { layers, spare: Mutex::new(Vec::new()) }, width))
    }

    pub(crate) fn on_device_at(&mut self, device: Device, precision: Precision) -> Result<()> {
        if self.layers.is_empty()
            || (self.layers[0].lstm.device() == device && self.layers[0].lstm.precision() == precision)
        {
            return Ok(());
        }
        let rebuilt: Vec<Lstm> = self
            .layers
            .iter()
            .map(|l| Layer::make(device, precision, l.input, l.hidden, &l.weights))
            .collect::<Result<_>>()?;
        for (layer, lstm) in self.layers.iter_mut().zip(rebuilt) {
            layer.lstm = lstm;
        }
        self.spare.lock().unwrap_or_else(PoisonError::into_inner).clear();
        Ok(())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub(crate) fn device(&self) -> Option<Device> {
        self.layers.first().map(|l| l.lstm.device())
    }

    pub(crate) fn precision(&self) -> Option<Precision> {
        self.layers.first().map(|l| l.lstm.precision())
    }

    /// Runs every layer over `lengths` (sorted longest-first, all positive,
    /// summing to `rows`) in `pooled`, leaving `[rows, width]` in `pooled`.
    pub(crate) fn run(
        &self,
        lengths: &[usize],
        rows: usize,
        pooled: &mut Vec<f32>,
        adapted: &mut Vec<f32>,
        cpu_work: &mut LstmWork,
    ) -> Result<()> {
        if self.layers.is_empty() || rows == 0 {
            return Ok(());
        }
        if self.layers[0].lstm.device() == Device::Cpu {
            return self.run_layers(lengths, rows, pooled, adapted, cpu_work);
        }
        let mut lw = self.spare.lock().unwrap_or_else(PoisonError::into_inner).pop().unwrap_or_default();
        let r = self.run_layers(lengths, rows, pooled, adapted, &mut lw);
        let mut guard = self.spare.lock().unwrap_or_else(PoisonError::into_inner);
        if guard.len() < 16 {
            guard.push(lw);
        }
        drop(guard);
        r
    }

    fn run_layers(
        &self,
        lengths: &[usize],
        rows: usize,
        pooled: &mut Vec<f32>,
        adapted: &mut Vec<f32>,
        lw: &mut LstmWork,
    ) -> Result<()> {
        for layer in &self.layers {
            let width = layer.lstm.width();
            adapted.clear();
            adapted.resize(rows * width, 0.0);
            layer.lstm.run(pooled, lengths, adapted, lw)?;
            std::mem::swap(pooled, adapted);
        }
        Ok(())
    }
}
