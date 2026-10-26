//! Ops the analysis crates run on a device from Rust, outside the C
//! backend table: the taggers are Rust and never load a model through
//! `turbo_model_load`, so their recurrences reach a GPU here instead of
//! through a new model family (#860).
//!
//! Each op has the shape of its CPU form in [`crate::cpu`] and the same
//! refusals; a device the library was built without, or one that is not
//! there, is refused when the op is made, never answered on the CPU.
//! CUDA's LSTM is core/cuda/lstm.cu; Level Zero's is in
//! core/src/levelzero/lstm.rs, on kernels in core/levelzero/encoder.cl.

#[cfg(feature = "cuda")]
mod cuda_lstm;
mod gpu;
pub mod lstm;

use crate::status::{Error, INVALID_ARGUMENT, Result};

/// Where an op runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// The CPU's packed kernels, the reference every device is held to.
    Cpu,
    /// An NVIDIA GPU, by CUDA ordinal.
    Cuda(u32),
    /// An Intel GPU, by the levelzero backend's ordinal.
    LevelZero(u32),
}

impl Device {
    /// `cpu`, `cuda` or `levelzero` (ordinal 0), or `cuda:N` or
    /// `levelzero:N`, as TURBO_TEST_DEVICE names a device.
    pub fn parse(name: &str) -> Result<Device> {
        let bad = || {
            let message = format!("device {name:?}: expected cpu, cuda, cuda:N, levelzero or levelzero:N");
            Error::new(INVALID_ARGUMENT, message)
        };
        match name.split_once(':') {
            None if name == "cpu" => Ok(Device::Cpu),
            None if name == "cuda" => Ok(Device::Cuda(0)),
            None if name == "levelzero" => Ok(Device::LevelZero(0)),
            Some(("cuda", n)) => n.parse().map(Device::Cuda).map_err(|_| bad()),
            Some(("levelzero", n)) => n.parse().map(Device::LevelZero).map_err(|_| bad()),
            _ => Err(bad()),
        }
    }

    /// The backend's name, as `turbo_version` lists it.
    pub fn backend(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::LevelZero(_) => "levelzero",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Device;

    #[test]
    fn names_parse() {
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("cuda").unwrap(), Device::Cuda(0));
        assert_eq!(Device::parse("cuda:3").unwrap(), Device::Cuda(3));
        assert_eq!(Device::parse("levelzero").unwrap(), Device::LevelZero(0));
        assert_eq!(Device::parse("levelzero:1").unwrap(), Device::LevelZero(1));
        for bad in ["", "gpu", "cuda:", "cuda:x", "cpu:0", "cuda:-1"] {
            assert!(Device::parse(bad).is_err(), "{bad:?} parsed");
        }
    }
}
