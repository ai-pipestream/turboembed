//! Cargo's settings for the benchmark crate, not a guessed release label.

use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CargoBuild {
    /// Cargo PROFILE: the inherited debug/release family, not a custom name.
    pub profile: String,
    /// Cargo OPT_LEVEL for turbo-bench, independent of debug assertions.
    pub opt_level: String,
    pub debug_assertions: bool,
}

impl CargoBuild {
    pub fn current() -> Self {
        Self {
            profile: env!("TURBO_BENCH_CARGO_PROFILE").into(),
            opt_level: env!("TURBO_BENCH_CARGO_OPT_LEVEL").into(),
            debug_assertions: cfg!(debug_assertions),
        }
    }

    pub(super) fn check(&self) -> Result<()> {
        if !matches!(self.profile.as_str(), "debug" | "release") {
            return Err(format!("unknown Cargo PROFILE {:?}", self.profile));
        }
        if !matches!(self.opt_level.as_str(), "1" | "2" | "3" | "s" | "z") {
            return Err(format!(
                "this turbo-bench has Cargo profile={} opt-level={}; throughput requires a nonzero Cargo opt-level",
                self.profile, self.opt_level
            ));
        }
        Ok(())
    }
}
