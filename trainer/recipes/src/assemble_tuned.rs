//! Public bundle assembly entry points, including tuned encoder replacement.

use std::path::Path;

use crate::Result;
use crate::assemble::{AssembledBundle, assemble_with_tuned};

/// Assemble verified heads over their original encoder bundle.
pub fn assemble_bundle(encoder_bundle: &Path, run_dirs: &[&Path], out_dir: &Path) -> Result<AssembledBundle> {
    assemble_with_tuned(encoder_bundle, run_dirs, None, out_dir)
}

/// Assemble verified heads over `tuned_run`'s digest-verified tuned encoder.
pub fn assemble_tuned_bundle(
    encoder_bundle: &Path,
    run_dirs: &[&Path],
    tuned_run: &Path,
    out_dir: &Path,
) -> Result<AssembledBundle> {
    assemble_with_tuned(encoder_bundle, run_dirs, Some(tuned_run), out_dir)
}
