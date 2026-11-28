use std::ffi::CStr;
use std::path::Path;

use turbo::{turbo_model, turbo_model_info};

use super::{Progress, Recipe, Result, Staged, api, stage_impl};

/// Stage using the already loaded teacher, without re-opening its weights.
/// When an accelerator is required, every teacher batch must report an
/// encode on that runtime index. The static table's CPU seal is separate.
pub fn stage_with_teacher(
    recipe: &Recipe,
    base_dir: &Path,
    bundle: &Path,
    terms: &[String],
    observe: &mut dyn FnMut(Progress),
    teacher: &turbo_model,
    require_accelerator: Option<u32>,
) -> Result<Staged> {
    stage_impl(recipe, base_dir, bundle, terms, observe, Some(api::ModelRef::new(teacher, require_accelerator)))
}

pub(super) fn verify_manifest(info: &turbo_model_info, expected: &str) -> Result<()> {
    // The public model-info buffer always contains a terminated hex digest.
    let bytes = info.manifest_sha256.map(|c| c as u8);
    if CStr::from_bytes_until_nul(&bytes).is_ok_and(|s| s.to_bytes() == expected.as_bytes()) {
        Ok(())
    } else {
        Err("teacher manifest changed since the model was loaded".into())
    }
}
