//! Where things are in the repository, found from this crate's directory.

use std::path::PathBuf;

/// `analysis/goldens-tool`.
pub fn tool_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The repository's root.
pub fn workspace_dir() -> PathBuf {
    tool_dir().join("../..")
}

/// `analysis/goldens/data`: the inputs and the checked in goldens.
pub fn data_dir() -> PathBuf {
    tool_dir().join("../goldens/data")
}

/// `analysis/goldens/data/inputs`.
pub fn inputs_dir() -> PathBuf {
    data_dir().join("inputs")
}

/// The Maven project the container builds, which is the image's build context.
pub fn java_dir() -> PathBuf {
    tool_dir().join("java")
}

/// The container recipe.
pub fn dockerfile() -> PathBuf {
    tool_dir().join("container/Dockerfile")
}

/// Scratch space under the workspace's `target` directory: the default
/// model cache and the temporary output directories. It is not under
/// /tmp, which some Docker installations cannot mount.
pub fn scratch_dir() -> PathBuf {
    workspace_dir().join("target/turbo-goldens")
}
