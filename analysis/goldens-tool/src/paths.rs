//! Where things are in the repository, found from this crate's directory.

use std::path::{Path, PathBuf};

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

/// `analysis/goldens/data/fixtures`: the checked in models and dictionaries
/// the generator loads besides the published models.
pub fn fixtures_dir() -> PathBuf {
    data_dir().join("fixtures")
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

/// The extension of a golden layer file.
pub const GOLDEN_EXTENSION: &str = "golden";

/// The extension of a decision point dump.
pub const DUMP_EXTENSION: &str = "dump";

/// `<root>/<annotator>/<model>/<set>/<id>.golden`: the golden of one
/// annotator, model and input.
pub fn golden_file(root: &Path, annotator: &str, model: &str, input_id: &str) -> PathBuf {
    root.join(annotator).join(model).join(format!("{input_id}.{GOLDEN_EXTENSION}"))
}

/// `<root>/<annotator>/<model>/<set>/<id>.dump`: the decision point dump of
/// one annotator, model and input, beside its golden.
pub fn dump_file(root: &Path, annotator: &str, model: &str, input_id: &str) -> PathBuf {
    root.join(annotator).join(model).join(format!("{input_id}.{DUMP_EXTENSION}"))
}

/// The dump beside a golden: the same path with the extension `dump`, or
/// nothing when `golden` is not a golden's path.
pub fn dump_beside(golden: &Path) -> Option<PathBuf> {
    (golden.extension()? == GOLDEN_EXTENSION).then(|| golden.with_extension(DUMP_EXTENSION))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dump_sits_beside_its_golden() {
        let root = Path::new("data");
        let golden = golden_file(root, "tokenize", "m-1.3", "multi/de");
        let dump = dump_file(root, "tokenize", "m-1.3", "multi/de");
        assert_eq!(golden, Path::new("data/tokenize/m-1.3/multi/de.golden"));
        assert_eq!(dump, Path::new("data/tokenize/m-1.3/multi/de.dump"));
        assert_eq!(dump_beside(&golden), Some(dump.clone()));
        assert_eq!(dump_beside(&dump), None);
        assert_eq!(dump_beside(Path::new("data/inputs/en/a.txt")), None);
    }
}
