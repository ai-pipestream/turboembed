//! Classical recipes fail before reading any corpus or reference file.

use std::process::Command;

#[test]
fn classical_commands_are_explicitly_disabled_before_io() {
    for command in ["masc", "masc-namefind-select", "ud", "depparse", "zoo-index", "tatoeba", "tatoeba-fineweb"] {
        let output = Command::new(env!("CARGO_BIN_EXE_turbo-train-recipes"))
            .args([command, "--references", "unavailable-models", "--out", "unavailable-output"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "{command}: {output:?}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("disabled by #1511"), "{command}: {error}");
        assert!(output.stdout.is_empty(), "{command}: {output:?}");
    }
}

#[test]
fn classical_workflows_and_reference_download_lists_are_removed() {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for workflow in ["models.yml", "models-ud.yml", "models-tatoeba.yml", "models-fineweb.yml", "models-publish.yml"] {
        assert!(!crate_dir.join("../../.github/workflows").join(workflow).exists(), "{workflow}");
    }
    for list in ["masc-references.txt", "ud-references.txt", "tatoeba-references.txt", "fineweb-references.txt"] {
        assert!(!crate_dir.join(list).exists(), "{list}");
    }
}

#[test]
fn dataset_commands_still_validate_their_required_arguments() {
    for command in ["fetch-masc", "fetch-ud"] {
        let output = Command::new(env!("CARGO_BIN_EXE_turbo-train-recipes")).arg(command).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).starts_with("--home is required"));
    }
}
