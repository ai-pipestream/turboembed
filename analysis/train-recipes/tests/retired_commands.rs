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
