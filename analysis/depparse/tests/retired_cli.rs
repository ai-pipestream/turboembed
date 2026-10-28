use std::process::Command;

#[test]
fn retired_commands_fail_before_accessing_model_or_output_paths() {
    for args in [
        vec!["inspect", "feedforward", "missing-input.bin"],
        vec!["convert", "linear", "missing-input.bin", "must-not-create.bin"],
        vec!["train", "linear", "en", "missing-corpus", "missing-settings", "must-not-create.bin"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_turbo-depparse")).args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8(output.stderr).unwrap().contains("retired under #1511"));
        assert!(output.stdout.is_empty());
    }
}
