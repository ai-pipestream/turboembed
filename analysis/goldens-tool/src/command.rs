//! Running an external program and turning its failure into a message that
//! names the command and what it wrote to stderr.

use std::process::{Command, Output};

/// Runs `cmd` to completion. Fails when it cannot start or exits
/// unsuccessfully; the error names the command line, the exit status and
/// its stderr.
pub fn run(cmd: &mut Command) -> Result<Output, String> {
    let line = describe(cmd);
    let out = cmd.output().map_err(|e| format!("{line}: cannot start: {e}"))?;
    if !out.status.success() {
        return Err(format!("{line}: {}\n{}", out.status, String::from_utf8_lossy(&out.stderr).trim_end()));
    }
    Ok(out)
}

/// The program and its arguments, separated by spaces, for messages.
pub fn describe(cmd: &Command) -> String {
    let mut s = cmd.get_program().to_string_lossy().into_owned();
    for a in cmd.get_args() {
        s.push(' ');
        s.push_str(&a.to_string_lossy());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failing_command_is_reported_with_its_line_and_stderr() {
        let err = run(Command::new("sh").args(["-c", "echo broken >&2; exit 3"])).unwrap_err();
        assert!(err.starts_with("sh -c echo broken >&2; exit 3: "), "{err}");
        assert!(err.contains("broken"), "{err}");
    }

    #[test]
    fn a_missing_program_is_reported() {
        let err = run(&mut Command::new("turbo-goldens-no-such-program")).unwrap_err();
        assert!(err.contains("cannot start"), "{err}");
    }
}
