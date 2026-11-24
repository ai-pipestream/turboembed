//! The evaluation records under analysis/eval/records are clean text, as
//! the records under benchmarks/records are: UTF-8 with LF line ends, no
//! path on the machine that made them and no error wrapper of the shell
//! that ran the tool. Outside the logs, which are a run's whole output, a
//! record also carries no line of the build tool: a report is the
//! evaluation tool's standard output and nothing else.

use turbo::record::HOST_PATHS;

use crate::common::workspace;

const RECORDS: &str = "analysis/eval/records";

/// Text only PowerShell writes, around a native program's standard error.
const SHELL_WRAPPERS: [&str; 3] = ["RemoteException", "FullyQualifiedErrorId", "CategoryInfo"];

/// The status words Cargo prints, right-aligned, before what it is doing.
const CARGO_STATUS: [&str; 6] = ["Compiling ", "Downloading ", "Downloaded ", "Finished ", "Running ", "Blocking "];

/// How the compiler and Cargo start a diagnostic.
const DIAGNOSTICS: [&str; 3] = ["warning: ", "error: ", "error["];

/// What Cargo says of a program it ran that exited nonzero.
const FAILED_RUN: &str = "process didn't exit successfully";

/// The drive letter of a Windows path (`C:\dir`) in `line`, if it has one.
/// In JSON a backslash is written twice, so `"Q:\n"` there is a line end
/// inside a string, not a path.
fn drive_path(line: &str, json: bool) -> Option<char> {
    let b = line.as_bytes();
    line.match_indices(":\\").find_map(|(at, _)| {
        let letter = *b.get(at.checked_sub(1)?)?;
        let starts_a_word = at < 2 || !b[at - 2].is_ascii_alphanumeric();
        let escaped = !json || b.get(at + 2) == Some(&b'\\');
        (letter.is_ascii_alphabetic() && starts_a_word && escaped).then_some(letter as char)
    })
}

/// Whether `line` is a line Cargo or the compiler wrote, not the tool.
fn build_noise(line: &str) -> bool {
    let word = line.trim_start();
    (line.starts_with("  ") && CARGO_STATUS.iter().any(|s| word.starts_with(s)))
        || DIAGNOSTICS.iter().any(|s| line.starts_with(s))
        || line.contains(FAILED_RUN)
}

/// Whether `text` holds anything a defect is made of. Most records hold
/// none of it and are not read line by line.
fn suspect(text: &str) -> bool {
    let marks = [&["\r", "\0", ":\\", FAILED_RUN][..], &HOST_PATHS, &SHELL_WRAPPERS, &CARGO_STATUS, &DIAGNOSTICS];
    marks.iter().any(|set| set.iter().any(|m| text.contains(m)))
}

/// What is wrong with the record named `name` holding `bytes`: nothing for
/// a clean record, else one line for each kind of defect, with the first
/// line it is on.
fn defects(name: &str, bytes: &[u8]) -> Vec<String> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return vec!["starts with a byte order mark".into()];
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        return vec!["is not UTF-8".into()];
    };
    if !suspect(text) {
        return Vec::new();
    }
    let json = name.ends_with(".json") || name.ends_with(".jsonl");
    let log = name.ends_with(".log");
    let mut found: Vec<(&str, String)> = Vec::new();
    let mut note = |kind: &'static str, n: usize, what: String| {
        if !found.iter().any(|(k, _)| *k == kind) {
            found.push((kind, format!("line {}: {what}", n + 1)));
        }
    };
    for (n, line) in text.split('\n').enumerate() {
        if line.contains('\0') {
            note("nul", n, "a NUL character".into());
        }
        if line.contains('\r') {
            note("cr", n, "a carriage return; records end their lines with LF".into());
        }
        if let Some(h) = HOST_PATHS.iter().find(|h| line.contains(*h)) {
            note("home", n, format!("host path {h}"));
        }
        if let Some(d) = drive_path(line, json) {
            note("drive", n, format!("host path {d}:\\"));
        }
        if let Some(w) = SHELL_WRAPPERS.iter().find(|w| line.contains(*w)) {
            note("shell", n, format!("shell error wrapper {w}"));
        }
        if !log && build_noise(line) {
            note("build", n, format!("build tool output {:?}", line.trim()));
        }
    }
    found.into_iter().map(|(_, what)| what).collect()
}

#[test]
fn the_committed_evaluation_records_are_clean_text() {
    let root = workspace();
    let out = std::process::Command::new("git")
        .current_dir(&root)
        .args(["ls-files", "-z", "--", RECORDS])
        .output()
        .expect("git lists the evaluation records");
    assert!(out.status.success(), "git could not list {RECORDS}");
    let mut bad = Vec::new();
    let mut seen = 0;
    for name in String::from_utf8(out.stdout).unwrap().split('\0').filter(|n| !n.is_empty()) {
        seen += 1;
        let bytes = std::fs::read(root.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        bad.extend(defects(name, &bytes).into_iter().map(|d| format!("{name}: {d}")));
    }
    assert!(seen > 600, "only {seen} evaluation records scanned");
    assert!(bad.is_empty(), "evaluation records that are not clean text:\n  {}", bad.join("\n  "));
}

#[test]
fn a_clean_report_and_a_clean_log_have_no_defect() {
    let report = "arguana: 8674 documents, 19840 shared cuts, 0 documents differ (cpu/model against npu/model)\n  \
                  doc-1 level 0 after Some(0) before None: A cuts at [907], B at [731], margin A 18 B -18\n";
    assert_eq!(defects("reports/cuts-vs-cpu-x.txt", report.as_bytes()), Vec::<String>::new());
    // A log is the whole output of a run, the build included.
    let log = "   Compiling turbo v0.1.0 (<records>)\n    Finished `release` profile [optimized] target(s) in 41.2s\n     \
               Running `target/release/turbo-eval gate <records>/scratch`\nwarning: unused variable\n\
               error: process didn't exit successfully: `target/release/turbo-eval gate` (exit code: 1)\n";
    assert_eq!(defects("logs/gate.log", log.as_bytes()), Vec::<String>::new());
    // A line end written inside a JSON string is not a drive path.
    assert_eq!(defects("x.json", br#"{"text": "Q:\nA:\tyes", "url": "https://example.org/a"}"#), Vec::<String>::new());
}

#[test]
fn a_report_made_on_windows_and_left_unscrubbed_is_refused() {
    // CRLF line ends.
    let crlf = defects("reports/x.txt", b"arguana: 1 documents\r\nnfcorpus: 2 documents\r\n");
    assert_eq!(crlf, ["line 1: a carriage return; records end their lines with LF"]);
    // UTF-16LE with its byte order mark, and UTF-8 with one.
    assert_eq!(defects("reports/x.txt", &[0xFF, 0xFE, b'a', 0, b'\r', 0, b'\n', 0]), ["is not UTF-8"]);
    assert_eq!(defects("reports/x.txt", &[0xEF, 0xBB, 0xBF, b'a', b'\n']), ["starts with a byte order mark"]);
    assert_eq!(defects("reports/x.txt", b"a\0b\n"), ["line 1: a NUL character"]);
    // The operands of a wrapped `Running` line, and a home directory.
    let paths = "ok\nD:\\scratch\\out\\dumps\\turbo-chunker-254-200-hybrid \n";
    assert_eq!(defects("reports/x.txt", paths.as_bytes()), ["line 2: host path D:\\"]);
    assert_eq!(defects("x.json", br#"{"dir": "C:\\scratch\\out"}"#), ["line 1: host path C:\\"]);
    assert_eq!(defects("run.log", b"wrote /home/someone/out.json\n"), ["line 1: host path /home/"]);
    // Cargo's lines, the compiler's, and the failure block after a nonzero exit.
    for noise in [
        "   Compiling turbo v0.1.0 (<records>)",
        "    Finished `release` profile [optimized] target(s) in 41.2s",
        "     Running `target/release/turbo-eval cuts <dumps>/a <dumps>/b`",
        "warning: unused variable: `x`",
        "error: could not compile `turbo-eval`",
        "error[E0425]: cannot find value `x` in this scope",
        "<dumps>/b` (exit code: 1) error: process didn't exit successfully",
    ] {
        let found = defects("reports/x.txt", format!("arguana: 1 documents\n{noise}\n").as_bytes());
        assert_eq!(found, [format!("line 2: build tool output {:?}", noise.trim())], "{noise}");
    }
    // PowerShell's wrapper is refused in a log as well.
    let wrapped =
        "turbo-eval.exe : warning\n    + CategoryInfo          : NotSpecified: (warning:String) [], RemoteException\n";
    assert_eq!(defects("logs/gate.log", wrapped.as_bytes()), ["line 2: shell error wrapper RemoteException"]);
    // Each kind of defect is named once, at its first line.
    let two = defects("reports/x.txt", b"a\r\nb\r\nE:\\x\r\n");
    assert_eq!(two, ["line 1: a carriage return; records end their lines with LF", "line 3: host path E:\\"]);
}
