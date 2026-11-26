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

/// The drive letter of a Windows path (`C:\dir` or `C:/dir`) in `line`,
/// if it has one. In JSON a backslash is written twice, so `"Q:\n"` there
/// is a line end inside a string, not a path. A URL is no drive path: the
/// letter before its colon ends a longer word, and two slashes follow.
fn drive_path(line: &str, json: bool) -> Option<char> {
    let b = line.as_bytes();
    let drive = |at: usize, escaped: bool| {
        let letter = *b.get(at.checked_sub(1)?)?;
        let starts_a_word = at < 2 || !b[at - 2].is_ascii_alphanumeric();
        (letter.is_ascii_alphabetic() && starts_a_word && escaped).then_some(letter as char)
    };
    let back = line.match_indices(":\\").find_map(|(at, _)| drive(at, !json || b.get(at + 2) == Some(&b'\\')));
    back.or_else(|| line.match_indices(":/").find_map(|(at, _)| drive(at, b.get(at + 2) != Some(&b'/'))))
}

/// Whether `line` holds a Windows network path (`\\server\share`): two
/// backslashes, four in JSON, that start a word and stand before a name.
fn network_path(line: &str, json: bool) -> bool {
    let b = line.as_bytes();
    let lead = if json { "\\\\\\\\" } else { "\\\\" };
    line.match_indices(lead).any(|(at, _)| {
        let starts_a_word = at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'\\');
        starts_a_word && b.get(at + lead.len()).is_some_and(u8::is_ascii_alphanumeric)
    })
}

/// The schema of a receipt `analysis/eval/reference/throughput_receipt.py`
/// writes.
const RECEIPT_SCHEMA: &str = "turboembed.throughput-receipt.v1";

/// Whether `path` names a place by its root: `/dir`, `C:...` or `\\server`.
fn rooted(path: &str) -> bool {
    path.starts_with(['/', '\\']) || path.as_bytes().get(1) == Some(&b':')
}

/// The host directories a throughput receipt records as they were on the
/// machine that made it: the host side of a docker bind mount in `command`,
/// and `output_file`. The tool writes each as a placeholder (`<out-dir>`),
/// and a receipt written by hand or by an older tool is held to the same.
/// Nothing for a record that is no receipt.
fn receipt_host_paths(text: &str) -> Vec<String> {
    if !text.contains(RECEIPT_SCHEMA) {
        return Vec::new();
    }
    let Ok(receipt) = serde_json::from_str::<serde_json::Value>(text) else {
        return vec!["a throughput receipt that is not JSON".into()];
    };
    if receipt["schema"] != RECEIPT_SCHEMA {
        return Vec::new();
    }
    let Some(command) = receipt["command"].as_array() else {
        return vec!["a throughput receipt with no command".into()];
    };
    let args: Vec<&str> = command.iter().filter_map(|a| a.as_str()).collect();
    let mut found = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        let spec = match arg.strip_prefix("--volume=") {
            Some(spec) => spec,
            None if i > 0 && matches!(args[i - 1], "-v" | "--volume") => arg,
            None => continue,
        };
        // The container side starts at the last `:/`; a named volume has
        // no separator before it.
        let host = spec.rfind(":/").map_or("", |at| &spec[..at]);
        if host.contains(['/', '\\']) && !host.starts_with('<') {
            found.push(format!("command mounts host directory {host}; a receipt names it by a placeholder"));
        }
    }
    if let Some(file) = receipt["output_file"].as_str()
        && rooted(file)
    {
        found.push(format!("output_file is host path {file}; a receipt names its directory by a placeholder"));
    }
    found
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
    let marks =
        [&["\r", "\0", ":\\", ":/", "\\\\", FAILED_RUN][..], &HOST_PATHS, &SHELL_WRAPPERS, &CARGO_STATUS, &DIAGNOSTICS];
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
        if network_path(line, json) {
            note("network", n, "a network path (\\\\server\\share)".into());
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
    let (mut seen, mut receipts) = (0, 0);
    for name in String::from_utf8(out.stdout).unwrap().split('\0').filter(|n| !n.is_empty()) {
        seen += 1;
        let bytes = std::fs::read(root.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        bad.extend(defects(name, &bytes).into_iter().map(|d| format!("{name}: {d}")));
        if let Ok(text) = std::str::from_utf8(&bytes) {
            receipts += usize::from(text.contains(RECEIPT_SCHEMA));
            bad.extend(receipt_host_paths(text).into_iter().map(|d| format!("{name}: {d}")));
        }
    }
    assert!(seen > 600, "only {seen} evaluation records scanned");
    assert!(receipts >= 7, "only {receipts} throughput receipts scanned");
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
    // Nor is a URL, a mount of a placeholder, or a backslash before a letter.
    let plain =
        "see s3://bucket/a and file:///srv/a; -v <out-dir>:/out; a ratio 1:/2; \\alpha is one escaped backslash\n";
    assert_eq!(defects("x.json", plain.as_bytes()), Vec::<String>::new());
}

/// A receipt as the tool writes it, with `mount` as its first bind mount
/// and `output` as its `output_file` member.
fn receipt(mount: &str, output: &str) -> String {
    format!(
        r#"{{"schema": "{RECEIPT_SCHEMA}", "command": ["docker", "run", "-v", "{mount}", "--volume=<out-dir>:/out",
            "-v", "cache:/root/.cache", "image", "/gold/test.conllu", "/out/pred.conllu"], "output_file": {output}}}"#
    )
}

#[test]
fn a_receipt_names_host_directories_by_placeholders() {
    let none = Vec::<String>::new();
    assert_eq!(receipt_host_paths(&receipt("<gold-dir>:/gold:ro", r#""<out-dir>/pred.conllu""#)), none);
    // No output file, or one beside the working directory, names no host directory.
    assert_eq!(receipt_host_paths(&receipt("<gold-dir>:/gold:ro", "null")), none);
    assert_eq!(receipt_host_paths(&receipt("<gold-dir>:/gold:ro", r#""out/pred.conllu""#)), none);
    // A record of another schema is not read as a receipt.
    assert_eq!(receipt_host_paths(r#"{"schema": "other", "output_file": "/srv/out.txt"}"#), none);

    let mounted = receipt_host_paths(&receipt("/srv/ud-data:/gold:ro", r#""<out-dir>/pred.conllu""#));
    assert_eq!(mounted, ["command mounts host directory /srv/ud-data; a receipt names it by a placeholder"]);
    let drive = receipt_host_paths(&receipt("D:/ud:/gold", "null"));
    assert_eq!(drive, ["command mounts host directory D:/ud; a receipt names it by a placeholder"]);
    let output = receipt_host_paths(&receipt("<gold-dir>:/gold:ro", r#""/scratch/run/pred.conllu""#));
    assert_eq!(
        output,
        ["output_file is host path /scratch/run/pred.conllu; a receipt names its directory by a placeholder"]
    );
    let both = receipt_host_paths(&receipt("/srv/ud-data:/gold", r#""/scratch/run/pred.conllu""#));
    assert_eq!(both.len(), 2);
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
    // A drive path written with forward slashes, and a network path.
    assert_eq!(defects("reports/x.txt", b"wrote D:/scratch/out/a.json\n"), ["line 1: host path D:\\"]);
    assert_eq!(defects("x.json", br#"{"dir": "D:/scratch/out"}"#), ["line 1: host path D:\\"]);
    let share = ["line 1: a network path (\\\\server\\share)"];
    assert_eq!(defects("reports/x.txt", b"wrote \\\\files01\\eval\\a.json\n"), share);
    assert_eq!(defects("x.json", br#"{"dir": "\\\\files01\\eval"}"#), share);
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
