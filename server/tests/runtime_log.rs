//! The runtime's log reaches the operator (issue #2512). The server used
//! to create its runtime with no log sink, so everything the library
//! said about a backend's probe, a device it could not read or a kernel
//! path it gave up on was discarded. Now every message of info level and
//! above is a line on stderr, and the warnings and errors are kept and
//! listed, typed, by `Capabilities`.

mod common;

use common::*;
use turbo_kserve::proto::result::{CapabilitiesRequest, RuntimeLogEntry};

/// The binary's stderr up to its ready line.
#[cfg(unix)]
fn startup_lines() -> Vec<String> {
    use std::io::{BufRead, BufReader};
    let model = format!("bundle={},device={},sessions=1", tiny().display(), cpu());
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-kserve"))
        .args(["--listen", "127.0.0.1:0", "--model", &model])
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut lines = Vec::new();
    loop {
        let mut line = String::new();
        assert!(err.read_line(&mut line).unwrap() > 0, "the server exited before ready: {lines:?}");
        let ready = line.trim() == "turbo-kserve: ready";
        lines.push(line.trim().to_owned());
        if ready {
            break;
        }
    }
    child.kill().unwrap();
    child.wait().unwrap();
    lines
}

/// What the runtime says while it lists its devices is on stderr: here
/// the one line a CPU build gives, the CPU backend's device count.
#[cfg(unix)]
#[test]
fn the_binary_prints_what_the_runtime_logs() {
    let lines = startup_lines();
    let said = "turbo-kserve: runtime: info: the cpu backend lists 1 device";
    assert!(lines.iter().any(|l| l == said), "no {said:?} in {lines:#?}");
}

/// A CPU runtime that loads the tiny bundle has nothing to warn about,
/// and the listing says so with typed fields: no entry, none dropped.
#[tokio::test]
async fn the_capability_listing_carries_the_runtime_warnings() {
    let (s, mut c) = serve().await;
    let caps = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(caps.runtime_warnings, Vec::<RuntimeLogEntry>::new());
    assert_eq!(caps.runtime_warnings_dropped, 0);
    s.stop().await;
}
