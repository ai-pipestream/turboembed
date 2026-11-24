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

/// A closed stderr does not take the server down. The runtime's callback
/// used eprintln!, which panics when the write fails, and a panic cannot
/// unwind out of the extern "C" callback, so the first runtime line after
/// the reader went away aborted the process. The runtime is made by the
/// first load, whose device listing says an info line on stderr.
#[cfg(unix)]
#[tokio::test]
async fn a_closed_stderr_does_not_abort_the_server() {
    use std::io::{BufRead, BufReader};
    use turbo_kserve::proto::result::LoadBundleRequest;
    use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-kserve"))
        .args(["--admin", "--listen", "127.0.0.1:0"])
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut addr = None;
    loop {
        let mut line = String::new();
        assert!(err.read_line(&mut line).unwrap() > 0, "the server exited before ready");
        if let Some(a) = line.trim().strip_prefix("turbo-kserve: listening on ") {
            addr = Some(a.to_owned());
        }
        if line.trim() == "turbo-kserve: ready" {
            break;
        }
    }
    // Only the child's stderr closes; the harness's own stays open.
    drop(err);
    let mut a = AnalysisServiceClient::connect(format!("http://{}", addr.expect("a listening line"))).await.unwrap();
    let req = LoadBundleRequest {
        name: "tiny".into(),
        bundle: tiny().to_str().unwrap().into(),
        sessions: 1,
        device: cpu(),
        ..Default::default()
    };
    let got = a.load_bundle(req).await;
    let exited = child.try_wait().unwrap();
    let _ = child.kill();
    let _ = child.wait();
    assert_eq!(exited, None, "the server died after its stderr closed");
    got.unwrap();
}

/// The listing carries the runtime's warnings, typed, and no others. A
/// runtime warns while it lists its devices when a linked backend's probe
/// fails (a loader with no driver behind it, a driver older than the
/// runtime), so what the server lists is what a runtime made here logged:
/// loading the tiny bundle on the CPU adds nothing. A build with the cpu
/// backend alone has no probe to fail, and lists no entry, none dropped.
#[tokio::test]
async fn the_capability_listing_carries_the_runtime_warnings() {
    let (own, own_dropped) = turbo_kserve::api::Runtime::create().unwrap().log().warnings();
    if turbo_kserve::api::version().split_whitespace().skip(1).eq(["cpu"]) {
        assert_eq!((&own, own_dropped), (&Vec::<RuntimeLogEntry>::new(), 0));
    }
    let (s, mut c) = serve().await;
    let caps = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(caps.runtime_warnings, own);
    assert_eq!(caps.runtime_warnings_dropped, own_dropped);
    s.stop().await;
}

/// The log is the callback's user data, and the core keeps a runtime
/// alive as long as anything made from it, any of which can log. So the
/// log lives until the last handle made from the runtime is released,
/// not until the runtime handle is dropped.
#[test]
fn the_log_outlives_the_runtime_handle_while_its_children_live() {
    use turbo_kserve::api::Runtime;
    let rt = Runtime::create().unwrap();
    let log = std::sync::Arc::downgrade(rt.log());
    let ctx = rt.context(cpu()).unwrap();
    let model = ctx.load(tiny().to_str().unwrap()).unwrap();
    // Every field 0 is the model's own value; only the size is set.
    let mut desc: turbo::turbo_session_desc = unsafe { std::mem::zeroed() };
    desc.struct_size = size_of::<turbo::turbo_session_desc>() as u32;
    let session = model.session(&desc).unwrap();
    drop(rt);
    assert!(log.upgrade().is_some(), "the log was freed while a context, model and session live");
    drop(ctx);
    drop(model);
    assert!(log.upgrade().is_some(), "the log was freed while a session lives");
    // The session still writes and runs; its result holds the log too.
    let t = "hello";
    let text = turbo::turbo_text { ptr: t.as_ptr().cast(), len: t.len() as u64 };
    let mut opts: turbo::turbo_embed_options = unsafe { std::mem::zeroed() };
    opts.struct_size = size_of::<turbo::turbo_embed_options>() as u32;
    session.write_text(&[text], &opts).unwrap();
    let out = session.run().unwrap();
    drop(session);
    assert!(log.upgrade().is_some(), "the log was freed while a result lives");
    drop(out);
    assert!(log.upgrade().is_none(), "the last handle released the log");
}
