//! `turbo-kserve zoo-start`: the zoo container entrypoint.
//!
//! Launches `turbo-kserve` (forwarding any server flags) alongside
//! `turbo-kserve-demo`, shares resolved environment settings with both, stops
//! the sibling process as soon as either exits, and forwards `SIGINT`/`SIGTERM`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{ExitCode, ExitStatus};

use tokio::process::{Child, Command};

use super::env::{DEFAULT_DEMO_USER, ENV_DEMO_TOKEN, ENV_DEMO_TOKEN_FILE, ENV_DEMO_USER, check_start_args, resolve};

const SERVER_BIN_NAME: &str = "turbo-kserve";
const DEMO_BIN_NAME: &str = "turbo-kserve-demo";
const TOKEN_RANDOM_BYTES: usize = 24;

pub(super) async fn run(args: Vec<String>) -> ExitCode {
    if let Err(msg) = check_start_args(&args) {
        eprintln!("zoo-start: {msg}");
        return ExitCode::from(2);
    }
    let mut child_env = match resolve(|k| std::env::var(k).ok()) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("zoo-env: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = populate_demo_defaults(&mut child_env, |k| std::env::var(k).ok()) {
        eprintln!("zoo-start: {e}");
        return ExitCode::FAILURE;
    }

    let (server_exe, demo_exe) = binaries();
    let mut server_cmd = server_command(&server_exe, &args, &child_env);
    let mut demo_cmd = demo_command(&demo_exe, &child_env);
    supervise(&mut server_cmd, &mut demo_cmd, crate::stop_signal()).await
}

fn populate_demo_defaults(
    out: &mut BTreeMap<String, String>,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<(), String> {
    let non_empty = |k: &str| lookup(k).filter(|v| !v.is_empty());
    if non_empty(ENV_DEMO_USER).is_none() {
        out.insert(ENV_DEMO_USER.to_owned(), DEFAULT_DEMO_USER.to_owned());
    }
    if non_empty(ENV_DEMO_TOKEN).is_none() && non_empty(ENV_DEMO_TOKEN_FILE).is_none() {
        let token = generate_token()?;
        eprintln!("zoo-start: the page's token for this run is {token} (set {ENV_DEMO_TOKEN} to choose one)");
        out.insert(ENV_DEMO_TOKEN.to_owned(), token);
    }
    Ok(())
}

#[cfg(unix)]
fn generate_token() -> Result<String, String> {
    use std::io::Read;
    let mut file = std::fs::File::open("/dev/urandom").map_err(|e| format!("/dev/urandom: {e}"))?;
    let mut buf = [0u8; TOKEN_RANDOM_BYTES];
    file.read_exact(&mut buf).map_err(|e| format!("/dev/urandom: {e}"))?;
    let mut hex = String::with_capacity(TOKEN_RANDOM_BYTES * 2);
    for byte in buf {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

#[cfg(not(unix))]
fn generate_token() -> Result<String, String> {
    Err(format!("set {ENV_DEMO_TOKEN} or {ENV_DEMO_TOKEN_FILE}"))
}

fn binaries() -> (PathBuf, PathBuf) {
    let server = std::env::current_exe()
        .and_then(|p| p.canonicalize().or(Ok(p)))
        .unwrap_or_else(|_| PathBuf::from(SERVER_BIN_NAME));
    let demo_name = format!("{DEMO_BIN_NAME}{}", std::env::consts::EXE_SUFFIX);
    let demo = server
        .parent()
        .map(|dir| dir.join(&demo_name))
        .filter(|candidate| candidate.is_file())
        .unwrap_or_else(|| PathBuf::from(demo_name));
    (server, demo)
}

fn server_command(exe: &Path, args: &[String], child_env: &BTreeMap<String, String>) -> Command {
    let mut std_cmd = std::process::Command::new(exe);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        std_cmd.arg0(SERVER_BIN_NAME);
    }
    std_cmd.args(args).envs(child_env);
    Command::from(std_cmd)
}

fn demo_command(exe: &Path, child_env: &BTreeMap<String, String>) -> Command {
    let mut cmd = Command::new(exe);
    cmd.envs(child_env);
    cmd
}

async fn supervise(
    server_cmd: &mut Command,
    demo_cmd: &mut Command,
    stop: impl std::future::Future<Output = &'static str>,
) -> ExitCode {
    let mut server = match server_cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("zoo-start: {SERVER_BIN_NAME}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut demo = match demo_cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("zoo-start: {DEMO_BIN_NAME}: {e}");
            terminate(&mut server);
            let _ = server.wait().await;
            return ExitCode::FAILURE;
        }
    };

    tokio::select! {
        res = server.wait() => {
            terminate(&mut demo);
            let _ = demo.wait().await;
            status_to_code(res)
        }
        res = demo.wait() => {
            terminate(&mut server);
            let _ = server.wait().await;
            status_to_code(res)
        }
        _ = stop => {
            terminate(&mut server);
            terminate(&mut demo);
            let s = server.wait().await;
            let d = demo.wait().await;
            let s_code = status_to_code(s);
            if s_code != ExitCode::SUCCESS {
                s_code
            } else {
                status_to_code(d)
            }
        }
    }
}

fn terminate(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id()
        && let Ok(pid) = i32::try_from(pid)
    {
        // SAFETY: `child.id()` is `Some` only while `tokio` has not yet reaped
        // this child process, so `pid` refers to our own unwaited child.
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        return;
    }
    let _ = child.start_kill();
}

fn status_to_code(res: std::io::Result<ExitStatus>) -> ExitCode {
    let Ok(status) = res else { return ExitCode::FAILURE };
    if let Some(code) = status.code() {
        return ExitCode::from(code as u8);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return ExitCode::from(128u8.saturating_add(sig as u8));
        }
    }
    ExitCode::FAILURE
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn populate_defaults_sets_zoo_user_and_48_hex_token_when_unset() {
        let mut out = BTreeMap::new();
        populate_demo_defaults(&mut out, |_| None).unwrap();
        assert_eq!(out.get(ENV_DEMO_USER).map(String::as_str), Some("zoo"));
        let token = out.get(ENV_DEMO_TOKEN).expect("generated token");
        assert_eq!(token.len(), 48);
        assert!(token.bytes().all(|b| b.is_ascii_hexdigit()));

        let mut kept = BTreeMap::new();
        populate_demo_defaults(&mut kept, |k| match k {
            ENV_DEMO_USER => Some("alice".into()),
            ENV_DEMO_TOKEN_FILE => Some("/run/secrets/token".into()),
            _ => None,
        })
        .unwrap();
        assert!(!kept.contains_key(ENV_DEMO_USER));
        assert!(!kept.contains_key(ENV_DEMO_TOKEN));
    }

    #[tokio::test]
    async fn supervise_stops_sibling_and_returns_first_child_status() {
        let mut failing = Command::new("sh");
        failing.args(["-c", "exit 7"]);
        let mut sleeper = Command::new("sleep");
        sleeper.arg("30");
        let code = supervise(&mut failing, &mut sleeper, std::future::pending()).await;
        assert_eq!(code, ExitCode::from(7));

        let mut sleeper_a = Command::new("sh");
        sleeper_a.args(["-c", "trap 'exit 0' TERM; sleep 30 & wait $!"]);
        let mut sleeper_b = Command::new("sh");
        sleeper_b.args(["-c", "trap 'exit 0' TERM; sleep 30 & wait $!"]);
        let stop = async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            "SIGTERM"
        };
        let clean = supervise(&mut sleeper_a, &mut sleeper_b, stop).await;
        assert_eq!(clean, ExitCode::SUCCESS);
    }
}
