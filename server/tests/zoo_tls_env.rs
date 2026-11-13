//! Exercise the zoo container's `turbo-kserve` subcommands and multicall
//! entrypoints without a shell or `curl`.
#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::os::unix::process::CommandExt;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_turbo-kserve");

fn run_subcommand(sub: &str, args: &[&str], values: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(BIN);
    cmd.arg(sub).args(args).env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        cmd.env("PATH", path);
    }
    cmd.envs(values.iter().copied()).output().unwrap()
}

fn run_multicall(name: &str, args: &[&str], values: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(BIN);
    cmd.arg0(name).args(args).env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        cmd.env("PATH", path);
    }
    cmd.envs(values.iter().copied()).output().unwrap()
}

fn tls() -> Vec<(&'static str, &'static str)> {
    vec![
        ("TURBO_KSERVE_TLS_CERT", "server.pem"),
        ("TURBO_KSERVE_TLS_KEY", "server.key"),
        ("TURBO_ZOO_TLS_CA", "server-ca.pem"),
    ]
}

#[test]
fn incomplete_credentials_and_plaintext_downgrades_fail_before_launch() {
    let mut bad = vec![
        vec![("TURBO_KSERVE_TLS_CERT", "server.pem")],
        vec![("TURBO_KSERVE_TLS_KEY", "server.key")],
        vec![("TURBO_KSERVE_TLS_CLIENT_CA", "ca.pem")],
        vec![("TURBO_ZOO_TLS_CA", "ca.pem")],
        tls()[..2].to_vec(),
    ];
    for extra in [
        ("TURBO_KSERVE_TLS_CLIENT_CA", "client-ca.pem"),
        ("TURBO_ZOO_TLS_CERT", "client.pem"),
        ("TURBO_ZOO_TLS_KEY", "client.key"),
        ("TURBO_DEMO_SERVER", "http://127.0.0.1:8081"),
        ("TURBO_ZOO_TLS_CA", " "),
    ] {
        let mut config = tls();
        config.push(extra);
        bad.push(config);
    }
    for config in bad {
        for sub in ["zoo-start", "zoo-health"] {
            let output = run_subcommand(sub, &[], &config);
            assert!(!output.status.success(), "{sub} accepted {config:?}");
            assert!(String::from_utf8_lossy(&output.stderr).contains("zoo-env:"), "{sub}: {output:?}");
        }
    }
}

#[test]
fn image_listener_flags_cannot_hide_configuration_from_docker_health() {
    for arg in ["--listen=127.0.0.1:1", "--listen", "--tls-cert=x", "--tls-key", "--tls-client-ca=x"] {
        for output in [run_subcommand("zoo-start", &[arg], &[]), run_multicall("zoo-start", &[arg], &[])] {
            assert_eq!(output.status.code(), Some(2), "{arg}: {output:?}");
            assert!(String::from_utf8_lossy(&output.stderr).contains("environment variables"), "{output:?}");
        }
    }
    for sub in ["zoo-health", "zoo-bucket"] {
        let output = run_subcommand(sub, &["--unexpected"], &[]);
        assert_eq!(output.status.code(), Some(2), "{sub}: {output:?}");
    }
}

#[cfg(feature = "catalog-http")]
#[test]
fn zoo_bucket_subcommand_and_multicall_create_store_bucket() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        for reply in ["200 OK", "409 Conflict"] {
            let (mut conn, _) = listener.accept().unwrap();
            for line in BufReader::new(conn.try_clone().unwrap()).lines() {
                if line.unwrap().is_empty() {
                    break;
                }
            }
            write!(conn, "HTTP/1.1 {reply}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
        }
    });

    let env = [
        ("TURBO_KSERVE_CATALOG_STORE", "s3://turbo-zoo/zoo"),
        ("AWS_ENDPOINT_URL", endpoint.as_str()),
        ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
        ("AWS_SECRET_KEY_UNUSED", "ignored"),
        ("AWS_SECRET_ACCESS_KEY", "secret"),
    ];
    let first = run_subcommand("zoo-bucket", &[], &env);
    assert!(first.status.success(), "{first:?}");
    assert!(String::from_utf8_lossy(&first.stdout).contains("made turbo-zoo"), "{first:?}");

    let second = run_multicall("zoo-bucket", &[], &env);
    assert!(second.status.success(), "{second:?}");
    assert!(String::from_utf8_lossy(&second.stdout).contains("already there"), "{second:?}");
    server.join().unwrap();
}
