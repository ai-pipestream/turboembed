//! Exercise the image's shared environment resolver without launching processes.
#![cfg(unix)]
use std::process::{Command, Output};

fn resolve(values: &[(&str, &str)]) -> Output {
    Command::new("bash")
        .args(["-euc", "source server/zoo/zoo-env; env | sort"])
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap())
        .envs(values.iter().copied())
        .output()
        .unwrap()
}

fn tls() -> Vec<(&'static str, &'static str)> {
    vec![
        ("TURBO_KSERVE_TLS_CERT", "server.pem"),
        ("TURBO_KSERVE_TLS_KEY", "server.key"),
        ("TURBO_ZOO_TLS_CA", "server-ca.pem"),
    ]
}

fn has(output: &Output, pair: &str) {
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stdout).lines().any(|line| line == pair), "missing {pair}");
}

#[test]
fn plain_and_tls_defaults_follow_the_listener() {
    has(&resolve(&[]), "TURBO_DEMO_SERVER=http://127.0.0.1:8081");
    let mut config = tls();
    config.push(("TURBO_KSERVE_LISTEN", "[::]:9001"));
    config.push(("TURBO_ZOO_TLS_NAME", "zoo.example"));
    let output = resolve(&config);
    has(&output, "TURBO_DEMO_SERVER=https://[::1]:9001");
    for prefix in ["TURBO_DEMO_SERVER_TLS", "TURBO_KSERVE_HEALTH_TLS"] {
        has(&output, &format!("{prefix}_CA=server-ca.pem"));
        has(&output, &format!("{prefix}_NAME=zoo.example"));
    }
}

#[test]
fn mtls_requires_explicit_client_keys_and_preserves_per_client_overrides() {
    let mut config = tls();
    config.extend([
        ("TURBO_KSERVE_TLS_CLIENT_CA", "client-ca.pem"),
        ("TURBO_ZOO_TLS_CERT", "client.pem"),
        ("TURBO_ZOO_TLS_KEY", "client.key"),
        ("TURBO_DEMO_SERVER_TLS_CERT", "page-client.pem"),
        ("TURBO_DEMO_SERVER_TLS_KEY", "page-client.key"),
    ]);
    let output = resolve(&config);
    has(&output, "TURBO_KSERVE_HEALTH_TLS_CERT=client.pem");
    has(&output, "TURBO_KSERVE_HEALTH_TLS_KEY=client.key");
    has(&output, "TURBO_DEMO_SERVER_TLS_CERT=page-client.pem");
    has(&output, "TURBO_DEMO_SERVER_TLS_KEY=page-client.key");
    has(&output, "TURBO_KSERVE_TLS_KEY=server.key");
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
        let output = resolve(&config);
        assert!(!output.status.success(), "accepted {config:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("zoo-env:"), "{output:?}");
    }
}

#[test]
fn image_listener_flags_cannot_hide_configuration_from_docker_health() {
    for arg in ["--listen=127.0.0.1:1", "--listen", "--tls-cert=x", "--tls-key", "--tls-client-ca=x"] {
        let output = Command::new("bash")
            .args(["server/zoo/zoo-start", arg])
            .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap())
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("environment variables"));
    }
}
