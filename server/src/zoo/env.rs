//! Shared environment resolution for `zoo-start` and `zoo-health`.
//!
//! Listener TLS keys never become client keys. Shared `TURBO_ZOO_TLS_*`
//! variables supply defaults for both loopback clients (`turbo-kserve-demo`
//! and `turbo-kserve health`), while per-binary variables override them.

use std::collections::BTreeMap;

use turbo_kserve::config::ENV_LISTEN;

pub(super) const DEFAULT_SERVER_LISTEN: &str = "0.0.0.0:8081";
pub(super) const DEFAULT_DEMO_USER: &str = "zoo";
pub(super) const ENV_DEMO_SERVER: &str = "TURBO_DEMO_SERVER";
pub(super) const ENV_DEMO_USER: &str = "TURBO_DEMO_USER";
pub(super) const ENV_DEMO_TOKEN: &str = "TURBO_DEMO_TOKEN";
pub(super) const ENV_DEMO_TOKEN_FILE: &str = "TURBO_DEMO_TOKEN_FILE";

const ENV_SERVER_TLS_CERT: &str = "TURBO_KSERVE_TLS_CERT";
const ENV_SERVER_TLS_KEY: &str = "TURBO_KSERVE_TLS_KEY";
const ENV_SERVER_TLS_CLIENT_CA: &str = "TURBO_KSERVE_TLS_CLIENT_CA";
const ENV_ZOO_TLS_PREFIX: &str = "TURBO_ZOO_TLS";
const ENV_DEMO_SERVER_TLS_PREFIX: &str = "TURBO_DEMO_SERVER_TLS";
const ENV_HEALTH_TLS_PREFIX: &str = "TURBO_KSERVE_HEALTH_TLS";
const ENV_SERVER_TLS_PREFIX: &str = "TURBO_KSERVE_TLS";

const TLS_PREFIXES: [&str; 4] =
    [ENV_ZOO_TLS_PREFIX, ENV_DEMO_SERVER_TLS_PREFIX, ENV_HEALTH_TLS_PREFIX, ENV_SERVER_TLS_PREFIX];
const CLIENT_PREFIXES: [&str; 2] = [ENV_DEMO_SERVER_TLS_PREFIX, ENV_HEALTH_TLS_PREFIX];
const ALL_TLS_SUFFIXES: [&str; 5] = ["CA", "CERT", "KEY", "NAME", "CLIENT_CA"];
const CLIENT_TLS_SUFFIXES: [&str; 4] = ["CA", "CERT", "KEY", "NAME"];

const SCHEME_HTTP: &str = "http";
const SCHEME_HTTPS: &str = "https";
const WILDCARD_V4_PREFIX: &str = "0.0.0.0:";
const WILDCARD_V6_PREFIX: &str = "[::]:";
const LOOPBACK_V4_HOST: &str = "127.0.0.1";
const LOOPBACK_V6_HOST: &str = "[::1]";

/// Rejects `--listen` and `--tls-*` command-line flags on `zoo-start` so
/// listener and TLS settings remain visible to Docker's independent health
/// check process through environment variables.
pub(super) fn check_start_args(args: &[String]) -> Result<(), &'static str> {
    for arg in args {
        if arg == "--listen" || arg.starts_with("--listen=") || arg.starts_with("--tls-") {
            return Err("configure listener and TLS through TURBO_KSERVE_* environment variables");
        }
    }
    Ok(())
}

/// Resolves the derived environment overrides for `zoo-start` and `zoo-health`
/// from `lookup`, validating listener/client TLS consistency without mutating
/// the current process environment.
pub(super) fn resolve(lookup: impl Fn(&str) -> Option<String>) -> Result<BTreeMap<String, String>, String> {
    for prefix in TLS_PREFIXES {
        for suffix in ALL_TLS_SUFFIXES {
            let name = format!("{prefix}_{suffix}");
            if let Some(value) = lookup(&name)
                && !value.is_empty()
                && value.trim().is_empty()
            {
                return Err(format!("{name} must not be blank"));
            }
        }
    }

    let get = |key: &str| lookup(key).filter(|v| !v.is_empty());
    let mut out = BTreeMap::new();

    let server_cert = get(ENV_SERVER_TLS_CERT);
    let server_key = get(ENV_SERVER_TLS_KEY);
    let client_ca = get(ENV_SERVER_TLS_CLIENT_CA);
    let scheme = if server_cert.is_some() || server_key.is_some() || client_ca.is_some() {
        if server_cert.is_none() || server_key.is_none() {
            return Err(format!("{ENV_SERVER_TLS_CERT} and {ENV_SERVER_TLS_KEY} are required together"));
        }
        SCHEME_HTTPS
    } else {
        SCHEME_HTTP
    };

    let listen = get(ENV_LISTEN).unwrap_or_else(|| DEFAULT_SERVER_LISTEN.to_owned());
    let dial = loopback_dial(&listen);
    let demo_server = get(ENV_DEMO_SERVER).unwrap_or_else(|| format!("{scheme}://{dial}"));
    let required_prefix = format!("{scheme}://");
    if !demo_server.starts_with(&required_prefix) {
        return Err(format!("{ENV_DEMO_SERVER} must use the server listener TLS scheme"));
    }
    out.insert(ENV_LISTEN.to_owned(), listen);
    out.insert(ENV_DEMO_SERVER.to_owned(), demo_server);

    for prefix in CLIENT_PREFIXES {
        for suffix in CLIENT_TLS_SUFFIXES {
            let source = format!("{ENV_ZOO_TLS_PREFIX}_{suffix}");
            let target = format!("{prefix}_{suffix}");
            if get(&target).is_none()
                && let Some(val) = get(&source)
            {
                out.insert(target, val);
            }
        }
        let effective = |suffix: &str| {
            let key = format!("{prefix}_{suffix}");
            out.get(&key).cloned().or_else(|| get(&key))
        };
        let ca = effective("CA");
        let cert = effective("CERT");
        let key = effective("KEY");
        let name = effective("NAME");
        if scheme == SCHEME_HTTP && (ca.is_some() || cert.is_some() || key.is_some() || name.is_some()) {
            return Err("client TLS settings require server listener TLS".into());
        }
        if scheme == SCHEME_HTTPS && ca.is_none() {
            return Err(format!("{prefix} needs a CA (or {ENV_ZOO_TLS_PREFIX}_CA)"));
        }
        if cert.is_some() != key.is_some() {
            return Err(format!("{prefix} client certificate and key are required together"));
        }
        if client_ca.is_some() && cert.is_none() {
            return Err(format!("mutual TLS requires a client identity for {prefix}"));
        }
    }

    Ok(out)
}

fn loopback_dial(listen: &str) -> String {
    if let Some(rest) = listen.strip_prefix(WILDCARD_V4_PREFIX) {
        let port = rest.rsplit(':').next().unwrap_or(rest);
        format!("{LOOPBACK_V4_HOST}:{port}")
    } else if let Some(rest) = listen.strip_prefix(WILDCARD_V6_PREFIX) {
        let port = rest.rsplit(':').next().unwrap_or(rest);
        format!("{LOOPBACK_V6_HOST}:{port}")
    } else {
        listen.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| pairs.iter().rev().find(|(k, _)| *k == key).map(|(_, v)| (*v).to_owned())
    }

    #[test]
    fn plain_and_tls_defaults_follow_the_listener() {
        let plain = resolve(lookup(&[])).unwrap();
        assert_eq!(plain.get(ENV_LISTEN).map(String::as_str), Some("0.0.0.0:8081"));
        assert_eq!(plain.get(ENV_DEMO_SERVER).map(String::as_str), Some("http://127.0.0.1:8081"));

        let tls = resolve(lookup(&[
            ("TURBO_KSERVE_TLS_CERT", "server.pem"),
            ("TURBO_KSERVE_TLS_KEY", "server.key"),
            ("TURBO_ZOO_TLS_CA", "server-ca.pem"),
            ("TURBO_KSERVE_LISTEN", "[::]:9001"),
            ("TURBO_ZOO_TLS_NAME", "zoo.example"),
        ]))
        .unwrap();
        assert_eq!(tls.get(ENV_DEMO_SERVER).map(String::as_str), Some("https://[::1]:9001"));
        for prefix in ["TURBO_DEMO_SERVER_TLS", "TURBO_KSERVE_HEALTH_TLS"] {
            assert_eq!(tls.get(&format!("{prefix}_CA")).map(String::as_str), Some("server-ca.pem"));
            assert_eq!(tls.get(&format!("{prefix}_NAME")).map(String::as_str), Some("zoo.example"));
        }
    }

    #[test]
    fn mtls_preserves_per_client_overrides_without_touching_listener_keys() {
        let resolved = resolve(lookup(&[
            ("TURBO_KSERVE_TLS_CERT", "server.pem"),
            ("TURBO_KSERVE_TLS_KEY", "server.key"),
            ("TURBO_ZOO_TLS_CA", "server-ca.pem"),
            ("TURBO_KSERVE_TLS_CLIENT_CA", "client-ca.pem"),
            ("TURBO_ZOO_TLS_CERT", "client.pem"),
            ("TURBO_ZOO_TLS_KEY", "client.key"),
            ("TURBO_DEMO_SERVER_TLS_CERT", "page-client.pem"),
            ("TURBO_DEMO_SERVER_TLS_KEY", "page-client.key"),
        ]))
        .unwrap();
        assert_eq!(resolved.get("TURBO_KSERVE_HEALTH_TLS_CERT").map(String::as_str), Some("client.pem"));
        assert_eq!(resolved.get("TURBO_KSERVE_HEALTH_TLS_KEY").map(String::as_str), Some("client.key"));
        assert_eq!(resolved.get("TURBO_DEMO_SERVER_TLS_CERT"), None);
        assert_eq!(resolved.get("TURBO_DEMO_SERVER_TLS_KEY"), None);
    }

    #[test]
    fn incomplete_credentials_and_plaintext_downgrades_are_rejected() {
        let base = [
            ("TURBO_KSERVE_TLS_CERT", "server.pem"),
            ("TURBO_KSERVE_TLS_KEY", "server.key"),
            ("TURBO_ZOO_TLS_CA", "server-ca.pem"),
        ];
        let mut bad: Vec<Vec<(&str, &str)>> = vec![
            vec![("TURBO_KSERVE_TLS_CERT", "server.pem")],
            vec![("TURBO_KSERVE_TLS_KEY", "server.key")],
            vec![("TURBO_KSERVE_TLS_CLIENT_CA", "ca.pem")],
            vec![("TURBO_ZOO_TLS_CA", "ca.pem")],
            vec![("TURBO_KSERVE_TLS_CERT", "server.pem"), ("TURBO_KSERVE_TLS_KEY", "server.key")],
        ];
        for extra in [
            ("TURBO_KSERVE_TLS_CLIENT_CA", "client-ca.pem"),
            ("TURBO_ZOO_TLS_CERT", "client.pem"),
            ("TURBO_ZOO_TLS_KEY", "client.key"),
            ("TURBO_DEMO_SERVER", "http://127.0.0.1:8081"),
            ("TURBO_ZOO_TLS_CA", " "),
        ] {
            let mut cfg = base.to_vec();
            cfg.push(extra);
            bad.push(cfg);
        }
        for cfg in bad {
            assert!(resolve(lookup(&cfg)).is_err(), "accepted {cfg:?}");
        }
    }

    #[test]
    fn start_args_refuse_listener_and_tls_flags() {
        for bad in ["--listen", "--listen=127.0.0.1:1", "--tls-cert=x", "--tls-key", "--tls-client-ca=x"] {
            assert!(check_start_args(&[bad.to_owned()]).is_err(), "{bad}");
        }
        assert!(check_start_args(&["--catalog-prefetch=permissive".to_owned(), "--admin".to_owned()]).is_ok());
    }
}
