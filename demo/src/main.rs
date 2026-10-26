//! turbo-kserve-demo: a web page for trying a running turbo-kserve
//! (docs/demo.md).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use turbo_kserve_demo::{Allow, Demo, Options, Tls, gate};

const USAGE: &str = "usage: turbo-kserve-demo [--listen ADDR:PORT] [--server http://HOST:PORT] [--allow METHOD]...
         [--token-file PATH] [--cors-origin ORIGIN]... [--tls-cert PEM --tls-key PEM]
  --listen       where the page is served (TURBO_DEMO_LISTEN; default 127.0.0.1:8080)
  --server       the turbo-kserve to call (TURBO_DEMO_SERVER; default http://127.0.0.1:8081)
  --allow        also forward a method that changes the server, named
                 /<package>.<Service>/<Method>; repeat for more (TURBO_DEMO_ALLOW,
                 comma-separated); the methods that only read are always forwarded
  --token-file   a file holding the bearer token every call must carry
                 (TURBO_DEMO_TOKEN_FILE, or the token itself in TURBO_DEMO_TOKEN);
                 required off loopback when a method that changes the server is allowed
  --cors-origin  another origin whose pages may call, http(s)://HOST[:PORT]; repeat
                 for more (TURBO_DEMO_CORS_ORIGINS, comma-separated); none by default
  --tls-cert     serve HTTPS with this PEM certificate chain (TURBO_DEMO_TLS_CERT)
  --tls-key      and this PEM private key (TURBO_DEMO_TLS_KEY); both or neither";

#[derive(Debug, PartialEq, Eq)]
struct Args {
    listen: SocketAddr,
    server: String,
    allow: Vec<String>,
    token_file: Option<PathBuf>,
    origins: Vec<String>,
    tls: Option<Tls>,
}

fn add(list: &mut Vec<String>, v: &str) {
    let v = v.trim().to_string();
    if !v.is_empty() && !list.contains(&v) {
        list.push(v);
    }
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let env = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let (mut listen, mut server, mut allow) = (None, None, Vec::new());
    let (mut token_file, mut origins, mut cert, mut key) = (None, Vec::new(), None, None);
    let mut args = args.into_iter();
    while let Some(a) = args.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let mut value = || {
            let v = inline.clone().or_else(|| args.next()).ok_or_else(|| format!("{flag} needs a value"))?;
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return Err(format!("{flag} needs a value"));
            }
            Ok(trimmed.to_string())
        };
        match flag.as_str() {
            "--listen" => listen = Some(value()?),
            "--server" => server = Some(value()?),
            "--allow" => add(&mut allow, &value()?),
            "--token-file" => token_file = Some(value()?),
            "--cors-origin" => add(&mut origins, &gate::origin(&value()?)?),
            "--tls-cert" => cert = Some(value()?),
            "--tls-key" => key = Some(value()?),
            "-h" | "--help" => {
                if inline.is_some() {
                    return Err(format!("{flag} takes no value"));
                }
                return Err(String::new());
            }
            _ => return Err(format!("unknown argument `{a}`")),
        }
    }
    let listen = listen.or_else(|| env("TURBO_DEMO_LISTEN")).unwrap_or_else(|| "127.0.0.1:8080".into());
    let listen = listen.parse().map_err(|e| format!("--listen {listen}: {e}"))?;
    let server = server.or_else(|| env("TURBO_DEMO_SERVER")).unwrap_or_else(|| "http://127.0.0.1:8081".into());
    if allow.is_empty()
        && let Some(v) = env("TURBO_DEMO_ALLOW")
    {
        v.split(',').for_each(|m| add(&mut allow, m));
    }
    Allow::with(&allow)?;
    if origins.is_empty()
        && let Some(v) = env("TURBO_DEMO_CORS_ORIGINS")
    {
        for o in v.split(',').filter(|o| !o.trim().is_empty()) {
            add(&mut origins, &gate::origin(o)?);
        }
    }
    let token_file = token_file.or_else(|| env("TURBO_DEMO_TOKEN_FILE")).map(PathBuf::from);
    let tls = match (cert.or_else(|| env("TURBO_DEMO_TLS_CERT")), key.or_else(|| env("TURBO_DEMO_TLS_KEY"))) {
        (Some(cert), Some(key)) => Some(Tls { cert: cert.into(), key: key.into() }),
        (None, None) => None,
        _ => return Err("--tls-cert and --tls-key go together".into()),
    };
    Ok(Args { listen, server, allow, token_file, origins, tls })
}

/// The token from its file, else from TURBO_DEMO_TOKEN.
fn token(file: Option<&PathBuf>) -> Result<Option<String>, String> {
    match file {
        Some(f) => {
            let t = std::fs::read_to_string(f).map_err(|e| format!("--token-file {}: {e}", f.display()))?;
            Ok(Some(t.trim().to_string()))
        }
        None => Ok(std::env::var("TURBO_DEMO_TOKEN").ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty())),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) if e.is_empty() => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("turbo-kserve-demo: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let allow = Allow::with(&args.allow).expect("checked by parse");
    let token = match token(args.token_file.as_ref()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("turbo-kserve-demo: {e}");
            return ExitCode::from(2);
        }
    };
    let scheme = if args.tls.is_some() { "https" } else { "http" };
    if token.is_some() && args.tls.is_none() && !args.listen.ip().is_loopback() {
        eprintln!("turbo-kserve-demo: the token crosses the network in the clear; use --tls-cert and --tls-key");
    }
    let options = Options { allow, token, origins: args.origins, tls: args.tls };
    let demo = match Demo::start_with(args.listen, &args.server, options).await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("turbo-kserve-demo: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("turbo-kserve-demo: open {scheme}://{} (calling {})", demo.local_addr(), args.server);
    let _ = tokio::signal::ctrl_c().await;
    demo.stop().await;
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Result<Args, String> {
        parse(args.iter().map(|s| (*s).to_string()))
    }

    #[test]
    fn parse_rejects_inline_help_empty_flag_values_and_unknown_methods_and_deduplicates_allow() {
        assert_eq!(run(&["--help"]), Err(String::new()));
        assert_eq!(run(&["-h"]), Err(String::new()));
        assert_eq!(run(&["--help=1"]), Err("--help takes no value".into()));
        assert_eq!(run(&["--listen="]), Err("--listen needs a value".into()));
        assert_eq!(run(&["--listen", "   "]), Err("--listen needs a value".into()));
        assert_eq!(run(&["--server="]), Err("--server needs a value".into()));
        assert_eq!(run(&["--server", ""]), Err("--server needs a value".into()));
        assert_eq!(run(&["--allow="]), Err("--allow needs a value".into()));
        assert_eq!(run(&["--allow", "  "]), Err("--allow needs a value".into()));
        assert!(run(&["--allow", "/turbo.search.v1.TurboSearchService/IndexDocuments"]).is_err());
        let parsed = run(&[
            "--listen=127.0.0.1:9000",
            "--server",
            " http://127.0.0.1:9001 ",
            "--allow",
            "/turbo.search.v1.TurboSearchService/SetCollection",
            "--allow=turbo.search.v1.TurboSearchService/DeleteCollection",
            "--allow",
            " /turbo.search.v1.TurboSearchService/SetCollection ",
        ])
        .unwrap();
        assert_eq!(parsed.listen, "127.0.0.1:9000".parse().unwrap());
        assert_eq!(parsed.server, "http://127.0.0.1:9001");
        assert_eq!(
            parsed.allow,
            vec![
                "/turbo.search.v1.TurboSearchService/SetCollection".to_string(),
                "turbo.search.v1.TurboSearchService/DeleteCollection".to_string()
            ]
        );
    }

    #[test]
    fn parse_takes_origins_a_token_file_and_tls_in_pairs() {
        let parsed = run(&[
            "--cors-origin",
            "https://App.Example",
            "--cors-origin=https://app.example",
            "--token-file",
            "token",
            "--tls-cert=cert.pem",
            "--tls-key",
            "key.pem",
        ])
        .unwrap();
        assert_eq!(parsed.origins, vec!["https://app.example".to_string()]);
        assert_eq!(parsed.token_file, Some(PathBuf::from("token")));
        assert_eq!(parsed.tls, Some(Tls { cert: "cert.pem".into(), key: "key.pem".into() }));
        assert!(run(&["--cors-origin", "*"]).is_err());
        assert!(run(&["--cors-origin", "https://app.example/page"]).is_err());
        assert_eq!(run(&["--tls-cert", "cert.pem"]), Err("--tls-cert and --tls-key go together".into()));
        assert!(token(Some(&PathBuf::from("no-such-token-file"))).is_err());
    }
}
