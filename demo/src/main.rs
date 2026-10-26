//! turbo-kserve-demo: a web page for trying a running turbo-kserve
//! (docs/demo.md).

use std::net::SocketAddr;
use std::process::ExitCode;

use turbo_kserve_demo::{Allow, Demo};

const USAGE: &str = "usage: turbo-kserve-demo [--listen ADDR:PORT] [--server http://HOST:PORT] [--allow METHOD]...
  --listen  where the page is served (TURBO_DEMO_LISTEN; default 127.0.0.1:8080)
  --server  the turbo-kserve to call (TURBO_DEMO_SERVER; default http://127.0.0.1:8081)
  --allow   also forward a method that changes the server, named
            /<package>.<Service>/<Method>; repeat for more (TURBO_DEMO_ALLOW,
            comma-separated); the methods that only read are always forwarded";

#[derive(Debug, PartialEq, Eq)]
struct Args {
    listen: SocketAddr,
    server: String,
    allow: Vec<String>,
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let env = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let (mut listen, mut server, mut allow) = (None, None, Vec::new());
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
            "--allow" => {
                let m = value()?;
                if !allow.contains(&m) {
                    allow.push(m);
                }
            }
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
        for m in v.split(',').map(str::trim).filter(|m| !m.is_empty()) {
            let m = m.to_string();
            if !allow.contains(&m) {
                allow.push(m);
            }
        }
    }
    Allow::with(&allow)?;
    Ok(Args { listen, server, allow })
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
    let demo = match Demo::start(args.listen, &args.server, allow).await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("turbo-kserve-demo: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("turbo-kserve-demo: open http://{} (calling {})", demo.local_addr(), args.server);
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
}
