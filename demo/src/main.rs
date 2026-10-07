//! turbo-kserve-demo: a web page for trying a running turbo-kserve
//! (docs/demo.md).

use std::net::SocketAddr;
use std::process::ExitCode;

use turbo_kserve_demo::Demo;

const USAGE: &str = "usage: turbo-kserve-demo [--listen ADDR:PORT] [--server http://HOST:PORT] [--model NAME]...
  --listen  where the page is served (TURBO_DEMO_LISTEN; default 127.0.0.1:8080)
  --server  the turbo-kserve to call (TURBO_DEMO_SERVER; default http://127.0.0.1:8081)
  --model   a served model name the page offers; repeat for more
            (TURBO_DEMO_MODELS, comma-separated); more can be added on the page";

struct Args {
    listen: SocketAddr,
    server: String,
    models: Vec<String>,
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let (mut listen, mut server, mut models) = (None, None, Vec::new());
    let mut args = args.into_iter();
    while let Some(a) = args.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let mut value = || inline.clone().or_else(|| args.next()).ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--listen" => listen = Some(value()?),
            "--server" => server = Some(value()?),
            "--model" => models.push(value()?),
            "-h" | "--help" => return Err(String::new()),
            _ => return Err(format!("unknown argument `{a}`")),
        }
    }
    let listen = listen.or_else(|| env("TURBO_DEMO_LISTEN")).unwrap_or_else(|| "127.0.0.1:8080".into());
    let listen = listen.parse().map_err(|e| format!("--listen {listen}: {e}"))?;
    let server = server.or_else(|| env("TURBO_DEMO_SERVER")).unwrap_or_else(|| "http://127.0.0.1:8081".into());
    if models.is_empty() {
        models = env("TURBO_DEMO_MODELS")
            .map(|v| v.split(',').map(str::trim).filter(|m| !m.is_empty()).map(String::from).collect())
            .unwrap_or_default();
    }
    Ok(Args { listen, server, models })
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
    let demo = match Demo::start(args.listen, &args.server, args.models).await {
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
