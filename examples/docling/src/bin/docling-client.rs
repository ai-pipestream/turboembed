//! The docling example against a running `turbo-kserve`.
//!
//! ```text
//! docling-client seed DOCLING_JSON OUT
//! docling-client run ADDR DOCLING_JSON [--report]
//! ```
//!
//! `seed` writes the docling structure as `initial_graph` bytes, which the
//! Java and Python clients send. `run` sends it to the server at `ADDR`
//! (`http://127.0.0.1:8081`) and prints the summary every client prints;
//! `--report` prints the full report as JSON instead.

use std::process::ExitCode;

use turbo_example_docling::{remote, server};
use turbo_kserve::proto::result::OffsetMode;

const USAGE: &str = "usage: docling-client seed DOCLING_JSON OUT | run ADDR DOCLING_JSON [--report]";

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}

async fn run(addr: &str, path: &str, full: bool) -> Result<String, String> {
    let (d, seed) = server::seed(&read(path)?)?;
    let mut client = server::connect(addr).await?;
    let answer = server::run(&mut client, seed, OffsetMode::Utf8).await?;
    if !full {
        return remote::summary(&answer);
    }
    let doc = server::document(&answer)?;
    let report = remote::report(&d, &doc, &answer)?;
    serde_json::to_string_pretty(&report).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let done = match args.as_slice() {
        ["seed", json, out] => read(json)
            .and_then(|j| server::seed(&j))
            .and_then(|(_, bytes)| std::fs::write(out, bytes).map_err(|e| format!("{out}: {e}"))),
        ["run", addr, json, rest @ ..] if rest.is_empty() || rest == ["--report"] => {
            match tokio::runtime::Runtime::new() {
                Ok(rt) => rt.block_on(run(addr, json, !rest.is_empty())).map(|out| print!("{out}")),
                Err(e) => Err(e.to_string()),
            }
        }
        _ => Err(USAGE.to_owned()),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("docling-client: {e}");
            ExitCode::FAILURE
        }
    }
}
