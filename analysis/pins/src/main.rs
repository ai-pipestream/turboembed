//! `turbo-pins <path>...`: the models.txt line of each model, pinned or
//! not, so that a model meant for a test can be pinned by a reviewed line.

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut failed = false;
    for arg in std::env::args().skip(1) {
        let path = Path::new(&arg);
        match turbo_pins::digest(path) {
            Ok((kind, hex)) => {
                let name = path.file_name().map_or(arg.clone(), |n| n.to_string_lossy().into_owned());
                let state = match turbo_pins::lookup(&hex) {
                    Some(p) if p.kind == kind => format!("pinned as {}", p.name),
                    Some(p) => format!("pinned as a {} ({}), not as a {}", p.kind.name(), p.name, kind.name()),
                    None => "not pinned".to_string(),
                };
                println!("{}  {}  {name}\t# {state}", turbo_pins::line_key(&hex), kind.name());
            }
            Err(e) => {
                eprintln!("{e}");
                failed = true;
            }
        }
    }
    if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
