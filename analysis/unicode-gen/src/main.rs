//! Generates the tables in `analysis/unicode/src/tables/` from the Unicode
//! Character Database files in `analysis/unicode/ucd/`.

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: turbo-unicode-gen <ucd directory> <output directory>");
        return ExitCode::from(2);
    }
    let out = Path::new(&args[2]);
    match turbo_unicode_gen::generate(Path::new(&args[1]), out) {
        Ok(files) => {
            for file in files {
                println!("{} ({} bytes)", out.join(file.name).display(), file.contents.len());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("turbo-unicode-gen: cannot write to {}: {e}", out.display());
            ExitCode::FAILURE
        }
    }
}
