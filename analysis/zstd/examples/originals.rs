//! Writes the originals the test fixtures were compressed from, one file
//! each, into a directory: `cargo run -p turbo-zstd --example originals
//! -- <dir>`. `tests/data/README.md` has the `zstd` commands that make
//! the fixtures from them.

#[path = "../tests/common/mod.rs"]
mod common;

use std::path::Path;

fn main() {
    let Some(dir) = std::env::args().nth(1) else {
        eprintln!("usage: originals <dir>");
        std::process::exit(2);
    };
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir).expect("the directory");
    for (name, bytes) in common::originals() {
        let path = dir.join(name);
        std::fs::write(&path, &bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        println!("{} {}", bytes.len(), path.display());
    }
}
