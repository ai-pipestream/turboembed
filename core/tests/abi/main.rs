//! The header against the library: turbo.h compiles standalone as C11 and
//! C++17, the Rust mirrors of its structs have the C compiler's layout, and
//! a C program linked against libturbo makes a context and buffers on the
//! CPU, loads a model there, gets the upstream ids, and embeds two texts.

#[path = "../common/mod.rs"]
mod common;

mod constants;
mod header;
mod program;

use std::path::{Path, PathBuf};
use std::process::Command;

fn include() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../include")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-abi-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(cmd: &mut Command) -> String {
    let out = cmd.output().unwrap_or_else(|e| panic!("{cmd:?}: {e} (a C compiler is required)"));
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // A C program's stdout is in text mode: on Windows each \n is \r\n.
    String::from_utf8(out.stdout).unwrap().replace("\r\n", "\n")
}
