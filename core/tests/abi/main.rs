//! The header against the library: turbo.h compiles standalone as C11 and
//! C++17, the Rust mirrors of its structs have the C compiler's layout, and
//! a C program linked against libturbo makes a context and buffers on the
//! CPU, loads a model there, gets the upstream ids, and embeds two texts;
//! another returns int8 and ubinary codes and checks them itself; a third
//! calls every analyzer and analysis result function.

#[path = "../cdylib/mod.rs"]
mod cdylib;
#[path = "../common/mod.rs"]
mod common;

mod analyzer;
mod constants;
mod header;
mod program;
mod quantized;

use std::path::{Path, PathBuf};
use std::process::Command;
use turbo_scratch::Scratch;

fn include() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../include")
}

fn scratch(name: &str) -> Scratch {
    let d = Scratch::at(std::env::temp_dir().join(format!("turbo-abi-{}-{name}", std::process::id())));
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

/// `source` built as a C11 program linked against libturbo as it is now,
/// in a scratch directory, which is returned with the program's path.
fn compile(name: &str, source: &str) -> (Scratch, PathBuf) {
    let lib_dir = cdylib::build();
    let d = scratch(name);
    std::fs::write(d.join("p.c"), source).unwrap();
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(d.join("p"))
        .arg(d.join("p.c"))
        .arg("-I")
        .arg(include())
        .arg("-L")
        .arg(&lib_dir)
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-lturbo")
        .arg("-lm"));
    let program = d.join("p");
    (d, program)
}
