//! #2207: a binary built with `levelzero-onednn` starts without the build
//! directory on the library path. Cargo puts every build script's output
//! directory on `LD_LIBRARY_PATH` for the tests it runs, so this test takes
//! those entries out again and runs the real `turbo-kserve` binary. Run
//! that way, a binary that links `libturbo_onednn.so` exits 127 with the
//! dynamic loader's "cannot open shared object file" before `main`. The
//! oneAPI entries (oneDNN and the SYCL runtime) are kept, as docs/levelzero.md
//! asks of a program run outside the oneAPI environment.

#![cfg(feature = "levelzero-onednn")]

use std::path::Path;
use std::process::Command;

#[test]
fn the_binary_starts_without_the_build_directory_on_the_library_path() {
    let bin = Path::new(env!("CARGO_BIN_EXE_turbo-kserve"));
    // target/<profile>/turbo-kserve: the build directories are under target.
    let target = bin.parent().and_then(Path::parent).expect("the binary is under target/<profile>");
    let kept: Vec<String> = std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|entry| !entry.is_empty() && !Path::new(entry).starts_with(target))
        .map(str::to_owned)
        .collect();
    let out = Command::new(bin)
        .env("LD_LIBRARY_PATH", kept.join(":"))
        .env_remove("TURBO_KSERVE_LISTEN")
        .env_remove("TURBO_KSERVE_MODELS")
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!err.contains("libturbo_onednn.so"), "{err}");
    // With no listen address and no models the binary prints its usage and
    // exits 2: it started.
    assert_eq!(out.status.code(), Some(2), "{err}");
}
